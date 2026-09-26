"""Reference CPU BM25 with immutable full-corpus statistics.

Replace this implementation behind Retriever for large collections. Document input
is structural (doc_id/language/title/text), not tied to a data-loader class.
"""

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float


class Retriever(Protocol):
    identity: str
    languages: dict[str, str]

    def search(self, query: str, *, candidates=None, k=100) -> list[Hit]: ...


class Analyzer:
    def __init__(self, mode="word", stopwords=(), ngram=2):
        if mode not in {"word", "character"} or ngram < 1:
            raise ValueError("Analyzer requires word/character and a positive ngram")
        self.mode, self.stopwords, self.ngram = mode, set(stopwords), ngram

    def __call__(self, text):
        text = unicodedata.normalize("NFKC", text).casefold()
        words = re.findall(r"\w+", text)
        words = [w for w in words if w not in self.stopwords]
        if self.mode == "character":
            return [
                w[i : i + self.ngram] for w in words for i in range(max(1, len(w) - self.ngram + 1))
            ]
        return words


class BM25:
    def __init__(self, documents, *, k1=1.2, b=0.75, analyzer=None):
        if k1 <= 0 or not 0 <= b <= 1 or not documents:
            raise ValueError("BM25 needs corpus, k1 > 0, 0 <= b <= 1")
        self.analyzer = Analyzer(**(analyzer or {}))
        self.k1, self.b = k1, b
        self.languages = {d.doc_id: d.language for d in documents}
        if len(self.languages) != len(documents):
            raise ValueError("Duplicate document ID")
        self.counts = {d.doc_id: Counter(self.analyzer(d.title + " " + d.text)) for d in documents}
        self.lengths = {key: sum(count.values()) for key, count in self.counts.items()}
        self.avgdl = sum(self.lengths.values()) / len(documents) or 1.0
        frequencies = Counter(term for count in self.counts.values() for term in count)
        self.idf = {
            term: math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            for term, df in frequencies.items()
        }
        payload = {
            "docs": sorted((d.doc_id, d.language, d.title, d.text) for d in documents),
            "analyzer": analyzer,
            "k1": k1,
            "b": b,
            "version": "bm25-v1",
        }
        self.identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def search(self, query, *, candidates=None, k=100):
        ids = self.counts if candidates is None else sorted(set(candidates))
        if any(doc_id not in self.counts for doc_id in ids):
            raise ValueError("Candidate absent from fixed corpus")
        terms = set(self.analyzer(query))
        hits = []
        for doc_id in ids:
            count = self.counts[doc_id]
            norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / self.avgdl)
            score = sum(
                self.idf.get(t, 0) * count[t] * (self.k1 + 1) / (count[t] + norm)
                for t in terms
                if count[t]
            )
            hits.append(Hit(doc_id, score))
        return sorted(hits, key=lambda hit: (-hit.score, hit.doc_id))[:k]
