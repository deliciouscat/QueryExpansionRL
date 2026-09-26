"""Normalized data owned by the data team; no model or retriever dependencies."""

import json
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Query:
    query_id: str
    query: str
    language: str
    source: str = "local"
    split: str = "train"
    answers: tuple[str, ...] = ()
    positive_doc_ids: tuple[str, ...] = ()
    negative_doc_ids: tuple[str, ...] = ()
    expansion_target: str | None = None
    group_id: str | None = None
    qrels: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Document:
    doc_id: str
    language: str
    text: str
    title: str = ""


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_queries(path, qrels_path=None):
    rows = [Query(**row) for row in read_jsonl(path)]
    by_id = {row.query_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate query_id")
    if qrels_path:
        for rel in read_jsonl(qrels_path):
            if rel["query_id"] not in by_id:
                raise ValueError(f"qrels refers to unknown query: {rel['query_id']}")
            labels = by_id[rel["query_id"]].qrels
            if rel["doc_id"] in labels:
                raise ValueError("Duplicate qrel")
            labels[rel["doc_id"]] = float(rel["relevance"])
    seen = {}
    for row in rows:
        if not row.query.strip() or not row.language or not row.query_id:
            raise ValueError("Query requires nonempty ID, language and query")
        for doc_id in row.positive_doc_ids:
            if doc_id in row.qrels and row.qrels[doc_id] <= 0:
                raise ValueError("positive_doc_ids conflicts with qrels")
            row.qrels.setdefault(doc_id, 1.0)
        if any(not math.isfinite(v) or v < 0 for v in row.qrels.values()):
            raise ValueError("Relevance must be finite and nonnegative")
        normalized = " ".join(unicodedata.normalize("NFKC", row.query).casefold().split())
        for key in [("text", normalized), ("group", row.group_id)]:
            if key[1] is None:
                continue
            if key in seen and seen[key] != row.split:
                raise ValueError("Query/group leakage across splits")
            seen[key] = row.split
    return rows


def load_corpus(path):
    rows = [Document(**row) for row in read_jsonl(path)]
    if len({row.doc_id for row in rows}) != len(rows):
        raise ValueError("Duplicate doc_id")
    return rows


def synthetic_data():
    """100 training queries, disjoint dev/test queries; never evidence for model quality."""
    docs = [
        Document("toy:battery", "en", "car battery cold winter range"),
        Document("toy:engine", "en", "car engine fuel petrol"),
        Document("toy:ko", "ko", "전기차 겨울 배터리 저온 난방"),
        Document("toy:ko-noise", "ko", "자동차 엔진 휘발유"),
    ]
    rows = []
    for split, count in [("train", 100), ("dev", 4), ("test", 4)]:
        for i in range(count):
            ko = i % 2 == 0
            positive = "toy:ko" if ko else "toy:battery"
            rows.append(
                Query(
                    f"toy:{split}:{i}",
                    f"전기차 겨울 {split} {i}" if ko else f"car winter {split} {i}",
                    "ko" if ko else "en",
                    source="toy",
                    split=split,
                    expansion_target="배터리" if ko else "battery",
                    positive_doc_ids=(positive,),
                    qrels={positive: 1.0},
                )
            )
    return rows, docs
