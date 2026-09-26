"""One prompt and one output policy shared by SFT, RL and inference."""

import re
import unicodedata
from dataclasses import dataclass

PROMPT = "language: {language}\nquery: {query}\nexpansion:"
POSTPROCESS_VERSION = "nfkc-whitespace-v1"


def prompt(query, language):
    return PROMPT.format(language=language, query=query)


@dataclass(frozen=True)
class Completion:
    tokens: list[int]
    text: str
    ended: bool


@dataclass(frozen=True)
class Expansion:
    query: str
    text: str
    invalid: bool
    fallback: bool


@dataclass(frozen=True)
class Rollout:
    completion: Completion
    expansion: Expansion


def postprocess(query, text, ended=True):
    """Conservative single-line format guard; semantic answer detection needs review."""
    invalid = (
        not ended
        or any(c in text for c in "\n\r{}[]")
        or bool(re.search(r"<[^>]+>|^(answer|explanation|답변|설명)\s*:", text, re.I))
    )
    if invalid:
        return Expansion(query, "", True, True)

    def normalize(s):
        return unicodedata.normalize("NFKC", s).casefold()

    seen = set(normalize(query).split())
    terms = []
    for term in unicodedata.normalize("NFKC", text).split():
        key = normalize(term)
        if key not in seen:
            terms.append(term)
            seen.add(key)
    expansion = " ".join(terms)
    return Expansion(
        query + (" " + expansion if expansion else ""), expansion, False, not bool(expansion)
    )
