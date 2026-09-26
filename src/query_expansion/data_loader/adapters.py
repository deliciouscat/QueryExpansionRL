"""Adapter registry: add a dataset here without changing the training entrypoint.

Adapters return (queries, corpus). Dataset-specific validation, licensing and
stable evidence-ID mapping belong here; no answer-to-fake-document conversion.
"""

from .schema import load_corpus, load_queries, synthetic_data

ADAPTERS = {}


def register_adapter(name, adapter):
    if name in ADAPTERS:
        raise ValueError(f"Duplicate data adapter: {name}")
    ADAPTERS[name] = adapter


def load_dataset(config):
    name = config.get("adapter", "synthetic")
    if name not in ADAPTERS:
        raise ValueError(f"Unknown data adapter {name}; available: {list(ADAPTERS)}")
    return ADAPTERS[name]({key: value for key, value in config.items() if key != "adapter"})


def jsonl_adapter(config):
    return load_queries(config["queries"], config.get("qrels")), load_corpus(config["corpus"])


register_adapter("synthetic", lambda config: synthetic_data())
register_adapter("jsonl", jsonl_adapter)
