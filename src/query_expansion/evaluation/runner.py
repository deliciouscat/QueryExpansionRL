"""Injected expansion callback -> full-corpus metrics; never inserts positives."""

import time

import numpy as np

from .metrics import aggregate, retrieval_metrics


def evaluate(records, retriever, expand, *, split="dev", language_weights=None):
    rows, generation, search = [], [], []
    started = time.perf_counter()
    for record in records:
        if record.split != split or not any(r > 0 for r in record.qrels.values()):
            continue
        baseline = retriever.search(record.query, k=100)
        start = time.perf_counter()
        output, token_count = expand(record.query, record.language)
        generation.append(time.perf_counter() - start)
        start = time.perf_counter()
        expanded = retriever.search(output.query, k=100)
        search.append(time.perf_counter() - start)
        rows.append(
            {
                "query_id": record.query_id,
                "language": record.language,
                "source": record.source,
                "fallback": output.fallback,
                "invalid": output.invalid,
                "generated_tokens": token_count,
                "baseline": retrieval_metrics([h.doc_id for h in baseline], record.qrels),
                "expanded": retrieval_metrics([h.doc_id for h in expanded], record.qrels),
            }
        )
    report = aggregate(rows, language_weights)
    report.update(
        scope="full_corpus",
        split=split,
        queries=rows,
        generation_seconds_p50_p95=np.quantile(generation, [0.5, 0.95]).tolist(),
        search_seconds_p50_p95=np.quantile(search, [0.5, 0.95]).tolist(),
        queries_per_second=len(rows) / (time.perf_counter() - started),
    )
    return report
