"""Evaluation owns its metric implementation independently from training rewards."""

import math
from collections import defaultdict

import numpy as np


def retrieval_metrics(ranking, qrels):
    ranking = list(dict.fromkeys(ranking))
    positives = {d for d, r in qrels.items() if r > 0}
    if not positives:
        raise ValueError("Cannot evaluate query without positive qrels")

    def dcg(values):
        return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(values))

    ideal = dcg(sorted(qrels.values(), reverse=True)[:10])
    return {
        "ndcg@10": dcg([qrels.get(d, 0) for d in ranking[:10]]) / ideal,
        "mrr@10": next((1 / (i + 1) for i, d in enumerate(ranking[:10]) if d in positives), 0.0),
        "recall@100": len(positives & set(ranking[:100])) / len(positives),
    }


def paired_bootstrap(deltas, seed=17, samples=1000):
    values = np.asarray(deltas)
    if len(values) == 0:
        raise ValueError("Bootstrap needs paired observations")
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)]
    return np.quantile(means, [0.025, 0.975]).tolist()


def aggregate(rows, language_weights=None):
    def mean(subset, field):
        return {
            metric: sum(row[field][metric] for row in subset) / len(subset)
            for metric in ("ndcg@10", "mrr@10", "recall@100")
        }

    if not rows:
        raise ValueError("No evaluation queries")
    by_language, by_source = defaultdict(list), defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row)
        by_source[row["source"]].append(row)

    def groups(mapping):
        return {
            key: {field: mean(group, field) for field in ("baseline", "expanded")}
            for key, group in mapping.items()
        }

    languages, sources = groups(by_language), groups(by_source)
    macro = {
        field: {
            metric: sum(s[field][metric] for s in sources.values()) / len(sources)
            for metric in ("ndcg@10", "mrr@10", "recall@100")
        }
        for field in ("baseline", "expanded")
    }
    deltas = [r["expanded"]["ndcg@10"] - r["baseline"]["ndcg@10"] for r in rows]
    weighted = None
    if language_weights and set(language_weights) <= languages.keys():
        weighted = {
            field: {
                metric: sum(
                    languages[lang][field][metric] * weight
                    for lang, weight in language_weights.items()
                )
                / sum(language_weights.values())
                for metric in ("ndcg@10", "mrr@10", "recall@100")
            }
            for field in ("baseline", "expanded")
        }
    return {
        "by_language": languages,
        "by_source": sources,
        "macro": macro,
        "language_weighted": weighted,
        "paired_ndcg_gain_ci95": paired_bootstrap(deltas),
        "fallback_rate": sum(r["fallback"] for r in rows) / len(rows),
        "invalid_rate": sum(r["invalid"] for r in rows) / len(rows),
        "degradation_rate": sum(d < 0 for d in deltas) / len(rows),
    }
