"""Rank-gain reward; only the Retriever public API crosses module boundaries."""

import math
from dataclasses import dataclass

from query_expansion.retriever import Retriever

from .objectives import group_advantages


def ndcg(ranking, qrels, k=10):
    def dcg(values):
        return sum((2**value - 1) / math.log2(i + 2) for i, value in enumerate(values))

    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    return dcg([qrels.get(doc, 0) for doc in ranking[:k]]) / ideal if ideal else 0.0


@dataclass(frozen=True)
class RewardContext:
    environments: tuple[int, ...]
    candidates: tuple[tuple[str, ...], ...]
    baselines: tuple[float, ...]


@dataclass(frozen=True)
class RewardGroup:
    advantages: object
    metrics: dict

    @property
    def active(self):
        return bool(self.advantages.any())


class RetrievalReward:
    def __init__(
        self,
        retrievers: list[Retriever],
        *,
        negatives=255,
        length_penalty=0.02,
        invalid_penalty=0.10,
        environments_per_group=None,
    ):
        if not retrievers or negatives < 0:
            raise ValueError("Need retrievers and nonnegative negative count")
        self.retrievers, self.negatives = retrievers, negatives
        self.length_penalty, self.invalid_penalty = length_penalty, invalid_penalty
        self.environments_per_group = environments_per_group or len(retrievers)
        if not 1 <= self.environments_per_group <= len(retrievers):
            raise ValueError("Invalid environment sample size")

    def validate(self, record):
        positives = {d for d, r in record.qrels.items() if r > 0}
        if not positives:
            raise ValueError("Query has no positive qrels")
        for index in self.retrievers:
            if not positives <= index.languages.keys():
                raise ValueError("Fixed corpus is missing positive documents")

    def prepare(self, record, rng):
        self.validate(record)
        environments = tuple(
            sorted(rng.sample(range(len(self.retrievers)), self.environments_per_group))
        )
        positives = {d for d, r in record.qrels.items() if r > 0}
        candidates, baselines = [], []
        for environment in environments:
            index = self.retrievers[environment]
            eligible = {
                d
                for d, lang in index.languages.items()
                if lang == record.language and d not in positives
            }
            hard = [
                h.doc_id
                for h in index.search(
                    record.query, candidates=eligible, k=(self.negatives + 1) // 2
                )
            ]
            rest = sorted(eligible - set(hard))
            random_docs = rng.sample(rest, min(len(rest), self.negatives - len(hard)))
            docs = tuple(sorted(positives | set(hard) | set(random_docs)))
            baseline = index.search(record.query, candidates=docs, k=10)
            candidates.append(docs)
            baselines.append(ndcg([h.doc_id for h in baseline], record.qrels))
        return RewardContext(environments, tuple(candidates), tuple(baselines))

    def score_group(self, record, rollouts, rng):
        """Structural rollout input: completion.tokens and expansion.{query,invalid,fallback}.

        No model import/callback: every candidate shares this one sampled context.
        Metrics are one observation per query, including constant-reward groups.
        """
        context = self.prepare(record, rng)
        scores, invalid, fallback = [], 0, 0
        for rollout in rollouts:
            completion, expansion = rollout.completion, rollout.expansion
            score, _ = self(
                record,
                expanded_query=expansion.query,
                generated_tokens=len(completion.tokens),
                invalid=expansion.invalid,
                context=context,
            )
            scores.append(score)
            invalid += expansion.invalid
            fallback += expansion.fallback
        advantages = group_advantages(scores)
        return RewardGroup(
            advantages,
            {
                "reward": sum(scores) / len(scores),
                "zero_advantage_rate": float(not advantages.any()),
                "invalid_rate": invalid / len(scores),
                "fallback_rate": fallback / len(scores),
                "candidate_counts": list(map(len, context.candidates)),
                "environments": [context.environments],
            },
        )

    def __call__(self, record, *, expanded_query, generated_tokens, invalid, context):
        gains = []
        for env, candidates, baseline in zip(
            context.environments, context.candidates, context.baselines, strict=True
        ):
            hits = self.retrievers[env].search(expanded_query, candidates=candidates, k=10)
            gains.append(ndcg([hit.doc_id for hit in hits], record.qrels) - baseline)
        gain = sum(gains) / len(gains)
        value = (
            gain
            - self.length_penalty * min(generated_tokens / 64, 1)
            - self.invalid_penalty * int(invalid)
        )
        return value, {
            "gain": gain,
            "candidate_counts": list(map(len, context.candidates)),
            "environments": context.environments,
        }
