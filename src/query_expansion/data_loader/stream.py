"""Strategy preprocessing and resumable query sampling.

Batch inputs are {query, language} dictionaries. Labels contain supervised targets
or full Query records for reward computation, never policy inputs.
"""

import random
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Batch:
    inputs: list[dict[str, str]]
    labels: list
    save_flag: bool = False

    def __iter__(self):
        return iter((self.inputs, self.labels, self.save_flag))


class DatasetStream:
    preprocessors = {}

    @classmethod
    def register(cls, name, function):
        if name in cls.preprocessors:
            raise ValueError(f"Duplicate preprocessing strategy: {name}")
        cls.preprocessors[name] = function

    def __init__(
        self,
        data,
        strategy,
        batch_size=1,
        *,
        split="train",
        seed=17,
        language_weights=None,
        max_exposures=1,
        shuffle=True,
    ):
        if strategy not in self.preprocessors:
            raise ValueError(
                f"Unsupported strategy {strategy}; available: {list(self.preprocessors)}"
            )
        if batch_size < 1 or max_exposures < 1:
            raise ValueError("batch_size and max_exposures must be positive")
        self.strategy, self.batch_size, self.seed = strategy, batch_size, seed
        self.rows = [row for row in data if row.split == split]
        before = len(self.rows)
        self.rows = [row for row in self.rows if self.preprocessors[strategy](row) is not None]
        self.excluded = before - len(self.rows)
        if not self.rows:
            raise ValueError(f"No eligible {split} queries for {strategy}")
        self.epoch, self.position = 0, 0
        self.weights, self.max_exposures, self.shuffle = language_weights, max_exposures, shuffle
        if language_weights and any(v <= 0 for v in language_weights.values()):
            raise ValueError("Language weights must be positive")
        self.order = self._order()

    def _order(self):
        rng = random.Random(self.seed + self.epoch)
        if not self.weights:
            order = list(range(len(self.rows)))
            if self.shuffle:
                rng.shuffle(order)
            return order
        pools = defaultdict(lambda: defaultdict(list))
        for i, row in enumerate(self.rows):
            pools[row.language][row.source].extend([i] * self.max_exposures)
        # Fixed requested quotas; exhaustion is reported, not silently redistributed.
        order, total = [], sum(self.weights.values())
        for lang, weight in self.weights.items():
            sources = pools[lang]
            for pool in sources.values():
                rng.shuffle(pool)
            for _ in range(round(len(self.rows) * weight / total)):
                available = [source for source, pool in sources.items() if pool]
                if not available:
                    break
                order.append(sources[rng.choice(available)].pop())
        rng.shuffle(order)
        if not order:
            raise ValueError("Requested languages have no eligible data")
        return order

    def __iter__(self):
        while self.position < len(self.order):
            indices = self.order[self.position : self.position + self.batch_size]
            rows = [self.rows[i] for i in indices]
            self.position += len(indices)
            yield Batch(
                [{"query": r.query, "language": r.language} for r in rows],
                [self.preprocessors[self.strategy](r) for r in rows],
            )

    @property
    def at_epoch_end(self):
        return self.position == len(self.order)

    def next_epoch(self):
        if self.position != len(self.order):
            raise RuntimeError("Cannot advance an unfinished epoch")
        self.epoch += 1
        self.position = 0
        self.order = self._order()

    def state_dict(self):
        return {"epoch": self.epoch, "position": self.position, "order": self.order}

    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        if self._order() != state["order"] or not 0 <= state["position"] <= len(state["order"]):
            raise ValueError("Sampler state does not match dataset/config")
        self.order, self.position = list(state["order"]), state["position"]

    def report(self):
        counts = Counter(self.rows[i].language for i in self.order)
        exposures = Counter(self.rows[i].query_id for i in self.order)
        return {
            "target_language_weights": self.weights,
            "language_counts": dict(counts),
            "unique_queries": len(exposures),
            "exposures": dict(exposures),
            "excluded_queries": self.excluded,
        }


DatasetStream.register("sft", lambda row: row.expansion_target)
DatasetStream.register("rl", lambda row: row if any(v > 0 for v in row.qrels.values()) else None)
