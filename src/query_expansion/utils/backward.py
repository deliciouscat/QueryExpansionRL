"""Backward/update only. The caller owns every model call and the visible loop.

LossTerm is the public forward/decorator contract. A generator is consumed one
term at a time, immediately freeing saved backward activations. TrainingLoop owns
the accumulation boundary and normalizes once before clipping/updating. A wholly
skipped RL window advances neither optimizer nor scheduler. Explicitly normalized
windows remain available for low-level experiments.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from inspect import isgenerator
from numbers import Real

import torch


@dataclass(frozen=True)
class LossTerm:
    """One streaming contribution to sum(weight * loss) / sum(normalizer).

    The experiment chooses the denominator's meaning. Use normalizer=token_count
    with weight=token_count for a token-mean SFT loss. For RL, emit a loss=None
    observation with normalizer=1 per query (even a skipped group), then candidate
    losses with weight=1/G, normalizer=0. Metrics must be detached Python values:
    numeric observations are averaged, lists concatenated within the window.
    """

    loss: torch.Tensor | None = None
    weight: float = 1.0
    normalizer: float = 0.0
    metrics: dict = field(default_factory=dict)

    @classmethod
    def mean(cls, loss, count):
        return cls(loss, weight=count, normalizer=count)


class GradientEngine:
    def __init__(self, model, optimizers, schedulers=None, max_grad_norm=1.0):
        self.model, self.optimizers = model, optimizers
        self.schedulers, self.max_grad_norm = schedulers or {}, max_grad_norm
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        ids = [
            id(p)
            for opt in optimizers.values()
            for group in opt.param_groups
            for p in group["params"]
        ]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in self.parameters}:
            raise ValueError("Optimizers must partition trainable parameters exactly")
        self.global_step, self.active, self.terms, self.calls = 0, False, 0, 0
        self.last = {"updated": False}

    def zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

    @contextmanager
    def window(self, normalizer=None):
        if self.active or (
            normalizer is not None
            and (
                not isinstance(normalizer, Real) or not math.isfinite(normalizer) or normalizer <= 0
            )
        ):
            raise ValueError("Need a positive normalizer and no nested window")
        self.zero_grad()
        self.active, self.terms, self.failed = True, 0, False
        self.fixed_normalizer = normalizer
        self.normalizer, self.loss_sum = normalizer or 0.0, 0.0
        self.metric_sums, self.metric_counts, self.metric_lists = {}, {}, {}
        self.last = {"updated": False}
        try:
            yield
            if self.failed:
                raise RuntimeError("A failed loss cannot commit an optimizer window")
            if self.fixed_normalizer is None and (
                not math.isfinite(self.normalizer) or self.normalizer <= 0
            ):
                raise ValueError("A dynamic window requires a positive total normalizer")
            self.last.update(
                {key: total / self.metric_counts[key] for key, total in self.metric_sums.items()}
            )
            self.last.update(self.metric_lists)
            if self.terms:
                if self.fixed_normalizer is None:
                    for parameter in self.parameters:
                        if parameter.grad is not None:
                            parameter.grad.div_(self.normalizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    self.parameters, self.max_grad_norm, error_if_nonfinite=True
                )
                for optimizer in self.optimizers.values():
                    optimizer.step()
                for scheduler in self.schedulers.values():
                    scheduler.step()
                self.global_step += 1
                self.last.update(
                    {
                        "updated": True,
                        "loss": self.loss_sum / self.normalizer,
                        "grad_norm": float(norm),
                    }
                )
        finally:
            self.zero_grad()
            self.active = False

    def backward(self, loss, weight):
        if not self.active:
            raise RuntimeError("gradient must be called inside engine.window")
        if not isinstance(weight, Real) or not math.isfinite(weight) or weight <= 0:
            self.failed = True
            raise ValueError("Gradient weight must be finite and positive")
        if loss.ndim != 0 or not torch.isfinite(loss):
            self.failed = True
            raise FloatingPointError("Loss must be a finite scalar")
        (loss * (weight / (self.fixed_normalizer or 1.0))).backward()
        self.loss_sum += float(loss.detach()) * weight
        self.terms += 1

    def consume(self, term):
        if not self.active:
            raise RuntimeError("gradient must be called inside an accumulation window")
        if not isinstance(term, LossTerm):
            raise TypeError("forward must return a Tensor/LossTerm or yield LossTerm objects")
        if (
            not isinstance(term.normalizer, Real)
            or not math.isfinite(term.normalizer)
            or term.normalizer < 0
        ):
            raise ValueError("LossTerm normalizer must be finite and nonnegative")
        if self.fixed_normalizer is None:
            self.normalizer += term.normalizer
        for key, value in term.metrics.items():
            if key in {"updated", "loss", "grad_norm"}:
                raise ValueError(f"Reserved infrastructure metric: {key}")
            if isinstance(value, list):
                if key in self.metric_sums:
                    raise TypeError("Metric type changed within window")

                def plain(item):
                    if isinstance(item, (list, tuple)):
                        return all(plain(child) for child in item)
                    return (
                        isinstance(item, str)
                        or item is None
                        or isinstance(item, (int, float))
                        and math.isfinite(item)
                    )

                if not plain(value):
                    raise TypeError("List metrics cannot retain tensors or nonfinite values")
                self.metric_lists.setdefault(key, []).extend(value)
            elif isinstance(value, (int, float)) and math.isfinite(value):
                if key in self.metric_lists:
                    raise TypeError("Metric type changed within window")
                self.metric_sums[key] = self.metric_sums.get(key, 0.0) + value
                self.metric_counts[key] = self.metric_counts.get(key, 0) + 1
            else:
                raise TypeError("Metrics must be detached finite numbers or lists")
        if term.loss is not None:
            self.backward(term.loss, term.weight)
            return float(term.loss.detach()) * term.weight
        return 0.0


def gradient(engine):
    """Consume forward's lazy loss stream; backward finishes before the next yield.

    No optimizer step occurs here. TrainingLoop owns the accumulation boundary.
    Scalar Tensor returns remain supported for explicitly normalized windows.
    Lists of losses are rejected to avoid accidentally retaining all graphs.
    """

    def decorate(forward_block):
        @wraps(forward_block)
        def wrapped(*args, weight=1.0, **kwargs):
            stream = None
            try:
                output = forward_block(*args, **kwargs)
                if isgenerator(output):
                    if weight != 1.0:
                        raise ValueError("Set weights on yielded LossTerm objects")
                    stream = output
                    total, count = 0.0, 0
                    for term in stream:
                        total += engine.consume(term)
                        count += 1
                        del term
                    if not count:
                        raise ValueError("Empty loss stream; report skipped groups explicitly")
                else:
                    if isinstance(output, torch.Tensor):
                        output = LossTerm.mean(output, weight)
                    elif weight != 1.0:
                        raise ValueError("Set weight on the returned LossTerm")
                    total = engine.consume(output)
                engine.calls += 1
                return total
            except BaseException:
                engine.failed = True
                raise
            finally:
                if stream is not None:
                    stream.close()

        return wrapped

    return decorate


def make_optimizers(groups, *, lora_lr=1e-4, upper_lr=1e-5, weight_decay=0.01):
    return {
        "lora": torch.optim.AdamW(groups["lora"], lr=lora_lr, weight_decay=weight_decay),
        "upper": torch.optim.SGD(groups["upper"], lr=upper_lr, momentum=0, weight_decay=0),
    }


def make_schedulers(optimizers, max_steps, warmup_ratio=0.03):
    if max_steps < 1 or not 0 <= warmup_ratio < 1:
        raise ValueError("Invalid scheduler horizon/warmup")
    warmup = int(max_steps * warmup_ratio)

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0, (max_steps - step) / max(1, max_steps - warmup))

    return {name: torch.optim.lr_scheduler.LambdaLR(opt, scale) for name, opt in optimizers.items()}
