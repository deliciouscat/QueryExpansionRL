"""Run artifacts and lifecycle. Domain evaluation is an injected callback.

This module never constructs an agent, dataset, retriever or objective. The
integration owner provides those and chooses the normalization in LossTerm.
"""

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .backward import GradientEngine, make_optimizers, make_schedulers
from .checkpoint import (
    atomic_json,
    checkpoint_digest,
    load_checkpoint,
    read_checkpoint,
    restore_rng,
    rng_state,
    save_checkpoint,
)
from .iteration import TrainingLoop


@dataclass(frozen=True)
class EvaluationResult:
    """Integration-selected score/stop decision and a serializable evaluation report."""

    report: dict
    score: float
    stop: bool = False


class Experiment:
    def __init__(
        self,
        config,
        *,
        model,
        groups,
        stream,
        identity,
        evaluate,
        resume=None,
        init_from=None,
        stop_after=None,
    ):
        if resume and init_from:
            raise ValueError("resume and init_from are mutually exclusive")
        self.config, self.training, self.model = config, config["training"], model
        self.stream, self.identity, self.evaluate = stream, identity, evaluate
        self.stop_after = stop_after
        self.root = Path(config["output_dir"])
        if not resume and (self.root / "manifest.json").exists():
            raise FileExistsError(
                f"Run already exists: {self.root}; use --resume or new output_dir"
            )
        optimizers = make_optimizers(groups, **config.get("optimizer", {}))
        schedulers = make_schedulers(
            optimizers, self.training["max_steps"], self.training.get("warmup_ratio", 0.03)
        )
        self.engine = GradientEngine(
            model, optimizers, schedulers, self.training.get("max_grad_norm", 1.0)
        )
        self.rng = random.Random(config.get("seed", 17) + 1)
        self.extra = {"best_score": -1.0, "windows": 0}
        identity["initial_weights"] = None
        if init_from:
            initial = read_checkpoint(init_from)
            if initial["identity"]["config"]["model"] != config["model"]:
                raise ValueError("init_from requires identical model configuration")
            model.load_state_dict(initial["model"], strict=True)
            identity["initial_weights"] = checkpoint_digest(init_from)
            identity["initial_step"] = initial["global_step"]
        if resume:
            saved = read_checkpoint(resume)["identity"]
            identity["initial_weights"] = saved.get("initial_weights")
            if "initial_step" in saved:
                identity["initial_step"] = saved["initial_step"]
            self.extra = load_checkpoint(resume, model, self.engine, stream, self.rng, identity)
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_json(self.root / "manifest.json", identity)
        atomic_json(self.root / "sampling.json", stream.report())
        (self.root / "environment.lock").write_text(
            "\n".join(f"{name}=={version}" for name, version in identity["environment"].items())
            + "\n"
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.started = self.last_save = time.monotonic()
        self.stop = False

    def steps(self):
        return TrainingLoop(
            self.stream,
            self.engine,
            epochs=self.training["epochs"],
            accumulation_steps=self.training["accumulation_steps"],
            max_steps=self.training["max_steps"],
            on_boundary=self.boundary,
            should_stop=lambda: self.stop,
        )

    def evaluate_dev(self):
        rng = rng_state()
        try:
            evaluation = self.evaluate()
            report = evaluation.report
            report["global_step"] = self.engine.global_step
            report["peak_vram_bytes"] = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )
            atomic_json(self.root / f"dev-{self.engine.global_step:08d}.json", report)
            best = evaluation.score > self.extra["best_score"]
            if best:
                self.extra["best_score"] = evaluation.score
            self.stop |= evaluation.stop
            return best
        finally:
            restore_rng(rng)

    def save(self, best=False):
        path = save_checkpoint(
            self.root / "checkpoints",
            self.model,
            self.engine,
            self.stream,
            self.rng,
            self.identity,
            extra=self.extra,
            best=best,
        )
        self.last_save = time.monotonic()
        return path

    def boundary(self, metrics, save_requested=False):
        self.extra["windows"] += 1
        elapsed = time.monotonic() - self.started
        metrics.update(
            global_step=self.engine.global_step, epoch=self.stream.epoch, elapsed_seconds=elapsed
        )
        with (self.root / "metrics.jsonl").open("a") as output:
            output.write(json.dumps(metrics, allow_nan=False) + "\n")
        step = self.engine.global_step
        evaluate_every = self.training.get("eval_every", 100)
        best = False
        if metrics["updated"] and evaluate_every and step % evaluate_every == 0:
            best = self.evaluate_dev()
        time_limit = self.training.get("max_seconds")
        self.stop |= (
            step >= self.training["max_steps"]
            or self.stop_after is not None
            and step >= self.stop_after
            or time_limit is not None
            and elapsed >= time_limit
        )
        if (
            self.stop
            or save_requested
            or best
            or metrics["updated"]
            and step % self.training.get("save_every", 100) == 0
            or time.monotonic() - self.last_save >= self.training.get("save_seconds", 600)
        ):
            self.save(best)

    def finish(self):
        best = self.evaluate_dev()
        path = self.save(best)
        return {
            "global_step": self.engine.global_step,
            "checkpoint": str(path),
            "windows": self.extra["windows"],
            "best_dev_score": self.extra["best_score"],
        }
