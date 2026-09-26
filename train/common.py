"""Integration helpers: data preparation, provenance and evaluation wiring only."""

from dataclasses import asdict
from pathlib import Path

import torch

from query_expansion.agents import POSTPROCESS_VERSION, PROMPT, postprocess
from query_expansion.data_loader import load_dataset
from query_expansion.evaluation import evaluate
from query_expansion.utils.checkpoint import seed_everything
from query_expansion.utils.config import manifest
from query_expansion.utils.experiment import EvaluationResult, Experiment


def load_data(config):
    return load_dataset(config.get("data", {"adapter": "synthetic"}))


def prepare(config):
    """Seed before explicit model construction in the experiment file."""
    training = config["training"]
    if min(training["max_steps"], training["epochs"], training["accumulation_steps"]) < 1:
        raise ValueError("Training limits must be positive")
    seed_everything(config.get("seed", 17))
    if config.get("num_threads"):
        torch.set_num_threads(config["num_threads"])
    return load_data(config)


def configure(config, model, stream, records, indexes, **execution):
    identity = manifest(
        config,
        records=[asdict(row) for row in records],
        indexes=[index.identity for index in indexes],
        prompt=PROMPT,
        postprocess=POSTPROCESS_VERSION,
        entrypoint_root=Path(__file__).parent,
    )
    identity["strategy"] = stream.strategy

    def expand(query, language):
        completion = model.generate(query, language, sample=False)
        return postprocess(query, completion.text, completion.ended), len(completion.tokens)

    def evaluate_dev():
        report = evaluate(
            records,
            indexes[0],
            expand,
            split="dev",
            language_weights=config.get("sampling", {}).get("language_weights"),
        )

        score = report["macro"]["expanded"]["ndcg@10"]
        safety = config.get("early_stop", {})
        stop = (
            "min_dev_ndcg" in safety
            and score < safety["min_dev_ndcg"]
            or "max_invalid_rate" in safety
            and report["invalid_rate"] > safety["max_invalid_rate"]
        )
        return EvaluationResult(report, score, stop)

    return Experiment(
        config,
        model=model.model,
        groups=model.groups,
        stream=stream,
        identity=identity,
        evaluate=evaluate_dev,
        **execution,
    )
