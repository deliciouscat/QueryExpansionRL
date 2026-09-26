"""Evaluate a selected checkpoint once on dev or test, over the entire corpus."""

import argparse
from dataclasses import asdict

from .agents import Agents, postprocess
from .evaluation import evaluate
from .utils.checkpoint import atomic_json, read_checkpoint
from .utils.config import digest


def main():
    from train.common import load_data

    from .retriever import BM25

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    payload = read_checkpoint(args.checkpoint)
    config = payload["identity"]["config"]
    records, corpus = load_data(config)
    if digest([asdict(row) for row in records]) != payload["identity"]["data_hash"]:
        raise ValueError("Evaluation query/qrels data changed")
    agent = Agents(**dict(config["model"], device=args.device))
    agent.model.load_state_dict(payload["model"])
    index = BM25(corpus, **config.get("retrievers", [{}])[0])
    if index.identity != payload["identity"]["index_hashes"][0]:
        raise ValueError("Evaluation corpus/index changed")

    def expand(query, language):
        completion = agent.generate(query, language, sample=False)
        return postprocess(query, completion.text, completion.ended), len(completion.tokens)

    report = evaluate(
        records,
        index,
        expand,
        split=args.split,
        language_weights=config.get("sampling", {}).get("language_weights"),
    )
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
