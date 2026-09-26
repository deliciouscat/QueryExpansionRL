"""Merge full trained policy, reload and verify logits, outputs and retrieval."""

import argparse
from dataclasses import asdict

import torch

from .agents import Agents, postprocess
from .agents.artifacts import export_policy
from .utils.checkpoint import atomic_json, read_checkpoint
from .utils.config import digest


def export(checkpoint, output):
    from train.common import load_data

    from .evaluation import evaluate
    from .retriever import BM25

    payload = read_checkpoint(checkpoint)
    config = payload["identity"]["config"]
    agent = Agents(**dict(config["model"], device="cpu"))
    agent.model.load_state_dict(payload["model"])
    agent.model.eval()
    records, corpus = load_data(config)
    if digest([asdict(row) for row in records]) != payload["identity"]["data_hash"]:
        raise ValueError("Evaluation query/qrels data changed")
    index = BM25(corpus, **config.get("retrievers", [{}])[0])
    if index.identity != payload["identity"]["index_hashes"][0]:
        raise ValueError("Export verification corpus changed")
    reloaded = export_policy(agent, config["model"], output)
    probes = [r for r in records if r.split == "dev"][:4]
    if not probes:
        raise ValueError("Export verification requires fixed dev queries")
    max_error = 0.0
    with torch.no_grad():
        for record in probes:
            ids = torch.tensor([agent.prompt_ids(record.query, record.language)])
            before, after = agent.model(ids).logits, reloaded.model(ids).logits
            tolerance = 0.05 if before.dtype == torch.bfloat16 else 1e-5
            torch.testing.assert_close(before, after, atol=tolerance, rtol=tolerance)
            max_error = max(max_error, float((before - after).abs().max()))
            first = agent.generate(record.query, record.language, sample=False)
            second = reloaded.generate(record.query, record.language, sample=False)
            if first.tokens != second.tokens:
                raise ValueError("Merged/reloaded greedy outputs differ; export incomplete")
            q1 = postprocess(record.query, first.text, first.ended).query
            q2 = postprocess(record.query, second.text, second.ended).query
            if index.search(q1) != index.search(q2):
                raise ValueError("Export retrieval verification failed")

    def expand(query, language):
        completion = reloaded.generate(query, language, sample=False)
        return postprocess(query, completion.text, completion.ended), len(completion.tokens)

    atomic_json(str(output) + "/evaluation.json", evaluate(records, index, expand, split="dev"))
    atomic_json(str(output) + "/manifest.json", payload["identity"])
    atomic_json(
        str(output) + "/COMPLETE.json",
        {"max_logit_error": max_error, "verified_queries": len(probes)},
    )
    return {"output": str(output), "max_logit_error": max_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
