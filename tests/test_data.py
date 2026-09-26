import json

import pytest
import torch

from query_expansion.agents import prompt
from query_expansion.data_loader import DatasetStream, Query, load_queries, synthetic_data


def test_prompt_no_evidence_and_masking(make_policy):
    agent, _ = make_policy()
    inputs = [{"query": "question", "language": "en"}]
    tokens, labels = agent.supervised(inputs, ["secret"])
    prefix = len(agent.prompt_ids(**inputs[0]))
    assert "secret" not in prompt(**inputs[0])
    assert torch.all(labels[:, :prefix] == -100)
    assert labels[0, -1] == agent.tokenizer.eos_token_id
    assert torch.equal(tokens["input_ids"][:, prefix:], labels[:, prefix:])
    agent.max_prompt_tokens = 2
    with pytest.raises(ValueError, match="prompt length"):
        agent.prompt_ids(**inputs[0])


def test_split_leakage_and_graded_qrels(tmp_path):
    path = tmp_path / "queries.jsonl"
    rows = [
        {"query_id": "q1", "query": "car", "language": "en", "split": "train"},
        {"query_id": "q2", "query": " CAR ", "language": "en", "split": "test"},
    ]
    path.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="leakage"):
        load_queries(path)
    path.write_text(json.dumps(rows[0]))
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(json.dumps({"query_id": "q1", "doc_id": "d1", "relevance": 2}))
    assert load_queries(path, qrels)[0].qrels == {"d1": 2}


def test_sampler_resume_and_exclusions():
    rows, _ = synthetic_data()
    rows.append(Query("missing", "no target", "en"))
    stream = DatasetStream(rows, "sft", batch_size=3)
    iterator = iter(stream)
    next(iterator)
    state = stream.state_dict()
    next_inputs = next(iterator).inputs
    restored = DatasetStream(rows, "sft", batch_size=3)
    restored.load_state_dict(state)
    assert next(iter(restored)).inputs == next_inputs
    assert stream.excluded == 1
    with pytest.raises(ValueError, match="Unsupported"):
        DatasetStream(rows, "dpo")


def test_language_shortfall_is_not_unlimited_duplication():
    rows, _ = synthetic_data()
    stream = DatasetStream(rows, "sft", language_weights={"ko": 0.8, "en": 0.1, "de": 0.1})
    report = stream.report()
    assert report["language_counts"] == {"ko": 50, "en": 10}
    assert max(report["exposures"].values()) == 1
