import json

import pytest
import torch

from query_expansion.contracts import Record
from query_expansion.data_loader import dataloader, load_records
from query_expansion.data_loader.encoding import Encoder, prompt
from query_expansion.models.tiny import ByteTokenizer
from query_expansion.utils.registry import Registry


def test_prompt_has_no_answers_or_labels_and_eos_masking():
    row = Record("1", "query", "en", expansion_target="secret", chosen="hidden")
    assert "secret" not in prompt(row) and "hidden" not in prompt(row)
    encoder = Encoder(ByteTokenizer())
    inputs, labels = encoder.supervised([row], [row.expansion_target])
    n = len(encoder.prompt_ids(row))
    assert torch.all(labels[:, :n] == -100)
    assert labels[0, -1] == ByteTokenizer.eos_token_id
    assert torch.equal(inputs["input_ids"][:, n:], labels[:, n:])


def test_unknown_and_duplicate_registry_names():
    registry = Registry("test")
    registry.register("one")(object())
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register("one")(object())
    with pytest.raises(ValueError, match="available"):
        registry.get("unknown")


def test_strategy_schema_errors_and_query_length():
    with pytest.raises(ValueError, match="expansion_target"):
        dataloader(
            records=[Record("1", "q", "en")],
            batch_size=1,
            strategy="sft",
            tokenizer=ByteTokenizer(),
        )
    with pytest.raises(ValueError, match="prompt length"):
        Encoder(ByteTokenizer(), max_prompt_tokens=2).prompt_ids(Record("1", "q", "en"))


def test_split_duplicate_detection(tmp_path):
    path = tmp_path / "rows.jsonl"
    row = {"query_id": "1", "query": "q", "language": "en", "split": "test"}
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "split": "train"}))
    assert len(load_records(path=path)) == 1
    path.write_text((json.dumps({**row, "split": "train"}) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_records(path=path)


def test_loader_order_resume_does_not_consume_global_rng():
    records = [Record(str(i), str(i), "en", expansion_target=str(i)) for i in range(7)]
    kwargs = dict(records=records, batch_size=2, strategy="sft", tokenizer=ByteTokenizer(), seed=9)
    before = torch.get_rng_state().clone()
    full = list(dataloader(**kwargs))
    resumed = list(dataloader(**kwargs, start_batch=2))
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(full[2].inputs["input_ids"], resumed[0].inputs["input_ids"])
