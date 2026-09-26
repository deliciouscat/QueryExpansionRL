import copy
from pathlib import Path

import pytest
import torch

from query_expansion.train import run
from query_expansion.utils.checkpoint import read_checkpoint
from query_expansion.utils.config import read_config

ROOT = Path(__file__).resolve().parents[1]


def test_cli_resume_and_stage_initialization(tmp_path, monkeypatch):
    from query_expansion.utils.trainer import Trainer

    def hidden_loop(*args, **kwargs):
        raise AssertionError("CLI must execute the public for loop, not Trainer.fit")

    monkeypatch.setattr(Trainer, "fit", hidden_loop)
    config = read_config(ROOT / "configs/smoke_sft.yaml")
    config["output_dir"] = str(tmp_path / "uninterrupted")
    config["training"]["max_seconds"] = None
    run(config)
    expected = read_checkpoint(Path(config["output_dir"]) / "latest.pt")
    # Moving config into utils must not narrow provenance to just utils/*.py.
    source = expected["identity"]["manifest"]["source"]
    assert {
        "train.py",
        "utils/experiment.py",
        "contracts/strategy.py",
        "strategies/sft.py",
        "retriever/bm25.py",
        "utils/trainer.py",
    } <= source.keys()
    config["output_dir"] = str(tmp_path / "resumed")
    state = run(config, stop_after=1)
    assert state["global_step"] == 1
    checkpoint = Path(config["output_dir"]) / "latest.pt"
    run(config, resume=checkpoint)
    actual = read_checkpoint(checkpoint)
    assert actual["state"] == expected["state"]
    for key, value in expected["model"].items():
        assert torch.equal(value, actual["model"][key]), key
    for key in expected["schedulers"]:
        assert expected["schedulers"][key] == actual["schedulers"][key]
    dpo = read_config(ROOT / "configs/smoke_dpo.yaml")
    dpo["output_dir"] = str(tmp_path / "dpo")
    run(dpo, init_from=checkpoint, stop_after=1)
    dpo_checkpoint = Path(dpo["output_dir"]) / "latest.pt"
    first = read_checkpoint(dpo_checkpoint)
    run(dpo, resume=dpo_checkpoint)
    second = read_checkpoint(dpo_checkpoint)
    assert second["state"]["global_step"] == 2
    assert second["identity"]["manifest"]["initial_checkpoint_sha256"]
    for key, value in first["strategy"]["reference"].items():
        assert torch.equal(value, second["strategy"]["reference"][key])
    changed = copy.deepcopy(dpo)
    changed["optimizer"]["upper_lr"] = 0.7
    with pytest.raises(ValueError, match="manifest differs"):
        run(changed, resume=dpo_checkpoint)


def test_config_rejects_unset_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("QE_MISSING_TEST_VAR", raising=False)
    file = tmp_path / "config.yaml"
    file.write_text("data: ${QE_MISSING_TEST_VAR}")
    with pytest.raises(ValueError, match="environment variables"):
        read_config(file)


def test_custom_dataset_and_criterion_plugin(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    csv = tmp_path / "train.csv"
    csv.write_text("id,question,language,expansion,split\n1,car,en,battery,train\n")
    config = read_config(ROOT / "configs/smoke_sft.yaml")
    config["plugins"] = ["custom_experiment"]
    config["data"].update(name="qa_csv", path=str(csv))
    config["strategy"] = {"name": "scaled_sft"}
    config["output_dir"] = str(tmp_path / "plugin-run")
    config["training"]["max_steps"] = 1
    assert run(config)["global_step"] == 1
    payload = read_checkpoint(Path(config["output_dir"]) / "latest.pt")
    assert payload["identity"]["manifest"]["plugins"]


def test_experiment_decorator_finalizes_short_loop_and_preserves_checkpoint_on_error(tmp_path):
    from query_expansion.utils.config import file_hash
    from query_expansion.utils.execution import experiment

    @experiment
    def research(train, dataloader, criterion):
        @train
        def objective(outputs, labels):
            return criterion(outputs, labels)

        for inputs, labels in dataloader:
            objective(inputs, labels)
            break

    config = read_config(ROOT / "configs/smoke_sft.yaml")
    config["output_dir"] = str(tmp_path / "research")
    state = research(config)
    assert state["global_step"] == state["next_batch"] == 1
    checkpoint = Path(config["output_dir"]) / "latest.pt"
    before = file_hash(checkpoint)

    @experiment
    def broken(train, dataloader, criterion):
        @train
        def objective(outputs, labels):
            raise ValueError("failed loss")

        for inputs, labels in dataloader:
            objective(inputs, labels)

    with pytest.raises(ValueError, match="failed loss"):
        broken(config, resume=checkpoint)
    assert file_hash(checkpoint) == before
