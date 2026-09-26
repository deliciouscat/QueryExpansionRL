import copy
from pathlib import Path

import pytest
import torch

from query_expansion.export import export
from query_expansion.train import run
from query_expansion.utils.checkpoint import read_checkpoint
from query_expansion.utils.config import read_config

ROOT = Path(__file__).resolve().parents[1]


def config_for(tmp_path, name, strategy="sft"):
    config = read_config(ROOT / f"configs/smoke_{strategy}.yaml")
    config["output_dir"] = str(tmp_path / name)
    config["training"].update(max_steps=3, accumulation_steps=2, epochs=2, eval_every=0)
    return config


def test_exact_sft_resume_and_manifest_guard(tmp_path):
    config = config_for(tmp_path, "whole")
    # Cross an epoch with a partial accumulation window (64 + 36 queries).
    config["training"]["batch_size"] = 32
    whole = read_checkpoint(run(config)["checkpoint"])
    config["output_dir"] = str(tmp_path / "resume")
    first = run(config, stop_after=1)
    actual = read_checkpoint(run(config, resume=first["checkpoint"])["checkpoint"])
    assert whole["global_step"] == actual["global_step"] == 3
    assert whole["sampler"] == actual["sampler"]
    assert whole["schedulers"] == actual["schedulers"]
    assert torch.equal(whole["rng"]["torch"], actual["rng"]["torch"])
    for key, value in whole["model"].items():
        assert torch.equal(value, actual["model"][key]), key
    changed = copy.deepcopy(config)
    changed["optimizer"]["upper_lr"] = 0.7
    with pytest.raises(ValueError, match="manifest differs"):
        run(changed, resume=first["checkpoint"])
    assert "train/sft.py" in whole["identity"]["source"]
    assert "agents/policy.py" in whole["identity"]["source"]


def test_rl_exact_resume_and_full_model_export(tmp_path):
    sft = run(config_for(tmp_path, "sft"))
    config = config_for(tmp_path, "rl", "rl")
    whole = read_checkpoint(run(config, init_from=sft["checkpoint"])["checkpoint"])
    assert whole["global_step"] == 3  # Actual updates, not just attempted groups.
    config["output_dir"] = str(tmp_path / "rl-resume")
    first = run(config, init_from=sft["checkpoint"], stop_after=1)
    result = run(config, resume=first["checkpoint"])
    actual = read_checkpoint(result["checkpoint"])
    assert whole["environment_rng"] == actual["environment_rng"]
    assert whole["sampler"] == actual["sampler"]
    for key, value in whole["model"].items():
        assert torch.equal(value, actual["model"][key]), key
    output = tmp_path / "export"
    report = export(result["checkpoint"], output)
    assert report["max_logit_error"] < 1e-4
    assert (output / "COMPLETE.json").exists()


def test_config_and_completion_marker(tmp_path, monkeypatch):
    monkeypatch.delenv("QE_MISSING", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text("data: ${QE_MISSING}")
    with pytest.raises(ValueError, match="environment variables"):
        read_config(path)
    with pytest.raises(ValueError, match="completion marker"):
        read_checkpoint(tmp_path)
