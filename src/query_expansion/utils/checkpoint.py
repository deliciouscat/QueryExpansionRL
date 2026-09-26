"""Atomic, optimizer-boundary snapshots including both optimizer and RNG states.

Only load trusted local checkpoints: torch serialization is not a safe interchange
format. A completed directory is atomically published before latest.json changes.
"""

import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_checkpoint(
    root, model, engine, sampler, environment_rng, identity, *, extra=None, best=False
):
    if engine.active:
        raise RuntimeError("Checkpoint requires optimizer-step boundary")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Full model + conservative allowance for Adam moments, optimizer tensors and next write.
    estimate = sum(t.numel() * t.element_size() for t in model.state_dict().values()) * 5
    if shutil.disk_usage(root).free < estimate:
        raise OSError("Insufficient space for a complete checkpoint")
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    payload = {
        "format": 1,
        "model": model.state_dict(),
        "optimizers": {k: o.state_dict() for k, o in engine.optimizers.items()},
        "schedulers": {k: s.state_dict() for k, s in engine.schedulers.items()},
        "global_step": engine.global_step,
        "sampler": sampler.state_dict(),
        "environment_rng": environment_rng.getstate(),
        "rng": rng_state(),
        "identity": identity,
        "extra": extra or {},
    }
    try:
        with (temporary / "state.pt").open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(temporary / "COMPLETE.json", {"sha256": file_hash(temporary / "state.pt")})
        destination = root / f"step-{engine.global_step:08d}-{temporary.name[9:]}"
        os.replace(temporary, destination)
        atomic_json(root / "latest.json", {"checkpoint": destination.name})
        if best:
            atomic_json(root / "best.json", {"checkpoint": destination.name})
        # Retain newest two plus dev-best. Never remove another directory type.
        checkpoints = sorted(root.glob("step-*/COMPLETE.json"), key=lambda p: p.stat().st_mtime_ns)
        # A boundary save followed by final evaluation may publish the same step
        # twice. Keep two distinct optimizer steps, not two copies of one step.
        keep, steps = set(), set()
        for marker in reversed(checkpoints):
            step = marker.parent.name.split("-")[1]
            if step not in steps and len(steps) < 2:
                keep.add(marker.parent.name)
                steps.add(step)
        if (root / "best.json").exists():
            keep.add(json.loads((root / "best.json").read_text())["checkpoint"])
        for marker in checkpoints:
            if marker.parent.name not in keep:
                shutil.rmtree(marker.parent)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def resolve_checkpoint(path):
    path = Path(path)
    if path.is_file() and path.suffix == ".json":
        return path.parent / json.loads(path.read_text())["checkpoint"]
    if not (path / "COMPLETE.json").exists() and (path / "latest.json").exists():
        return resolve_checkpoint(path / "latest.json")
    return path


def checkpoint_digest(path):
    return json.loads((resolve_checkpoint(path) / "COMPLETE.json").read_text())["sha256"]


def read_checkpoint(path):
    path = resolve_checkpoint(path)
    marker = path / "COMPLETE.json"
    if not marker.exists():
        raise ValueError("Checkpoint has no completion marker")
    if file_hash(path / "state.pt") != json.loads(marker.read_text())["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    return torch.load(path / "state.pt", map_location="cpu", weights_only=False)


def load_checkpoint(path, model, engine, sampler, environment_rng, identity):
    payload = read_checkpoint(path)
    if payload["identity"] != identity:
        raise ValueError("Checkpoint manifest differs; use init_from for a new run")
    model.load_state_dict(payload["model"], strict=True)
    for name, optimizer in engine.optimizers.items():
        optimizer.load_state_dict(payload["optimizers"][name])
    for name, scheduler in engine.schedulers.items():
        scheduler.load_state_dict(payload["schedulers"][name])
    sampler.load_state_dict(payload["sampler"])
    environment_rng.setstate(payload["environment_rng"])
    engine.global_step = payload["global_step"]
    restore_rng(payload["rng"])
    return payload["extra"]
