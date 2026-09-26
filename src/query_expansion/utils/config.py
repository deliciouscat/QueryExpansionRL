"""Configuration and provenance only; no domain-module imports."""

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
from pathlib import Path

import torch
import yaml

from .checkpoint import file_hash


def read_config(path):
    text = os.path.expandvars(Path(path).read_text())
    if re.search(r"\$\{[^}]+\}", text):
        raise ValueError("Unresolved environment variables in config")
    config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    return config


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def manifest(config, *, records, indexes, prompt, postprocess, entrypoint_root):
    effective = copy.deepcopy(config)
    effective.pop("output_dir", None)
    source_root = Path(__file__).resolve().parents[1]
    sources = {
        str(p.relative_to(source_root)): file_hash(p) for p in sorted(source_root.rglob("*.py"))
    }
    sources.update(
        {"train/" + p.name: file_hash(p) for p in sorted(Path(entrypoint_root).glob("*.py"))}
    )
    environment = {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "PyYAML", "query-expansion")
    }
    for name in ("transformers", "accelerate", "safetensors"):
        try:
            environment[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "config": effective,
        "config_hash": digest(effective),
        "data_hash": digest(records),
        "index_hashes": indexes,
        "prompt": prompt,
        "prompt_hash": digest(prompt),
        "postprocess": postprocess,
        "source": sources,
        "environment": environment,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "image_digest": config.get("provenance", {}).get("image_digest"),
        "data_provenance": config.get("provenance", {}).get("data"),
        "cost": config.get("cost", {"gpu_hourly_usd": None}),
    }
