"""Standalone merged-policy export and reload (no training utilities required)."""

import copy
import json
from pathlib import Path

import torch

from .lora import merge_lora
from .policy import Agents
from .text import POSTPROCESS_VERSION, PROMPT
from .tiny import ByteTokenizer, TinyLM


def export_policy(agent, model_config, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model = merge_lora(copy.deepcopy(agent.model).cpu()).eval()
    metadata = {"model": model_config, "prompt": PROMPT, "postprocess": POSTPROCESS_VERSION}
    if model_config["backend"] == "tiny":
        torch.save(model.state_dict(), path / "model.pt")
        (path / "tokenizer.json").write_text(json.dumps({"kind": "utf8-byte-v1", "eos": 1}))
    else:
        model.save_pretrained(path, safe_serialization=True)
        agent.tokenizer.save_pretrained(path)
    (path / "policy.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return load_export(path)


def load_export(path, device="cpu"):
    path = Path(path)
    metadata = json.loads((path / "policy.json").read_text())
    if metadata["prompt"] != PROMPT or metadata["postprocess"] != POSTPROCESS_VERSION:
        raise ValueError("Export prompt/postprocess version differs from runtime")
    config = metadata["model"]
    policy = object.__new__(Agents)
    policy.device = torch.device(device)
    policy.max_prompt_tokens = config.get("max_prompt_tokens", 256)
    policy.max_new_tokens = config.get("max_new_tokens", 64)
    if config["backend"] == "tiny":
        policy.model = TinyLM(config.get("width", 16))
        policy.model.load_state_dict(torch.load(path / "model.pt", weights_only=True))
        policy.tokenizer = ByteTokenizer()
    else:
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM

        policy.model = Qwen3_5ForCausalLM.from_pretrained(path)
        policy.tokenizer = AutoTokenizer.from_pretrained(path)
        if policy.tokenizer.pad_token_id is None:
            policy.tokenizer.pad_token_id = policy.tokenizer.eos_token_id
    policy.model.to(device).eval()
    return policy
