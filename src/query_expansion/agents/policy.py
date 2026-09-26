"""Policy facade: encoding, generation and differentiable completion scoring.

Only query/language enter prompt_ids. Evidence never reaches this API.
"""

import re
from pathlib import Path

import torch
from torch import nn

from .lora import configure_parameters
from .text import Completion, Rollout, postprocess, prompt
from .tiny import ByteTokenizer, TinyLM


class Agents:
    def __init__(
        self,
        backend="tiny",
        name="Qwen/Qwen3.5-2B-Base",
        revision=None,
        device="cpu",
        dtype="float32",
        width=16,
        rank=16,
        alpha=32,
        max_prompt_tokens=256,
        max_new_tokens=64,
        gradient_checkpointing=True,
    ):
        if max_prompt_tokens < 1 or not 1 <= max_new_tokens <= 64:
            raise ValueError("Require positive prompt limit and 1..64 output tokens")
        self.max_prompt_tokens, self.max_new_tokens = max_prompt_tokens, max_new_tokens
        self.device = torch.device(device)
        if backend == "tiny":
            self.tokenizer, self.model = ByteTokenizer(), TinyLM(width)
            layers, split = self.model.layers, 2
        elif backend == "qwen":
            from transformers import AutoTokenizer, Qwen3_5ForCausalLM

            if not revision or (
                not Path(name).exists() and not re.fullmatch(r"[0-9a-f]{40}", revision)
            ):
                raise ValueError("Qwen requires a pinned model revision")
            self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
            self.model = Qwen3_5ForCausalLM.from_pretrained(
                name, revision=revision, dtype=getattr(torch, dtype)
            )
            layers, split = self.model.model.layers, 12
            if len(layers) != 24:
                raise ValueError("Expected 24 text decoder layers")
            if self.model.lm_head.weight is not self.model.model.embed_tokens.weight:
                raise ValueError("Expected tied input embedding and LM head")
        else:
            raise ValueError(f"Unknown model backend: {backend}")
        self.model.to(self.device)
        self.groups = configure_parameters(self.model, layers, split, rank, alpha)
        if backend == "qwen" and gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.enable_input_require_grads()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must provide EOS; adding tokens is forbidden")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def prompt_ids(self, query, language):
        ids = self.tokenizer.encode(prompt(query, language), add_special_tokens=False)
        if len(ids) > self.max_prompt_tokens:
            raise ValueError(f"prompt length {len(ids)} exceeds {self.max_prompt_tokens}")
        return ids

    def supervised(self, inputs, targets):
        sequences, labels = [], []
        for record, target in zip(inputs, targets, strict=True):
            prefix = self.prompt_ids(**record)
            output = self.tokenizer.encode(target, add_special_tokens=False)
            if self.tokenizer.eos_token_id in output:
                raise ValueError("Target contains an embedded EOS")
            output += [self.tokenizer.eos_token_id]
            if len(output) > self.max_new_tokens:
                raise ValueError("Supervised target exceeds completion length limit")
            sequences.append(prefix + output)
            labels.append([-100] * len(prefix) + output)
        size = max(map(len, sequences))
        ids = [s + [self.tokenizer.pad_token_id] * (size - len(s)) for s in sequences]
        mask = [[1] * len(s) + [0] * (size - len(s)) for s in sequences]
        labels = [s + [-100] * (size - len(s)) for s in labels]

        def tensor(x):
            return torch.tensor(x, device=self.device, dtype=torch.long)

        return {"input_ids": tensor(ids), "attention_mask": tensor(mask)}, tensor(labels)

    def __call__(self, inputs):
        return self.model(**inputs, use_cache=False).logits

    @torch.no_grad()
    def generate(self, query, language, *, sample=True):
        # Explicit categorical sampling: no top-k/top-p/processors changing the policy.
        prefix = self.prompt_ids(query, language)
        tokens = []
        was_training = self.model.training
        self.model.eval()
        try:
            for _ in range(self.max_new_tokens):
                ids = torch.tensor([prefix + tokens], device=self.device)
                logits = self.model(input_ids=ids, use_cache=False).logits[0, -1].float()
                token = (
                    torch.multinomial(logits.softmax(-1), 1).item()
                    if sample
                    else logits.argmax().item()
                )
                tokens.append(token)
                if token == self.tokenizer.eos_token_id:
                    break
        finally:
            self.model.train(was_training)
        return Completion(
            tokens,
            self.tokenizer.decode(tokens, skip_special_tokens=True),
            tokens[-1] == self.tokenizer.eos_token_id,
        )

    def generate_group(self, inputs, size):
        """Sequential, no-grad rollouts; retain only token IDs and processed text."""
        if size < 2:
            raise ValueError("A rollout group requires at least two candidates")
        outputs = []
        for _ in range(size):
            completion = self.generate(**inputs)
            expansion = postprocess(inputs["query"], completion.text, completion.ended)
            outputs.append(Rollout(completion, expansion))
        return outputs

    def completion_log_prob(self, inputs, tokens):
        prefix = self.prompt_ids(**inputs)
        if not tokens:
            raise ValueError("Empty token sequence")
        ids = torch.tensor([prefix + tokens], device=self.device)
        logits = self.model(input_ids=ids, use_cache=False).logits
        selected = logits[0, len(prefix) - 1 : -1].float().log_softmax(-1)
        return selected.gather(1, torch.tensor(tokens, device=self.device)[:, None]).sum()
