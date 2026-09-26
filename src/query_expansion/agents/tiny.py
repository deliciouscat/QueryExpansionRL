"""Download-free causal CPU fixture, deliberately not a research model."""

from types import SimpleNamespace

import torch
from torch import nn


class ByteTokenizer:
    pad_token_id, eos_token_id = 0, 1

    def encode(self, text, add_special_tokens=False):
        return [b + 2 for b in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens=True):
        return bytes(i - 2 for i in ids if i >= 2).decode("utf-8", errors="replace")


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate_proj = nn.Linear(width, width)
        self.up_proj = nn.Linear(width, width)
        self.down_proj = nn.Linear(width, width)

    def forward(self, x):
        return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = MLP(width)

    def forward(self, x):
        # Prefix mean mixes context causally without an attention implementation dependency.
        length = torch.arange(1, x.shape[1] + 1, device=x.device)[None, :, None]
        return x + self.mlp(self.norm(x.cumsum(1) / length))


class TinyLM(nn.Module):
    def __init__(self, width=16, depth=4):
        super().__init__()
        self.embed_tokens = nn.Embedding(258, width)
        nn.init.normal_(self.embed_tokens.weight, std=0.05)
        self.layers = nn.ModuleList(Layer(width) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, 258, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(logits=self.lm_head(self.norm(x)))
