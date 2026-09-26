"""Explicit MLP-only LoRA. No PEFT side effects on upper-layer trainability."""

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.base, self.scale = base, alpha / rank
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def configure_parameters(model, layers, split, rank, alpha):
    if not 0 < split < len(layers):
        raise ValueError("Split must leave lower and upper decoder layers")
    model.requires_grad_(False)
    lora = []
    for layer in layers[:split]:
        for name in ("gate_proj", "up_proj", "down_proj"):
            base = getattr(layer.mlp, name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Expected Linear at mlp.{name}")
            module = LoRALinear(base, rank, alpha)
            setattr(layer.mlp, name, module)
            lora.extend([module.lora_A, module.lora_B])
    upper = []
    for layer in layers[split:]:
        layer.requires_grad_(True)
        upper.extend(layer.parameters())
    groups = {"lora": lora, "upper": upper}
    ids = [{id(p) for p in params} for params in groups.values()]
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if ids[0] & ids[1] or ids[0] | ids[1] != trainable:
        raise RuntimeError("Optimizer parameter partition is not exact")
    return groups


def merge_lora(model):
    for name, child in list(model.named_children()):
        if isinstance(child, LoRALinear):
            with torch.no_grad():
                child.base.weight.add_((child.lora_B @ child.lora_A) * child.scale)
            setattr(model, name, child.base)
        else:
            merge_lora(child)
    return model
