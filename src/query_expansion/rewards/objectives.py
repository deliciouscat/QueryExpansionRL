"""Objective registry. Math only; the experiment owns forward and rollout loops."""

import torch
import torch.nn.functional as F


def sequence_log_probs(logits, labels):
    labels = labels[:, 1:]
    mask = labels != -100
    if not mask.any(-1).all():
        raise ValueError("Every sequence requires completion labels")
    logps = logits[:, :-1].float().log_softmax(-1)
    selected = logps.gather(-1, labels.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    return (selected * mask).sum(-1), mask.sum(-1)


def sft_loss(logits, labels):
    if not (labels[:, 1:] != -100).any():
        raise ValueError("Missing completion labels")
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def group_advantages(rewards):
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.numel() < 2 or not torch.isfinite(values).all():
        raise ValueError("A group requires at least two finite rewards")
    return (values - values.mean()) / (values.std(unbiased=False) + 1e-6)


def reinforce_loss(log_probability, advantage):
    return -log_probability * torch.as_tensor(advantage, device=log_probability.device).detach()


class Rewards:
    registry = {"sft": sft_loss, "rl": reinforce_loss}

    @classmethod
    def register(cls, name, objective):
        if name in cls.registry:
            raise ValueError(f"Duplicate objective: {name}")
        cls.registry[name] = objective

    def __init__(self, strategy):
        if strategy not in self.registry:
            raise ValueError(f"Unsupported objective {strategy}; available: {list(self.registry)}")
        self.objective = self.registry[strategy]

    def __call__(self, *args, **kwargs):
        return self.objective(*args, **kwargs)


def target_token_count(labels):
    """Normalizer for causal completion-only SFT (prompt/padding excluded)."""
    return int((labels[:, 1:] != -100).sum())
