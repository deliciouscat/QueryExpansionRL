import copy

import pytest
import torch

from query_expansion.agents import LoRALinear, merge_lora
from query_expansion.rewards import sft_loss
from query_expansion.utils.backward import gradient
from query_expansion.utils.checkpoint import seed_everything


def batch(agent, queries=("q",), targets=("battery",)):
    return agent.supervised([{"query": q, "language": "en"} for q in queries], targets)


def test_two_optimizers_frozen_weights_and_merge(make_policy):
    agent, engine = make_policy()
    before = copy.deepcopy(agent.model.state_dict())

    @gradient(engine)
    def forward(inputs, labels):
        return sft_loss(agent(inputs), labels)

    with engine.window(1):
        forward(*batch(agent))
    changed = [n for n, p in agent.model.named_parameters() if not torch.equal(before[n], p)]
    assert any("lora_B" in n for n in changed)
    assert any(n.startswith("layers.2.") for n in changed)
    for n, p in agent.model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(before[n], p)
    merged = merge_lora(copy.deepcopy(agent.model))
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    ids = torch.tensor([[20, 25, 30]])
    torch.testing.assert_close(agent.model(ids).logits, merged(ids).logits)


def test_token_weighted_accumulation_handles_unequal_lengths(make_policy):
    a, ea = make_policy()
    seed_everything(17)
    b, eb = make_policy()
    micro = [batch(a, ("q",), ("a",)), batch(a, ("longer query",), ("lengthy target",))]
    lengths = [int((labels != -100).sum()) for _, labels in micro]

    @gradient(ea)
    def first(inputs, labels):
        return sft_loss(a(inputs), labels)

    @gradient(eb)
    def second(inputs, labels):
        return sft_loss(b(inputs), labels)

    with ea.window(sum(lengths)):
        for (inputs, labels), length in zip(micro, lengths, strict=True):
            first(inputs, labels, weight=length)
    with eb.window(1):
        second(*batch(b, ("q", "longer query"), ("a", "lengthy target")))
    for p, q in zip(a.model.parameters(), b.model.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-6)


def test_nonfinite_does_not_step_and_caught_exception_cannot_commit(make_policy):
    agent, engine = make_policy()
    before = copy.deepcopy(agent.model.state_dict())

    @gradient(engine)
    def bad():
        return next(p for p in agent.model.parameters() if p.requires_grad).sum() * float("nan")

    with pytest.raises(FloatingPointError):
        with engine.window(1):
            bad()
    assert engine.global_step == 0
    assert all(p.grad is None for p in agent.model.parameters())
    assert all(torch.equal(v, agent.model.state_dict()[k]) for k, v in before.items())
    with pytest.raises(RuntimeError, match="failed loss"):
        with engine.window(1):
            try:
                bad()
            except FloatingPointError:
                pass
