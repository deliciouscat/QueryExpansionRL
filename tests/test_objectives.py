import math

import pytest
import torch

from query_expansion.criterion import dpo_loss, group_advantages, sequence_log_probs, sft_loss


def test_causal_shift_and_completion_only():
    logits = torch.zeros(1, 4, 3, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, -100]])
    loss = sft_loss(logits, labels)
    assert loss.item() == pytest.approx(math.log(3))
    loss.backward()
    assert logits.grad[:, 0].abs().sum() == 0
    assert logits.grad[:, 1].abs().sum() > 0
    assert logits.grad[:, 2:].abs().sum() == 0
    logps, lengths = sequence_log_probs(logits, labels)
    assert lengths.tolist() == [1]
    assert logps.item() == pytest.approx(-math.log(3))


def test_empty_completion_rejected():
    with pytest.raises(ValueError, match="completion"):
        sft_loss(torch.zeros(1, 3, 4), torch.full((1, 3), -100))


def test_dpo_reference_and_gradient_direction():
    chosen = torch.tensor([-2.0], requires_grad=True)
    rejected = torch.tensor([-4.0], requires_grad=True)
    rc, rr = chosen.detach().clone(), rejected.detach().clone()
    loss = dpo_loss(chosen, rejected, rc, rr)
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert chosen.grad.item() < 0 < rejected.grad.item()


def test_advantage_constant_and_population_std():
    assert torch.equal(group_advantages(torch.ones(4)), torch.zeros(4))
    assert torch.allclose(group_advantages(torch.tensor([0.0, 2.0])), torch.tensor([-1.0, 1.0]))
