import math

import pytest
import torch

from query_expansion.rewards import Rewards, group_advantages, sequence_log_probs, sft_loss


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


def test_empty_completion_and_unsupported_strategy():
    with pytest.raises(ValueError, match="completion"):
        sft_loss(torch.zeros(1, 3, 4), torch.full((1, 3), -100))
    with pytest.raises(ValueError, match="available"):
        Rewards("dpo")


def test_population_advantage_and_detached_policy_gradient():
    assert torch.equal(group_advantages([1, 1, 1, 1]), torch.zeros(4))
    assert torch.allclose(group_advantages([0, 2]), torch.tensor([-1.0, 1.0]))
    logp = torch.tensor(-2.0, requires_grad=True)
    advantage = torch.tensor(1.0, requires_grad=True)
    Rewards("rl")(logp, advantage).backward()
    assert logp.grad == -1 and advantage.grad is None
