import pytest
import torch

from query_expansion.agents import Agents
from query_expansion.utils.backward import GradientEngine, make_optimizers
from query_expansion.utils.checkpoint import seed_everything


@pytest.fixture(autouse=True)
def deterministic():
    seed_everything(17)
    torch.set_num_threads(1)


@pytest.fixture
def make_policy():
    def make():
        agent = Agents(width=8, rank=2, alpha=4)
        optimizers = make_optimizers(agent.groups, lora_lr=0.01, upper_lr=0.03)
        return agent, GradientEngine(agent.model, optimizers, max_grad_norm=1000)

    return make
