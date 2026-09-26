import pytest
import torch

from query_expansion.models import build_model, make_optimizers
from query_expansion.utils.checkpoint import seed_everything
from query_expansion.utils.trainer import Trainer


@pytest.fixture(autouse=True)
def deterministic():
    seed_everything(17)
    torch.set_num_threads(1)


@pytest.fixture
def make_trainer():
    def make(accumulation=1):
        bundle = build_model({"backend": "tiny", "width": 8, "rank": 2, "alpha": 4})
        optimizers = make_optimizers(bundle.groups, {"lora_lr": 0.01, "upper_lr": 0.03})
        return bundle, Trainer(
            bundle.model, optimizers, accumulation_steps=accumulation, max_grad_norm=1000
        )

    return make
