import copy

import pytest
import torch

from query_expansion.contracts import Record
from query_expansion.data_loader import dataloader
from query_expansion.data_loader.encoding import Encoder
from query_expansion.models.lora import LoRALinear, merge_lora
from query_expansion.strategies import DPO, SFT
from query_expansion.utils.checkpoint import load_checkpoint, save_checkpoint, seed_everything


def rows():
    return [
        Record(str(i), f"query {i}", "en", expansion_target="search", chosen="yes", rejected="no")
        for i in range(5)
    ]


def factory(bundle, strategy, batch_size=1):
    return lambda epoch, start: dataloader(
        records=rows(),
        batch_size=batch_size,
        strategy=strategy,
        tokenizer=bundle.tokenizer,
        seed=12,
        epoch=epoch,
        start_batch=start,
    )


def test_two_optimizers_frozen_weights_and_merge(make_trainer):
    bundle, trainer = make_trainer()
    before = {n: p.detach().clone() for n, p in bundle.model.named_parameters()}
    trainer.fit(factory(bundle, SFT()), SFT(), epochs=1, max_steps=2)
    changed = [n for n, p in bundle.model.named_parameters() if not torch.equal(before[n], p)]
    assert any("lora_B" in n for n in changed)
    assert any(n.startswith("layers.2.") for n in changed)
    for n, p in bundle.model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(before[n], p), n
    x = torch.tensor([[20, 25, 30]])
    merged = merge_lora(copy.deepcopy(bundle.model))
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    assert torch.allclose(bundle.model(x).logits, merged(x).logits, atol=1e-5)


def test_accumulation_matches_combined_batch_including_partial(make_trainer):
    a, ta = make_trainer(accumulation=3)
    seed_everything(17)
    b, tb = make_trainer()
    strategy = SFT()
    encoder = Encoder(a.tokenizer)
    micro = [strategy.collate(rows()[i : i + 1], encoder) for i in range(2)]
    ta.update(micro, strategy)
    tb.update([strategy.collate(rows()[:2], encoder)], strategy)
    for pa, pb in zip(a.model.parameters(), b.model.parameters(), strict=True):
        assert torch.allclose(pa, pb, atol=1e-6)


def test_resume_matches_uninterrupted_with_partial_epoch(make_trainer, tmp_path):
    a, ta = make_trainer(accumulation=2)
    seed_everything(17)
    b, tb = make_trainer(accumulation=2)
    strategy = SFT()
    ta.fit(factory(a, strategy), strategy, epochs=2, max_steps=5)
    tb.fit(factory(b, strategy), strategy, epochs=2, max_steps=5, stop_after=2)
    path = tmp_path / "state.pt"
    save_checkpoint(path, tb, strategy, {"experiment": "test"})
    c, tc = make_trainer(accumulation=2)
    load_checkpoint(path, tc, strategy, {"experiment": "test"})
    tc.fit(factory(c, strategy), strategy, epochs=2, max_steps=5)
    assert tc.state == ta.state
    for pa, pc in zip(a.model.parameters(), c.model.parameters(), strict=True):
        assert torch.equal(pa, pc)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_checkpoint(path, tc, strategy, {"experiment": "changed"})


def test_dpo_reference_is_frozen_and_persisted(make_trainer, tmp_path):
    bundle, trainer = make_trainer()
    reference = copy.deepcopy(bundle.model)
    strategy = DPO(reference)
    before = {k: v.clone() for k, v in reference.state_dict().items()}
    trainer.fit(factory(bundle, strategy), strategy, epochs=1, max_steps=1)
    assert all(p.grad is None and not p.requires_grad for p in reference.parameters())
    assert all(torch.equal(v, reference.state_dict()[k]) for k, v in before.items())
    path = tmp_path / "dpo.pt"
    save_checkpoint(path, trainer, strategy, {})
    restored = DPO(copy.deepcopy(bundle.model))
    load_checkpoint(path, trainer, restored, {})
    assert all(torch.equal(v, restored.reference_model.state_dict()[k]) for k, v in before.items())


def test_decorator_matches_strategy(make_trainer):
    from query_expansion.criterion import sft_loss

    a, ta = make_trainer()
    seed_everything(17)
    b, tb = make_trainer()
    batch = SFT().collate(rows()[:2], Encoder(a.tokenizer))

    @ta.train
    def objective(outputs, labels):
        return sft_loss(outputs.logits, labels)

    objective(*batch)
    tb.update([batch], SFT())
    for pa, pb in zip(a.model.parameters(), b.model.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_nonfinite_loss_does_not_update_either_optimizer(make_trainer):
    from query_expansion.contracts import LossTerm

    bundle, trainer = make_trainer()
    before = copy.deepcopy(bundle.model.state_dict())

    class Bad:
        def loss_terms(self, model, batch):
            yield LossTerm(
                next(p for p in model.parameters() if p.requires_grad).sum() * float("nan")
            )

    batch = SFT().collate(rows()[:1], Encoder(bundle.tokenizer))
    with pytest.raises(FloatingPointError):
        trainer.update([batch], Bad())
    assert trainer.state.global_step == 0
    assert all(p.grad is None for p in bundle.model.parameters())
    assert all(torch.equal(v, bundle.model.state_dict()[k]) for k, v in before.items())
