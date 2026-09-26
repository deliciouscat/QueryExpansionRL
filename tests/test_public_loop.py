import copy

import pytest
import torch

from query_expansion.contracts import Record
from query_expansion.data_loader import dataloader
from query_expansion.data_loader.encoding import Encoder
from query_expansion.strategies import SFT
from query_expansion.utils.checkpoint import seed_everything
from query_expansion.utils.execution import TrainingSession


def batches(bundle):
    rows = [Record(str(i), "query", "en", expansion_target="term") for i in range(3)]
    encoder = Encoder(bundle.tokenizer)
    return [SFT().collate(rows[:2], encoder), SFT().collate(rows[2:], encoder)]


def test_public_decorated_loop_matches_update_and_backprops_immediately(make_trainer):
    bundle, trainer = make_trainer(accumulation=4)
    seed_everything(17)
    reference, reference_trainer = make_trainer()
    window = batches(bundle)
    before = copy.deepcopy(bundle.model.state_dict())
    step = trainer.train(SFT().criterion, strategy=SFT())
    with trainer.accumulate(window) as metrics:
        for index, (inputs, labels) in enumerate(window):
            step(inputs, labels)
            # Backward happened inside the visible for loop, but step has not.
            assert any(p.grad is not None for p in bundle.model.parameters())
            assert all(torch.equal(v, bundle.model.state_dict()[k]) for k, v in before.items())
            assert trainer.state.global_step == 0
    assert metrics["updated"] == 1
    reference_trainer.update(window, SFT())
    for left, right in zip(bundle.model.parameters(), reference.model.parameters(), strict=True):
        assert torch.allclose(left, right, atol=1e-6)


def test_incomplete_or_failed_window_never_steps(make_trainer):
    bundle, trainer = make_trainer()
    before = copy.deepcopy(bundle.model.state_dict())
    window = batches(bundle)
    step = trainer.train(SFT().criterion, strategy=SFT())
    with pytest.raises(RuntimeError, match="Every batch"):
        with trainer.accumulate(window):
            step(*window[0])
    assert trainer.state.global_step == 0
    assert all(p.grad is None for p in bundle.model.parameters())
    assert all(torch.equal(v, bundle.model.state_dict()[k]) for k, v in before.items())
    # Infrastructure remains usable after an aborted window.
    with trainer.accumulate(window):
        for inputs, labels in window:
            step(inputs, labels)
    assert trainer.state.global_step == 1


def test_loop_uses_editable_loss_callback(make_trainer):
    bundle, trainer = make_trainer()
    rows = [Record("one", "query", "en", expansion_target="term")]
    calls, events = [], []
    with TrainingSession(
        trainer,
        lambda epoch, start: dataloader(
            records=rows,
            batch_size=1,
            strategy="sft",
            tokenizer=bundle.tokenizer,
            epoch=epoch,
            start_batch=start,
        ),
        SFT(),
        epochs=1,
        max_steps=1,
        on_boundary=lambda _, metrics: events.append(metrics),
    ) as session:

        @session.train
        def objective(outputs, labels):
            calls.append(labels.shape)
            return session.criterion(outputs, labels) * 0.5

        for inputs, labels in session:
            objective(inputs, labels)
    assert len(calls) == 1 and len(events) == 1
    assert events[0]["updated"] == 1


def session_for(bundle, trainer, *, on_boundary=None):
    rows = [Record(str(i), "query", "en", expansion_target="term") for i in range(3)]
    return TrainingSession(
        trainer,
        lambda epoch, start: dataloader(
            records=rows,
            batch_size=1,
            strategy="sft",
            tokenizer=bundle.tokenizer,
            epoch=epoch,
            start_batch=start,
            shuffle=False,
        ),
        SFT(),
        epochs=1,
        max_steps=10,
        on_boundary=on_boundary,
    )


def test_clean_break_flushes_partial_window_with_correct_weight(make_trainer):
    bundle, trainer = make_trainer(accumulation=3)
    seed_everything(17)
    reference, reference_trainer = make_trainer()
    consumed = []
    with session_for(bundle, trainer) as session:
        objective = session.train(session.criterion)
        for inputs, labels in session:
            consumed.append((inputs, labels))
            objective(inputs, labels)
            break
    from query_expansion.contracts import Batch

    reference_trainer.update([Batch(*consumed[0], 1)], SFT())
    assert trainer.state.global_step == 1
    assert trainer.state.next_batch == 1
    assert trainer.state.batches_seen == 1
    for left, right in zip(bundle.model.parameters(), reference.model.parameters(), strict=True):
        assert torch.allclose(left, right, atol=1e-6)
    # Resume from the consumed prefix; no already-trained row is repeated.
    with session_for(bundle, trainer) as session:
        objective = session.train(session.criterion)
        for inputs, labels in session:
            objective(inputs, labels)
    assert trainer.state.global_step == 2
    assert trainer.state.batches_seen == 3
    assert trainer.state.epoch == 1


def test_exception_discards_partial_window_and_clears_gradients(make_trainer):
    bundle, trainer = make_trainer(accumulation=3)
    before = copy.deepcopy(bundle.model.state_dict())
    with pytest.raises(ValueError, match="research error"):
        with session_for(bundle, trainer) as session:
            objective = session.train(session.criterion)
            for inputs, labels in session:
                objective(inputs, labels)
                raise ValueError("research error")
    assert trainer.state.global_step == trainer.state.next_batch == 0
    assert not trainer._updating
    assert all(p.grad is None for p in bundle.model.parameters())
    assert all(torch.equal(v, bundle.model.state_dict()[k]) for k, v in before.items())


def test_skipped_objective_rejected_instead_of_losing_batch(make_trainer):
    bundle, trainer = make_trainer(accumulation=3)
    with pytest.raises(RuntimeError, match="once for each yielded batch"):
        with session_for(bundle, trainer) as session:
            for inputs, labels in session:
                pass
    assert trainer.state.batches_seen == 0
    assert not trainer._updating


def test_caught_callback_error_still_prevents_commit(make_trainer):
    bundle, trainer = make_trainer(accumulation=3)
    with pytest.raises(RuntimeError, match="refusing to save"):
        with session_for(bundle, trainer) as session:

            @session.train
            def objective(outputs, labels):
                raise ValueError("bad criterion")

            for inputs, labels in session:
                try:
                    objective(inputs, labels)
                except ValueError:
                    break
    assert trainer.state.global_step == 0
    assert all(p.grad is None for p in bundle.model.parameters())
