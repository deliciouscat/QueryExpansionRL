"""Contract tests: graph lifetime, update boundaries and partial-window semantics."""

import copy
import weakref

import pytest
import torch

from query_expansion.data_loader import DatasetStream, Query
from query_expansion.rewards import sft_loss, target_token_count
from query_expansion.utils.backward import LossTerm, gradient, make_schedulers
from query_expansion.utils.checkpoint import seed_everything
from query_expansion.utils.iteration import TrainingLoop


def make_stream():
    return DatasetStream(
        [Query(str(i), f"query {i}", "en", expansion_target="a" * (i + 1)) for i in range(3)],
        "sft",
        shuffle=False,
    )


def make_loop(stream, engine, callback=None):
    return TrainingLoop(
        stream, engine, epochs=1, accumulation_steps=2, max_steps=10, on_boundary=callback
    )


def supervised_forward(agent, engine):
    @gradient(engine)
    def forward(inputs, targets):
        inputs, labels = agent.supervised(inputs, targets)
        return LossTerm.mean(sft_loss(agent(inputs), labels), target_token_count(labels))

    return forward


def test_generator_releases_activations_before_next_candidate_and_never_steps_mid_group(
    make_policy,
):
    agent, engine = make_policy()
    parameter = agent.groups["upper"][0]
    before = parameter.detach().clone()
    references, events = [], []

    class Probe(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            activation = x.detach().clone()
            ctx.save_for_backward(activation)
            references.append(weakref.ref(activation))
            return x.square().sum()

        @staticmethod
        def backward(ctx, grad):
            events.append("backward")
            return 2 * ctx.saved_tensors[0] * grad

    @gradient(engine)
    def forward():
        yield LossTerm(normalizer=1)
        for _ in range(3):
            yield LossTerm(Probe.apply(parameter), weight=1 / 3)
            assert references[-1]() is None  # Saved activations already released.
            assert torch.equal(parameter, before)
            assert engine.global_step == 0

    with engine.window():
        forward()
        assert events == ["backward"] * 3
    assert engine.global_step == 1
    assert not torch.equal(parameter, before)


def test_dynamic_token_normalization_matches_combined_batch(make_policy):
    a, ea = make_policy()
    seed_everything(17)
    b, eb = make_policy()
    inputs = [{"query": "q", "language": "en"}, {"query": "long query", "language": "en"}]
    targets = ["a", "many target tokens"]
    forward = supervised_forward(a, ea)
    reference = supervised_forward(b, eb)
    with ea.window():
        for row, target in zip(inputs, targets, strict=True):
            forward([row], [target])
    with eb.window():
        reference(inputs, targets)
    for p, q in zip(a.model.parameters(), b.model.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-6)


def test_skipped_groups_still_count_in_denominator(make_policy):
    agent, engine = make_policy()
    parameter = agent.groups["upper"][0]
    before = parameter.detach().clone()

    @gradient(engine)
    def forward():
        yield LossTerm(normalizer=1, metrics={"zero_advantage_rate": 1.0})
        yield LossTerm(normalizer=1, metrics={"zero_advantage_rate": 0.0})
        yield LossTerm(parameter.sum())

    with engine.window():
        forward()
    torch.testing.assert_close(parameter, before - 0.03 / 2)
    assert engine.last["zero_advantage_rate"] == 0.5


def test_all_skipped_preserves_optimizers_and_schedulers(make_policy):
    agent, engine = make_policy()
    engine.schedulers = make_schedulers(engine.optimizers, 10)
    before = copy.deepcopy(agent.model.state_dict())
    schedules = {k: s.state_dict() for k, s in engine.schedulers.items()}

    @gradient(engine)
    def forward():
        yield LossTerm(normalizer=1, metrics={"zero_advantage_rate": 1.0})

    with engine.window():
        forward()
        forward()
    assert engine.global_step == 0 and not engine.last["updated"]
    assert all(torch.equal(v, agent.model.state_dict()[k]) for k, v in before.items())
    assert all(not o.state for o in engine.optimizers.values())
    assert schedules == {k: s.state_dict() for k, s in engine.schedulers.items()}


def test_clean_break_flushes_consumed_prefix_and_can_continue(make_policy):
    agent, engine = make_policy()
    stream = make_stream()
    forward = supervised_forward(agent, engine)
    callbacks = []
    with make_loop(stream, engine, lambda m, s: callbacks.append((m, s))) as batches:
        for inputs, labels, _ in batches:
            forward(inputs, labels)
            break
    assert stream.position == 1 and engine.global_step == 1
    assert len(callbacks) == 1 and not engine.active
    with make_loop(stream, engine) as batches:
        for inputs, labels, _ in batches:
            forward(inputs, labels)
    assert engine.global_step == 2 and stream.epoch == 1


@pytest.mark.parametrize("failure", ["skip", "twice", "caught", "uncaught"])
def test_incomplete_or_failed_batch_never_commits(make_policy, failure):
    agent, engine = make_policy()
    stream = make_stream()
    before = copy.deepcopy(agent.model.state_dict())
    good = supervised_forward(agent, engine)

    @gradient(engine)
    def broken():
        yield LossTerm(agent.groups["upper"][0].sum(), normalizer=1)
        raise ValueError("failure after first candidate")

    expected = ValueError if failure == "uncaught" else RuntimeError
    with pytest.raises(expected):
        with make_loop(stream, engine) as batches:
            for inputs, labels, _ in batches:
                if failure == "twice":
                    good(inputs, labels)
                    good(inputs, labels)
                elif failure in {"caught", "uncaught"}:
                    try:
                        broken()
                    except ValueError:
                        if failure == "uncaught":
                            raise
                break
    assert engine.global_step == 0 and not engine.active
    assert all(p.grad is None for p in agent.model.parameters())
    assert all(torch.equal(v, agent.model.state_dict()[k]) for k, v in before.items())


def test_save_request_is_deferred_to_completed_window(make_policy):
    from dataclasses import replace

    agent, engine = make_policy()

    class SavingStream(DatasetStream):
        def __iter__(self):
            for batch in super().__iter__():
                yield replace(batch, save_flag=self.position == 1)

    stream = SavingStream(make_stream().rows, "sft", shuffle=False)
    forward = supervised_forward(agent, engine)
    events = []

    def callback(metrics, save_requested):
        assert not engine.active
        events.append((engine.global_step, save_requested))

    with make_loop(stream, engine, callback) as batches:
        for inputs, labels, _ in batches:
            forward(inputs, labels)
    assert events == [(1, True), (2, False)]


def test_list_of_graphs_and_empty_generators_are_rejected(make_policy):
    agent, engine = make_policy()

    @gradient(engine)
    def eager():
        return [LossTerm(agent.groups["upper"][0].sum(), normalizer=1)]

    @gradient(engine)
    def empty():
        yield from ()

    for forward, error in [(eager, TypeError), (empty, ValueError)]:
        with pytest.raises(error):
            with engine.window():
                forward()
    assert engine.global_step == 0 and not engine.active
