import copy
import random

import pytest
import torch

from query_expansion.contracts import Completion, Document, Record
from query_expansion.data_loader.encoding import Encoder
from query_expansion.retriever import BM25Reward, expanded_query, ndcg
from query_expansion.strategies import RL


def test_reward_improves_ranking_and_preserves_statistics():
    docs = [
        Document("a", "en", "car"),
        Document("b", "en", "car battery battery"),
        Document("c", "en", "unrelated"),
    ]
    reward = BM25Reward(docs, negatives=2, length_penalty=0)
    row = Record("q", "car", "en", positive_doc_ids=("b",))
    before = copy.deepcopy(reward.indexes[0].idf)
    ctx = reward.prepare(row, random.Random(1))
    assert "b" in ctx.candidates and len(ctx.candidates) == 3
    value, _ = reward(row, Completion([2, 1], "battery", True), ctx)
    assert value > 0
    assert reward.indexes[0].idf == before
    assert ndcg(["b", "a"], row.qrels) == 1
    assert expanded_query("car", "car car battery battery") == ("car battery", False)
    assert expanded_query("car", "battery", ended=False) == ("car", True)
    with pytest.raises(ValueError, match="missing positive"):
        reward.validate(Record("bad", "q", "en", positive_doc_ids=("missing",)))


class SimpleReward:
    def validate(self, record):
        pass

    def prepare(self, record, rng):
        return rng.random()

    def __call__(self, record, completion, context):
        return float(completion.tokens[0] == 3), {"gain": float(completion.tokens[0] == 3)}


class Alternating:
    def __init__(self):
        self.i = 0

    def __call__(self, model, prefix, tokenizer, max_new_tokens):
        token = 2 + self.i % 2
        self.i += 1
        return Completion([token, 1], str(token), True)


def test_rl_nonzero_advantage_updates_policy(make_trainer):
    bundle, trainer = make_trainer()
    strategy = RL(SimpleReward(), bundle.tokenizer, group_size=2, generator=Alternating())
    row = Record("q", "car", "en", positive_doc_ids=("b",))
    batch = strategy.collate([row], Encoder(bundle.tokenizer))
    before = copy.deepcopy(bundle.model.state_dict())
    metrics = trainer.update([batch], strategy)
    assert metrics["reward"] == 0.5
    assert metrics["updated"] == 1 and trainer.state.global_step == 1
    assert any(not torch.equal(v, bundle.model.state_dict()[k]) for k, v in before.items())


def test_constant_reward_skips_weight_decay_and_step(make_trainer):
    bundle, trainer = make_trainer()

    def constant(*args):
        return Completion([2, 1], "2", True)

    strategy = RL(SimpleReward(), bundle.tokenizer, group_size=2, generator=constant)
    batch = strategy.collate(
        [Record("q", "car", "en", positive_doc_ids=("b",))], Encoder(bundle.tokenizer)
    )
    before = copy.deepcopy(bundle.model.state_dict())
    metrics = trainer.update([batch], strategy)
    assert metrics["updated"] == 0 and trainer.state.global_step == 0
    assert metrics["zero_advantage"] == 1
    assert all(torch.equal(v, bundle.model.state_dict()[k]) for k, v in before.items())


def test_rl_rng_resume_matches_uninterrupted(make_trainer, tmp_path):
    from query_expansion.data_loader import dataloader
    from query_expansion.utils.checkpoint import load_checkpoint, save_checkpoint, seed_everything

    def stochastic(model, prefix, tokenizer, max_new_tokens):
        token = int(torch.randint(2, 4, (1,)).item())
        return Completion([token, 1], str(token), True)

    rows = [Record(str(i), "car", "en", positive_doc_ids=("b",)) for i in range(20)]

    def setup():
        seed_everything(17)
        bundle, trainer = make_trainer()
        strategy = RL(SimpleReward(), bundle.tokenizer, group_size=4, generator=stochastic)

        def factory(epoch, start):
            return dataloader(
                records=rows,
                batch_size=1,
                strategy=strategy,
                tokenizer=bundle.tokenizer,
                epoch=epoch,
                start_batch=start,
            )

        return bundle, trainer, strategy, factory

    a, ta, sa, fa = setup()
    ta.fit(fa, sa, epochs=2, max_steps=4)
    expected_rng = torch.get_rng_state().clone()
    b, tb, sb, fb = setup()
    tb.fit(fb, sb, epochs=2, max_steps=4, stop_after=2)
    path = tmp_path / "rl.pt"
    save_checkpoint(path, tb, sb, {})
    c, tc, sc, fc = setup()
    load_checkpoint(path, tc, sc, {})
    tc.fit(fc, sc, epochs=2, max_steps=4)
    assert ta.state == tc.state
    assert sa.state_dict() == sc.state_dict()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for pa, pc in zip(a.model.parameters(), c.model.parameters(), strict=True):
        assert torch.equal(pa, pc)
