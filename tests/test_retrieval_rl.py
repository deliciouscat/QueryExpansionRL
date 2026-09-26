import copy
import random

import pytest
import torch

from query_expansion.agents import postprocess
from query_expansion.data_loader import Document, Query
from query_expansion.retriever import BM25
from query_expansion.rewards import RetrievalReward, Rewards, group_advantages
from query_expansion.utils.backward import gradient


def test_reward_fixed_corpus_statistics_and_common_candidates():
    docs = [
        Document("a", "en", "car"),
        Document("b", "en", "car battery battery"),
        Document("c", "en", "unrelated"),
    ]
    index = BM25(docs)
    reward = RetrievalReward([index], negatives=2, length_penalty=0)
    record = Query("q", "car", "en", qrels={"b": 2})
    before = copy.deepcopy(index.idf)
    context = reward.prepare(record, random.Random(1))
    assert context.candidates == (("a", "b", "c"),)
    value, _ = reward(
        record, expanded_query="car battery", generated_tokens=3, invalid=False, context=context
    )
    assert value > 0
    assert index.idf == before
    assert postprocess("car", "car car battery battery").query == "car battery"
    assert postprocess("car", "battery", ended=False).invalid
    assert not postprocess("car", "").invalid
    with pytest.raises(ValueError, match="missing positive"):
        reward.validate(Query("bad", "q", "en", qrels={"missing": 1}))


def test_constant_advantage_window_skips_weight_decay(make_policy):
    agent, engine = make_policy()
    before = copy.deepcopy(agent.model.state_dict())
    with engine.window(1):
        assert not group_advantages([1, 1, 1, 1]).any()
    assert engine.global_step == 0 and not engine.last["updated"]
    assert all(torch.equal(value, agent.model.state_dict()[key]) for key, value in before.items())


def test_reinforce_updates_with_completion_only_log_prob(make_policy):
    agent, engine = make_policy()
    inputs = {"query": "car", "language": "en"}
    criterion = Rewards("rl")
    before = copy.deepcopy(agent.model.state_dict())

    @gradient(engine)
    def forward(tokens, advantage):
        return criterion(agent.completion_log_prob(inputs, tokens), advantage)

    with engine.window(1):
        for tokens, advantage in zip(([3, 1], [4, 1]), group_advantages([0, 1]), strict=True):
            forward(tokens, advantage, weight=0.5)
    assert engine.global_step == 1
    assert any(
        not torch.equal(value, agent.model.state_dict()[key]) for key, value in before.items()
    )
