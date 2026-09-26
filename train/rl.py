"""Stage B: group-relative REINFORCE. Each yield is immediately backpropagated."""

from query_expansion.agents import Agents
from query_expansion.data_loader import DatasetStream
from query_expansion.retriever import BM25
from query_expansion.rewards import RetrievalReward, Rewards
from query_expansion.utils.backward import LossTerm, gradient
from train.common import configure, prepare


def run(config, *, resume=None, init_from=None, stop_after=None):
    if not resume and not init_from:
        raise ValueError("RL requires --init-from SFT checkpoint or --resume")
    records, corpus = prepare(config)
    model = Agents(**config["model"])
    dataset_stream = DatasetStream(
        records,
        "rl",
        config["training"].get("batch_size", 1),
        seed=config.get("seed", 17),
        **config.get("sampling", {}),
    )
    indexes = [BM25(corpus, **options) for options in config.get("retrievers", [{}])]
    criterion = Rewards("rl")
    reward = RetrievalReward(indexes, **config.get("reward", {}))
    group_size = config.get("group_size", 4)
    experiment = configure(
        config,
        model,
        dataset_stream,
        records,
        indexes,
        resume=resume,
        init_from=init_from,
        stop_after=stop_after,
    )
    engine = experiment.engine

    @gradient(engine)
    def forward(inputs, records):
        for policy_input, record in zip(inputs, records, strict=True):
            rollouts = model.generate_group(policy_input, group_size)
            group = reward.score_group(record, rollouts, experiment.rng)
            yield LossTerm(normalizer=1, metrics=group.metrics)  # Include skipped groups.
            if group.active:
                for rollout, advantage in zip(rollouts, group.advantages, strict=True):
                    log_probability = model.completion_log_prob(
                        policy_input, rollout.completion.tokens
                    )
                    loss = criterion(log_probability, advantage)
                    yield LossTerm(loss, weight=1 / group_size)

    with experiment.steps() as batches:
        for inputs, labels, save_flag in batches:
            forward(inputs, labels)
    return experiment.finish()


if __name__ == "__main__":
    from query_expansion.train import main

    main(default_strategy="rl")
