"""Stage A: token-mean SFT. Edit model/data selection and forward here."""

from query_expansion.agents import Agents
from query_expansion.data_loader import DatasetStream
from query_expansion.retriever import BM25
from query_expansion.rewards import Rewards, target_token_count
from query_expansion.utils.backward import LossTerm, gradient
from train.common import configure, prepare


def run(config, *, resume=None, init_from=None, stop_after=None):
    records, corpus = prepare(config)
    model = Agents(**config["model"])
    dataset_stream = DatasetStream(
        records,
        "sft",
        config["training"].get("batch_size", 1),
        seed=config.get("seed", 17),
        **config.get("sampling", {}),
    )
    indexes = [BM25(corpus, **options) for options in config.get("retrievers", [{}])]
    criterion = Rewards("sft")
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
    def forward(inputs, targets):
        inputs, labels = model.supervised(inputs, targets)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        return LossTerm.mean(loss, count=target_token_count(labels))

    with experiment.steps() as batches:
        for inputs, labels, save_flag in batches:
            forward(inputs, labels)
    return experiment.finish()


if __name__ == "__main__":
    from query_expansion.train import main

    main(default_strategy="sft")
