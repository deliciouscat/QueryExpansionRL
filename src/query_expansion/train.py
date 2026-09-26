"""CLI dispatch only; see train/sft.py and train/rl.py for editable learning loops."""

import argparse
import json

from .utils.config import read_config


def run(config, *, resume=None, init_from=None, stop_after=None):
    from train import rl, sft

    strategy = config.get("strategy", "sft")
    if strategy not in {"sft", "rl"}:
        raise ValueError(f"Unsupported training strategy {strategy}; available: sft, rl")
    return {"sft": sft.run, "rl": rl.run}[strategy](
        config, resume=resume, init_from=init_from, stop_after=stop_after
    )


def main(default_strategy=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--init-from")
    args = parser.parse_args()
    config = read_config(args.config)
    if default_strategy:
        config["strategy"] = default_strategy
    print(json.dumps(run(config, resume=args.resume, init_from=args.init_from), indent=2))


if __name__ == "__main__":
    main()
