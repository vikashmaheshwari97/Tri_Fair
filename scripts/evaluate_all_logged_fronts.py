"""Evaluate every real logged optimizer state on the fixed v3 holdout set.

This reuses evaluate_checkpoint_fronts.run. Each run's own actual cumulative
downstream-token positions become exact checkpoint targets, so every mapping has
zero token error. Unique prompts are evaluated once and cached.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from scripts.evaluate_checkpoint_fronts import _step_token_table, run
from scripts._common import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--selection",
        choices=("current", "current_incumbents", "incumbents", "current_and_incumbents"),
        default="current_incumbents",
    )
    parser.add_argument("--minimum-actual-tokens", type=int, default=0)
    parser.add_argument("--maximum-actual-tokens", type=int, default=5_000_000)
    parser.add_argument("--manifest-dir", default="data/splits_v3")
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--output-file", default="eval_checkpoints.parquet")
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument(
        "--backup-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if args.minimum_actual_tokens < 0:
        raise ValueError("--minimum-actual-tokens must be non-negative")
    if args.maximum_actual_tokens <= args.minimum_actual_tokens:
        raise ValueError(
            "--maximum-actual-tokens must be greater than --minimum-actual-tokens"
        )

    log_path = Path(args.log_path).expanduser().resolve()
    step_path = log_path / "step_results.parquet"
    if not step_path.is_file():
        raise FileNotFoundError(step_path)

    frame = pd.read_parquet(step_path)
    per_step = _step_token_table(frame)
    selected = per_step[
        (per_step >= int(args.minimum_actual_tokens))
        & (per_step <= int(args.maximum_actual_tokens))
    ]

    actual_tokens = sorted({int(value) for value in selected.to_numpy() if int(value) > 0})
    if not actual_tokens:
        raise RuntimeError(
            "No real logged state lies inside the requested actual-token interval"
        )

    args.checkpoints = ",".join(str(value) for value in actual_tokens)
    args.checkpoint_policy = "nearest"
    args.maximum_checkpoint_relative_error = 0.0
    args.minimum_checkpoint_utilization = 1.0

    print(
        f"Evaluating {len(actual_tokens)} real logged states from "
        f"{actual_tokens[0]} to {actual_tokens[-1]} downstream tokens"
    )
    run(args)


if __name__ == "__main__":
    main()
