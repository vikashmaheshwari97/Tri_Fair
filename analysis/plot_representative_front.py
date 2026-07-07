"""Create the Tri-Fair replacement for MO-CAPO's annotated Figure 1.

Instead of a 2-D performance-cost curve, the Tri-Fair figure shows three
pairwise projections and a 3-D quality-cost-unfairness front. Representative
prompts are selected exclusively from development objectives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .inspect_results import resolve_step
from .io import read_eval_results
from .objectives import BoundsStore
from .plots import plot_test_front_projections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bounds-file", default="analysis/normalization_bounds.json")
    parser.add_argument("--budget", type=int, default=1_000_000)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output-dir", default="analysis/output/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    step = resolve_step(run_dir, requested_step=args.step, requested_budget=args.budget)
    frame = read_eval_results(run_dir / "eval.parquet")
    frame = frame[frame["chosen_step"] == step].copy().reset_index(drop=True)

    args_path = run_dir / "args.json"
    metadata = (
        json.loads(args_path.read_text(encoding="utf-8")) if args_path.exists() else {}
    )
    dataset = str(metadata.get("dataset", frame["dataset"].iloc[0]))
    model = str(metadata.get("model", frame["model"].iloc[0]))
    optimizer = str(metadata.get("optimizer", "unknown"))
    seed = int(
        metadata.get(
            "random_seed", frame.get("evaluation_seed", pd.Series([0])).iloc[0]
        )
    )

    bounds = BoundsStore.load(args.bounds_file)[(dataset, model)]
    title = f"{optimizer} | {dataset} | {model} | seed {seed} | step {step}"
    stem = f"{dataset}_{model}_{optimizer}_seed{seed}_step{step}"
    paths = plot_test_front_projections(
        frame,
        bounds,
        args.output_dir,
        title=title,
        file_stem=stem,
        annotate_representatives=True,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
