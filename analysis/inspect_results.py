"""Inspect one Tri-Fair run without inventing a single scalar 'best prompt'.

The original MO-CAPO notebooks selected the maximum score.  Tri-Fair has three
objectives, so this utility reports representative development-selected prompts:
quality-first, cost-first, fairness-first, and balanced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .io import read_eval_results, read_step_results, step_token_table
from .metrics import representative_solutions
from .objectives import BoundsStore, objective_matrix, pareto_mask, valid_objective_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bounds-file", default="analysis/normalization_bounds.json")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--show-prompts", action="store_true")
    return parser.parse_args()


def resolve_step(
    run_dir: Path, *, requested_step: int | None, requested_budget: int | None
) -> int:
    evaluation = read_eval_results(run_dir / "eval.parquet")
    available_steps = sorted(evaluation["chosen_step"].astype(int).unique())
    if requested_step is not None:
        if requested_step not in available_steps:
            raise ValueError(
                f"Step {requested_step} is not evaluated; available: {available_steps}"
            )
        return requested_step
    if requested_budget is None:
        return available_steps[-1]

    steps = read_step_results(run_dir / "step_results.parquet")
    table = step_token_table(steps)
    reached = table[table["total_tokens_downstream"] >= int(requested_budget)]
    if reached.empty:
        raise ValueError(
            f"Run never reached budget {requested_budget:,}; maximum is "
            f"{int(table['total_tokens_downstream'].max()):,}"
        )
    target_step = int(reached.iloc[0]["step"])
    if target_step in available_steps:
        return target_step
    closest = min(available_steps, key=lambda value: abs(value - target_step))
    return int(closest)


def _metadata(run_dir: Path, frame: pd.DataFrame) -> tuple[str, str]:
    args_path = run_dir / "args.json"
    args = (
        json.loads(args_path.read_text(encoding="utf-8")) if args_path.exists() else {}
    )
    dataset = str(
        args.get("dataset", frame.get("dataset", pd.Series(["unknown"])).iloc[0])
    )
    model = str(args.get("model", frame.get("model", pd.Series(["unknown"])).iloc[0]))
    return dataset, model


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    step = resolve_step(run_dir, requested_step=args.step, requested_budget=args.budget)
    frame = read_eval_results(run_dir / "eval.parquet")
    frame = frame[frame["chosen_step"] == step].copy().reset_index(drop=True)
    mask = valid_objective_rows(frame, require_test=True)
    frame = frame.loc[mask].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"No fairness-ready evaluated prompts at step {step}")

    dataset, model = _metadata(run_dir, frame)
    bounds = BoundsStore.load(args.bounds_file)[(dataset, model)]
    representatives = representative_solutions(frame, bounds)

    dev = objective_matrix(frame, "dev")
    front = frame.loc[pareto_mask(dev)].reset_index(drop=True)
    rows = []
    for policy, selection in representatives.items():
        rows.append(selection.to_dict())
    selected = pd.DataFrame(rows)

    columns = [
        "policy",
        "dev_quality",
        "dev_cost",
        "dev_fairness",
        "test_quality",
        "test_cost",
        "test_fairness",
    ]
    print(f"Run: {run_dir}")
    print(f"Checkpoint step: {step}")
    print(f"Fairness-ready candidates: {len(frame)}")
    print(f"Development Pareto prompts: {len(front)}\n")
    print(selected[columns].to_string(index=False))

    if args.show_prompts:
        for _, row in selected.iterrows():
            print("\n" + "=" * 80)
            print(row["policy"])
            print("-" * 80)
            print(row["prompt"])

    output = (
        Path(args.output)
        if args.output
        else run_dir / f"representative_prompts_step{step}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
