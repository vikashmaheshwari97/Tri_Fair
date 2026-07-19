"""Audit per-step downstream token increments in completed Tri-Fair runs.

Run from the repository root:

    python -m analysis.audit_token_budget_v2_2 \
      --results-root results/tri_fair_v2_2/qwen-3-30b \
      --output-dir analysis/output/budget_audit_v2_2

The script is read-only.  It follows each run's logging_dir.txt pointer, computes
one downstream-token value per optimizer step, and reports step increments,
final budget differences, and the largest budget-crossing steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results/tri_fair_v2_2/qwen-3-30b",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/output/budget_audit_v2_2",
    )
    parser.add_argument("--requested-budget", type=int, default=5_000_000)
    return parser.parse_args()


def resolve_logging_dir(pointer: Path) -> Path:
    text = pointer.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"Empty logging_dir pointer: {pointer}")
    raw = Path(text).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        Path.cwd() / raw,
        pointer.parent / raw,
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot resolve logging directory from {pointer}; tried {candidates}"
    )


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def markdown_table(frame: pd.DataFrame) -> str:
    def clean(value: object) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", r"\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(args.results_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    pointers = sorted(root.glob("*/*/seed*/logging_dir.txt"))
    if not pointers:
        raise RuntimeError(f"No run pointers found below {root}")

    step_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for pointer in pointers:
        relative = pointer.relative_to(root)
        dataset, optimizer, seed_dir, _ = relative.parts
        seed = int(seed_dir.removeprefix("seed"))
        logging_dir = resolve_logging_dir(pointer)
        step_path = logging_dir / "step_results.parquet"
        if not step_path.is_file():
            raise FileNotFoundError(step_path)

        frame = pd.read_parquet(
            step_path,
            columns=["step", "total_tokens_downstream"],
        )
        if frame.empty:
            raise RuntimeError(f"Empty step log: {step_path}")

        per_step = (
            frame.groupby("step", as_index=False)["total_tokens_downstream"]
            .max()
            .sort_values("step", kind="stable")
            .reset_index(drop=True)
        )
        per_step["step_increment_tokens"] = (
            per_step["total_tokens_downstream"]
            .diff()
            .fillna(per_step["total_tokens_downstream"])
            .astype(int)
        )
        per_step["previous_total_tokens"] = (
            per_step["total_tokens_downstream"]
            - per_step["step_increment_tokens"]
        )
        per_step["crossed_requested_budget"] = (
            (per_step["previous_total_tokens"] < args.requested_budget)
            & (per_step["total_tokens_downstream"] >= args.requested_budget)
        )

        for row in per_step.itertuples(index=False):
            step_rows.append(
                {
                    "dataset": dataset,
                    "optimizer": optimizer,
                    "seed": seed,
                    "step": int(row.step),
                    "previous_total_tokens": int(row.previous_total_tokens),
                    "total_tokens_downstream": int(row.total_tokens_downstream),
                    "step_increment_tokens": int(row.step_increment_tokens),
                    "crossed_requested_budget": bool(
                        row.crossed_requested_budget
                    ),
                }
            )

        summary = load_summary(logging_dir / "run_summary.json")
        final_tokens = int(per_step["total_tokens_downstream"].iloc[-1])
        increments = per_step["step_increment_tokens"].astype(float)
        crossing = per_step[per_step["crossed_requested_budget"]]
        crossing_increment = (
            int(crossing["step_increment_tokens"].iloc[0])
            if len(crossing)
            else None
        )
        run_rows.append(
            {
                "dataset": dataset,
                "optimizer": optimizer,
                "seed": seed,
                "steps": len(per_step),
                "requested_budget": args.requested_budget,
                "final_tokens": final_tokens,
                "budget_difference_tokens": final_tokens - args.requested_budget,
                "overshoot_percent": max(
                    0.0,
                    100.0 * (final_tokens - args.requested_budget)
                    / args.requested_budget,
                ),
                "mean_step_increment": float(increments.mean()),
                "median_step_increment": float(increments.median()),
                "p90_step_increment": float(increments.quantile(0.90)),
                "p95_step_increment": float(increments.quantile(0.95)),
                "maximum_step_increment": int(increments.max()),
                "budget_crossing_step_increment": crossing_increment,
                "run_summary_final_tokens": (
                    summary.get("final_downstream_tokens", {})
                    .get("total_tokens")
                ),
                "logging_dir": str(logging_dir),
            }
        )

    steps = pd.DataFrame(step_rows).sort_values(
        ["dataset", "optimizer", "seed", "step"]
    )
    runs = pd.DataFrame(run_rows).sort_values(
        ["dataset", "optimizer", "seed"]
    )
    method = (
        runs.groupby(["dataset", "optimizer"], as_index=False)
        .agg(
            runs=("seed", "count"),
            mean_final_tokens=("final_tokens", "mean"),
            mean_budget_difference=("budget_difference_tokens", "mean"),
            maximum_overshoot_percent=("overshoot_percent", "max"),
            mean_step_increment=("mean_step_increment", "mean"),
            mean_p95_step_increment=("p95_step_increment", "mean"),
            maximum_step_increment=("maximum_step_increment", "max"),
        )
        .sort_values(["dataset", "optimizer"])
    )

    steps.to_csv(output / "qwen_v2_2_step_token_increments.csv", index=False)
    runs.to_csv(output / "qwen_v2_2_run_budget_summary.csv", index=False)
    method.to_csv(output / "qwen_v2_2_method_budget_summary.csv", index=False)
    (output / "qwen_v2_2_run_budget_summary.md").write_text(
        markdown_table(runs),
        encoding="utf-8",
    )
    (output / "qwen_v2_2_method_budget_summary.md").write_text(
        markdown_table(method),
        encoding="utf-8",
    )

    print(f"Audited runs: {len(runs)}")
    print(f"Step rows:    {len(steps)}")
    print(f"Output:       {output}")
    print()
    print(method.to_string(index=False))


if __name__ == "__main__":
    main()
