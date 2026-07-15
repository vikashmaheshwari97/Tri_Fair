#!/usr/bin/env python
"""Select non-dominated Tri-Fair-GR policies from policy_eval_summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--quality-col",
        default="quality",
        help="Quality column to maximize. Falls back to answer_mean if missing.",
    )
    parser.add_argument(
        "--cost-col",
        default="cost_usd",
        help="Cost column to minimize. Falls back to cost_proxy_context_chars if missing.",
    )
    parser.add_argument(
        "--unfairness-col",
        default="unfairness",
        help="Unfairness column to minimize. Falls back to loss if missing.",
    )
    return parser.parse_args()


def resolve_columns(df: pd.DataFrame, quality_col: str, cost_col: str, unfairness_col: str) -> tuple[str, str, str]:
    if quality_col not in df.columns:
        quality_col = "answer_mean"

    if cost_col not in df.columns:
        cost_col = "cost_proxy_context_chars"

    if unfairness_col not in df.columns:
        unfairness_col = "loss"

    missing = [c for c in [quality_col, cost_col, unfairness_col] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required objective columns: {missing}")

    return quality_col, cost_col, unfairness_col


def dominates(a: pd.Series, b: pd.Series, *, quality_col: str, cost_col: str, unfairness_col: str) -> bool:
    # Objectives: maximize quality; minimize cost; minimize unfairness.
    better_or_equal = (
        a[quality_col] >= b[quality_col]
        and a[cost_col] <= b[cost_col]
        and a[unfairness_col] <= b[unfairness_col]
    )
    strictly_better = (
        a[quality_col] > b[quality_col]
        or a[cost_col] < b[cost_col]
        or a[unfairness_col] < b[unfairness_col]
    )
    return bool(better_or_equal and strictly_better)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary)
    if df.empty:
        raise SystemExit("summary is empty")

    quality_col, cost_col, unfairness_col = resolve_columns(
        df,
        args.quality_col,
        args.cost_col,
        args.unfairness_col,
    )

    keep = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            if dominates(
                other,
                row,
                quality_col=quality_col,
                cost_col=cost_col,
                unfairness_col=unfairness_col,
            ):
                dominated = True
                break
        keep.append(not dominated)

    out = df.loc[keep].copy()
    out = out.sort_values(
        [unfairness_col, quality_col, cost_col],
        ascending=[True, False, True],
    )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    print(path)
    print("objectives:")
    print("  maximize", quality_col)
    print("  minimize", cost_col)
    print("  minimize", unfairness_col)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
