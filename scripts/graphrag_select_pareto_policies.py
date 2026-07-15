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
    return parser.parse_args()


def dominates(a: pd.Series, b: pd.Series) -> bool:
    # Objectives: maximize answer_mean; minimize cost_proxy_context_chars; minimize loss.
    better_or_equal = (
        a["answer_mean"] >= b["answer_mean"]
        and a["cost_proxy_context_chars"] <= b["cost_proxy_context_chars"]
        and a["loss"] <= b["loss"]
    )
    strictly_better = (
        a["answer_mean"] > b["answer_mean"]
        or a["cost_proxy_context_chars"] < b["cost_proxy_context_chars"]
        or a["loss"] < b["loss"]
    )
    return bool(better_or_equal and strictly_better)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary)
    if df.empty:
        raise SystemExit("summary is empty")

    keep = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            if dominates(other, row):
                dominated = True
                break
        keep.append(not dominated)

    out = df.loc[keep].copy()
    out = out.sort_values(["loss", "answer_mean", "cost_proxy_context_chars"], ascending=[True, False, True])
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(path)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
