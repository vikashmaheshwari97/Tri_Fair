"""Statistical comparisons for Tri-Fair optimizers.

Why this replaces ``cd_plot.ipynb``
-----------------------------------
The initial Tri-Fair study has two methods, so a Nemenyi critical-difference
plot is not the right primary test.  This module uses paired Wilcoxon tests for
two methods.  When three or more methods are later available, it adds a
Friedman omnibus test, Holm-corrected pairwise Wilcoxon tests, and an average-
rank plot.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from .config import METRIC_DIRECTIONS, parse_csv_strings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-metrics", default="analysis/output/run_metrics.csv")
    parser.add_argument("--metric", default="noisy_r2_3d")
    parser.add_argument("--budget", type=int, default=1_000_000)
    parser.add_argument("--optimizers", default="Tri-Fair,NSGAII-PO-Fair")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", default="analysis/output/statistics")
    return parser.parse_args()


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def rank_biserial_from_differences(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if not len(differences):
        return 0.0
    ranks = rankdata(np.abs(differences))
    positive = float(ranks[differences > 0].sum())
    negative = float(ranks[differences < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def complete_performance_matrix(
    frame: pd.DataFrame,
    *,
    metric: str,
    budget: int,
    optimizers: Sequence[str],
) -> pd.DataFrame:
    subset = frame[
        (frame["budget_checkpoint"] == int(budget))
        & frame["optimizer"].isin(optimizers)
    ].copy()
    block_columns = ["model", "dataset", "seed"]
    matrix = subset.pivot_table(
        index=block_columns,
        columns="optimizer",
        values=metric,
        aggfunc="mean",
    )
    missing = [optimizer for optimizer in optimizers if optimizer not in matrix]
    if missing:
        raise ValueError(f"No values for optimizers: {missing}")
    return matrix[list(optimizers)].dropna(axis=0, how="any")


def paired_wilcoxon(matrix: pd.DataFrame, direction: str) -> pd.DataFrame:
    if matrix.shape[1] != 2:
        raise ValueError("paired_wilcoxon requires exactly two optimizer columns")
    first, second = matrix.columns
    first_values = matrix[first].to_numpy(dtype=float)
    second_values = matrix[second].to_numpy(dtype=float)
    # Positive differences always mean the first method is better.
    differences = (
        first_values - second_values
        if direction == "max"
        else second_values - first_values
    )
    if np.allclose(differences, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return pd.DataFrame(
        [
            {
                "optimizer_a": first,
                "optimizer_b": second,
                "n_blocks": len(matrix),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial_a_better": rank_biserial_from_differences(differences),
                "median_a": float(np.median(first_values)),
                "median_b": float(np.median(second_values)),
            }
        ]
    )


def multi_method_tests(
    matrix: pd.DataFrame, direction: str, alpha: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    arrays = [matrix[column].to_numpy(dtype=float) for column in matrix.columns]
    omnibus = friedmanchisquare(*arrays)
    omnibus_df = pd.DataFrame(
        [
            {
                "n_blocks": len(matrix),
                "n_methods": matrix.shape[1],
                "friedman_statistic": float(omnibus.statistic),
                "p_value": float(omnibus.pvalue),
                "alpha": float(alpha),
                "significant": bool(omnibus.pvalue < alpha),
            }
        ]
    )

    rows: list[dict[str, Any]] = []
    for first, second in itertools.combinations(matrix.columns, 2):
        first_values = matrix[first].to_numpy(dtype=float)
        second_values = matrix[second].to_numpy(dtype=float)
        differences = (
            first_values - second_values
            if direction == "max"
            else second_values - first_values
        )
        if np.allclose(differences, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            result = wilcoxon(
                differences, alternative="two-sided", zero_method="wilcox"
            )
            statistic, p_value = float(result.statistic), float(result.pvalue)
        rows.append(
            {
                "optimizer_a": first,
                "optimizer_b": second,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial_a_better": rank_biserial_from_differences(differences),
            }
        )
    pairwise = pd.DataFrame(rows)
    pairwise["p_value_holm"] = holm_adjust(pairwise["p_value"].to_numpy(dtype=float))
    pairwise["significant_holm"] = pairwise["p_value_holm"] < alpha
    return omnibus_df, pairwise


def average_ranks(matrix: pd.DataFrame, direction: str) -> pd.Series:
    values = matrix.to_numpy(dtype=float)
    if direction == "max":
        values = -values
    ranks = np.vstack([rankdata(row, method="average") for row in values])
    return pd.Series(ranks.mean(axis=0), index=matrix.columns).sort_values()


def plot_average_ranks(ranks: pd.Series, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, max(2.5, 0.5 * len(ranks) + 1.5)))
    y = np.arange(len(ranks))
    axis.scatter(ranks.to_numpy(), y)
    for position, (name, rank) in enumerate(ranks.items()):
        axis.text(rank + 0.03, position, str(name), va="center")
    axis.set_yticks([])
    axis.set_xlabel("Average rank (lower is better)")
    axis.set_xlim(0.8, max(len(ranks) + 0.2, float(ranks.max()) + 0.5))
    axis.grid(True, axis="x", linestyle="--", alpha=0.3)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    optimizers = parse_csv_strings(args.optimizers)
    frame = pd.read_csv(args.run_metrics)
    if args.metric not in frame:
        raise ValueError(f"Metric {args.metric!r} is not in {args.run_metrics}")
    direction = METRIC_DIRECTIONS.get(args.metric)
    if direction is None:
        raise ValueError(
            f"Unknown direction for {args.metric!r}; add it to analysis.config.METRIC_DIRECTIONS"
        )
    matrix = complete_performance_matrix(
        frame,
        metric=args.metric,
        budget=args.budget,
        optimizers=optimizers,
    )
    if len(matrix) < 5:
        raise RuntimeError(
            f"Only {len(matrix)} complete paired blocks are available. "
            "Do not report a cross-task significance test with fewer than five blocks."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_dir / f"matrix_{args.metric}_{args.budget}.csv")

    if len(optimizers) == 2:
        result = paired_wilcoxon(matrix, direction)
        result.to_csv(
            output_dir / f"wilcoxon_{args.metric}_{args.budget}.csv", index=False
        )
        print(result.to_string(index=False))
    else:
        omnibus, pairwise = multi_method_tests(matrix, direction, args.alpha)
        omnibus.to_csv(
            output_dir / f"friedman_{args.metric}_{args.budget}.csv", index=False
        )
        pairwise.to_csv(
            output_dir / f"pairwise_holm_{args.metric}_{args.budget}.csv", index=False
        )
        ranks = average_ranks(matrix, direction)
        ranks.rename("average_rank").to_csv(
            output_dir / f"average_ranks_{args.metric}_{args.budget}.csv"
        )
        plot_average_ranks(
            ranks,
            output_dir / f"average_ranks_{args.metric}_{args.budget}.pdf",
        )
        print(omnibus.to_string(index=False))
        print("\n" + pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
