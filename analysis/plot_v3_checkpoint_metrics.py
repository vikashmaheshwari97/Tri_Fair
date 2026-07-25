"""Plot exact holdout nR2, hypervolume, and approximation-gap trajectories.

Input is ``v3_checkpoint_run_metrics.csv`` produced by
``analysis/summarize_v3_checkpoints.py``.  Development-selected checkpoint
archives are evaluated on holdout data; the fixed Initial Instructions baseline
is drawn as a horizontal reference band, following the MO-CAPO presentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ("Tri-Fair-v3", "NSGAII-PO-Fair")
DISPLAY = {
    "Tri-Fair-v3": "Tri-Fair v3",
    "NSGAII-PO-Fair": "NSGA-II-PO-Fair",
    "Initial": "Initial Instructions",
}
COLORS = {
    "Tri-Fair-v3": "black",
    "NSGAII-PO-Fair": "#E69F00",
    "Initial": "#777777",
}
MARKERS = {"Tri-Fair-v3": "o", "NSGAII-PO-Fair": "s"}
DATASET_LABELS = {
    "bbq": "BBQ",
    "civil_comments": "Civil Comments",
    "bias_in_bios": "Bias in Bios",
}
MODEL_LABELS = {
    "qwen-3-30b": "Qwen-3-30B",
    "gpt-oss-120b": "GPT-OSS-120B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    return parser.parse_args()


def _finite(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    output = frame.copy()
    output[column] = pd.to_numeric(output[column], errors="coerce")
    output["budget_checkpoint"] = pd.to_numeric(
        output["budget_checkpoint"], errors="coerce"
    )
    return output.dropna(subset=[column, "budget_checkpoint"])


def _aggregate(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = _finite(frame, metric)
    grouped = (
        data.groupby(["optimizer", "budget_checkpoint"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped.sort_values(["optimizer", "budget_checkpoint"])


def _draw_metric(
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    outdir: Path,
    stem: str,
    output_format: str,
) -> None:
    aggregate = _aggregate(frame, metric)
    methods = aggregate[aggregate["optimizer"].isin(METHOD_ORDER)]
    if methods.empty:
        return

    max_budget = float(methods["budget_checkpoint"].max())
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)

    initial = aggregate[aggregate["optimizer"] == "Initial"]
    if not initial.empty:
        initial_mean = float(initial["mean"].iloc[0])
        initial_std = float(initial["std"].iloc[0])
        ax.axhline(
            initial_mean,
            color=COLORS["Initial"],
            linestyle="--",
            linewidth=1.7,
            label=DISPLAY["Initial"],
        )
        if initial_std > 0:
            ax.fill_between(
                [0.0, max_budget / 1_000_000.0],
                [initial_mean - initial_std] * 2,
                [initial_mean + initial_std] * 2,
                color=COLORS["Initial"],
                alpha=0.10,
            )

    for method in METHOD_ORDER:
        data = methods[methods["optimizer"] == method]
        if data.empty:
            continue
        x = data["budget_checkpoint"].to_numpy(dtype=float) / 1_000_000.0
        mean = data["mean"].to_numpy(dtype=float)
        std = data["std"].to_numpy(dtype=float)
        ax.plot(
            x,
            mean,
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2.0,
            markersize=5,
            label=DISPLAY[method],
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=COLORS[method],
            alpha=0.16,
        )

    ax.set_title(title)
    ax.set_xlabel("Downstream Token Checkpoint [×10⁶]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0.0, right=max_budget / 1_000_000.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    if output_format in {"png", "both"}:
        fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    if output_format in {"pdf", "both"}:
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source = Path(args.run_metrics).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    required = {"model", "dataset", "optimizer", "budget_checkpoint"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Run-metric table is missing {sorted(missing)}")

    output_root = Path(args.output_dir).expanduser().resolve()
    for (model, dataset), group in frame.groupby(["model", "dataset"], sort=True):
        outdir = output_root / str(model) / str(dataset)
        outdir.mkdir(parents=True, exist_ok=True)
        heading = (
            f"{DATASET_LABELS.get(str(dataset), dataset)} — "
            f"{MODEL_LABELS.get(str(model), model)}"
        )
        specifications = (
            (
                "noisy_r2_3d",
                f"{heading}: Exact Holdout nR2",
                "nR2 ↓",
                "exact_holdout_nr2",
            ),
            (
                "hv_test_optimistic_3d",
                f"{heading}: Optimistic Holdout Hypervolume",
                "Optimistic HV ↑",
                "exact_holdout_hv_optimistic",
            ),
            (
                "hv_test_pessimistic_3d",
                f"{heading}: Pessimistic Holdout Hypervolume",
                "Pessimistic HV ↑",
                "exact_holdout_hv_pessimistic",
            ),
            (
                "approximation_gap_3d",
                f"{heading}: Holdout Approximation Gap",
                "Approximation Gap ↓",
                "exact_holdout_gap",
            ),
        )
        for metric, title, ylabel, stem in specifications:
            if metric not in group:
                continue
            _draw_metric(
                group,
                metric=metric,
                title=title,
                ylabel=ylabel,
                outdir=outdir,
                stem=stem,
                output_format=args.format,
            )
        print(f"Wrote v3 checkpoint figures to {outdir}")


if __name__ == "__main__":
    main()
