"""Plot dense post-hoc holdout trajectories in a MO-CAPO-style layout.

The input is v3_checkpoint_run_metrics.csv produced after evaluating a dense
nominal checkpoint grid (for example, 2.00M, 2.25M, ..., 5.00M).  Every plotted
value is computed from a real development-selected logged state evaluated on the
fixed holdout set.  No objective interpolation or synthetic result is used.

The evaluator may map adjacent nominal checkpoints to the same logged step.  In
that case the right-continuous staircase correctly displays a plateau.
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
    "Initial": "#888888",
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

SPECS = (
    ("noisy_r2_3d", "Holdout nR2 ↓", "nR2 ↓", "dense_holdout_nr2"),
    (
        "hv_test_optimistic_3d",
        "Optimistic Holdout Hypervolume ↑",
        "Optimistic HV ↑",
        "dense_holdout_hv_optimistic",
    ),
    (
        "hv_test_pessimistic_3d",
        "Pessimistic Holdout Hypervolume ↑",
        "Pessimistic HV ↑",
        "dense_holdout_hv_pessimistic",
    ),
    (
        "approximation_gap_3d",
        "Holdout Approximation Gap ↓",
        "Approximation Gap ↓",
        "dense_holdout_gap",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    parser.add_argument(
        "--strict-three-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require all method/checkpoint groups to contain exactly three seeds.",
    )
    parser.add_argument(
        "--minimum-budget",
        type=int,
        default=None,
        help="Optional lower nominal checkpoint limit, for example 2000000.",
    )
    parser.add_argument(
        "--maximum-budget",
        type=int,
        default=None,
        help="Optional upper nominal checkpoint limit, for example 5000000.",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def aggregate(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = frame.copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data["budget_checkpoint"] = pd.to_numeric(
        data["budget_checkpoint"], errors="coerce"
    )
    data["seed"] = pd.to_numeric(data["seed"], errors="coerce")
    data = data.dropna(subset=[metric, "budget_checkpoint", "seed"])
    grouped = (
        data.groupby(["optimizer", "budget_checkpoint"], sort=True)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


def save_figure(fig: plt.Figure, outdir: Path, stem: str, output_format: str) -> None:
    if output_format in {"png", "both"}:
        fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    if output_format in {"pdf", "both"}:
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def draw_metric(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
    strict_three_seeds: bool,
) -> pd.DataFrame:
    grouped = aggregate(frame, metric)
    methods = grouped[grouped["optimizer"].isin(METHOD_ORDER)].copy()
    if methods.empty:
        raise RuntimeError(f"No method rows available for {metric}")

    if strict_three_seeds:
        bad = methods[methods["n"] != 3]
        if not bad.empty:
            details = bad[
                ["optimizer", "budget_checkpoint", "n"]
            ].to_string(index=False)
            raise RuntimeError(
                f"{metric} does not have three seeds at every checkpoint:\n{details}"
            )

    xmin = float(methods["budget_checkpoint"].min()) / 1_000_000.0
    xmax = float(methods["budget_checkpoint"].max()) / 1_000_000.0

    initial = grouped[grouped["optimizer"] == "Initial"]
    if not initial.empty:
        initial_mean = float(initial["mean"].iloc[0])
        initial_std = float(initial["std"].iloc[0])
        ax.axhline(
            initial_mean,
            color=COLORS["Initial"],
            linestyle="--",
            linewidth=1.6,
            label=DISPLAY["Initial"],
            zorder=1,
        )
        if initial_std > 0:
            ax.fill_between(
                [xmin, xmax],
                [initial_mean - initial_std] * 2,
                [initial_mean + initial_std] * 2,
                color=COLORS["Initial"],
                alpha=0.08,
                zorder=0,
            )

    for method in METHOD_ORDER:
        data = methods[methods["optimizer"] == method].sort_values(
            "budget_checkpoint"
        )
        if data.empty:
            continue
        x = data["budget_checkpoint"].to_numpy(dtype=float) / 1_000_000.0
        mean = data["mean"].to_numpy(dtype=float)
        std = data["std"].to_numpy(dtype=float)

        ax.step(
            x,
            mean,
            where="post",
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2.0,
            markersize=4.5,
            label=DISPLAY[method],
            zorder=3,
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            step="post",
            color=COLORS[method],
            alpha=0.16,
            zorder=2,
        )

    padding = max(0.03, (xmax - xmin) * 0.02)
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_title(title)
    ax.set_xlabel("Nominal Downstream Token Checkpoint [×10⁶]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    return grouped


def main() -> None:
    args = parse_args()
    configure_style()

    source = Path(args.run_metrics).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    frame = pd.read_csv(source)
    required = {
        "model",
        "dataset",
        "optimizer",
        "seed",
        "budget_checkpoint",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Run-metric table is missing {sorted(missing)}")

    checkpoint = pd.to_numeric(frame["budget_checkpoint"], errors="coerce")
    methods_mask = frame["optimizer"].isin(METHOD_ORDER)
    keep = (~methods_mask) | (checkpoint > 0)

    if args.minimum_budget is not None:
        keep &= (~methods_mask) | (checkpoint >= args.minimum_budget)
    if args.maximum_budget is not None:
        keep &= (~methods_mask) | (checkpoint <= args.maximum_budget)
    frame = frame[keep].copy()

    output_root = Path(args.output_dir).expanduser().resolve()

    for (model, dataset), group in frame.groupby(["model", "dataset"], sort=True):
        outdir = output_root / str(model) / str(dataset)
        outdir.mkdir(parents=True, exist_ok=True)

        heading = (
            f"{DATASET_LABELS.get(str(dataset), dataset)} — "
            f"{MODEL_LABELS.get(str(model), model)}"
        )

        aggregate_tables: list[pd.DataFrame] = []

        for metric, title, ylabel, stem in SPECS:
            if metric not in group:
                continue
            fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
            table = draw_metric(
                ax,
                group,
                metric=metric,
                title=f"{heading}: {title}",
                ylabel=ylabel,
                strict_three_seeds=args.strict_three_seeds,
            )
            ax.legend(frameon=False)
            ax.text(
                0.01,
                0.01,
                "Post-hoc holdout; nearest real logged states (≤12% token deviation).",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7,
                alpha=0.72,
            )
            save_figure(fig, outdir, stem, args.format)
            aggregate_tables.append(table.assign(metric=metric))

        fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
        for ax, (metric, title, ylabel, _) in zip(axes.ravel(), SPECS):
            if metric not in group:
                ax.set_axis_off()
                continue
            draw_metric(
                ax,
                group,
                metric=metric,
                title=title,
                ylabel=ylabel,
                strict_three_seeds=args.strict_three_seeds,
            )

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=3,
                frameon=False,
                bbox_to_anchor=(0.5, 1.02),
            )
        fig.suptitle(
            f"{heading}: Post-hoc Holdout Trajectories",
            y=1.055,
            fontsize=14,
        )
        fig.text(
            0.5,
            -0.01,
            "Nearest real logged state at each nominal checkpoint; "
            "lines and bands show mean ± standard deviation across three seeds.",
            ha="center",
            fontsize=8,
        )
        save_figure(
            fig,
            outdir,
            "dense_holdout_nr2_hv_gap_2x2",
            args.format,
        )

        if aggregate_tables:
            pd.concat(aggregate_tables, ignore_index=True).to_csv(
                outdir / "dense_holdout_aggregate_mean_std.csv",
                index=False,
            )

        print(f"Wrote dense holdout figures to {outdir}")


if __name__ == "__main__":
    main()
