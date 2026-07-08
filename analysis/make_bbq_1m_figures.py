"""Curated BBQ/Qwen-3-30B figures for the 1M Tri-Fair pilot.

This script creates a compact, paper-style figure package from:
  analysis/output/run_metrics.csv
  analysis/output/all_evaluations.parquet

It intentionally uses Accuracy/Test Accuracy terminology instead of Quality.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL = "qwen-3-30b"
DATASET = "bbq"
FINAL_BUDGET = 1_000_000

OPTIMIZER_ORDER = ["Tri-Fair", "NSGAII-PO-Fair"]
COLORS = {
    "Tri-Fair": "black",
    "NSGAII-PO-Fair": "#E69F00",
}
MARKERS = {
    "Tri-Fair": "o",
    "NSGAII-PO-Fair": "s",
}


def clean_run_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove archived pilot copies if they were accidentally ingested."""
    out = frame.copy()
    out = out[~out["run_dir"].astype(str).str.contains("pilot_reports", regex=False)]
    out = out[~out["run_key"].astype(str).str.endswith("/logging")]
    out = out[(out["dataset"] == DATASET) & (out["model"] == MODEL)]
    return out.reset_index(drop=True)


def clean_evaluations(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove archived pilot copies if they were accidentally ingested."""
    out = frame.copy()
    out = out[~out["run_dir"].astype(str).str.contains("pilot_reports", regex=False)]
    out = out[~out["run_key"].astype(str).str.endswith("/logging")]
    out = out[(out["dataset"] == DATASET) & (out["model"] == MODEL)]
    out = out[out["test_fairness_ready"].fillna(False).astype(bool)]
    return out.reset_index(drop=True)


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    """Return non-dominated mask for minimize-all objective vectors."""
    n = len(values)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates_i = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            keep[i] = False
    return keep


def add_metric_panel(run_metrics: pd.DataFrame, outdir: Path) -> None:
    metrics = [
        ("noisy_r2_3d", "Noisy R2 ↓"),
        ("hv_test_optimistic_3d", "Optimistic Test Hypervolume ↑"),
        ("hv_test_pessimistic_3d", "Pessimistic Test Hypervolume ↑"),
        ("approximation_gap_3d", "Approximation Gap ↓"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes = axes.ravel()

    for ax, (column, label) in zip(axes, metrics):
        for optimizer in OPTIMIZER_ORDER:
            data = run_metrics[run_metrics["optimizer"] == optimizer].copy()
            if data.empty:
                continue

            grouped = (
                data.groupby("budget_checkpoint")[column]
                .agg(["mean", "min", "max"])
                .reset_index()
                .sort_values("budget_checkpoint")
            )

            x = grouped["budget_checkpoint"].to_numpy(dtype=float) / 1_000_000.0
            y = grouped["mean"].to_numpy(dtype=float)
            ymin = grouped["min"].to_numpy(dtype=float)
            ymax = grouped["max"].to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                marker=MARKERS[optimizer],
                color=COLORS[optimizer],
                label=optimizer,
                linewidth=2,
                markersize=5,
            )
            ax.fill_between(x, ymin, ymax, color=COLORS[optimizer], alpha=0.14)

        ax.set_title(label)
        ax.set_xlabel("Evaluation-token budget [×10⁶]")
        ax.grid(True, alpha=0.25)
        ax.set_xticks([0.25, 1.0])
        ax.set_xticklabels(["250k", "1M"])

    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("BBQ — Qwen-3-30B: 3-objective front metrics", y=1.02, fontsize=13)

    fig.savefig(outdir / "bbq_qwen3_1m_metric_panel.pdf", bbox_inches="tight")
    fig.savefig(outdir / "bbq_qwen3_1m_metric_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_front_projection_figure(evaluations: pd.DataFrame, outdir: Path) -> None:
    final = evaluations[evaluations["budget_checkpoint"] == FINAL_BUDGET].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    projections = [
        ("test_cost", "test_quality", "Avg. Cost [$] per 1M Calls", "Test Accuracy"),
        ("test_fairness", "test_quality", "Test Unfairness", "Test Accuracy"),
        ("test_cost", "test_fairness", "Avg. Cost [$] per 1M Calls", "Test Unfairness"),
    ]

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, projections):
        for optimizer in OPTIMIZER_ORDER:
            data = final[final["optimizer"] == optimizer].copy()
            if data.empty:
                continue

            objective_values = np.column_stack(
                [
                    1.0 - data["test_quality"].to_numpy(dtype=float),
                    data["test_cost"].to_numpy(dtype=float),
                    data["test_fairness"].to_numpy(dtype=float),
                ]
            )
            mask = pareto_mask_minimize(objective_values)
            pareto = data.loc[mask].copy()

            ax.scatter(
                data[xcol],
                data[ycol],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                alpha=0.18,
                s=28,
            )
            ax.scatter(
                pareto[xcol],
                pareto[ycol],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.95,
                s=48,
                label=optimizer,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("BBQ — Qwen-3-30B at 1M: Test Pareto projections", y=1.03, fontsize=13)

    fig.savefig(outdir / "bbq_qwen3_1m_test_front_projections.pdf", bbox_inches="tight")
    fig.savefig(outdir / "bbq_qwen3_1m_test_front_projections.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_accuracy_cost_unfairness_scatter(evaluations: pd.DataFrame, outdir: Path) -> None:
    final = evaluations[evaluations["budget_checkpoint"] == FINAL_BUDGET].copy()

    # Plot only test non-dominated rows per optimizer to avoid clutter.
    rows = []
    for optimizer in OPTIMIZER_ORDER:
        data = final[final["optimizer"] == optimizer].copy()
        if data.empty:
            continue
        obj = np.column_stack(
            [
                1.0 - data["test_quality"].to_numpy(dtype=float),
                data["test_cost"].to_numpy(dtype=float),
                data["test_fairness"].to_numpy(dtype=float),
            ]
        )
        rows.append(data.loc[pareto_mask_minimize(obj)].copy())

    if not rows:
        return

    front = pd.concat(rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)

    vmin = float(front["test_fairness"].min())
    vmax = float(front["test_fairness"].max())

    for optimizer in OPTIMIZER_ORDER:
        data = front[front["optimizer"] == optimizer]
        if data.empty:
            continue

        sc = ax.scatter(
            data["test_cost"],
            data["test_quality"],
            c=data["test_fairness"],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            marker=MARKERS[optimizer],
            edgecolor="black",
            linewidth=0.5,
            s=75,
            label=optimizer,
        )

        for _, row in data.iterrows():
            ax.text(
                row["test_cost"],
                row["test_quality"] + 0.002,
                str(int(row["seed"])),
                fontsize=7,
                ha="center",
                va="bottom",
            )

    ax.set_xlabel("Avg. Cost [$] per 1M Calls")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("BBQ — Qwen-3-30B at 1M: Pareto candidates")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Test Unfairness ↓")

    fig.savefig(outdir / "bbq_qwen3_1m_accuracy_cost_unfairness_scatter.pdf", bbox_inches="tight")
    fig.savefig(outdir / "bbq_qwen3_1m_accuracy_cost_unfairness_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_summary_tables(run_metrics: pd.DataFrame, outdir: Path) -> None:
    selected = run_metrics[run_metrics["budget_checkpoint"] == FINAL_BUDGET].copy()

    rows = []
    for optimizer in OPTIMIZER_ORDER:
        data = selected[selected["optimizer"] == optimizer]
        rows.append(
            {
                "Method": optimizer,
                "Actual tokens mean": data["actual_budget_tokens"].mean(),
                "Noisy R2 ↓": data["noisy_r2_3d"].mean(),
                "HVopt ↑": data["hv_test_optimistic_3d"].mean(),
                "HVpes ↑": data["hv_test_pessimistic_3d"].mean(),
                "Gap ↓": data["approximation_gap_3d"].mean(),
                "Balanced Test Accuracy ↑": data["balanced_test_quality"].mean(),
                "Balanced Test Cost ↓": data["balanced_test_cost"].mean(),
                "Balanced Test Unfairness ↓": data["balanced_test_fairness"].mean(),
                "Seeds": len(data),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "bbq_qwen3_1m_summary_table.csv", index=False)
    (outdir / "bbq_qwen3_1m_summary_table.md").write_text(
        table.to_markdown(index=False, floatfmt=".4f") + "\n",
        encoding="utf-8",
    )

    print("\n1M summary table")
    print(table.to_string(index=False))


def main() -> None:
    output_root = Path("analysis/output")
    outdir = output_root / "curated_figures"
    outdir.mkdir(parents=True, exist_ok=True)

    run_metrics_path = output_root / "run_metrics.csv"
    eval_path = output_root / "all_evaluations.parquet"

    if not run_metrics_path.is_file():
        raise FileNotFoundError(run_metrics_path)
    if not eval_path.is_file():
        raise FileNotFoundError(eval_path)

    run_metrics = clean_run_metrics(pd.read_csv(run_metrics_path))
    evaluations = clean_evaluations(pd.read_parquet(eval_path))

    # Basic guard: after cleaning, there should be 3 seeds per optimizer and budget.
    counts = (
        run_metrics.groupby(["optimizer", "budget_checkpoint"])
        .size()
        .rename("n_runs")
        .reset_index()
    )
    print("\nRun counts after cleaning")
    print(counts.to_string(index=False))

    for _, row in counts.iterrows():
        if int(row["n_runs"]) != 3:
            raise RuntimeError(
                f"Expected 3 runs for {row['optimizer']} budget {row['budget_checkpoint']}, "
                f"got {row['n_runs']}"
            )

    add_metric_panel(run_metrics, outdir)
    add_front_projection_figure(evaluations, outdir)
    add_accuracy_cost_unfairness_scatter(evaluations, outdir)
    add_summary_tables(run_metrics, outdir)

    print("\nCurated figures written to:")
    print(outdir.resolve())


if __name__ == "__main__":
    main()