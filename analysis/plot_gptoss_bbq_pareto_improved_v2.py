from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OPTIMIZERS = ("Tri-Fair", "NSGAII-PO-Fair")
DISPLAY = {"Tri-Fair": "Tri-Fair", "NSGAII-PO-Fair": "NSGA-II-PO-Fair"}
COLORS = {"Tri-Fair": "black", "NSGAII-PO-Fair": "#E69F00"}
MARKERS = {"Tri-Fair": "o", "NSGAII-PO-Fair": "s"}


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for i in range(len(values)):
        dominates = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        dominates[i] = False
        if np.any(dominates):
            keep[i] = False
    return keep


def objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            1.0 - pd.to_numeric(frame["test_quality"], errors="coerce").to_numpy(float),
            pd.to_numeric(frame["test_cost"], errors="coerce").to_numpy(float),
            pd.to_numeric(frame["test_fairness"], errors="coerce").to_numpy(float),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "analysis/output/bbq_gptoss120b_final5m/publication_figures/"
            "bbq_gptoss120b_5m_final_publication_valid_evaluations.parquet"
        ),
    )
    parser.add_argument(
        "--output-stem",
        default=(
            "analysis/output/bbq_gptoss120b_final5m/publication_figures/"
            "bbq_gptoss120b_5m_final_test_pareto_3d_improved"
        ),
    )
    args = parser.parse_args()

    source = Path(args.input)
    output_stem = Path(args.output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = pd.read_parquet(source)
    except Exception:
        import pyarrow.parquet as pq

        data = pd.DataFrame(pq.read_table(source).to_pylist())
    required = {"optimizer", "seed", "test_quality", "test_cost", "test_fairness"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")

    fronts: list[pd.DataFrame] = []
    for optimizer in OPTIMIZERS:
        for seed, group in data[data["optimizer"] == optimizer].groupby("seed", sort=True):
            group = group.copy()
            matrix = objective_matrix(group)
            finite = np.all(np.isfinite(matrix), axis=1)
            group = group.loc[finite].reset_index(drop=True)
            matrix = matrix[finite]
            if group.empty:
                continue
            front = group.loc[pareto_mask_minimize(matrix)].copy()
            front["optimizer"] = optimizer
            front["seed"] = int(seed)
            fronts.append(front)

    if not fronts:
        raise RuntimeError("No finite Pareto-front points were found")
    front_data = pd.concat(fronts, ignore_index=True)

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(12.8, 7.2), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.45, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.055,
        right=0.965,
        bottom=0.085,
        top=0.90,
        wspace=0.23,
        hspace=0.30,
    )
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_cost = fig.add_subplot(grid[0, 1])
    ax_fair = fig.add_subplot(grid[1, 1])

    for optimizer in OPTIMIZERS:
        group = front_data[front_data["optimizer"] == optimizer]
        ax3d.scatter(
            group["test_cost"],
            group["test_fairness"],
            group["test_quality"],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            s=52,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.7,
            depthshade=True,
            label=DISPLAY[optimizer],
        )
        ax_cost.scatter(
            group["test_cost"],
            group["test_quality"],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            s=38,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.6,
            label=DISPLAY[optimizer],
        )
        ax_fair.scatter(
            group["test_fairness"],
            group["test_quality"],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            s=38,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.6,
            label=DISPLAY[optimizer],
        )

    ax3d.set_xlabel("Weighted Mean-Token Cost ↓", labelpad=10)
    ax3d.set_ylabel("Statistical BBQ Unfairness ↓", labelpad=10)
    ax3d.zaxis.set_rotate_label(False)
    ax3d.set_zlabel("Test Accuracy ↑", rotation=90, labelpad=12)
    ax3d.view_init(elev=24, azim=-57)
    ax3d.set_box_aspect((1.28, 1.0, 0.88))
    ax3d.tick_params(axis="x", pad=2)
    ax3d.tick_params(axis="y", pad=2)
    ax3d.tick_params(axis="z", pad=4)
    ax3d.grid(True, alpha=0.24)
    ax3d.legend(loc="upper left", frameon=False)
    ax_cost.set_title("Accuracy–Cost Projection")
    ax_cost.set_xlabel("Weighted Mean-Token Cost ↓")
    ax_cost.set_ylabel("Test Accuracy ↑")
    ax_cost.grid(True, alpha=0.25)

    ax_fair.set_title("Accuracy–Unfairness Projection")
    ax_fair.set_xlabel("Statistical BBQ Unfairness ↓")
    ax_fair.set_ylabel("Test Accuracy ↑")
    ax_fair.grid(True, alpha=0.25)

    fig.suptitle("BBQ — GPT-OSS-120B at 5M: Test Pareto Fronts", fontsize=14)
    fig.text(
        0.055,
        0.025,
        "Points are per-seed 3-objective Pareto candidates; lower cost/unfairness and higher accuracy are preferred.",
        fontsize=8.5,
        alpha=0.78,
    )

    for suffix in (".png", ".pdf"):
        fig.savefig(output_stem.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_stem.with_suffix('.png')}")
    print(f"Wrote {output_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
