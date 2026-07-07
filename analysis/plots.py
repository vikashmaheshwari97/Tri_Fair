"""Plotting helpers for Tri-Fair objective fronts and budget trajectories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DATASET_LABELS, OPTIMIZER_LABELS, budget_label
from .metrics import representative_solutions
from .objectives import Bounds, objective_matrix, pareto_mask, valid_objective_rows


def safe_filename(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def _front_frame(frame: pd.DataFrame) -> pd.DataFrame:
    mask = valid_objective_rows(frame, require_test=True)
    valid = frame.loc[mask].reset_index(drop=True)
    if valid.empty:
        return valid
    dev = objective_matrix(valid, "dev")
    return valid.loc[pareto_mask(dev)].reset_index(drop=True)


def plot_test_front_projections(
    frame: pd.DataFrame,
    bounds: Bounds,
    output_dir: str | Path,
    *,
    title: str,
    file_stem: str,
    annotate_representatives: bool = True,
) -> list[Path]:
    """Create one 3-D and three pairwise plots for a development-selected front."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    front = _front_frame(frame)
    if front.empty:
        return []

    quality = pd.to_numeric(front["test_quality"], errors="coerce").to_numpy(
        dtype=float
    )
    cost = pd.to_numeric(front["test_cost"], errors="coerce").to_numpy(dtype=float)
    fairness = pd.to_numeric(front["test_fairness"], errors="coerce").to_numpy(
        dtype=float
    )
    saved: list[Path] = []

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(cost, fairness, quality, c=quality)
    axis.set_xlabel("Inference cost")
    axis.set_ylabel("Unfairness")
    axis.set_zlabel("Quality")
    axis.set_title(title)
    figure.colorbar(scatter, ax=axis, label="Quality", shrink=0.7)
    figure.tight_layout()
    path = output / f"{safe_filename(file_stem)}_pareto_3d.pdf"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    saved.append(path)

    representatives = representative_solutions(
        frame.loc[valid_objective_rows(frame)], bounds
    )
    representative_prompts = {
        selection.prompt: policy for policy, selection in representatives.items()
    }

    projections = (
        (
            cost,
            quality,
            fairness,
            "Inference cost",
            "Quality",
            "Unfairness",
            "cost_quality",
        ),
        (
            fairness,
            quality,
            cost,
            "Unfairness",
            "Quality",
            "Inference cost",
            "fairness_quality",
        ),
        (
            cost,
            fairness,
            quality,
            "Inference cost",
            "Unfairness",
            "Quality",
            "cost_fairness",
        ),
    )
    for x, y, color, xlabel, ylabel, color_label, suffix in projections:
        figure, axis = plt.subplots(figsize=(7, 5))
        scatter = axis.scatter(x, y, c=color)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, linestyle="--", alpha=0.3)
        figure.colorbar(scatter, ax=axis, label=color_label)

        if annotate_representatives:
            for row_index, row in front.iterrows():
                policy = representative_prompts.get(str(row.get("prompt", "")))
                if policy is None:
                    continue
                axis.annotate(
                    policy.replace("_", " "),
                    (x[row_index], y[row_index]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        figure.tight_layout()
        path = output / f"{safe_filename(file_stem)}_{suffix}.pdf"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        saved.append(path)
    return saved


def plot_budget_metric(
    summary: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    metric: str,
    output_path: str | Path,
    ylabel: str | None = None,
) -> Path:
    subset = summary[
        (summary["dataset"] == dataset) & (summary["model"] == model)
    ].copy()
    if subset.empty:
        raise ValueError(f"No summary rows for {dataset}/{model}")
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    if mean_column not in subset:
        raise ValueError(f"Summary does not contain {mean_column}")

    figure, axis = plt.subplots(figsize=(7, 5))
    for optimizer, group in subset.groupby("optimizer", sort=True):
        group = group.sort_values("budget_checkpoint")
        x = group["budget_checkpoint"].to_numpy(dtype=float)
        y = group[mean_column].to_numpy(dtype=float)
        std = (
            group[std_column].to_numpy(dtype=float)
            if std_column in group
            else np.zeros(len(group))
        )
        label = OPTIMIZER_LABELS.get(str(optimizer), str(optimizer))
        line = axis.plot(x, y, marker="o", label=label)[0]
        axis.fill_between(x, y - std, y + std, alpha=0.15, color=line.get_color())

    ticks = sorted(subset["budget_checkpoint"].dropna().astype(int).unique())
    axis.set_xticks(ticks, [budget_label(value) for value in ticks])
    axis.set_xlabel("Evaluation-token budget")
    axis.set_ylabel(ylabel or metric)
    axis.set_title(f"{DATASET_LABELS.get(dataset, dataset)} — {model}")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend(frameon=False)
    figure.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target


def plot_trajectory(
    trajectory: pd.DataFrame,
    *,
    run_key: str,
    metric: str,
    output_path: str | Path,
) -> Path:
    group = trajectory[trajectory["run_key"] == run_key].sort_values(
        "actual_budget_tokens"
    )
    if group.empty:
        raise ValueError(f"No trajectory rows for {run_key}")
    if metric not in group:
        raise ValueError(f"Trajectory does not contain {metric}")
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(group["actual_budget_tokens"], group[metric], marker="o")
    axis.set_xlabel("Cumulative evaluation tokens")
    axis.set_ylabel(metric)
    axis.set_title(run_key)
    axis.grid(True, linestyle="--", alpha=0.3)
    figure.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target
