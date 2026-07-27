"""Generate publication-style Tri-Fair v5 / Qwen-3-30B / 5M figures.

This is the v5 replacement for the dataset-specific BBQ v2 figure script.  It
uses the completed all-real-step holdout evaluations and produces a separate
publication directory for each dataset:

    BBQ
    Civil Comments
    Bias-in-Bios

The script never edits experimental results and never fabricates checkpoints.
Trajectory means use last-observation-carried-forward only at real states at or
below a shared token target.  By default, trajectories stop at 4.5M tokens,
which every v5 run reached.  Final-state figures use the last complete real
state available under the strict 5M cap.

Default Rocket usage
--------------------

    python -m analysis.make_v5_5m_figures --strict

Explicit usage
--------------

    python -m analysis.make_v5_5m_figures \
      --results-root results/tri_fair_v5_qwen_5m/qwen-3-30b \
      --analysis-root analysis/output/tri_fair_v5_qwen_5m \
      --figure-root analysis/output/tri_fair_v5_qwen_5m/publication_figures \
      --datasets bbq,civil_comments,bias_in_bios \
      --matched-cap 4500000 \
      --grid-step 250000 \
      --strict

Generated per-dataset outputs include:

* exact holdout nR2/HV/gap trajectories;
* balanced quality/cost/unfairness trajectories;
* final nR2/HV/gap comparisons;
* final balanced operating-point comparisons;
* Initial Instructions versus final operating points;
* empirical attainment curves;
* 2-D cost-unfairness Pareto projections;
* 3-D quality-cost-unfairness Pareto fronts;
* high-quality operating-point comparisons;
* final multi-metric scorecards and token-usage diagnostics;
* CSV/Markdown summary tables and a provenance manifest.

The configured cost objective is a weighted mean-token objective, not a
currency amount or Rocket GPU charge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL = "qwen-3-30b"
FINAL_BUDGET = 5_000_000
DEFAULT_MATCHED_CAP = 4_500_000
DEFAULT_GRID_STEP = 250_000
EXPECTED_SEEDS = (52, 53, 54)

DATASET_ORDER = ("bbq", "civil_comments", "bias_in_bios")
DATASET_DISPLAY = {
    "bbq": "BBQ",
    "civil_comments": "Civil Comments",
    "bias_in_bios": "Bias-in-Bios",
}
DATASET_SLUG = {
    "bbq": "bbq",
    "civil_comments": "civil_comments",
    "bias_in_bios": "bias_in_bios",
}
DEFAULT_QUALITY_THRESHOLDS = {
    "bbq": 0.90,
    "civil_comments": 0.80,
    "bias_in_bios": 0.45,
}

OPTIMIZER_ORDER = ("Tri-Fair-v5", "NSGAII-PO-Fair")
DISPLAY_NAME = {
    "Tri-Fair-v5": "Tri-Fair v5",
    "NSGAII-PO-Fair": "NSGA-II-PO-Fair",
    "Initial": "Initial Instructions",
}
COLORS = {
    "Tri-Fair-v5": "black",
    "NSGAII-PO-Fair": "#E69F00",
    "Initial": "0.55",
}
MARKERS = {
    "Tri-Fair-v5": "o",
    "NSGAII-PO-Fair": "s",
    "Initial": "*",
}

COST_AXIS_LABEL = "Weighted Mean-Token Cost ↓"
COST_SHORT_LABEL = "Cost Objective ↓"
QUALITY_AXIS_LABEL = "Test Quality ↑"
UNFAIRNESS_AXIS_LABEL = "Test Unfairness ↓"

TRAJECTORY_METRICS = (
    ("noisy_r2_3d", "Exact Holdout nR2 ↓", "nR2"),
    ("hv_test_optimistic_3d", "Optimistic Test HV ↑", "Hypervolume"),
    ("hv_test_pessimistic_3d", "Pessimistic Test HV ↑", "Hypervolume"),
    ("approximation_gap_3d", "Approximation Gap ↓", "Gap"),
)
BALANCED_METRICS = (
    ("balanced_test_quality", "Balanced Test Quality ↑", "Test Quality"),
    ("balanced_test_cost", "Balanced Test Cost ↓", COST_SHORT_LABEL),
    ("balanced_test_fairness", "Balanced Test Unfairness ↓", UNFAIRNESS_AXIS_LABEL),
)
SCORECARD_METRICS = (
    ("noisy_r2_3d", "nR2 ↓"),
    ("hv_test_optimistic_3d", "Optimistic HV ↑"),
    ("hv_test_pessimistic_3d", "Pessimistic HV ↑"),
    ("approximation_gap_3d", "Gap ↓"),
    ("balanced_test_quality", "Balanced Quality ↑"),
    ("balanced_test_cost", "Balanced Cost ↓"),
    ("balanced_test_fairness", "Balanced Unfairness ↓"),
    ("fairness_generalization_gap_abs", "|Fairness Gen. Gap| ↓"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="results/tri_fair_v5_qwen_5m/qwen-3-30b",
        help="Root containing the 18 v5 optimization runs and eval_checkpoints.parquet files.",
    )
    parser.add_argument(
        "--analysis-root",
        default="analysis/output/tri_fair_v5_qwen_5m",
        help="Root containing v5_checkpoint_run_metrics.csv and final_tables/.",
    )
    parser.add_argument(
        "--figure-root",
        default=None,
        help="Output root (default: <analysis-root>/publication_figures).",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASET_ORDER),
        help="Comma-separated datasets: bbq,civil_comments,bias_in_bios.",
    )
    parser.add_argument(
        "--matched-cap",
        type=int,
        default=DEFAULT_MATCHED_CAP,
        help="Shared token cap for trajectory figures.",
    )
    parser.add_argument(
        "--grid-step",
        type=int,
        default=DEFAULT_GRID_STEP,
        help="Token spacing for LOCF trajectory aggregation.",
    )
    parser.add_argument(
        "--quality-thresholds",
        default="bbq=0.90,civil_comments=0.80,bias_in_bios=0.45",
        help="Dataset-specific high-quality thresholds.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf,svg",
        help="Comma-separated figure formats.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on incomplete run matrices, missing evaluation files, or invalid rows.",
    )
    parser.add_argument(
        "--skip-pareto",
        action="store_true",
        help="Generate metric figures only; do not load raw eval_checkpoints.parquet files.",
    )
    return parser.parse_args(argv)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 400,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_csv(raw: str | Iterable[str]) -> tuple[str, ...]:
    values = (
        [piece.strip() for piece in raw.split(",") if piece.strip()]
        if isinstance(raw, str)
        else [str(value).strip() for value in raw if str(value).strip()]
    )
    return tuple(dict.fromkeys(values))


def parse_thresholds(raw: str) -> dict[str, float]:
    result = dict(DEFAULT_QUALITY_THRESHOLDS)
    for piece in str(raw).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(
                "--quality-thresholds entries must use dataset=value syntax"
            )
        dataset, value = piece.split("=", 1)
        dataset = dataset.strip()
        threshold = float(value)
        if dataset not in DATASET_ORDER:
            raise ValueError(f"Unknown threshold dataset: {dataset}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must lie in [0,1]: {dataset}={threshold}")
        result[dataset] = threshold
    return result


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def format_budget(value: int | float) -> str:
    number = float(value)
    if number >= 1_000_000:
        return f"{number / 1_000_000:g}M"
    if number >= 1_000:
        return f"{number / 1_000:g}K"
    return f"{number:g}"


def safe_markdown(frame: pd.DataFrame, path: Path, *, floatfmt: str = ".4f") -> None:
    try:
        text = frame.to_markdown(index=False, floatfmt=floatfmt) + "\n"
    except ImportError:
        text = "```text\n" + frame.to_string(index=False) + "\n```\n"
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "model" in out:
        out = out[out["model"].astype(str) == MODEL]
    if "dataset" in out:
        out = out[out["dataset"].astype(str).isin(DATASET_ORDER)]
    if "optimizer" in out:
        out = out[
            out["optimizer"].astype(str).isin((*OPTIMIZER_ORDER, "Initial"))
        ]
    if "seed" in out:
        seed = pd.to_numeric(out["seed"], errors="coerce")
        out = out[seed.isin(EXPECTED_SEEDS)].copy()
        out["seed"] = pd.to_numeric(out["seed"], errors="raise").astype(int)
    for column in ("budget_checkpoint", "actual_budget_tokens", "chosen_step"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.reset_index(drop=True)


def load_metric_inputs(analysis_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint_path = analysis_root / "v5_checkpoint_run_metrics.csv"
    final_path = analysis_root / "final_tables" / "final_available_run_metrics.csv"
    initial_path = analysis_root / "final_tables" / "initial_run_metrics.csv"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Run analysis.summarize_v5_checkpoints first."
        )

    checkpoint = clean_metric_frame(pd.read_csv(checkpoint_path))
    method_checkpoint = checkpoint[
        checkpoint["optimizer"].isin(OPTIMIZER_ORDER)
    ].copy()
    initial_from_checkpoint = checkpoint[
        checkpoint["optimizer"].eq("Initial")
    ].copy()

    if final_path.is_file():
        final = clean_metric_frame(pd.read_csv(final_path))
    else:
        final = select_final_available(method_checkpoint)

    if initial_path.is_file():
        initial = clean_metric_frame(pd.read_csv(initial_path))
    else:
        initial = initial_from_checkpoint

    return method_checkpoint, final, initial


def select_final_available(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frame,
        [
            "model",
            "dataset",
            "optimizer",
            "seed",
            "budget_checkpoint",
            "actual_budget_tokens",
        ],
        "checkpoint metrics",
    )
    ordered = frame.sort_values(
        [
            "model",
            "dataset",
            "optimizer",
            "seed",
            "actual_budget_tokens",
            "budget_checkpoint",
            "chosen_step",
        ],
        kind="mergesort",
    )
    return (
        ordered.groupby(
            ["model", "dataset", "optimizer", "seed"],
            sort=True,
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )


def validate_metric_matrix(
    final: pd.DataFrame,
    initial: pd.DataFrame,
    datasets: Sequence[str],
    *,
    strict: bool,
) -> None:
    expected = {
        (dataset, optimizer, seed)
        for dataset in datasets
        for optimizer in OPTIMIZER_ORDER
        for seed in EXPECTED_SEEDS
    }
    observed = {
        (str(row.dataset), str(row.optimizer), int(row.seed))
        for row in final[["dataset", "optimizer", "seed"]].itertuples(index=False)
    }
    missing = sorted(expected - observed)
    duplicates = (
        final.groupby(["dataset", "optimizer", "seed"]).size().loc[lambda x: x > 1]
    )
    if missing or not duplicates.empty:
        message = []
        if missing:
            message.append("Missing final runs: " + ", ".join(map(str, missing)))
        if not duplicates.empty:
            message.append("Duplicate final runs:\n" + duplicates.to_string())
        if strict:
            raise RuntimeError("\n".join(message))
        warnings.warn("\n".join(message))

    if not initial.empty:
        expected_initial = {
            (dataset, seed) for dataset in datasets for seed in EXPECTED_SEEDS
        }
        observed_initial = {
            (str(row.dataset), int(row.seed))
            for row in initial[["dataset", "seed"]].itertuples(index=False)
        }
        missing_initial = sorted(expected_initial - observed_initial)
        if missing_initial:
            message = "Missing Initial rows: " + ", ".join(map(str, missing_initial))
            if strict:
                raise RuntimeError(message)
            warnings.warn(message)


def _infer_path_metadata(path: Path) -> dict[str, object]:
    parts = path.parts
    metadata: dict[str, object] = {}
    for dataset in DATASET_ORDER:
        if dataset in parts:
            metadata["dataset"] = dataset
            index = parts.index(dataset)
            if index + 1 < len(parts):
                metadata["optimizer"] = parts[index + 1]
            if index + 2 < len(parts) and parts[index + 2].startswith("seed"):
                metadata["seed"] = int(parts[index + 2].removeprefix("seed"))
            break
    metadata["model"] = MODEL
    return metadata


def load_all_evaluations(
    results_root: Path,
    datasets: Sequence[str],
    *,
    strict: bool,
) -> pd.DataFrame:
    files = sorted(results_root.rglob("eval_checkpoints.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No eval_checkpoints.parquet files found beneath {results_root}"
        )

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for path in files:
        try:
            frame = pd.read_parquet(path).copy()
        except Exception as error:
            errors.append(f"{path}: {type(error).__name__}: {error}")
            continue

        inferred = _infer_path_metadata(path)
        for key, value in inferred.items():
            if key not in frame:
                frame[key] = value
        frame["source_file"] = str(path.resolve())
        frame["run_dir"] = str(path.parent.resolve())
        frames.append(frame)

    if errors:
        message = "Unreadable evaluation files:\n" + "\n".join(errors)
        if strict:
            raise RuntimeError(message)
        warnings.warn(message)
    if not frames:
        raise RuntimeError("No readable eval_checkpoints.parquet files")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined[
        combined["dataset"].astype(str).isin(datasets)
        & combined["optimizer"].astype(str).isin(OPTIMIZER_ORDER)
    ].copy()
    combined["seed"] = pd.to_numeric(combined["seed"], errors="coerce")
    combined = combined[combined["seed"].isin(EXPECTED_SEEDS)].copy()
    combined["seed"] = combined["seed"].astype(int)
    for column in ("budget_checkpoint", "actual_budget_tokens", "chosen_step"):
        if column in combined:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined.reset_index(drop=True)


def _json_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, float) and math.isnan(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def attach_validity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dev_ready = (
        out["dev_fairness_ready"].fillna(False).astype(bool)
        if "dev_fairness_ready" in out
        else pd.Series(True, index=out.index)
    )
    test_ready = (
        out["test_fairness_ready"].fillna(False).astype(bool)
        if "test_fairness_ready" in out
        else pd.Series(True, index=out.index)
    )
    out["dev_coverage_valid"] = True
    out["test_coverage_valid"] = True

    bbq_mask = out["dataset"].astype(str).eq("bbq")
    for split in ("dev", "test"):
        column = f"{split}_fairness_diagnostics_json"
        if column in out:
            parsed = out.loc[bbq_mask, column].map(
                lambda value: bool(_json_object(value).get("coverage_valid", False))
            )
            out.loc[bbq_mask, f"{split}_coverage_valid"] = parsed.astype(bool)

    out["publication_valid"] = (
        dev_ready
        & test_ready
        & out["dev_coverage_valid"].astype(bool)
        & out["test_coverage_valid"].astype(bool)
    )
    return out


def select_final_evaluations(
    evaluations: pd.DataFrame,
    final_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        evaluations,
        [
            "dataset",
            "optimizer",
            "seed",
            "budget_checkpoint",
            "test_quality",
            "test_cost",
            "test_fairness",
        ],
        "eval_checkpoints.parquet",
    )
    selected_frames: list[pd.DataFrame] = []
    for row in final_metrics.itertuples(index=False):
        group = evaluations[
            (evaluations["dataset"].astype(str) == str(row.dataset))
            & (evaluations["optimizer"].astype(str) == str(row.optimizer))
            & (pd.to_numeric(evaluations["seed"], errors="coerce") == int(row.seed))
        ].copy()
        if group.empty:
            continue
        budget = int(row.budget_checkpoint)
        exact = group[
            pd.to_numeric(group["budget_checkpoint"], errors="coerce") == budget
        ].copy()
        if exact.empty and "actual_budget_tokens" in group:
            exact = group[
                pd.to_numeric(group["actual_budget_tokens"], errors="coerce")
                == int(row.actual_budget_tokens)
            ].copy()
        if exact.empty:
            available = pd.to_numeric(
                group["budget_checkpoint"], errors="coerce"
            ).dropna()
            if not available.empty:
                chosen = int(available.max())
                exact = group[
                    pd.to_numeric(group["budget_checkpoint"], errors="coerce")
                    == chosen
                ].copy()
        if not exact.empty:
            selected_frames.append(exact)

    if not selected_frames:
        raise RuntimeError("Could not select any final evaluation candidates")

    raw = attach_validity(
        pd.concat(selected_frames, ignore_index=True, sort=False)
    )
    valid = raw[raw["publication_valid"]].copy().reset_index(drop=True)
    return raw.reset_index(drop=True), valid


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Pareto input must be two-dimensional")
    keep = np.all(np.isfinite(values), axis=1)
    for index in np.flatnonzero(keep):
        candidate = values[index]
        others = values[keep]
        dominated = np.all(others <= candidate, axis=1) & np.any(
            others < candidate, axis=1
        )
        if np.any(dominated):
            keep[index] = False
    return keep


def test_objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            1.0 - numeric(frame["test_quality"]),
            numeric(frame["test_cost"]),
            numeric(frame["test_fairness"]),
        ]
    )


def final_pareto_rows(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = test_objective_matrix(frame)
    valid = np.all(np.isfinite(matrix), axis=1)
    subset = frame.loc[valid].reset_index(drop=True)
    matrix = matrix[valid]
    if subset.empty:
        return subset
    return subset.loc[pareto_mask_minimize(matrix)].reset_index(drop=True)


def build_locf_grid(
    checkpoint: pd.DataFrame,
    dataset: str,
    *,
    matched_cap: int,
    grid_step: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = checkpoint[
        (checkpoint["dataset"].astype(str) == dataset)
        & checkpoint["optimizer"].isin(OPTIMIZER_ORDER)
    ].copy()
    require_columns(
        data,
        ["optimizer", "seed", "actual_budget_tokens"],
        "checkpoint metrics",
    )
    starts = []
    for _, group in data.groupby(["optimizer", "seed"], sort=True):
        values = pd.to_numeric(
            group["actual_budget_tokens"], errors="coerce"
        ).dropna()
        if not values.empty:
            starts.append(int(values.min()))
    if not starts:
        raise RuntimeError(f"No trajectory states for {dataset}")

    # Start at the earliest real state seen in the dataset.  Each method's
    # aggregate curve appears only after all three of its seeds have reached a
    # real state, so early Tri-Fair progress is not discarded merely because
    # NSGA-II's first generation is larger.
    start = int(math.ceil(min(starts) / grid_step) * grid_step)
    if start > matched_cap:
        start = int(min(starts))
    targets = np.arange(start, matched_cap + 1, grid_step, dtype=int)
    if len(targets) == 0 or targets[-1] != matched_cap:
        targets = np.unique(np.append(targets, matched_cap)).astype(int)

    seed_rows: list[dict[str, object]] = []
    metric_columns = [
        metric for metric, _, _ in (*TRAJECTORY_METRICS, *BALANCED_METRICS)
    ]

    for optimizer in OPTIMIZER_ORDER:
        for seed in EXPECTED_SEEDS:
            run = data[
                (data["optimizer"] == optimizer)
                & (pd.to_numeric(data["seed"], errors="coerce") == seed)
            ].copy()
            run = run.sort_values(
                ["actual_budget_tokens", "budget_checkpoint", "chosen_step"],
                kind="mergesort",
            )
            for target in targets:
                eligible = run[
                    pd.to_numeric(
                        run["actual_budget_tokens"], errors="coerce"
                    )
                    <= target
                ]
                if eligible.empty:
                    continue
                selected = eligible.iloc[-1]
                row: dict[str, object] = {
                    "dataset": dataset,
                    "optimizer": optimizer,
                    "seed": seed,
                    "target_tokens": int(target),
                    "selected_actual_tokens": int(selected["actual_budget_tokens"]),
                    "selected_budget_checkpoint": int(selected["budget_checkpoint"]),
                    "selected_step": int(selected.get("chosen_step", 0)),
                }
                for metric in metric_columns:
                    row[metric] = selected.get(metric, np.nan)
                seed_rows.append(row)

    seed_grid = pd.DataFrame(seed_rows)
    if seed_grid.empty:
        raise RuntimeError(f"No LOCF trajectory rows for {dataset}")

    summary_rows: list[dict[str, object]] = []
    for (optimizer, target), group in seed_grid.groupby(
        ["optimizer", "target_tokens"], sort=True
    ):
        if int(group["seed"].nunique()) < len(EXPECTED_SEEDS):
            continue
        row = {
            "dataset": dataset,
            "optimizer": optimizer,
            "target_tokens": int(target),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summary_rows.append(row)
    return seed_grid, pd.DataFrame(summary_rows)


def save_figure(
    fig: plt.Figure,
    outdir: Path,
    stem: str,
    formats: Sequence[str],
) -> None:
    for extension in formats:
        kwargs: dict[str, object] = {"bbox_inches": "tight"}
        if extension.lower() == "png":
            kwargs["dpi"] = 400
        fig.savefig(outdir / f"{stem}.{extension}", **kwargs)
    plt.close(fig)


def add_initial_point(
    ax: plt.Axes,
    initial: pd.DataFrame,
    metric: str,
    *,
    label: bool,
) -> None:
    if initial.empty or metric not in initial:
        return
    values = pd.to_numeric(initial[metric], errors="coerce").dropna()
    if values.empty:
        return
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    ax.errorbar(
        [0.0],
        [mean],
        yerr=[std],
        color=COLORS["Initial"],
        marker=MARKERS["Initial"],
        markersize=8,
        capsize=3,
        linestyle="none",
        label=DISPLAY_NAME["Initial"] if label else None,
        zorder=6,
    )


def plot_metric_trajectories(
    seed_grid: pd.DataFrame,
    summary_grid: pd.DataFrame,
    initial: pd.DataFrame,
    dataset: str,
    outdir: Path,
    formats: Sequence[str],
    *,
    matched_cap: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.7), constrained_layout=True)
    for metric_index, (ax, (metric, title, ylabel)) in enumerate(
        zip(axes.flat, TRAJECTORY_METRICS)
    ):
        add_initial_point(ax, initial, metric, label=metric_index == 0)
        for optimizer in OPTIMIZER_ORDER:
            method_seed = seed_grid[seed_grid["optimizer"] == optimizer]
            for _, group in method_seed.groupby("seed", sort=True):
                group = group.sort_values("target_tokens")
                ax.step(
                    group["target_tokens"] / 1_000_000.0,
                    group[metric],
                    where="post",
                    color=COLORS[optimizer],
                    alpha=0.18,
                    linewidth=0.9,
                )

            group = summary_grid[
                summary_grid["optimizer"] == optimizer
            ].sort_values("target_tokens")
            if group.empty:
                continue
            x = group["target_tokens"].to_numpy(dtype=float) / 1_000_000.0
            mean = pd.to_numeric(
                group[f"{metric}_mean"], errors="coerce"
            ).to_numpy(dtype=float)
            std = pd.to_numeric(
                group[f"{metric}_std"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
            valid = np.isfinite(mean)
            ax.step(
                x[valid],
                mean[valid],
                where="post",
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                markevery=max(1, int(valid.sum() / 7)),
                linewidth=2.2,
                markersize=4.5,
                label=DISPLAY_NAME[optimizer],
            )
            ax.fill_between(
                x[valid],
                (mean - std)[valid],
                (mean + std)[valid],
                step="post",
                color=COLORS[optimizer],
                alpha=0.14,
            )
        ax.set_title(title)
        ax.set_xlabel("Cumulative Downstream Tokens [×10⁶]")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, matched_cap / 1_000_000.0)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle(
        f"{DATASET_DISPLAY[dataset]} — Qwen-3-30B v5 Exact Holdout Trajectories"
    )
    stem = f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_holdout_nr2_hv_gap_trajectory"
    save_figure(fig, outdir, stem, formats)


def plot_balanced_trajectories(
    seed_grid: pd.DataFrame,
    summary_grid: pd.DataFrame,
    initial: pd.DataFrame,
    dataset: str,
    outdir: Path,
    formats: Sequence[str],
    *,
    matched_cap: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), constrained_layout=True)
    for metric_index, (ax, (metric, title, ylabel)) in enumerate(
        zip(axes, BALANCED_METRICS)
    ):
        add_initial_point(ax, initial, metric, label=metric_index == 0)
        for optimizer in OPTIMIZER_ORDER:
            method_seed = seed_grid[seed_grid["optimizer"] == optimizer]
            for _, group in method_seed.groupby("seed", sort=True):
                group = group.sort_values("target_tokens")
                ax.step(
                    group["target_tokens"] / 1_000_000.0,
                    group[metric],
                    where="post",
                    color=COLORS[optimizer],
                    alpha=0.18,
                    linewidth=0.9,
                )
            group = summary_grid[
                summary_grid["optimizer"] == optimizer
            ].sort_values("target_tokens")
            if group.empty:
                continue
            x = group["target_tokens"].to_numpy(dtype=float) / 1_000_000.0
            mean = pd.to_numeric(
                group[f"{metric}_mean"], errors="coerce"
            ).to_numpy(dtype=float)
            std = pd.to_numeric(
                group[f"{metric}_std"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype=float)
            valid = np.isfinite(mean)
            ax.step(
                x[valid],
                mean[valid],
                where="post",
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                markevery=max(1, int(valid.sum() / 7)),
                linewidth=2.2,
                markersize=4.5,
                label=DISPLAY_NAME[optimizer],
            )
            ax.fill_between(
                x[valid],
                (mean - std)[valid],
                (mean + std)[valid],
                step="post",
                color=COLORS[optimizer],
                alpha=0.14,
            )
        ax.set_title(title)
        ax.set_xlabel("Cumulative Downstream Tokens [×10⁶]")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, matched_cap / 1_000_000.0)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        f"{DATASET_DISPLAY[dataset]} — Qwen-3-30B v5 Balanced Holdout Trajectories"
    )
    stem = f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_balanced_holdout_trajectory"
    save_figure(fig, outdir, stem, formats)


def method_errorbar_panel(
    ax: plt.Axes,
    final: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    positions = np.arange(len(OPTIMIZER_ORDER), dtype=float)
    means: list[float] = []
    stds: list[float] = []
    for optimizer in OPTIMIZER_ORDER:
        values = pd.to_numeric(
            final.loc[final["optimizer"] == optimizer, metric],
            errors="coerce",
        ).dropna()
        means.append(float(values.mean()))
        stds.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)

    bars = ax.bar(
        positions,
        means,
        yerr=stds,
        capsize=4,
        color=[COLORS[optimizer] for optimizer in OPTIMIZER_ORDER],
        alpha=0.80,
        width=0.62,
    )
    seed_jitter = {-1: -0.07, 0: 0.0, 1: 0.07}
    for position, optimizer in zip(positions, OPTIMIZER_ORDER):
        group = final[final["optimizer"] == optimizer].sort_values("seed")
        centre = int(len(group) // 2)
        for index, (_, row) in enumerate(group.iterrows()):
            ax.scatter(
                position + seed_jitter.get(index - centre, 0.0),
                row[metric],
                color="white" if optimizer == "Tri-Fair-v5" else "black",
                edgecolor="black",
                marker=MARKERS[optimizer],
                s=32,
                linewidth=0.7,
                zorder=5,
            )
    ax.set_xticks(
        positions,
        [DISPLAY_NAME[optimizer] for optimizer in OPTIMIZER_ORDER],
        rotation=7,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    for bar in bars:
        bar.set_zorder(2)


def plot_final_mo_metrics(
    final: pd.DataFrame,
    dataset: str,
    outdir: Path,
    formats: Sequence[str],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    specs = (
        ("noisy_r2_3d", "Final Holdout nR2 ↓", "nR2"),
        ("hv_test_pessimistic_3d", "Final Pessimistic Test HV ↑", "Hypervolume"),
        ("approximation_gap_3d", "Final Approximation Gap ↓", "Gap"),
    )
    for ax, spec in zip(axes, specs):
        method_errorbar_panel(ax, final, *spec)
    fig.suptitle(
        f"{DATASET_DISPLAY[dataset]} — Qwen-3-30B v5 Final Available States"
    )
    stem = f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_final_nr2_hv_gap"
    save_figure(fig, outdir, stem, formats)


def plot_final_balanced(
    final: pd.DataFrame,
    dataset: str,
    outdir: Path,
    formats: Sequence[str],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes, BALANCED_METRICS):
        method_errorbar_panel(ax, final, metric, title, ylabel)
    fig.suptitle(
        f"{DATASET_DISPLAY[dataset]} — Qwen-3-30B v5 Final Balanced Operating Point"
    )
    stem = f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_final_balanced_metrics"
    save_figure(fig, outdir, stem, formats)


def initial_final_panel(
    ax: plt.Axes,
    initial: pd.DataFrame,
    final: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    categories = ("Initial", *OPTIMIZER_ORDER)
    positions = np.arange(len(categories))
    for position, category in zip(positions, categories):
        group = initial if category == "Initial" else final[final["optimizer"] == category]
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ax.errorbar(
            [position],
            [mean],
            yerr=[std],
            marker=MARKERS[category],
            color=COLORS[category],
            markersize=8,
            capsize=4,
            linestyle="none",
            zorder=4,
        )
        jitter = np.linspace(-0.07, 0.07, len(values))
        ax.scatter(
            position + jitter,
            values,
            color=COLORS[category],
            marker=MARKERS[category],
            alpha=0.45,
            s=28,
            zorder=3,
        )
    ax.set_xticks(
        positions,
        [DISPLAY_NAME[category] for category in categories],
        rotation=8,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def plot_initial_vs_final(
    initial: pd.DataFrame,
    final: pd.DataFrame,
    dataset: str,
    outdir: Path,
    formats: Sequence[str],
) -> None:
    if initial.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes, BALANCED_METRICS):
        initial_final_panel(ax, initial, final, metric, title, ylabel)
    fig.suptitle(
        f"{DATASET_DISPLAY[dataset]} — Shared Initial Instructions vs Final States"
    )
    stem = f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_initial_vs_final_balanced"
    save_figure(fig, outdir, stem, formats)


def y_attained_at_x(
    data: pd.DataFrame,
    x_grid: np.ndarray,
    x_col: str,
) -> np.ndarray:
    x = numeric(data[x_col])
    y = numeric(data["test_quality"])
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    result = np.full(len(x_grid), np.nan)
    for index, target in enumerate(x_grid):
        eligible = y[x <= target]
        if len(eligible):
            result[index] = float(np.max(eligible))
    return result


def plot_empirical_attainment(
    final: pd.DataFrame,
    dataset: str,
    outdir: Path,
    formats: Sequence[str],
    *,
    x_col: str,
    xlabel: str,
    suffix: str,
) -> None:
    values = pd.to_numeric(final[x_col], errors="coerce").dropna()
    if values.empty:
        return
    xmin, xmax = float(values.min()), float(values.max())
    padding = 0.03 * max(xmax - xmin, 1e-6)
    x_grid = np.linspace(xmin - padding, xmax + padding, 450)

    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    for optimizer in OPTIMIZER_ORDER:
        curves: list[np.ndarray] = []
        method = final[final["optimizer"] == optimizer]
        for _, seed_group in method.groupby("seed", sort=True):
            front = final_pareto_rows(seed_group)
            if not front.empty:
                curves.append(y_attained_at_x(front, x_grid, x_col))
        if not curves:
            continue
        matrix = np.vstack(curves)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(matrix, axis=0)
            lower = np.nanmin(matrix, axis=0)
            upper = np.nanmax(matrix, axis=0)
        valid = np.isfinite(median)
        ax.step(
            x_grid[valid],
            median[valid],
            where="post",
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            markevery=max(1, int(valid.sum() / 8)),
            linewidth=2.1,
            markersize=4.5,
            label=DISPLAY_NAME[optimizer],
        )
        ax.fill_between(
            x_grid[valid],
            lower[valid],
            upper[valid],
            step="post",
            color=COLORS[optimizer],
            alpha=0.15,
        )
    ax.set_title(f"{DATASET_DISPLAY[dataset]} — Final Empirical Attainment")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(QUALITY_AXIS_LABEL)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    stem = (
        f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_attainment_quality_{suffix}"
    )
    save_figure(fig, outdir, stem, formats)


def plot_cost_unfairness_pareto(
    final: pd.DataFrame,
    dataset: str,
    outdir: Path,
    formats: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.7), constrained_layout=True)
    for optimizer in OPTIMIZER_ORDER:
        pooled: list[pd.DataFrame] = []
        method = final[final["optimizer"] == optimizer]
        for _, seed_group in method.groupby("seed", sort=True):
            front = final_pareto_rows(seed_group)
            if front.empty:
                continue
            pooled.append(front)
            ax.scatter(
                front["test_cost"],
                front["test_fairness"],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                alpha=0.22,
                s=25,
            )
        if not pooled:
            continue
        combined = pd.concat(pooled, ignore_index=True)
        projection = np.column_stack(
            [numeric(combined["test_cost"]), numeric(combined["test_fairness"])]
        )
        valid = np.all(np.isfinite(projection), axis=1)
        combined = combined.loc[valid].reset_index(drop=True)
        projection = projection[valid]
        projected_front = combined.loc[
            pareto_mask_minimize(projection)
        ].sort_values("test_cost")
        ax.plot(
            projected_front["test_cost"],
            projected_front["test_fairness"],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            linewidth=2.1,
            markersize=5,
            label=DISPLAY_NAME[optimizer],
        )
    ax.set_title(f"{DATASET_DISPLAY[dataset]} — Final Cost vs Unfairness Pareto")
    ax.set_xlabel(COST_AXIS_LABEL)
    ax.set_ylabel(UNFAIRNESS_AXIS_LABEL)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    stem = f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_pareto_cost_unfairness"
    save_figure(fig, outdir, stem, formats)


def plot_three_objective_pareto(
    final: pd.DataFrame,
    dataset: str,
    outdir: Path,
    formats: Sequence[str],
) -> None:
    fig = plt.figure(figsize=(7.4, 5.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for optimizer in OPTIMIZER_ORDER:
        method = final[final["optimizer"] == optimizer]
        first = True
        for _, seed_group in method.groupby("seed", sort=True):
            front = final_pareto_rows(seed_group)
            if front.empty:
                continue
            ax.scatter(
                front["test_cost"],
                front["test_fairness"],
                front["test_quality"],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                alpha=0.76,
                s=34,
                label=DISPLAY_NAME[optimizer] if first else None,
            )
            first = False
    ax.set_title(
        f"{DATASET_DISPLAY[dataset]} — Qwen-3-30B v5 Final Test Pareto Fronts"
    )
    ax.set_xlabel(COST_SHORT_LABEL)
    ax.set_ylabel(UNFAIRNESS_AXIS_LABEL)
    ax.set_zlabel(QUALITY_AXIS_LABEL)
    ax.legend(frameon=False, loc="best")
    stem = f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_test_pareto_3d"
    save_figure(fig, outdir, stem, formats)


def high_quality_operating_points(
    final: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for optimizer in OPTIMIZER_ORDER:
        for seed in EXPECTED_SEEDS:
            group = final[
                (final["optimizer"] == optimizer)
                & (pd.to_numeric(final["seed"], errors="coerce") == seed)
            ].copy()
            group["_quality"] = pd.to_numeric(group["test_quality"], errors="coerce")
            group["_cost"] = pd.to_numeric(group["test_cost"], errors="coerce")
            group["_fairness"] = pd.to_numeric(
                group["test_fairness"], errors="coerce"
            )
            group = group.dropna(subset=["_quality", "_cost", "_fairness"])
            eligible = group[group["_quality"] >= threshold].copy()
            threshold_met = not eligible.empty
            candidate = eligible if threshold_met else group
            candidate = candidate.sort_values(
                ["_fairness", "_cost", "_quality"],
                ascending=[True, True, False],
            )
            if candidate.empty:
                continue
            selected = candidate.iloc[0]
            rows.append(
                {
                    "optimizer": optimizer,
                    "method": DISPLAY_NAME[optimizer],
                    "seed": seed,
                    "threshold": float(threshold),
                    "threshold_met": bool(threshold_met),
                    "test_quality": float(selected["_quality"]),
                    "test_cost": float(selected["_cost"]),
                    "test_fairness": float(selected["_fairness"]),
                    "prompt_id": selected.get("prompt_id", ""),
                    "prompt": selected.get("prompt", ""),
                }
            )
    return pd.DataFrame(rows)


def plot_high_quality_operating_points(
    final: pd.DataFrame,
    dataset: str,
    outdir: Path,
    formats: Sequence[str],
    *,
    threshold: float,
) -> pd.DataFrame:
    points = high_quality_operating_points(final, threshold=threshold)
    path = (
        outdir
        / f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_high_quality_operating_points.csv"
    )
    points.to_csv(path, index=False)
    if points.empty:
        return points

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    specs = (
        ("test_quality", "Quality ↑", QUALITY_AXIS_LABEL),
        ("test_cost", "Cost ↓", COST_SHORT_LABEL),
        ("test_fairness", "Unfairness ↓", UNFAIRNESS_AXIS_LABEL),
    )
    for ax, (column, title, ylabel) in zip(axes, specs):
        for optimizer in OPTIMIZER_ORDER:
            group = points[points["optimizer"] == optimizer].sort_values("seed")
            ax.plot(
                group["seed"],
                group[column],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                linewidth=2,
                markersize=6,
                label=DISPLAY_NAME[optimizer],
            )
        ax.set_xticks(EXPECTED_SEEDS)
        ax.set_xlabel("Seed")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].axhline(threshold, color="0.5", linestyle="--", linewidth=1)
    axes[0].legend(frameon=False)
    fig.suptitle(
        f"{DATASET_DISPLAY[dataset]} — Lowest Unfairness with Test Quality ≥ {threshold:.2f}"
    )
    stem = (
        f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_high_quality_operating_points"
    )
    save_figure(fig, outdir, stem, formats)
    return points


def plot_final_scorecard(
    final: pd.DataFrame,
    dataset: str,
    outdir: Path,
    formats: Sequence[str],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.0), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, SCORECARD_METRICS):
        if metric not in final:
            ax.set_axis_off()
            continue
        method_errorbar_panel(ax, final, metric, title, title)
    fig.suptitle(
        f"{DATASET_DISPLAY[dataset]} — Qwen-3-30B v5 Final Holdout Scorecard"
    )
    stem = f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_final_scorecard"
    save_figure(fig, outdir, stem, formats)


def plot_final_token_usage(
    final: pd.DataFrame,
    dataset: str,
    outdir: Path,
    formats: Sequence[str],
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    offsets = {"Tri-Fair-v5": -0.08, "NSGAII-PO-Fair": 0.08}
    for optimizer in OPTIMIZER_ORDER:
        group = final[final["optimizer"] == optimizer].sort_values("seed")
        x = group["seed"].to_numpy(dtype=float) + offsets[optimizer]
        y = pd.to_numeric(
            group["actual_budget_tokens"], errors="coerce"
        ).to_numpy(dtype=float) / 1_000_000.0
        ax.plot(
            x,
            y,
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            linewidth=1.8,
            markersize=6,
            label=DISPLAY_NAME[optimizer],
        )
    ax.axhline(5.0, color="0.5", linestyle="--", linewidth=1, label="5M cap")
    ax.axhline(4.5, color="0.7", linestyle=":", linewidth=1, label="4.5M matched cap")
    ax.set_xticks(EXPECTED_SEEDS)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Final Real State [million downstream tokens]")
    ax.set_title(f"{DATASET_DISPLAY[dataset]} — Final Token Utilization")
    ax.set_ylim(bottom=0.0, top=5.15)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    stem = f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_final_token_usage"
    save_figure(fig, outdir, stem, formats)


def plot_bbq_coverage_diagnostics(
    raw_final: pd.DataFrame,
    dataset: str,
    outdir: Path,
    formats: Sequence[str],
) -> None:
    if dataset != "bbq" or raw_final.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    coverage_rows = []
    for (optimizer, seed), group in raw_final.groupby(
        ["optimizer", "seed"], sort=True
    ):
        coverage_rows.append(
            {
                "optimizer": optimizer,
                "seed": seed,
                "coverage_fraction": float(
                    group["publication_valid"].fillna(False).astype(bool).mean()
                ),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    for optimizer in OPTIMIZER_ORDER:
        group = coverage[coverage["optimizer"] == optimizer].sort_values("seed")
        axes[0].plot(
            group["seed"],
            group["coverage_fraction"],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            linewidth=2,
            label=DISPLAY_NAME[optimizer],
        )
    axes[0].set_xticks(EXPECTED_SEEDS)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Publication-Valid Candidate Fraction")
    axes[0].set_title("BBQ Coverage Validity")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    invalid = raw_final[~raw_final["publication_valid"].astype(bool)]
    valid = raw_final[raw_final["publication_valid"].astype(bool)]
    for optimizer in OPTIMIZER_ORDER:
        for frame, alpha, label_suffix in (
            (invalid, 0.18, "invalid"),
            (valid, 0.65, "valid"),
        ):
            group = frame[frame["optimizer"] == optimizer]
            axes[1].scatter(
                group["test_fairness"],
                group["test_quality"],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                alpha=alpha,
                s=28,
                label=f"{DISPLAY_NAME[optimizer]} {label_suffix}",
            )
    axes[1].set_xlabel(UNFAIRNESS_AXIS_LABEL)
    axes[1].set_ylabel(QUALITY_AXIS_LABEL)
    axes[1].set_title("Coverage Validity and Test Objectives")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7.5)

    fig.suptitle("BBQ — Qwen-3-30B v5 Coverage Diagnostics")
    stem = f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_coverage_diagnostics"
    save_figure(fig, outdir, stem, formats)
    coverage.to_csv(
        outdir / f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_coverage_summary.csv",
        index=False,
    )


def summarize_final(
    final: pd.DataFrame,
    valid_final: pd.DataFrame,
    raw_final: pd.DataFrame,
    high_quality: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for optimizer in OPTIMIZER_ORDER:
        metrics = final[final["optimizer"] == optimizer]
        valid = (
            valid_final[valid_final["optimizer"] == optimizer]
            if "optimizer" in valid_final
            else pd.DataFrame()
        )
        raw = (
            raw_final[raw_final["optimizer"] == optimizer]
            if "optimizer" in raw_final
            else pd.DataFrame()
        )
        high = (
            high_quality[high_quality["optimizer"] == optimizer]
            if "optimizer" in high_quality
            else pd.DataFrame()
        )
        row: dict[str, object] = {
            "Method": DISPLAY_NAME[optimizer],
            "Seeds": int(metrics["seed"].nunique()),
            "Actual tokens mean": metrics["actual_budget_tokens"].mean(),
            "Actual tokens min": metrics["actual_budget_tokens"].min(),
            "Actual tokens max": metrics["actual_budget_tokens"].max(),
            "Raw final candidates": int(len(raw)),
            "Publication-valid final candidates": int(len(valid)),
        }
        for metric, label in SCORECARD_METRICS:
            if metric in metrics:
                values = pd.to_numeric(metrics[metric], errors="coerce")
                row[f"{label} mean"] = values.mean()
                row[f"{label} SD"] = values.std(ddof=1)
        if not high.empty:
            row["High-quality point quality mean ↑"] = high["test_quality"].mean()
            row["High-quality point cost mean ↓"] = high["test_cost"].mean()
            row["High-quality point unfairness mean ↓"] = high[
                "test_fairness"
            ].mean()
            row["High-quality threshold met fraction"] = high[
                "threshold_met"
            ].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def write_dataset_readme(
    outdir: Path,
    dataset: str,
    *,
    results_root: Path,
    analysis_root: Path,
    matched_cap: int,
    grid_step: int,
    quality_threshold: float,
    formats: Sequence[str],
) -> None:
    content = f"""# {DATASET_DISPLAY[dataset]} / Qwen-3-30B / Tri-Fair v5 publication figures

Generated with:

`python -m analysis.make_v5_5m_figures --strict`

## Sources

- Results root: `{results_root}`
- Metric root: `{analysis_root}`
- Seeds: {", ".join(str(seed) for seed in EXPECTED_SEEDS)}
- Methods: Tri-Fair v5 and NSGA-II-PO-Fair
- Final analysis: last complete real optimizer state under the strict 5M cap
- Trajectory analysis: LOCF over real evaluated states through {matched_cap:,} tokens
- Trajectory grid step: {grid_step:,} tokens
- High-quality threshold: {quality_threshold:.2f}
- Formats: {", ".join(formats)}

## Interpretation

- Higher is better for test quality and hypervolume.
- Lower is better for cost, unfairness, nR2, and approximation gap.
- Cost is the configured weighted mean-token objective, not a currency amount.
- Thin lines/points represent independent seeds; thick lines and bands are
  three-seed means and sample standard deviations.
- No checkpoint is interpolated or fabricated.
"""
    (outdir / "README.md").write_text(content, encoding="utf-8")


def build_manifest(
    outdir: Path,
    dataset: str,
    final: pd.DataFrame,
    *,
    results_root: Path,
    analysis_root: Path,
    matched_cap: int,
    grid_step: int,
    quality_threshold: float,
) -> None:
    files = sorted(path for path in outdir.iterdir() if path.is_file())
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "dataset": dataset,
        "results_root": str(results_root.resolve()),
        "analysis_root": str(analysis_root.resolve()),
        "optimizers": list(OPTIMIZER_ORDER),
        "seeds": list(EXPECTED_SEEDS),
        "strict_budget_cap": FINAL_BUDGET,
        "matched_trajectory_cap": int(matched_cap),
        "trajectory_grid_step": int(grid_step),
        "quality_threshold": float(quality_threshold),
        "selected_final_states": [
            {
                "optimizer": str(row.optimizer),
                "seed": int(row.seed),
                "budget_checkpoint": int(row.budget_checkpoint),
                "actual_budget_tokens": int(row.actual_budget_tokens),
                "chosen_step": int(row.chosen_step),
            }
            for row in final.sort_values(["optimizer", "seed"]).itertuples(index=False)
        ],
        "files": {
            path.name: {
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256(path),
            }
            for path in files
        },
    }
    (outdir / "figure_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def process_dataset(
    dataset: str,
    *,
    checkpoint: pd.DataFrame,
    final: pd.DataFrame,
    initial: pd.DataFrame,
    all_evaluations: pd.DataFrame | None,
    figure_root: Path,
    results_root: Path,
    analysis_root: Path,
    matched_cap: int,
    grid_step: int,
    quality_threshold: float,
    formats: Sequence[str],
    strict: bool,
) -> None:
    dataset_final = final[final["dataset"] == dataset].copy()
    dataset_initial = initial[initial["dataset"] == dataset].copy()
    dataset_checkpoint = checkpoint[checkpoint["dataset"] == dataset].copy()
    if dataset_final.empty:
        raise RuntimeError(f"No final metrics for {dataset}")

    outdir = figure_root / DATASET_SLUG[dataset]
    outdir.mkdir(parents=True, exist_ok=True)

    selected_columns = [
        "model",
        "dataset",
        "optimizer",
        "seed",
        "budget_checkpoint",
        "chosen_step",
        "actual_budget_tokens",
    ]
    dataset_final.to_csv(
        outdir / f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_selected_final_states.csv",
        index=False,
    )

    seed_grid, summary_grid = build_locf_grid(
        dataset_checkpoint,
        dataset,
        matched_cap=matched_cap,
        grid_step=grid_step,
    )
    seed_grid.to_csv(
        outdir / f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_trajectory_seed_grid.csv",
        index=False,
    )
    summary_grid.to_csv(
        outdir / f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_trajectory_summary.csv",
        index=False,
    )

    plot_metric_trajectories(
        seed_grid,
        summary_grid,
        dataset_initial,
        dataset,
        outdir,
        formats,
        matched_cap=matched_cap,
    )
    plot_balanced_trajectories(
        seed_grid,
        summary_grid,
        dataset_initial,
        dataset,
        outdir,
        formats,
        matched_cap=matched_cap,
    )
    plot_final_mo_metrics(dataset_final, dataset, outdir, formats)
    plot_final_balanced(dataset_final, dataset, outdir, formats)
    plot_initial_vs_final(dataset_initial, dataset_final, dataset, outdir, formats)
    plot_final_scorecard(dataset_final, dataset, outdir, formats)
    plot_final_token_usage(dataset_final, dataset, outdir, formats)

    raw_final = pd.DataFrame()
    valid_final = pd.DataFrame()
    high_quality = pd.DataFrame()

    if all_evaluations is not None:
        dataset_evaluations = all_evaluations[
            all_evaluations["dataset"].astype(str) == dataset
        ].copy()
        raw_final, valid_final = select_final_evaluations(
            dataset_evaluations,
            dataset_final,
        )
        if valid_final.empty:
            message = f"No publication-valid final candidates for {dataset}"
            if strict:
                raise RuntimeError(message)
            warnings.warn(message)
        else:
            raw_final.to_parquet(
                outdir
                / f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_raw_final_evaluations.parquet",
                index=False,
            )
            valid_final.to_parquet(
                outdir
                / f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_publication_valid_final_evaluations.parquet",
                index=False,
            )
            plot_empirical_attainment(
                valid_final,
                dataset,
                outdir,
                formats,
                x_col="test_cost",
                xlabel=COST_AXIS_LABEL,
                suffix="cost",
            )
            plot_empirical_attainment(
                valid_final,
                dataset,
                outdir,
                formats,
                x_col="test_fairness",
                xlabel=UNFAIRNESS_AXIS_LABEL,
                suffix="unfairness",
            )
            plot_cost_unfairness_pareto(
                valid_final, dataset, outdir, formats
            )
            plot_three_objective_pareto(
                valid_final, dataset, outdir, formats
            )
            high_quality = plot_high_quality_operating_points(
                valid_final,
                dataset,
                outdir,
                formats,
                threshold=quality_threshold,
            )
            plot_bbq_coverage_diagnostics(
                raw_final, dataset, outdir, formats
            )

    summary = summarize_final(
        dataset_final,
        valid_final,
        raw_final,
        high_quality,
    )
    summary.to_csv(
        outdir / f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_summary_table.csv",
        index=False,
    )
    safe_markdown(
        summary,
        outdir / f"{DATASET_SLUG[dataset]}_qwen3_5m_v5_summary_table.md",
    )

    write_dataset_readme(
        outdir,
        dataset,
        results_root=results_root,
        analysis_root=analysis_root,
        matched_cap=matched_cap,
        grid_step=grid_step,
        quality_threshold=quality_threshold,
        formats=formats,
    )
    build_manifest(
        outdir,
        dataset,
        dataset_final[selected_columns],
        results_root=results_root,
        analysis_root=analysis_root,
        matched_cap=matched_cap,
        grid_step=grid_step,
        quality_threshold=quality_threshold,
    )

    generated = sorted(path.name for path in outdir.iterdir() if path.is_file())
    print(f"\n{DATASET_DISPLAY[dataset]} generated files ({len(generated)}):")
    for name in generated:
        print(f"  {name}")
    print(f"Output: {outdir.resolve()}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_style()

    results_root = Path(args.results_root).expanduser()
    analysis_root = Path(args.analysis_root).expanduser()
    figure_root = (
        Path(args.figure_root).expanduser()
        if args.figure_root
        else analysis_root / "publication_figures"
    )
    datasets = parse_csv(args.datasets)
    formats = tuple(value.lower() for value in parse_csv(args.formats))
    thresholds = parse_thresholds(args.quality_thresholds)

    unknown = sorted(set(datasets) - set(DATASET_ORDER))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    invalid_formats = sorted(set(formats) - {"png", "pdf", "svg"})
    if invalid_formats:
        raise ValueError(f"Unsupported figure formats: {invalid_formats}")
    if args.matched_cap <= 0 or args.matched_cap > FINAL_BUDGET:
        raise ValueError("--matched-cap must lie in (0, 5,000,000]")
    if args.grid_step <= 0:
        raise ValueError("--grid-step must be positive")

    checkpoint, final, initial = load_metric_inputs(analysis_root)
    checkpoint = checkpoint[checkpoint["dataset"].isin(datasets)].copy()
    final = final[final["dataset"].isin(datasets)].copy()
    initial = initial[initial["dataset"].isin(datasets)].copy()
    validate_metric_matrix(
        final,
        initial,
        datasets,
        strict=bool(args.strict),
    )

    all_evaluations: pd.DataFrame | None = None
    if not args.skip_pareto:
        all_evaluations = load_all_evaluations(
            results_root,
            datasets,
            strict=bool(args.strict),
        )

    figure_root.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        process_dataset(
            dataset,
            checkpoint=checkpoint,
            final=final,
            initial=initial,
            all_evaluations=all_evaluations,
            figure_root=figure_root,
            results_root=results_root,
            analysis_root=analysis_root,
            matched_cap=int(args.matched_cap),
            grid_step=int(args.grid_step),
            quality_threshold=float(thresholds[dataset]),
            formats=formats,
            strict=bool(args.strict),
        )

    print("\nAll requested v5 dataset figure suites are complete.")
    print(f"Figure root: {figure_root.resolve()}")


if __name__ == "__main__":
    main()
