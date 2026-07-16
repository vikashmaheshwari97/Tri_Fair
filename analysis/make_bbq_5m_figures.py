"""Create publication-style BBQ 5M figures for Tri-Fair and NSGA-II-PO-Fair.

This module fixes two common failure modes of the original 5M patch:

1. It never reads the generic ``analysis/output`` directory by default.  It uses
   a dedicated ``analysis/output/bbq_5m`` directory, so stale 1M CSV/Parquet
   files cannot be mistaken for 5M analysis inputs.
2. Stepwise trajectories are computed only from the six selected 5M runs.  Old
   1M run histories are not mixed into the 5M per-seed trajectories.

Default one-command usage on Rocket::

    python -m analysis.make_bbq_5m_figures

If the dedicated 5M analysis tables are missing or stale, the script rebuilds
those tables from ``results/tri_fair`` using the repository analysis pipeline
and the frozen 1M normalization bounds.

Generated outputs include:

* development nR2-proxy trajectory (stepwise),
* development HV trajectory and checkpoint holdout gap,
* exact checkpoint nR2 / pessimistic HV / approximation-gap trajectories,
* balanced test accuracy / cost / unfairness budget trajectories,
* 5M accuracy-vs-cost and accuracy-vs-unfairness attainment plots,
* 5M cost-vs-unfairness Pareto projection,
* 5M three-objective Pareto scatter,
* 5M method-comparison bars for HV, nR2, and gap,
* Tri-Fair final-candidate diagnostic scatters,
* CSV/Markdown summary tables and a selected-run manifest.

Scientific terminology
----------------------
``noisy_r2_3d`` in ``run_metrics.csv`` is the checkpoint-level holdout nR2
metric produced by the main analysis pipeline.  The dense stepwise nR2 curve is
explicitly labelled a *development nR2 proxy*, because intermediate optimizer
steps were not all evaluated on the holdout set.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL = "qwen-3-30b"
DATASET = "bbq"
FINAL_BUDGET = 5_000_000
DEFAULT_BUDGETS = (250_000, 1_000_000, 5_000_000)
EXPECTED_SEEDS = (42, 43, 44)
N_PREFERENCES = 500
PREFERENCE_SEED = 2026
REFERENCE_POINT = (1.1, 1.1, 1.1)

# Qwen3-30B experiment objective weights.
# Cost = 0.11 * input_tokens + 0.41 * output_tokens.
QWEN_INPUT_WEIGHT = 0.11
QWEN_OUTPUT_WEIGHT = 0.41

OPTIMIZER_ORDER = ["Tri-Fair", "NSGAII-PO-Fair"]
DISPLAY_NAME = {
    "Tri-Fair": "Tri-Fair",
    "NSGAII-PO-Fair": "NSGA-II-PO-Fair",
}
COLORS = {
    "Tri-Fair": "black",
    "NSGAII-PO-Fair": "#E69F00",
}
MARKERS = {
    "Tri-Fair": "o",
    "NSGAII-PO-Fair": "s",
}


@dataclass(frozen=True)
class CostBounds:
    """Simple normalization bounds for development-side proxy metrics."""

    cost_max: float

    def normalize_minimize(self, raw: np.ndarray) -> np.ndarray:
        """Normalize minimize-all objectives [1-accuracy, cost, unfairness]."""
        out = np.asarray(raw, dtype=float).copy()
        out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
        out[:, 1] = np.clip(out[:, 1] / self.cost_max, 0.0, 1.1)
        out[:, 2] = np.clip(out[:, 2], 0.0, 1.0)
        return out


def parse_csv_ints(raw: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        values = [int(piece.strip().replace("_", "")) for piece in raw.split(",") if piece.strip()]
    else:
        values = [int(value) for value in raw]
    if not values:
        raise ValueError("At least one budget is required")
    return tuple(sorted(set(values)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="results/tri_fair",
        help="Root containing the extracted 1M and 5M run directories.",
    )
    parser.add_argument(
        "--analysis-dir",
        default="analysis/output/bbq_5m",
        help="Dedicated analysis-table directory. Do not point this at stale 1M output.",
    )
    parser.add_argument(
        "--figure-dir",
        default=None,
        help="Figure output directory (default: <analysis-dir>/curated_figures_mocapo_style_5m).",
    )
    parser.add_argument(
        "--bounds-file",
        default="analysis/normalization_bounds.json",
        help="Frozen normalization bounds created from the 1M analysis.",
    )
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in DEFAULT_BUDGETS),
        help="Checkpoint budgets used when rebuilding analysis tables.",
    )
    parser.add_argument("--n-preferences", type=int, default=1_000)
    parser.add_argument("--preference-seed", type=int, default=2026)
    parser.add_argument(
        "--reference-point",
        default="1.1,1.1,1.1",
        help="Normalized minimization reference point for hypervolume.",
    )
    parser.add_argument(
        "--rebuild-analysis",
        action="store_true",
        help="Force rebuilding the dedicated 5M analysis tables.",
    )
    parser.add_argument(
        "--no-auto-rebuild",
        action="store_true",
        help="Fail instead of rebuilding when the 5M tables are missing or stale.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on unreadable run artifacts during analysis reconstruction.",
    )
    return parser.parse_args(argv)


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


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the intended BBQ/Qwen runs and remove archived pilot copies."""
    out = frame.copy()
    if "run_dir" in out:
        run_dir = out["run_dir"].astype(str)
        out = out[~run_dir.str.contains("pilot_reports", case=False, regex=False)]
    if "dataset" in out:
        out = out[out["dataset"].astype(str) == DATASET]
    if "model" in out:
        out = out[out["model"].astype(str) == MODEL]
    if "optimizer" in out:
        out = out[out["optimizer"].astype(str).isin(OPTIMIZER_ORDER)]
    return out.reset_index(drop=True)


def analysis_paths(analysis_dir: Path) -> dict[str, Path]:
    return {
        "run_metrics": analysis_dir / "run_metrics.csv",
        "summary": analysis_dir / "summary.csv",
        "trajectory": analysis_dir / "trajectory_metrics.csv",
        "evaluations": analysis_dir / "all_evaluations.parquet",
    }


def analysis_has_5m(analysis_dir: Path) -> bool:
    paths = analysis_paths(analysis_dir)
    required = [paths["run_metrics"], paths["trajectory"], paths["evaluations"]]
    if any(not path.is_file() for path in required):
        return False
    try:
        run_metrics = clean_frame(pd.read_csv(paths["run_metrics"]))
    except Exception:
        return False
    if "budget_checkpoint" not in run_metrics or run_metrics.empty:
        return False
    budgets = set(
        pd.to_numeric(run_metrics["budget_checkpoint"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    return FINAL_BUDGET in budgets


def attach_configured_budget(frame: pd.DataFrame, evaluations: pd.DataFrame) -> pd.DataFrame:
    """Attach configured_budget to pipeline outputs when the source eval has it."""
    if frame.empty or "run_dir" not in frame or "configured_budget" not in evaluations:
        return frame
    mapping = (
        evaluations[["run_dir", "configured_budget"]]
        .copy()
        .assign(
            configured_budget=lambda x: pd.to_numeric(
                x["configured_budget"], errors="coerce"
            )
        )
        .dropna(subset=["configured_budget"])
        .groupby("run_dir", as_index=False)["configured_budget"]
        .max()
    )
    if "configured_budget" in frame:
        frame = frame.drop(columns=["configured_budget"])
    return frame.merge(mapping, on="run_dir", how="left")


def rebuild_analysis_outputs(
    *,
    results_root: Path,
    analysis_dir: Path,
    bounds_file: Path,
    budgets: tuple[int, ...],
    n_preferences: int,
    preference_seed: int,
    reference_point: tuple[float, float, float],
    strict: bool,
) -> None:
    """Rebuild BBQ-only 5M tables using the repository's main analysis code."""
    if not results_root.is_dir():
        raise FileNotFoundError(
            f"Results root does not exist: {results_root}\n"
            "Extract/copy the BBQ 5M bundle under results/tri_fair, or pass "
            "--results-root to its extracted run-tree root."
        )
    if not bounds_file.is_file():
        raise FileNotFoundError(
            f"Frozen 1M normalization bounds not found: {bounds_file}\n"
            "Do not estimate new bounds from 5M. Restore the bounds file used for the 1M paper analysis."
        )

    # Imports are local so ``python -m py_compile`` still gives a useful syntax
    # check even outside a fully installed project environment.
    from analysis.analysis_pipeline import (  # type: ignore
        compute_run_metrics,
        compute_trajectory_metrics,
        load_or_create_bounds,
    )
    from analysis.io import load_all_evaluations, load_all_step_results  # type: ignore
    from analysis.metrics import aggregate_run_metrics  # type: ignore

    print(f"\nRebuilding BBQ 5M analysis tables from: {results_root.resolve()}")
    evaluations = load_all_evaluations(
        results_root,
        budget_checkpoints=budgets,
        strict=strict,
    )
    evaluations = clean_frame(evaluations)
    if evaluations.empty:
        raise RuntimeError(
            f"No {DATASET}/{MODEL} evaluations were discovered beneath {results_root}"
        )

    bounds = load_or_create_bounds(
        evaluations,
        bounds_file,
        bounds_budget=1_000_000,
        rebuild=False,
    )
    run_metrics = compute_run_metrics(
        evaluations,
        bounds,
        n_preferences=n_preferences,
        preference_seed=preference_seed,
        reference_point=reference_point,
        strict=strict,
    )
    run_metrics = attach_configured_budget(run_metrics, evaluations)
    summary = aggregate_run_metrics(run_metrics)

    try:
        steps = clean_frame(load_all_step_results(results_root, strict=strict))
        trajectory = compute_trajectory_metrics(
            steps,
            bounds,
            budgets=budgets,
            reference_point=reference_point,
            strict=strict,
        )
        trajectory = attach_configured_budget(trajectory, evaluations)
    except FileNotFoundError:
        warnings.warn("No step logs found; trajectory_metrics.csv will be empty")
        trajectory = pd.DataFrame()

    analysis_dir.mkdir(parents=True, exist_ok=True)
    evaluations.to_parquet(analysis_dir / "all_evaluations.parquet", index=False)
    run_metrics.to_csv(analysis_dir / "run_metrics.csv", index=False)
    summary.to_csv(analysis_dir / "summary.csv", index=False)
    trajectory.to_csv(analysis_dir / "trajectory_metrics.csv", index=False)
    print(f"Rebuilt analysis tables in: {analysis_dir.resolve()}")


def read_inputs(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = analysis_paths(analysis_dir)
    missing = [str(path) for path in paths.values() if path.name != "summary.csv" and not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required 5M analysis outputs:\n"
            + "\n".join(missing)
            + "\nRun with --rebuild-analysis or allow the default auto-rebuild."
        )

    run_metrics = clean_frame(pd.read_csv(paths["run_metrics"]))
    try:
        trajectory = clean_frame(pd.read_csv(paths["trajectory"]))
    except pd.errors.EmptyDataError:
        trajectory = pd.DataFrame()
    evaluations = clean_frame(pd.read_parquet(paths["evaluations"]))

    if "test_fairness_ready" in evaluations:
        evaluations = evaluations[
            evaluations["test_fairness_ready"].fillna(False).astype(bool)
        ].reset_index(drop=True)

    return run_metrics, trajectory, evaluations


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def selected_final_runs(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Select exactly one 5M run per method/seed and report duplicates clearly."""
    require_columns(
        run_metrics,
        ["optimizer", "seed", "budget_checkpoint", "run_key", "run_dir", "actual_budget_tokens"],
        "run_metrics.csv",
    )
    final = run_metrics[
        pd.to_numeric(run_metrics["budget_checkpoint"], errors="coerce") == FINAL_BUDGET
    ].copy()
    if "configured_budget" in final and final["configured_budget"].notna().any():
        configured = pd.to_numeric(final["configured_budget"], errors="coerce")
        preferred = final[configured >= FINAL_BUDGET].copy()
        if not preferred.empty:
            final = preferred

    if final.empty:
        observed = sorted(
            set(
                pd.to_numeric(run_metrics["budget_checkpoint"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
        )
        raise RuntimeError(
            "No 5M checkpoint rows were found after cleaning. "
            f"Observed checkpoints: {observed}. The tables are stale or the 5M bundle "
            "is not under --results-root."
        )

    final["seed"] = pd.to_numeric(final["seed"], errors="raise").astype(int)
    final["actual_budget_tokens"] = pd.to_numeric(
        final["actual_budget_tokens"], errors="coerce"
    )

    chosen_rows: list[pd.Series] = []
    duplicate_notes: list[str] = []
    for optimizer in OPTIMIZER_ORDER:
        for seed in EXPECTED_SEEDS:
            group = final[(final["optimizer"] == optimizer) & (final["seed"] == seed)].copy()
            if group.empty:
                continue
            group = group.sort_values(
                ["actual_budget_tokens", "run_dir"], ascending=[False, True]
            )
            if len(group) > 1:
                duplicate_notes.append(
                    f"{optimizer} seed {seed}: selected {group.iloc[0]['run_dir']} from {len(group)} candidates"
                )
            chosen_rows.append(group.iloc[0])

    if chosen_rows:
        selected = pd.DataFrame(chosen_rows).reset_index(drop=True)
    else:
        selected = pd.DataFrame(columns=final.columns)

    missing_pairs = [
        (optimizer, seed)
        for optimizer in OPTIMIZER_ORDER
        for seed in EXPECTED_SEEDS
        if selected.empty
        or selected[
            (selected["optimizer"] == optimizer) & (selected["seed"] == seed)
        ].empty
    ]
    if missing_pairs:
        raise RuntimeError(
            "The complete six-run 5M grid was not found. Missing: "
            + ", ".join(f"{optimizer}/seed{seed}" for optimizer, seed in missing_pairs)
        )

    if duplicate_notes:
        warnings.warn("Duplicate 5M candidates were found:\n" + "\n".join(duplicate_notes))

    counts = (
        run_metrics.groupby(["optimizer", "budget_checkpoint"])
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["optimizer", "budget_checkpoint"])
    )
    print("\nRun-metric counts after cleaning")
    print(counts.to_string(index=False))
    print("\nSelected 5M runs")
    print(
        selected[
            ["optimizer", "seed", "actual_budget_tokens", "run_dir"]
        ].sort_values(["optimizer", "seed"]).to_string(index=False)
    )
    return selected


def filter_to_selected_runs(frame: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "run_key" not in frame:
        return frame.copy()
    keys = set(selected["run_key"].astype(str))
    return frame[frame["run_key"].astype(str).isin(keys)].reset_index(drop=True)


def final_checkpoint_evaluations(
    evaluations: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    require_columns(
        evaluations,
        [
            "optimizer",
            "seed",
            "run_key",
            "budget_checkpoint",
            "test_quality",
            "test_cost",
            "test_fairness",
        ],
        "all_evaluations.parquet",
    )
    data = filter_to_selected_runs(evaluations, selected)
    data = data[
        pd.to_numeric(data["budget_checkpoint"], errors="coerce") == FINAL_BUDGET
    ].copy()
    if data.empty:
        raise RuntimeError("No selected 5M evaluation rows were found")
    return data.reset_index(drop=True)


def checkpoint_run_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Choose one scientifically appropriate run for each method/seed/checkpoint.

    Exact configured-budget runs are preferred.  If unavailable, the smallest
    configured budget that crossed the checkpoint is used.  This prevents a 1M
    and a 5M run from both being counted as independent observations at 1M.
    """
    frame = run_metrics.copy()
    frame["budget_checkpoint"] = pd.to_numeric(
        frame["budget_checkpoint"], errors="coerce"
    )
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce")
    frame = frame.dropna(subset=["budget_checkpoint", "seed"])
    frame["budget_checkpoint"] = frame["budget_checkpoint"].astype(int)
    frame["seed"] = frame["seed"].astype(int)
    frame = frame[frame["budget_checkpoint"].isin(DEFAULT_BUDGETS)]

    selected_rows: list[pd.Series] = []
    for (_, _, checkpoint), group in frame.groupby(
        ["optimizer", "seed", "budget_checkpoint"], sort=True
    ):
        candidate = group.copy()
        if "configured_budget" in candidate and candidate["configured_budget"].notna().any():
            candidate["configured_budget"] = pd.to_numeric(
                candidate["configured_budget"], errors="coerce"
            )
            exact = candidate[candidate["configured_budget"] == checkpoint]
            if not exact.empty:
                candidate = exact
            else:
                eligible = candidate[candidate["configured_budget"] >= checkpoint]
                if not eligible.empty:
                    minimum = eligible["configured_budget"].min()
                    candidate = eligible[eligible["configured_budget"] == minimum]
        candidate = candidate.sort_values(
            ["actual_budget_tokens", "run_dir"], ascending=[False, True]
        )
        selected_rows.append(candidate.iloc[0])
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(len(values), dtype=bool)
    for i in range(len(values)):
        dominates_i = np.all(values <= values[i], axis=1) & np.any(
            values < values[i], axis=1
        )
        dominates_i[i] = False
        if np.any(dominates_i):
            keep[i] = False
    return keep


def make_preferences(n_preferences: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(3), size=n_preferences)


def chebyshev_r2(normalized_minimize_front: np.ndarray, weights: np.ndarray) -> float:
    """Lower is better; development-side proxy for nR2/R2-style utility."""
    if len(normalized_minimize_front) == 0:
        return float("nan")
    front = np.asarray(normalized_minimize_front, dtype=float)
    utilities = np.max(weights[:, None, :] * front[None, :, :], axis=2)
    return float(np.mean(np.min(utilities, axis=1)))


def finite_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def load_step_results_from_selected_runs(selected: pd.DataFrame) -> pd.DataFrame:
    unique_runs = selected[
        ["run_key", "run_dir", "model", "dataset", "optimizer", "seed"]
    ].drop_duplicates()
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for _, row in unique_runs.iterrows():
        path = Path(str(row["run_dir"])) / "step_results.parquet"
        if not path.is_file():
            missing.append(str(path))
            continue
        frame = pd.read_parquet(path).copy()
        frame["run_key"] = row["run_key"]
        frame["run_dir"] = row["run_dir"]
        frame["model"] = row["model"]
        frame["dataset"] = row["dataset"]
        frame["optimizer"] = row["optimizer"]
        frame["seed"] = int(row["seed"])
        frames.append(frame)
    if missing:
        raise FileNotFoundError(
            "Missing step_results.parquet for selected 5M runs:\n" + "\n".join(missing)
        )
    if not frames:
        raise RuntimeError("No selected 5M step_results.parquet files were loaded")
    return pd.concat(frames, ignore_index=True, sort=False)


def infer_cost_bounds(step_results: pd.DataFrame, evaluations: pd.DataFrame) -> CostBounds:
    candidates: list[float] = []
    if "cost" in step_results:
        value = np.nanmax(finite_numeric(step_results["cost"]))
        if np.isfinite(value):
            candidates.append(float(value))
    for column in ("dev_cost", "test_cost"):
        if column in evaluations:
            value = np.nanmax(finite_numeric(evaluations[column]))
            if np.isfinite(value):
                candidates.append(float(value))
    cost_max = max(candidates or [1.0])
    return CostBounds(cost_max=max(cost_max * 1.05, 1.0))


def stepwise_development_r2_proxy(
    step_results: pd.DataFrame,
    evaluations: pd.DataFrame,
) -> pd.DataFrame:
    bounds = infer_cost_bounds(step_results, evaluations)
    weights = make_preferences(N_PREFERENCES, PREFERENCE_SEED)
    rows: list[dict[str, object]] = []
    group_cols = ["run_key", "run_dir", "model", "dataset", "optimizer", "seed", "step"]

    for keys, group in step_results.groupby(group_cols, sort=True, dropna=False):
        meta = dict(zip(group_cols, keys))
        require_columns(group, ["quality", "cost", "fairness"], "step_results.parquet")
        quality = finite_numeric(group["quality"])
        cost = finite_numeric(group["cost"])
        fairness = finite_numeric(group["fairness"])
        valid_mask = np.isfinite(quality) & np.isfinite(cost) & np.isfinite(fairness)
        if "fairness_ready" in group:
            valid_mask &= group["fairness_ready"].fillna(False).astype(bool).to_numpy()
        if not valid_mask.any():
            continue

        raw = np.column_stack([1.0 - quality[valid_mask], cost[valid_mask], fairness[valid_mask]])
        normalized = bounds.normalize_minimize(raw)
        front_mask = pareto_mask_minimize(normalized)

        if "total_tokens_downstream" in group:
            tokens = int(np.nanmax(finite_numeric(group["total_tokens_downstream"])))
        else:
            input_tokens = (
                np.nanmax(finite_numeric(group["input_tokens_downstream"]))
                if "input_tokens_downstream" in group
                else 0.0
            )
            output_tokens = (
                np.nanmax(finite_numeric(group["output_tokens_downstream"]))
                if "output_tokens_downstream" in group
                else 0.0
            )
            tokens = int(input_tokens + output_tokens)

        rows.append(
            {
                **meta,
                "actual_budget_tokens": tokens,
                "dev_noisy_r2_proxy": chebyshev_r2(normalized[front_mask], weights),
                "dev_proxy_front_size": int(front_mask.sum()),
            }
        )
    if not rows:
        raise RuntimeError("Could not compute stepwise development nR2 proxy")
    return pd.DataFrame(rows)


def prepare_staircase_grid(data: pd.DataFrame, max_budget: int | None = None) -> np.ndarray:
    values = finite_numeric(data["actual_budget_tokens"])
    values = values[np.isfinite(values)]
    if max_budget is not None:
        values = values[values <= max_budget]
    values = np.unique(values.astype(int))
    return np.sort(values[values > 0])


def interpolate_step_values(run_data: pd.DataFrame, grid: np.ndarray, value_col: str) -> np.ndarray:
    data = run_data.sort_values("actual_budget_tokens")
    x = finite_numeric(data["actual_budget_tokens"])
    y = finite_numeric(data[value_col])
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return np.full(len(grid), np.nan)
    collapsed = (
        pd.DataFrame({"x": x[mask].astype(int), "y": y[mask]})
        .groupby("x", as_index=False)["y"]
        .last()
    )
    x_values = collapsed["x"].to_numpy(dtype=int)
    y_values = collapsed["y"].to_numpy(dtype=float)
    result = np.full(len(grid), np.nan)
    for index, grid_value in enumerate(grid):
        position = np.searchsorted(x_values, grid_value, side="right") - 1
        if position >= 0:
            result[index] = y_values[position]
    return result


def step_band_statistics(
    data: pd.DataFrame,
    value_col: str,
    *,
    max_budget: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    if data.empty or value_col not in data:
        return result
    for optimizer in OPTIMIZER_ORDER:
        optimizer_data = data[data["optimizer"] == optimizer].copy()
        if optimizer_data.empty:
            continue
        grid = prepare_staircase_grid(optimizer_data, max_budget=max_budget)
        if len(grid) == 0:
            continue
        seed_values = [
            interpolate_step_values(seed_group, grid, value_col)
            for _, seed_group in optimizer_data.groupby("seed", sort=True)
        ]
        matrix = np.vstack(seed_values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(matrix, axis=0)
            std = np.nanstd(matrix, axis=0)
        valid = np.isfinite(mean)
        result[optimizer] = (
            grid[valid],
            mean[valid],
            (mean - std)[valid],
            (mean + std)[valid],
        )
    return result


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_stepwise_nr2_proxy(step_proxy: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    stats = step_band_statistics(step_proxy, "dev_noisy_r2_proxy", max_budget=5_250_000)
    for optimizer in OPTIMIZER_ORDER:
        if optimizer not in stats:
            continue
        grid, mean, lower, upper = stats[optimizer]
        x = grid / 1_000_000.0
        ax.step(
            x,
            mean,
            where="post",
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            markevery=max(1, len(x) // 8),
            linewidth=2,
            markersize=4,
            label=DISPLAY_NAME[optimizer],
        )
        ax.fill_between(x, lower, upper, step="post", color=COLORS[optimizer], alpha=0.16)
    ax.set_title("BBQ — Qwen-3-30B")
    ax.set_xlabel("Token Budget [×10⁶]")
    ax.set_ylabel("Development nR2 Proxy ↓")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.02,
        "Stepwise proxy from development objectives;\nexact holdout nR2 is checkpoint-level.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        alpha=0.75,
    )
    save_figure(fig, outdir, "bbq_qwen3_5m_stepwise_dev_nr2_proxy")


def plot_stepwise_hv_and_gap(
    selected_trajectory: pd.DataFrame,
    checkpoint_metrics: pd.DataFrame,
    outdir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    hv_stats = step_band_statistics(selected_trajectory, "hv_dev_3d", max_budget=5_250_000)
    for optimizer in OPTIMIZER_ORDER:
        if optimizer not in hv_stats:
            continue
        grid, mean, lower, upper = hv_stats[optimizer]
        x = grid / 1_000_000.0
        axes[0].step(
            x,
            mean,
            where="post",
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            markevery=max(1, len(x) // 8),
            linewidth=2,
            markersize=4,
            label=DISPLAY_NAME[optimizer],
        )
        axes[0].fill_between(x, lower, upper, step="post", color=COLORS[optimizer], alpha=0.16)
    axes[0].set_title("Development Hypervolume ↑")
    axes[0].set_xlabel("Token Budget [×10⁶]")
    axes[0].set_ylabel("Development HV")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    plot_checkpoint_metric_on_axis(
        axes[1],
        checkpoint_metrics,
        metric="approximation_gap_3d",
        title="Holdout Approximation Gap ↓",
        ylabel="Approximation Gap",
    )
    fig.suptitle("BBQ — Qwen-3-30B")
    save_figure(fig, outdir, "bbq_qwen3_5m_stepwise_hv_checkpoint_gap")


def plot_checkpoint_metric_on_axis(
    ax: plt.Axes,
    checkpoint_metrics: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    if metric not in checkpoint_metrics:
        ax.text(0.5, 0.5, f"Missing {metric}", ha="center", va="center")
        ax.set_axis_off()
        return
    for optimizer in OPTIMIZER_ORDER:
        data = checkpoint_metrics[checkpoint_metrics["optimizer"] == optimizer]
        grouped = (
            data.groupby("budget_checkpoint")[metric]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("budget_checkpoint")
        )
        if grouped.empty:
            continue
        x = grouped["budget_checkpoint"].to_numpy(dtype=float) / 1_000_000.0
        mean = grouped["mean"].to_numpy(dtype=float)
        std = grouped["std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(
            x,
            mean,
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            linewidth=2,
            markersize=5,
            label=DISPLAY_NAME[optimizer],
        )
        ax.fill_between(x, mean - std, mean + std, color=COLORS[optimizer], alpha=0.16)
    ax.set_title(title)
    ax.set_xlabel("Token Budget")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0.25, 1.0, 5.0], ["250K", "1M", "5M"])
    ax.grid(True, alpha=0.25)


def plot_exact_checkpoint_mo_metrics(checkpoint_metrics: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    specifications = [
        ("noisy_r2_3d", "Holdout nR2 ↓", "nR2"),
        ("hv_test_pessimistic_3d", "Pessimistic Test HV ↑", "Hypervolume"),
        ("approximation_gap_3d", "Approximation Gap ↓", "Gap"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, specifications):
        plot_checkpoint_metric_on_axis(
            ax,
            checkpoint_metrics,
            metric=metric,
            title=title,
            ylabel=ylabel,
        )
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("BBQ — Qwen-3-30B: Exact Checkpoint MO Metrics")
    save_figure(fig, outdir, "bbq_qwen3_5m_checkpoint_nr2_hv_gap")


def plot_balanced_budget_trajectory(checkpoint_metrics: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    specifications = [
        ("balanced_test_quality", "Balanced Test Accuracy ↑", "Accuracy"),
        ("balanced_test_cost", "Balanced Test Cost ↓", "Avg. Cost [$] per 1M Calls"),
        ("balanced_test_fairness", "Balanced Test Unfairness ↓", "Unfairness"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, specifications):
        plot_checkpoint_metric_on_axis(
            ax,
            checkpoint_metrics,
            metric=metric,
            title=title,
            ylabel=ylabel,
        )
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("BBQ — Qwen-3-30B: Budget Trajectory")
    save_figure(fig, outdir, "bbq_qwen3_5m_balanced_budget_trajectory")


def test_objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    quality = finite_numeric(frame["test_quality"])
    cost = finite_numeric(frame["test_cost"])
    fairness = finite_numeric(frame["test_fairness"])
    return np.column_stack([1.0 - quality, cost, fairness])


def y_attained_at_x(data: pd.DataFrame, x_grid: np.ndarray, x_col: str) -> np.ndarray:
    x = finite_numeric(data[x_col])
    y = finite_numeric(data["test_quality"])
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    result = np.full(len(x_grid), np.nan)
    for index, grid_value in enumerate(x_grid):
        eligible = y[x <= grid_value]
        if len(eligible):
            result[index] = np.max(eligible)
    return result


def plot_empirical_attainment(
    final: pd.DataFrame,
    outdir: Path,
    *,
    x_col: str,
    xlabel: str,
    filename: str,
) -> None:
    xmin = float(np.nanmin(finite_numeric(final[x_col])))
    xmax = float(np.nanmax(finite_numeric(final[x_col])))
    padding = 0.03 * max(xmax - xmin, 1e-6)
    x_grid = np.linspace(xmin - padding, xmax + padding, 400)
    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)

    for optimizer in OPTIMIZER_ORDER:
        optimizer_data = final[final["optimizer"] == optimizer]
        curves: list[np.ndarray] = []
        for _, seed_group in optimizer_data.groupby("seed", sort=True):
            valid = np.all(np.isfinite(test_objective_matrix(seed_group)), axis=1)
            seed_group = seed_group.loc[valid].reset_index(drop=True)
            if seed_group.empty:
                continue
            front = seed_group.loc[pareto_mask_minimize(test_objective_matrix(seed_group))]
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
            linewidth=2,
            markersize=4,
            label=DISPLAY_NAME[optimizer],
        )
        ax.fill_between(
            x_grid[valid], lower[valid], upper[valid], step="post", color=COLORS[optimizer], alpha=0.16
        )
    ax.set_title("BBQ — Qwen-3-30B at 5M")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, outdir, filename)


def plot_cost_unfairness_projection(final: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    for optimizer in OPTIMIZER_ORDER:
        optimizer_data = final[final["optimizer"] == optimizer]
        pooled_fronts: list[pd.DataFrame] = []
        for seed, seed_group in optimizer_data.groupby("seed", sort=True):
            valid = np.all(np.isfinite(test_objective_matrix(seed_group)), axis=1)
            seed_group = seed_group.loc[valid].reset_index(drop=True)
            if seed_group.empty:
                continue
            front = seed_group.loc[pareto_mask_minimize(test_objective_matrix(seed_group))].copy()
            pooled_fronts.append(front)
            ax.scatter(
                front["test_cost"],
                front["test_fairness"],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                alpha=0.22,
                s=25,
            )
        if not pooled_fronts:
            continue
        pooled = pd.concat(pooled_fronts, ignore_index=True)
        projection = np.column_stack(
            [finite_numeric(pooled["test_cost"]), finite_numeric(pooled["test_fairness"])]
        )
        valid = np.all(np.isfinite(projection), axis=1)
        pooled = pooled.loc[valid].reset_index(drop=True)
        projection = projection[valid]
        front = pooled.loc[pareto_mask_minimize(projection)].sort_values("test_cost")
        ax.plot(
            front["test_cost"],
            front["test_fairness"],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            linewidth=2,
            markersize=5,
            label=DISPLAY_NAME[optimizer],
        )
    ax.set_title("BBQ — 5M Cost vs Test Unfairness")
    ax.set_xlabel("Avg. Cost [$] per 1M Calls")
    ax.set_ylabel("Test Unfairness ↓")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, outdir, "bbq_qwen3_5m_pareto_cost_unfairness")


def plot_three_objective_pareto(final: pd.DataFrame, outdir: Path) -> None:
    fig = plt.figure(figsize=(7.2, 5.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for optimizer in OPTIMIZER_ORDER:
        optimizer_data = final[final["optimizer"] == optimizer]
        for seed, seed_group in optimizer_data.groupby("seed", sort=True):
            matrix = test_objective_matrix(seed_group)
            valid = np.all(np.isfinite(matrix), axis=1)
            seed_group = seed_group.loc[valid].reset_index(drop=True)
            matrix = matrix[valid]
            if seed_group.empty:
                continue
            front = seed_group.loc[pareto_mask_minimize(matrix)]
            ax.scatter(
                front["test_cost"],
                front["test_fairness"],
                front["test_quality"],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                alpha=0.78,
                s=34,
                label=DISPLAY_NAME[optimizer] if seed == EXPECTED_SEEDS[0] else None,
            )
    ax.set_title("BBQ — Qwen-3-30B 5M Test Pareto Fronts")
    ax.set_xlabel("Cost ↓")
    ax.set_ylabel("Unfairness ↓")
    ax.set_zlabel("Accuracy ↑")
    ax.legend(frameon=False)
    save_figure(fig, outdir, "bbq_qwen3_5m_test_pareto_3d")


def plot_method_comparison_bars(final_metrics: pd.DataFrame, outdir: Path) -> None:
    specifications = [
        ("hv_test_optimistic_3d", "Optimistic Test HV ↑"),
        ("hv_test_pessimistic_3d", "Pessimistic Test HV ↑"),
        ("noisy_r2_3d", "Holdout nR2 ↓"),
        ("approximation_gap_3d", "Approximation Gap ↓"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), specifications):
        means: list[float] = []
        stds: list[float] = []
        labels: list[str] = []
        colors: list[str] = []
        for optimizer in OPTIMIZER_ORDER:
            values = pd.to_numeric(
                final_metrics.loc[final_metrics["optimizer"] == optimizer, metric],
                errors="coerce",
            ).dropna()
            means.append(float(values.mean()))
            stds.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
            labels.append(DISPLAY_NAME[optimizer])
            colors.append(COLORS[optimizer])
        positions = np.arange(len(labels))
        bars = ax.bar(positions, means, yerr=stds, capsize=4, color=colors, alpha=0.82)
        ax.set_xticks(positions, labels, rotation=8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("BBQ — Qwen-3-30B at 5M (mean ± SD, 3 seeds)")
    save_figure(fig, outdir, "bbq_qwen3_5m_method_comparison_hv_nr2_gap")


def parse_fewshot_count(value: object) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    if isinstance(value, list):
        return len(value)
    text = str(value).strip()
    if not text:
        return 0
    try:
        loaded = json.loads(text)
        return len(loaded) if isinstance(loaded, list) else 0
    except json.JSONDecodeError:
        return 0


def output_cost_share(frame: pd.DataFrame) -> np.ndarray:
    require_columns(frame, ["test_input_tokens", "test_output_tokens"], "evaluations")
    input_tokens = finite_numeric(frame["test_input_tokens"])
    output_tokens = finite_numeric(frame["test_output_tokens"])
    weighted_input = QWEN_INPUT_WEIGHT * input_tokens
    weighted_output = QWEN_OUTPUT_WEIGHT * output_tokens
    denominator = weighted_input + weighted_output
    return np.clip(
        np.divide(
            weighted_output,
            denominator,
            out=np.zeros_like(weighted_output, dtype=float),
            where=denominator > 0,
        ),
        0.0,
        1.0,
    )


def trifair_final_candidates(final: pd.DataFrame) -> pd.DataFrame:
    data = final[final["optimizer"] == "Tri-Fair"].copy()
    if "is_incumbent" in data and data["is_incumbent"].notna().any():
        incumbent = data[data["is_incumbent"].fillna(False).astype(bool)]
        if not incumbent.empty:
            data = incumbent.copy()
    if data.empty:
        raise RuntimeError("No Tri-Fair 5M candidate rows were found")
    if "few_shots_json" in data:
        data["fewshot_count"] = data["few_shots_json"].apply(parse_fewshot_count)
    else:
        data["fewshot_count"] = 0
    data["output_cost_share"] = output_cost_share(data)
    return data.reset_index(drop=True)


def plot_trifair_diagnostic(
    final: pd.DataFrame,
    outdir: Path,
    *,
    color_col: str,
    color_label: str,
    filename: str,
) -> None:
    data = trifair_final_candidates(final)
    values = finite_numeric(data[color_col])
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    scatter = ax.scatter(
        data["test_cost"],
        data["test_quality"],
        c=values,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolor="black",
        linewidth=0.5,
        s=72,
        alpha=0.95,
    )
    for _, row in data.iterrows():
        ax.text(
            float(row["test_cost"]),
            float(row["test_quality"]) + 0.002,
            str(int(row["fewshot_count"])),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_title("Tri-Fair on BBQ — Qwen-3-30B at 5M")
    ax.set_xlabel("Avg. Cost [$] per 1M Calls")
    ax.set_ylabel("Test Accuracy")
    ax.grid(True, alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(color_label)
    save_figure(fig, outdir, filename)


def write_summary_tables(
    final_metrics: pd.DataFrame,
    final_evaluations: pd.DataFrame,
    outdir: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for optimizer in OPTIMIZER_ORDER:
        metrics = final_metrics[final_metrics["optimizer"] == optimizer]
        evaluations = final_evaluations[final_evaluations["optimizer"] == optimizer]
        best_per_seed = (
            evaluations.groupby("seed", as_index=False)
            .apply(
                lambda group: group.loc[
                    pd.to_numeric(group["test_quality"], errors="coerce").idxmax()
                ]
            )
            .reset_index(drop=True)
        )
        rows.append(
            {
                "Method": DISPLAY_NAME[optimizer],
                "Seeds": int(metrics["seed"].nunique()),
                "Actual tokens mean": metrics["actual_budget_tokens"].mean(),
                "Holdout nR2 mean ↓": metrics["noisy_r2_3d"].mean(),
                "Holdout nR2 SD": metrics["noisy_r2_3d"].std(ddof=1),
                "HV optimistic mean ↑": metrics["hv_test_optimistic_3d"].mean(),
                "HV pessimistic mean ↑": metrics["hv_test_pessimistic_3d"].mean(),
                "Gap mean ↓": metrics["approximation_gap_3d"].mean(),
                "Balanced test accuracy mean ↑": metrics["balanced_test_quality"].mean(),
                "Balanced test cost mean ↓": metrics["balanced_test_cost"].mean(),
                "Balanced test unfairness mean ↓": metrics["balanced_test_fairness"].mean(),
                "Best-quality point mean accuracy ↑": best_per_seed["test_quality"].mean(),
                "Best-quality point mean cost ↓": best_per_seed["test_cost"].mean(),
                "Best-quality point mean unfairness ↓": best_per_seed["test_fairness"].mean(),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "bbq_qwen3_5m_summary_table.csv", index=False)
    (outdir / "bbq_qwen3_5m_summary_table.md").write_text(
        table.to_markdown(index=False, floatfmt=".4f") + "\n",
        encoding="utf-8",
    )
    print("\n5M method summary")
    print(table.to_string(index=False))


def write_readme(outdir: Path, analysis_dir: Path, results_root: Path) -> None:
    content = f"""# BBQ / Qwen-3-30B 5M curated figures

Generated by `python -m analysis.make_bbq_5m_figures`.

- Results root: `{results_root}`
- Dedicated analysis tables: `{analysis_dir}`
- Final checkpoint: `{FINAL_BUDGET:,}` downstream tokens
- Seeds: {', '.join(str(seed) for seed in EXPECTED_SEEDS)}

## Metric interpretation

- `noisy_r2_3d`: exact checkpoint-level holdout nR2 from the repository analysis pipeline (lower is better).
- `hv_test_optimistic_3d` / `hv_test_pessimistic_3d`: checkpoint-level test hypervolume (higher is better).
- `approximation_gap_3d`: checkpoint-level approximation gap (lower is better).
- `bbq_qwen3_5m_stepwise_dev_nr2_proxy.*`: development-only stepwise proxy, not exact holdout nR2.

## Main figure files

- `bbq_qwen3_5m_stepwise_dev_nr2_proxy.*`
- `bbq_qwen3_5m_stepwise_hv_checkpoint_gap.*`
- `bbq_qwen3_5m_checkpoint_nr2_hv_gap.*`
- `bbq_qwen3_5m_balanced_budget_trajectory.*`
- `bbq_qwen3_5m_attainment_accuracy_cost.*`
- `bbq_qwen3_5m_attainment_accuracy_unfairness.*`
- `bbq_qwen3_5m_pareto_cost_unfairness.*`
- `bbq_qwen3_5m_test_pareto_3d.*`
- `bbq_qwen3_5m_method_comparison_hv_nr2_gap.*`
- `bbq_qwen3_5m_trifair_fewshot_outputshare.*`
- `bbq_qwen3_5m_trifair_fewshot_unfairness.*`
"""
    (outdir / "README.md").write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_style()

    results_root = Path(args.results_root)
    analysis_dir = Path(args.analysis_dir)
    figure_dir = (
        Path(args.figure_dir)
        if args.figure_dir
        else analysis_dir / "curated_figures_mocapo_style_5m"
    )
    bounds_file = Path(args.bounds_file)
    budgets = parse_csv_ints(args.budgets)
    reference_point_values = tuple(
        float(piece.strip()) for piece in str(args.reference_point).split(",")
    )
    if len(reference_point_values) != 3:
        raise ValueError("--reference-point must contain exactly three values")
    reference_point = (
        float(reference_point_values[0]),
        float(reference_point_values[1]),
        float(reference_point_values[2]),
    )
    if FINAL_BUDGET not in budgets:
        budgets = tuple(sorted(set(budgets + (FINAL_BUDGET,))))

    needs_rebuild = args.rebuild_analysis or not analysis_has_5m(analysis_dir)
    if needs_rebuild:
        if args.no_auto_rebuild and not args.rebuild_analysis:
            raise RuntimeError(
                f"{analysis_dir} does not contain 5M tables. Re-run with --rebuild-analysis."
            )
        rebuild_analysis_outputs(
            results_root=results_root,
            analysis_dir=analysis_dir,
            bounds_file=bounds_file,
            budgets=budgets,
            n_preferences=int(args.n_preferences),
            preference_seed=int(args.preference_seed),
            reference_point=reference_point,
            strict=bool(args.strict),
        )

    run_metrics, trajectory, evaluations = read_inputs(analysis_dir)
    selected = selected_final_runs(run_metrics)
    final_evaluations = final_checkpoint_evaluations(evaluations, selected)
    checkpoint_metrics = checkpoint_run_metrics(run_metrics)
    final_metrics = checkpoint_metrics[
        checkpoint_metrics["budget_checkpoint"] == FINAL_BUDGET
    ].copy()

    # Validate before creating the figure directory.  A failed validation will
    # therefore not leave behind an empty "success-looking" folder.
    require_columns(
        final_metrics,
        [
            "noisy_r2_3d",
            "hv_test_optimistic_3d",
            "hv_test_pessimistic_3d",
            "approximation_gap_3d",
            "balanced_test_quality",
            "balanced_test_cost",
            "balanced_test_fairness",
        ],
        "5M run metrics",
    )

    selected_trajectory = filter_to_selected_runs(trajectory, selected)
    step_results = clean_frame(load_step_results_from_selected_runs(selected))
    step_proxy = stepwise_development_r2_proxy(step_results, final_evaluations)

    figure_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(figure_dir / "bbq_qwen3_5m_selected_runs.csv", index=False)
    checkpoint_metrics.to_csv(
        figure_dir / "bbq_qwen3_5m_checkpoint_run_metrics.csv", index=False
    )
    step_proxy.to_csv(
        figure_dir / "bbq_qwen3_5m_stepwise_dev_nr2_proxy.csv", index=False
    )

    plot_stepwise_nr2_proxy(step_proxy, figure_dir)
    plot_stepwise_hv_and_gap(selected_trajectory, checkpoint_metrics, figure_dir)
    plot_exact_checkpoint_mo_metrics(checkpoint_metrics, figure_dir)
    plot_balanced_budget_trajectory(checkpoint_metrics, figure_dir)
    plot_empirical_attainment(
        final_evaluations,
        figure_dir,
        x_col="test_cost",
        xlabel="Avg. Cost [$] per 1M Calls",
        filename="bbq_qwen3_5m_attainment_accuracy_cost",
    )
    plot_empirical_attainment(
        final_evaluations,
        figure_dir,
        x_col="test_fairness",
        xlabel="Test Unfairness",
        filename="bbq_qwen3_5m_attainment_accuracy_unfairness",
    )
    plot_cost_unfairness_projection(final_evaluations, figure_dir)
    plot_three_objective_pareto(final_evaluations, figure_dir)
    plot_method_comparison_bars(final_metrics, figure_dir)
    plot_trifair_diagnostic(
        final_evaluations,
        figure_dir,
        color_col="output_cost_share",
        color_label="Output Token Cost Share",
        filename="bbq_qwen3_5m_trifair_fewshot_outputshare",
    )
    plot_trifair_diagnostic(
        final_evaluations,
        figure_dir,
        color_col="test_fairness",
        color_label="Test Unfairness ↓",
        filename="bbq_qwen3_5m_trifair_fewshot_unfairness",
    )
    write_summary_tables(final_metrics, final_evaluations, figure_dir)
    write_readme(figure_dir, analysis_dir, results_root)

    generated = sorted(path.name for path in figure_dir.iterdir() if path.is_file())
    print("\nGenerated files")
    for name in generated:
        print(f"  {name}")
    print(f"\nBBQ 5M figures written to: {figure_dir.resolve()}")


if __name__ == "__main__":
    main()
