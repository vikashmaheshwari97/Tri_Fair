"""MO-CAPO-style curated figures for advanced Bias in Bios 1M runs.

This script keeps the same figure style and file naming used by
``analysis/make_bias_1m_figures.py`` for the original 49% Bias figures, but it
loads the prompt-only advanced Bias runs directly from ``jobs/bias_advanced.sbatch``
output directories instead of ``analysis/output/run_metrics.csv``.

Default input root:
  results/tri_fair_bias_high_macro_f1/compact_full_1m

Expected runs:
  qwen-3-30b/bias_in_bios/{Tri-Fair,NSGAII-PO-Fair}/seed{42,43,44}/budget1000000

Output:
  analysis/output/curated_figures_bias_advanced_1m_mocapo_style

Figures generated:
  1. Stepwise development nR2-proxy trajectory.
  2. Stepwise development HV trajectory plus 1M holdout gap proxy.
  3. 1M empirical-attainment-style Test Macro-F1 vs Cost.
  4. 1M empirical-attainment-style Test Macro-F1 vs Test Unfairness.
  5. Tri-Fair-only final incumbent plot colored by output-token cost share.
  6. Tri-Fair-only final incumbent plot colored by test unfairness.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL = "qwen-3-30b"
DATASET = "bias_in_bios"
FINAL_BUDGET = 1_000_000
TIER = "compact"
N_PREFERENCES = 500
PREFERENCE_SEED = 2026

# Qwen3-30B weights used by the experiment cost objective.
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

DEFAULT_RESULTS_ROOT = "results/tri_fair_bias_high_macro_f1/compact_full_1m"
DEFAULT_OUT_DIR = "analysis/output/curated_figures_bias_advanced_1m_mocapo_style"


@dataclass(frozen=True)
class CostBounds:
    """Simple normalization bounds for development-side stepwise metrics."""

    cost_max: float

    def normalize_minimize(self, raw: np.ndarray) -> np.ndarray:
        """Normalize minimize-all objectives [1-macro_f1, cost, unfairness]."""
        out = raw.astype(float).copy()
        out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
        out[:, 1] = np.clip(out[:, 1] / self.cost_max, 0.0, 1.1)
        out[:, 2] = np.clip(out[:, 2], 0.0, 1.0)
        return out


@dataclass
class RunData:
    optimizer: str
    seed: int
    run_key: str
    output_dir: Path
    logging_dir: Path
    step_results: pd.DataFrame
    evaluations: pd.DataFrame
    run_summary: dict[str, object]


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


def output_dir(args: argparse.Namespace) -> Path:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def finite_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def split_csv(raw: str, *, cast=str) -> list:
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.warn(f"Could not parse {path}: {exc}")
        return {}


def resolve_output_dir(args: argparse.Namespace, optimizer: str, seed: int) -> Path:
    return (
        Path(args.results_root)
        / args.model
        / args.dataset
        / optimizer
        / f"seed{seed}"
        / f"budget{args.budget}"
    )


def resolve_logging_dir(output_dir: Path) -> Path:
    pointer = output_dir / "logging_dir.txt"
    if not pointer.is_file():
        raise FileNotFoundError(f"Missing logging_dir.txt: {pointer}")
    raw = pointer.read_text(encoding="utf-8").strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = pointer.parent / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Resolved logging directory does not exist: {path}")
    return path


def load_run(args: argparse.Namespace, optimizer: str, seed: int) -> RunData:
    output = resolve_output_dir(args, optimizer, seed)
    logging_dir = resolve_logging_dir(output)
    step_path = logging_dir / "step_results.parquet"
    eval_path = logging_dir / "eval.parquet"

    if not step_path.is_file():
        raise FileNotFoundError(step_path)
    if not eval_path.is_file():
        raise FileNotFoundError(eval_path)

    run_key = f"{args.model}/{args.dataset}/{optimizer}/seed{seed}/budget{args.budget}"
    return RunData(
        optimizer=optimizer,
        seed=int(seed),
        run_key=run_key,
        output_dir=output,
        logging_dir=logging_dir,
        step_results=pd.read_parquet(step_path),
        evaluations=pd.read_parquet(eval_path),
        run_summary=read_json(logging_dir / "run_summary.json"),
    )


def load_runs(args: argparse.Namespace) -> list[RunData]:
    optimizers = split_csv(args.optimizers, cast=str)
    seeds = split_csv(args.seeds, cast=int)
    runs: list[RunData] = []
    missing: list[str] = []

    for optimizer in optimizers:
        for seed in seeds:
            try:
                runs.append(load_run(args, optimizer, seed))
            except Exception as exc:
                label = f"{optimizer} seed{seed}: {exc}"
                if args.allow_missing:
                    warnings.warn(label)
                else:
                    missing.append(label)

    if missing:
        raise FileNotFoundError("Missing required runs:\n  - " + "\n  - ".join(missing))
    if not runs:
        raise RuntimeError("No runs could be loaded")
    return runs


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    """Return non-dominated mask for minimization objective matrix."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.zeros(0, dtype=bool)

    keep = np.ones(len(values), dtype=bool)
    for i in range(len(values)):
        dominates_i = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            keep[i] = False
    return keep


def make_preferences(n_preferences: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(3), size=n_preferences)


def chebyshev_r2(normalized_minimize_front: np.ndarray, weights: np.ndarray) -> float:
    """Lower is better. Development-side proxy for nR2/R2-style utility."""
    if len(normalized_minimize_front) == 0:
        return float("nan")

    front = np.asarray(normalized_minimize_front, dtype=float)
    utilities = np.max(weights[:, None, :] * front[None, :, :], axis=2)
    chosen = np.min(utilities, axis=1)
    return float(np.mean(chosen))


def hypervolume_3d(
    normalized_minimize_front: np.ndarray,
    ref: tuple[float, float, float] = (1.05, 1.05, 1.05),
) -> float:
    """Exact grid-sweep hypervolume for small normalized 3D minimization fronts."""
    points = np.asarray(normalized_minimize_front, dtype=float)
    if points.size == 0:
        return 0.0

    ref_arr = np.asarray(ref, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    points = points[np.all(points <= ref_arr, axis=1)]
    if len(points) == 0:
        return 0.0

    points = points[pareto_mask_minimize(points)]
    coords = [np.sort(np.unique(np.r_[points[:, d], ref_arr[d]])) for d in range(3)]

    hv = 0.0
    for i in range(len(coords[0]) - 1):
        for j in range(len(coords[1]) - 1):
            for k in range(len(coords[2]) - 1):
                low = np.array([coords[0][i], coords[1][j], coords[2][k]])
                if np.any(np.all(points <= low, axis=1)):
                    hv += float(
                        (coords[0][i + 1] - coords[0][i])
                        * (coords[1][j + 1] - coords[1][j])
                        * (coords[2][k + 1] - coords[2][k])
                    )
    return hv


def step_tokens(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    if "total_tokens_downstream" in out:
        return out, "total_tokens_downstream"
    if "actual_budget_tokens" in out:
        return out, "actual_budget_tokens"
    if {"input_tokens_downstream", "output_tokens_downstream"}.issubset(out.columns):
        out["_total_tokens"] = finite_numeric(out["input_tokens_downstream"]) + finite_numeric(
            out["output_tokens_downstream"]
        )
        return out, "_total_tokens"
    raise ValueError(
        "step_results needs total_tokens_downstream or input/output token columns"
    )


def stage_eval(evaluations: pd.DataFrame, budget: int) -> pd.DataFrame:
    out = evaluations.copy()

    if "budget_checkpoint" in out:
        stage = out[out["budget_checkpoint"] == budget].copy()
        if not stage.empty:
            out = stage

    if "test_fairness_ready" in out:
        ready = out["test_fairness_ready"].fillna(False).astype(bool)
        if ready.any():
            out = out[ready].copy()

    return out.reset_index(drop=True)


def development_objectives(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            1.0 - finite_numeric(frame["quality"]),
            finite_numeric(frame["cost"]),
            finite_numeric(frame["fairness"]),
        ]
    )


def test_objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            1.0 - finite_numeric(frame["test_quality"]),
            finite_numeric(frame["test_cost"]),
            finite_numeric(frame["test_fairness"]),
        ]
    )


def infer_cost_bounds(step_results: pd.DataFrame, evaluations: pd.DataFrame) -> CostBounds:
    candidates: list[float] = []

    if "cost" in step_results:
        candidates.append(float(np.nanmax(finite_numeric(step_results["cost"]))))

    for column in ["dev_cost", "test_cost"]:
        if column in evaluations:
            candidates.append(float(np.nanmax(finite_numeric(evaluations[column]))))

    cost_max = max([value for value in candidates if np.isfinite(value)] or [1.0])
    cost_max = max(cost_max * 1.05, 1.0)
    return CostBounds(cost_max=cost_max)


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "dataset" in out:
        out = out[out["dataset"] == DATASET]
    if "model" in out:
        out = out[out["model"] == MODEL]
    return out.reset_index(drop=True)


def concatenate_step_results(runs: Sequence[RunData]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        frame = run.step_results.copy()
        frame["run_key"] = run.run_key
        frame["run_dir"] = str(run.logging_dir)
        frame["model"] = MODEL
        frame["dataset"] = DATASET
        frame["optimizer"] = run.optimizer
        frame["seed"] = int(run.seed)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def concatenate_evaluations(runs: Sequence[RunData], budget: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        frame = stage_eval(run.evaluations, budget).copy()
        frame["run_key"] = run.run_key
        frame["run_dir"] = str(run.logging_dir)
        frame["model"] = MODEL
        frame["dataset"] = DATASET
        frame["optimizer"] = run.optimizer
        frame["seed"] = int(run.seed)
        if "budget_checkpoint" not in frame:
            frame["budget_checkpoint"] = int(budget)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def check_clean_counts(run_metrics: pd.DataFrame) -> None:
    counts = (
        run_metrics.groupby(["optimizer", "budget_checkpoint"])
        .size()
        .rename("n_runs")
        .reset_index()
        .sort_values(["optimizer", "budget_checkpoint"])
    )

    print("\nRun-metric counts after cleaning")
    print(counts.to_string(index=False))

    for optimizer in OPTIMIZER_ORDER:
        n = int(
            counts[
                (counts["optimizer"] == optimizer)
                & (counts["budget_checkpoint"] == FINAL_BUDGET)
            ]["n_runs"].sum()
        )
        if n != 3:
            raise RuntimeError(
                f"Expected 3 clean runs for {optimizer} at budget {FINAL_BUDGET}, got {n}."
            )


def stepwise_development_metrics(
    step_results: pd.DataFrame,
    evaluations: pd.DataFrame,
) -> pd.DataFrame:
    """Compute stepwise development-side nR2 and HV proxies from raw step results.

    Exact holdout nR2/HV needs test evaluation of each step/front. This function
    uses development objectives only, so it is an optimization trajectory proxy.
    """
    bounds = infer_cost_bounds(step_results, evaluations)
    weights = make_preferences(N_PREFERENCES, PREFERENCE_SEED)

    rows: list[dict[str, object]] = []
    group_cols = ["run_key", "run_dir", "model", "dataset", "optimizer", "seed", "step"]

    data, tok = step_tokens(step_results)
    for keys, group in data.groupby(group_cols, sort=True, dropna=False):
        meta = dict(zip(group_cols, keys))

        required = {"quality", "cost", "fairness"}
        if missing := required - set(group.columns):
            raise ValueError(f"step_results missing required columns: {sorted(missing)}")

        quality = finite_numeric(group["quality"])
        cost = finite_numeric(group["cost"])
        fairness = finite_numeric(group["fairness"])

        mask = np.isfinite(quality) & np.isfinite(cost) & np.isfinite(fairness)

        if "fairness_ready" in group:
            mask &= group["fairness_ready"].fillna(False).astype(bool).to_numpy()

        valid = group.loc[mask].copy()
        if valid.empty:
            continue

        raw_minimize = np.column_stack(
            [
                1.0 - quality[mask],
                cost[mask],
                fairness[mask],
            ]
        )
        normalized = bounds.normalize_minimize(raw_minimize)
        front_mask = pareto_mask_minimize(normalized)
        front = normalized[front_mask]

        rows.append(
            {
                **meta,
                "actual_budget_tokens": int(np.nanmax(finite_numeric(group[tok]))),
                "dev_noisy_r2_proxy": chebyshev_r2(front, weights),
                "hv_dev_3d": hypervolume_3d(front),
                "dev_proxy_front_size": int(front_mask.sum()),
                "normalization_cost_max": float(bounds.cost_max),
            }
        )

    if not rows:
        raise RuntimeError("Could not compute any stepwise development proxy rows")

    return pd.DataFrame(rows)


def test_front_metrics(evaluations: pd.DataFrame, budget: int, bounds: CostBounds) -> dict[str, float]:
    final = stage_eval(evaluations, budget)
    raw = test_objective_matrix(final)
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    if len(raw) == 0:
        return {
            "test_noisy_r2_proxy": float("nan"),
            "hv_test_3d_proxy": float("nan"),
            "test_front_size": 0,
        }

    normalized = bounds.normalize_minimize(raw)
    front = normalized[pareto_mask_minimize(normalized)]
    weights = make_preferences(N_PREFERENCES, PREFERENCE_SEED)
    return {
        "test_noisy_r2_proxy": chebyshev_r2(front, weights),
        "hv_test_3d_proxy": hypervolume_3d(front),
        "test_front_size": int(len(front)),
    }


def best_eval_for_run(evaluations: pd.DataFrame, budget: int) -> pd.Series:
    final = stage_eval(evaluations, budget)
    required = {"test_quality", "test_cost", "test_fairness"}
    if missing := required - set(final.columns):
        raise ValueError(f"eval.parquet missing required columns: {sorted(missing)}")
    return final.sort_values(
        ["test_quality", "test_fairness", "test_cost"],
        ascending=[False, True, True],
    ).iloc[0]


def build_run_metrics(
    runs: Sequence[RunData],
    step_proxy: pd.DataFrame,
    evaluations: pd.DataFrame,
) -> pd.DataFrame:
    bounds = infer_cost_bounds(concatenate_step_results(runs), evaluations)
    rows: list[dict[str, object]] = []

    for run in runs:
        run_steps = step_proxy[
            (step_proxy["optimizer"] == run.optimizer) & (step_proxy["seed"] == run.seed)
        ].sort_values("actual_budget_tokens")
        if run_steps.empty:
            raise RuntimeError(f"No stepwise proxy rows for {run.optimizer} seed{run.seed}")

        final_dev = run_steps.iloc[-1]
        best = best_eval_for_run(run.evaluations, FINAL_BUDGET)
        test_metrics = test_front_metrics(run.evaluations, FINAL_BUDGET, bounds)

        actual_tokens = int(final_dev["actual_budget_tokens"])
        if "actual_tokens" in run.run_summary:
            actual_tokens = int(float(run.run_summary["actual_tokens"]))

        rows.append(
            {
                "run_key": run.run_key,
                "run_dir": str(run.logging_dir),
                "model": MODEL,
                "dataset": DATASET,
                "optimizer": run.optimizer,
                "seed": int(run.seed),
                "budget_checkpoint": FINAL_BUDGET,
                "actual_budget_tokens": actual_tokens,
                "noisy_r2_3d": test_metrics["test_noisy_r2_proxy"],
                "hv_test_optimistic_3d": test_metrics["hv_test_3d_proxy"],
                "hv_test_pessimistic_3d": test_metrics["hv_test_3d_proxy"],
                "approximation_gap_3d": test_metrics["test_noisy_r2_proxy"]
                - float(final_dev["dev_noisy_r2_proxy"]),
                "balanced_test_quality": float(best["test_quality"]),
                "balanced_test_cost": float(best["test_cost"]),
                "balanced_test_fairness": float(best["test_fairness"]),
                "test_front_size": test_metrics["test_front_size"],
            }
        )

    return pd.DataFrame(rows)


def prepare_staircase_grid(data: pd.DataFrame, max_budget: int | None = None) -> np.ndarray:
    values = finite_numeric(data["actual_budget_tokens"])
    values = values[np.isfinite(values)]
    if max_budget is not None:
        values = values[values <= max_budget]
    values = np.unique(values.astype(int))
    values = values[values > 0]
    if len(values) == 0:
        return np.array([], dtype=int)
    return np.sort(values)


def interpolate_step_values(
    run_data: pd.DataFrame,
    grid: np.ndarray,
    value_col: str,
) -> np.ndarray:
    data = run_data.sort_values("actual_budget_tokens")
    x = finite_numeric(data["actual_budget_tokens"])
    y = finite_numeric(data[value_col])

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].astype(int)
    y = y[mask]

    if len(x) == 0:
        return np.full(len(grid), np.nan)

    # Collapse duplicate token values by keeping the latest value.
    collapsed = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].last()
    x = collapsed["x"].to_numpy(dtype=int)
    y = collapsed["y"].to_numpy(dtype=float)

    out = np.full(len(grid), np.nan)
    for i, gx in enumerate(grid):
        idx = np.searchsorted(x, gx, side="right") - 1
        if idx >= 0:
            out[i] = y[idx]
    return out


def step_band_statistics(
    data: pd.DataFrame,
    value_col: str,
    *,
    max_budget: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return per-optimizer step grid, mean, lower, upper."""
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for optimizer in OPTIMIZER_ORDER:
        opt = data[data["optimizer"] == optimizer].copy()
        if opt.empty:
            continue

        grid = prepare_staircase_grid(opt, max_budget=max_budget)
        if len(grid) == 0:
            continue

        seed_values = []
        for _, seed_group in opt.groupby("seed", sort=True):
            seed_values.append(interpolate_step_values(seed_group, grid, value_col))

        matrix = np.vstack(seed_values)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        lower = mean - std
        upper = mean + std

        valid = np.isfinite(mean)
        result[optimizer] = (grid[valid], mean[valid], lower[valid], upper[valid])

    return result


def plot_stepwise_nR2_proxy(step_proxy: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)

    stats = step_band_statistics(
        step_proxy,
        "dev_noisy_r2_proxy",
        max_budget=1_250_000,
    )

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
        ax.fill_between(
            x,
            lower,
            upper,
            step="post",
            color=COLORS[optimizer],
            alpha=0.16,
        )

    ax.set_title("Bias in Bios — Qwen-3-30B")
    ax.set_xlabel("Token Budget [×10⁶]")
    ax.set_ylabel("Development nR2 Proxy ↓")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")

    note = (
        "Stepwise proxy from development objectives.\n"
        "Exact test nR2 requires holdout evaluation at each step."
    )
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        alpha=0.75,
    )

    fig.savefig(outdir / "bias_in_bios_qwen3_nR2_trajectory_stepwise.pdf", bbox_inches="tight")
    fig.savefig(outdir / "bias_in_bios_qwen3_nR2_trajectory_stepwise.png", bbox_inches="tight")
    plt.close(fig)


def plot_hv_gap_trajectory(
    trajectory: pd.DataFrame,
    run_metrics: pd.DataFrame,
    outdir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Stepwise development HV from raw step_results.
    hv_stats = step_band_statistics(
        trajectory,
        "hv_dev_3d",
        max_budget=1_250_000,
    )

    ax = axes[0]
    for optimizer in OPTIMIZER_ORDER:
        if optimizer not in hv_stats:
            continue
        grid, mean, lower, upper = hv_stats[optimizer]
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
        ax.fill_between(
            x,
            lower,
            upper,
            step="post",
            color=COLORS[optimizer],
            alpha=0.16,
        )

    ax.set_title("Development Hypervolume ↑")
    ax.set_xlabel("Token Budget [×10⁶]")
    ax.set_ylabel("Development HV")
    ax.grid(True, alpha=0.25)

    # The advanced sbatch evaluates the final 1M checkpoint only.
    # Keep the old panel style but plot the available holdout gap at 1M.
    ax = axes[1]
    for optimizer in OPTIMIZER_ORDER:
        data = run_metrics[run_metrics["optimizer"] == optimizer].copy()
        grouped = (
            data.groupby("budget_checkpoint")["approximation_gap_3d"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("budget_checkpoint")
        )
        x = grouped["budget_checkpoint"].to_numpy(dtype=float) / 1_000_000.0
        mean = grouped["mean"].to_numpy(dtype=float)
        std = grouped["std"].fillna(0.0).to_numpy(dtype=float)

        ax.step(
            x,
            mean,
            where="post",
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            linewidth=2,
            markersize=5,
            label=DISPLAY_NAME[optimizer],
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            step="post",
            color=COLORS[optimizer],
            alpha=0.16,
        )

    ax.set_title("Holdout Approximation Gap ↓")
    ax.set_xlabel("Evaluation-token checkpoint [×10⁶]")
    ax.set_ylabel("Approximation Gap")
    ax.grid(True, alpha=0.25)
    ax.set_xticks([1.0])
    ax.set_xticklabels(["1M"])

    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("Bias in Bios — Qwen-3-30B: Trajectory diagnostics", y=1.03)

    fig.savefig(outdir / "bias_in_bios_qwen3_hv_gap_trajectory_stepwise.pdf", bbox_inches="tight")
    fig.savefig(outdir / "bias_in_bios_qwen3_hv_gap_trajectory_stepwise.png", bbox_inches="tight")
    plt.close(fig)


def final_checkpoint_evaluations(evaluations: pd.DataFrame) -> pd.DataFrame:
    if "budget_checkpoint" in evaluations:
        final = evaluations[evaluations["budget_checkpoint"] == FINAL_BUDGET].copy()
    else:
        final = evaluations.copy()
    if final.empty:
        raise RuntimeError(f"No evaluations for budget_checkpoint={FINAL_BUDGET}")
    return final.reset_index(drop=True)


def y_attained_at_x(data: pd.DataFrame, x_grid: np.ndarray, x_col: str) -> np.ndarray:
    """For minimization x and maximization accuracy, return best accuracy at x <= grid."""
    x = finite_numeric(data[x_col])
    y = finite_numeric(data["test_quality"])
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    out = np.full(len(x_grid), np.nan)
    for i, gx in enumerate(x_grid):
        eligible = y[x <= gx]
        if len(eligible):
            out[i] = np.max(eligible)
    return out


def plot_empirical_attainment(
    evaluations: pd.DataFrame,
    outdir: Path,
    *,
    x_col: str,
    xlabel: str,
    filename: str,
) -> None:
    final = final_checkpoint_evaluations(evaluations)

    xmin = float(np.nanmin(finite_numeric(final[x_col])))
    xmax = float(np.nanmax(finite_numeric(final[x_col])))
    padding = 0.03 * max(xmax - xmin, 1e-6)
    x_grid = np.linspace(xmin - padding, xmax + padding, 400)

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)

    for optimizer in OPTIMIZER_ORDER:
        opt = final[final["optimizer"] == optimizer].copy()
        if opt.empty:
            continue

        seed_curves = []
        for _, seed_group in opt.groupby("seed", sort=True):
            # Use the 3D test non-dominated candidates from each seed.
            mask = pareto_mask_minimize(test_objective_matrix(seed_group))
            front = seed_group.loc[mask].copy()
            seed_curves.append(y_attained_at_x(front, x_grid, x_col))

        matrix = np.vstack(seed_curves)
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
            x_grid[valid],
            lower[valid],
            upper[valid],
            step="post",
            color=COLORS[optimizer],
            alpha=0.16,
        )

    ax.set_title("Bias in Bios — Qwen-3-30B at 1M")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test Macro-F1")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    fig.savefig(outdir / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{filename}.png", bbox_inches="tight")
    plt.close(fig)


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
        if isinstance(loaded, list):
            return len(loaded)
    except json.JSONDecodeError:
        return 0
    return 0


def output_cost_share(frame: pd.DataFrame) -> np.ndarray:
    if {"test_input_tokens", "test_output_tokens"}.issubset(frame.columns):
        input_tokens = finite_numeric(frame["test_input_tokens"])
        output_tokens = finite_numeric(frame["test_output_tokens"])
    elif {"input_tokens", "output_tokens"}.issubset(frame.columns):
        input_tokens = finite_numeric(frame["input_tokens"])
        output_tokens = finite_numeric(frame["output_tokens"])
    else:
        return np.zeros(len(frame), dtype=float)

    weighted_input = QWEN_INPUT_WEIGHT * input_tokens
    weighted_output = QWEN_OUTPUT_WEIGHT * output_tokens
    denom = weighted_input + weighted_output
    share = np.divide(
        weighted_output,
        denom,
        out=np.zeros_like(weighted_output, dtype=float),
        where=denom > 0,
    )
    return np.clip(share, 0.0, 1.0)


def trifair_final_incumbents(evaluations: pd.DataFrame) -> pd.DataFrame:
    final = final_checkpoint_evaluations(evaluations)
    data = final[final["optimizer"] == "Tri-Fair"].copy()

    if "is_incumbent" in data and data["is_incumbent"].notna().any():
        incumbent = data[data["is_incumbent"].fillna(False).astype(bool)].copy()
        if not incumbent.empty:
            data = incumbent

    if data.empty:
        raise RuntimeError("No Tri-Fair final incumbent/evaluation rows found")

    if "few_shots_json" in data:
        data["fewshot_count"] = data["few_shots_json"].apply(parse_fewshot_count)
    else:
        data["fewshot_count"] = 0
    data["output_cost_share"] = output_cost_share(data)
    return data.reset_index(drop=True)


def plot_trifair_fewshot_scatter(
    evaluations: pd.DataFrame,
    outdir: Path,
    *,
    color_col: str,
    color_label: str,
    filename: str,
    cmap: str,
) -> None:
    data = trifair_final_incumbents(evaluations)

    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)

    values = finite_numeric(data[color_col])
    vmin = float(np.nanmin(values)) if np.isfinite(values).any() else 0.0
    vmax = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    scatter = ax.scatter(
        data["test_cost"],
        data["test_quality"],
        c=values,
        cmap=cmap,
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

    ax.set_title("Tri-Fair on Bias in Bios — Qwen-3-30B at 1M")
    ax.set_xlabel("Avg. Cost [$] per 1M Calls")
    ax.set_ylabel("Test Macro-F1")
    ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(color_label)

    fig.savefig(outdir / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{filename}.png", bbox_inches="tight")
    plt.close(fig)


def write_summary_table(run_metrics: pd.DataFrame, outdir: Path) -> None:
    selected = run_metrics[run_metrics["budget_checkpoint"] == FINAL_BUDGET].copy()

    rows = []
    for optimizer in OPTIMIZER_ORDER:
        data = selected[selected["optimizer"] == optimizer].copy()
        rows.append(
            {
                "Method": DISPLAY_NAME[optimizer],
                "Actual tokens mean": data["actual_budget_tokens"].mean(),
                "nR2 ↓": data["noisy_r2_3d"].mean(),
                "HVopt ↑": data["hv_test_optimistic_3d"].mean(),
                "HVpes ↑": data["hv_test_pessimistic_3d"].mean(),
                "Gap ↓": data["approximation_gap_3d"].mean(),
                "Balanced Test Macro-F1 ↑": data["balanced_test_quality"].mean(),
                "Balanced Test Cost ↓": data["balanced_test_cost"].mean(),
                "Balanced Test Unfairness ↓": data["balanced_test_fairness"].mean(),
                "Seeds": len(data),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(outdir / "bias_in_bios_qwen3_1m_summary_table.csv", index=False)
    (outdir / "bias_in_bios_qwen3_1m_summary_table.md").write_text(
        table.to_markdown(index=False, floatfmt=".4f") + "\n",
        encoding="utf-8",
    )

    print("\n1M summary table")
    print(table.to_string(index=False))


def write_readme(outdir: Path, args: argparse.Namespace, runs: Sequence[RunData]) -> None:
    loaded = "\n".join(f"- {run.optimizer} seed{run.seed}: `{run.logging_dir}`" for run in runs)
    readme = f"""# Bias in Bios / Qwen-3-30B 1M curated figures

Files generated by `analysis.make_bias_170287_figures`.

This is the advanced prompt-only Bias 1M setup, but the visual style and file
names intentionally match the original `analysis.make_bias_1m_figures.py` output.

Results root: `{args.results_root}`

Loaded runs:
{loaded}

## Important note on nR2 trajectories

`bias_in_bios_qwen3_nR2_trajectory_stepwise.*` is a development-side stepwise nR2 proxy
computed from `step_results.parquet`.

Exact MO-CAPO-style stepwise holdout nR2 would require evaluating each
intermediate optimizer front on the holdout set. Current `eval.parquet` files
contain checkpoint-level holdout evaluations only.

## Figures

- `bias_in_bios_qwen3_nR2_trajectory_stepwise.*`
  Stepwise development nR2 proxy, mean ± std across seeds.

- `bias_in_bios_qwen3_hv_gap_trajectory_stepwise.*`
  Stepwise development hypervolume plus final-checkpoint holdout approximation gap proxy.

- `bias_in_bios_qwen3_1m_attainment_accuracy_cost.*`
  Empirical-attainment-style Test Macro-F1 vs Avg. Cost at the 1M checkpoint.

- `bias_in_bios_qwen3_1m_attainment_accuracy_unfairness.*`
  Empirical-attainment-style Test Macro-F1 vs Test Unfairness at the 1M checkpoint.

- `bias_in_bios_qwen3_1m_trifair_fewshot_outputshare.*`
  Tri-Fair final incumbent/evaluated candidates. Labels show few-shot count;
  color shows weighted output-token cost share.

- `bias_in_bios_qwen3_1m_trifair_fewshot_unfairness.*`
  Same Tri-Fair candidates. Labels show few-shot count; color shows test unfairness.
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--budget", type=int, default=FINAL_BUDGET)
    parser.add_argument("--tier", default=TIER)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--optimizers", default=",".join(OPTIMIZER_ORDER))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global MODEL, DATASET, FINAL_BUDGET, TIER
    MODEL = args.model
    DATASET = args.dataset
    FINAL_BUDGET = int(args.budget)
    TIER = args.tier

    configure_style()
    outdir = output_dir(args)

    runs = load_runs(args)
    step_results = clean_frame(concatenate_step_results(runs))
    evaluations = clean_frame(concatenate_evaluations(runs, args.budget))

    step_proxy = stepwise_development_metrics(step_results, evaluations)
    run_metrics = build_run_metrics(runs, step_proxy, evaluations)
    check_clean_counts(run_metrics)

    step_proxy.to_csv(outdir / "bias_in_bios_qwen3_stepwise_dev_nR2_proxy.csv", index=False)
    step_proxy.to_csv(outdir / "bias_in_bios_qwen3_stepwise_dev_metrics.csv", index=False)
    run_metrics.to_csv(outdir / "bias_in_bios_qwen3_1m_run_metrics_proxy.csv", index=False)
    evaluations.to_csv(outdir / "bias_in_bios_qwen3_1m_eval_rows.csv", index=False)

    plot_stepwise_nR2_proxy(step_proxy, outdir)
    plot_hv_gap_trajectory(step_proxy, run_metrics, outdir)

    plot_empirical_attainment(
        evaluations,
        outdir,
        x_col="test_cost",
        xlabel="Avg. Cost [$] per 1M Calls",
        filename="bias_in_bios_qwen3_1m_attainment_accuracy_cost",
    )

    plot_empirical_attainment(
        evaluations,
        outdir,
        x_col="test_fairness",
        xlabel="Test Unfairness",
        filename="bias_in_bios_qwen3_1m_attainment_accuracy_unfairness",
    )

    plot_trifair_fewshot_scatter(
        evaluations,
        outdir,
        color_col="output_cost_share",
        color_label="Output Token Cost Share",
        filename="bias_in_bios_qwen3_1m_trifair_fewshot_outputshare",
        cmap="viridis",
    )

    plot_trifair_fewshot_scatter(
        evaluations,
        outdir,
        color_col="test_fairness",
        color_label="Test Unfairness ↓",
        filename="bias_in_bios_qwen3_1m_trifair_fewshot_unfairness",
        cmap="viridis",
    )

    write_summary_table(run_metrics, outdir)
    write_readme(outdir, args, runs)

    pd.DataFrame(
        [
            {"file": p.name, "bytes": p.stat().st_size}
            for p in sorted(outdir.iterdir())
            if p.is_file()
        ]
    ).to_csv(outdir / "manifest.csv", index=False)

    print("\nCurated MO-CAPO-style figures written to:")
    print(outdir.resolve())


if __name__ == "__main__":
    main()
