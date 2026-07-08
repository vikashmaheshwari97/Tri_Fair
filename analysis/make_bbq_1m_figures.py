"""MO-CAPO-style curated figures for BBQ / Qwen-3-30B / Tri-Fair.

Inputs expected under analysis/output/:
  - run_metrics.csv
  - trajectory_metrics.csv
  - all_evaluations.parquet

This script creates a compact figure set:
  1. Stepwise development nR2-proxy trajectory.
  2. Stepwise development HV trajectory plus checkpoint approximation gap.
  3. 1M empirical-attainment-style Test Accuracy vs Cost.
  4. 1M empirical-attainment-style Test Accuracy vs Test Unfairness.
  5. Tri-Fair-only final incumbent plot colored by output-token cost share.
  6. Tri-Fair-only final incumbent plot colored by test unfairness.

Terminology:
  - User-facing plots use Accuracy / Test Accuracy, not quality.
  - "Development nR2 proxy" is not the exact holdout nR2 from the paper.
    Exact stepwise holdout nR2 would require holdout evaluation for every
    optimizer step/front. Current eval.parquet files contain checkpoint-level
    holdout evaluations only.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL = "qwen-3-30b"
DATASET = "bbq"
FINAL_BUDGET = 1_000_000
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


@dataclass(frozen=True)
class CostBounds:
    """Simple normalization bounds for development-side stepwise metrics."""

    cost_max: float

    def normalize_minimize(self, raw: np.ndarray) -> np.ndarray:
        """Normalize minimize-all objectives [1-accuracy, cost, unfairness]."""
        out = raw.astype(float).copy()
        out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
        out[:, 1] = np.clip(out[:, 1] / self.cost_max, 0.0, 1.1)
        out[:, 2] = np.clip(out[:, 2], 0.0, 1.0)
        return out


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


def output_dir() -> Path:
    out = Path("analysis/output/curated_figures_mocapo_style")
    out.mkdir(parents=True, exist_ok=True)
    return out


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove archived pilot copies accidentally located under results/tri_fair."""
    out = frame.copy()
    if "run_dir" in out:
        out = out[
            ~out["run_dir"].astype(str).str.contains("pilot_reports", regex=False)
        ]
    if "run_key" in out:
        out = out[~out["run_key"].astype(str).str.endswith("/logging")]
    if "dataset" in out:
        out = out[out["dataset"] == DATASET]
    if "model" in out:
        out = out[out["model"] == MODEL]
    return out.reset_index(drop=True)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path("analysis/output")
    run_metrics_path = root / "run_metrics.csv"
    trajectory_path = root / "trajectory_metrics.csv"
    evaluations_path = root / "all_evaluations.parquet"

    missing = [
        str(path)
        for path in [run_metrics_path, trajectory_path, evaluations_path]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required analysis outputs:\n"
            + "\n".join(missing)
            + "\nRun analysis.analysis_pipeline first."
        )

    run_metrics = clean_frame(pd.read_csv(run_metrics_path))
    trajectory = clean_frame(pd.read_csv(trajectory_path))
    evaluations = clean_frame(pd.read_parquet(evaluations_path))

    if "test_fairness_ready" in evaluations:
        evaluations = evaluations[
            evaluations["test_fairness_ready"].fillna(False).astype(bool)
        ].reset_index(drop=True)

    return run_metrics, trajectory, evaluations


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

    expected_budgets = {250_000, FINAL_BUDGET}
    observed_budgets = set(int(v) for v in run_metrics["budget_checkpoint"].unique())
    if not expected_budgets.issubset(observed_budgets):
        raise RuntimeError(
            f"Expected at least budgets {sorted(expected_budgets)}, "
            f"observed {sorted(observed_budgets)}"
        )

    for optimizer in OPTIMIZER_ORDER:
        for budget in [250_000, FINAL_BUDGET]:
            n = int(
                counts[
                    (counts["optimizer"] == optimizer)
                    & (counts["budget_checkpoint"] == budget)
                ]["n_runs"].sum()
            )
            if n != 3:
                raise RuntimeError(
                    f"Expected 3 clean runs for {optimizer} at budget {budget}, got {n}. "
                    "Move archived pilot_reports out of results/tri_fair and rebuild analysis."
                )


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    """Return non-dominated mask for minimization objective matrix."""
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
    weights = rng.dirichlet(np.ones(3), size=n_preferences)
    return weights


def chebyshev_r2(normalized_minimize_front: np.ndarray, weights: np.ndarray) -> float:
    """Lower is better. Development-side proxy for nR2/R2-style utility."""
    if len(normalized_minimize_front) == 0:
        return float("nan")

    front = np.asarray(normalized_minimize_front, dtype=float)
    utilities = np.max(weights[:, None, :] * front[None, :, :], axis=2)
    chosen = np.min(utilities, axis=1)
    return float(np.mean(chosen))


def finite_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def load_step_results_from_run_dirs(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Load raw step_results.parquet for each unique run directory."""
    unique_runs = (
        run_metrics[
            ["run_key", "run_dir", "model", "dataset", "optimizer", "seed"]
        ]
        .drop_duplicates()
        .sort_values(["optimizer", "seed", "run_key"])
    )

    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for _, row in unique_runs.iterrows():
        run_dir = Path(str(row["run_dir"]))
        path = run_dir / "step_results.parquet"
        if not path.is_file():
            missing.append(str(path))
            continue

        df = pd.read_parquet(path)
        df = df.copy()
        df["run_key"] = row["run_key"]
        df["run_dir"] = row["run_dir"]
        df["model"] = row["model"]
        df["dataset"] = row["dataset"]
        df["optimizer"] = row["optimizer"]
        df["seed"] = int(row["seed"])
        frames.append(df)

    if missing:
        warnings.warn(
            "Some step_results.parquet files were not found and were skipped:\n"
            + "\n".join(missing[:10])
        )

    if not frames:
        raise RuntimeError(
            "No raw step_results.parquet files were found. "
            "Run this on Rocket where result run directories exist."
        )

    return pd.concat(frames, ignore_index=True)


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


def stepwise_development_r2_proxy(
    step_results: pd.DataFrame,
    evaluations: pd.DataFrame,
) -> pd.DataFrame:
    """Compute stepwise development-side nR2 proxy from raw step results.

    Exact holdout nR2 needs test evaluation of each step/front. This function
    uses development objectives only, so it is an optimization trajectory proxy.
    """
    bounds = infer_cost_bounds(step_results, evaluations)
    weights = make_preferences(N_PREFERENCES, PREFERENCE_SEED)

    rows: list[dict[str, object]] = []
    group_cols = ["run_key", "run_dir", "model", "dataset", "optimizer", "seed", "step"]

    for keys, group in step_results.groupby(group_cols, sort=True, dropna=False):
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
                "dev_noisy_r2_proxy": chebyshev_r2(front, weights),
                "dev_proxy_front_size": int(front_mask.sum()),
            }
        )

    if not rows:
        raise RuntimeError("Could not compute any stepwise development nR2-proxy rows")

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

    ax.set_title("BBQ — Qwen-3-30B")
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

    fig.savefig(outdir / "bbq_qwen3_nR2_trajectory_stepwise.pdf", bbox_inches="tight")
    fig.savefig(outdir / "bbq_qwen3_nR2_trajectory_stepwise.png", bbox_inches="tight")
    plt.close(fig)


def plot_hv_gap_trajectory(
    trajectory: pd.DataFrame,
    run_metrics: pd.DataFrame,
    outdir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # Stepwise development HV from trajectory_metrics.csv.
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

    # Checkpoint-level approximation gap from holdout run metrics.
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
    ax.set_xticks([0.25, 1.0])
    ax.set_xticklabels(["250k", "1M"])

    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("BBQ — Qwen-3-30B: Trajectory diagnostics", y=1.03)

    fig.savefig(outdir / "bbq_qwen3_hv_gap_trajectory_stepwise.pdf", bbox_inches="tight")
    fig.savefig(outdir / "bbq_qwen3_hv_gap_trajectory_stepwise.png", bbox_inches="tight")
    plt.close(fig)


def test_objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            1.0 - finite_numeric(frame["test_quality"]),
            finite_numeric(frame["test_cost"]),
            finite_numeric(frame["test_fairness"]),
        ]
    )


def final_checkpoint_evaluations(evaluations: pd.DataFrame) -> pd.DataFrame:
    final = evaluations[evaluations["budget_checkpoint"] == FINAL_BUDGET].copy()
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

    ax.set_title("BBQ — Qwen-3-30B at 1M")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test Accuracy")
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
    input_tokens = finite_numeric(frame["test_input_tokens"])
    output_tokens = finite_numeric(frame["test_output_tokens"])
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

    data["fewshot_count"] = data["few_shots_json"].apply(parse_fewshot_count)
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
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
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

    ax.set_title("Tri-Fair on BBQ — Qwen-3-30B at 1M")
    ax.set_xlabel("Avg. Cost [$] per 1M Calls")
    ax.set_ylabel("Test Accuracy")
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
                "Balanced Test Accuracy ↑": data["balanced_test_quality"].mean(),
                "Balanced Test Cost ↓": data["balanced_test_cost"].mean(),
                "Balanced Test Unfairness ↓": data[
                    "balanced_test_fairness"
                ].mean(),
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


def write_readme(outdir: Path) -> None:
    readme = """# BBQ / Qwen-3-30B 1M curated figures

Files generated by `analysis.make_bbq_1m_figures`.

## Important note on nR2 trajectories

`bbq_qwen3_nR2_trajectory_stepwise.*` is a development-side stepwise nR2 proxy
computed from `step_results.parquet`.

Exact MO-CAPO-style stepwise holdout nR2 would require evaluating each
intermediate optimizer front on the holdout set. Current `eval.parquet` files
contain checkpoint-level holdout evaluations only.

## Figures

- `bbq_qwen3_nR2_trajectory_stepwise.*`
  Stepwise development nR2 proxy, mean ± std across seeds.

- `bbq_qwen3_hv_gap_trajectory_stepwise.*`
  Stepwise development hypervolume plus checkpoint-level holdout approximation gap.

- `bbq_qwen3_1m_attainment_accuracy_cost.*`
  Empirical-attainment-style Test Accuracy vs Avg. Cost at the 1M checkpoint.

- `bbq_qwen3_1m_attainment_accuracy_unfairness.*`
  Empirical-attainment-style Test Accuracy vs Test Unfairness at the 1M checkpoint.

- `bbq_qwen3_1m_trifair_fewshot_outputshare.*`
  Tri-Fair final incumbent/evaluated candidates. Labels show few-shot count;
  color shows weighted output-token cost share.

- `bbq_qwen3_1m_trifair_fewshot_unfairness.*`
  Same Tri-Fair candidates. Labels show few-shot count; color shows test unfairness.
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    configure_style()
    outdir = output_dir()

    run_metrics, trajectory, evaluations = read_inputs()
    check_clean_counts(run_metrics)

    step_results = load_step_results_from_run_dirs(run_metrics)
    step_results = clean_frame(step_results)

    step_proxy = stepwise_development_r2_proxy(step_results, evaluations)
    step_proxy.to_csv(outdir / "bbq_qwen3_stepwise_dev_nR2_proxy.csv", index=False)

    plot_stepwise_nR2_proxy(step_proxy, outdir)
    plot_hv_gap_trajectory(trajectory, run_metrics, outdir)

    plot_empirical_attainment(
        evaluations,
        outdir,
        x_col="test_cost",
        xlabel="Avg. Cost [$] per 1M Calls",
        filename="bbq_qwen3_1m_attainment_accuracy_cost",
    )

    plot_empirical_attainment(
        evaluations,
        outdir,
        x_col="test_fairness",
        xlabel="Test Unfairness",
        filename="bbq_qwen3_1m_attainment_accuracy_unfairness",
    )

    plot_trifair_fewshot_scatter(
        evaluations,
        outdir,
        color_col="output_cost_share",
        color_label="Output Token Cost Share",
        filename="bbq_qwen3_1m_trifair_fewshot_outputshare",
        cmap="viridis",
    )

    plot_trifair_fewshot_scatter(
        evaluations,
        outdir,
        color_col="test_fairness",
        color_label="Test Unfairness ↓",
        filename="bbq_qwen3_1m_trifair_fewshot_unfairness",
        cmap="viridis",
    )

    write_summary_table(run_metrics, outdir)
    write_readme(outdir)

    print("\nCurated MO-CAPO-style figures written to:")
    print(outdir.resolve())


if __name__ == "__main__":
    main()