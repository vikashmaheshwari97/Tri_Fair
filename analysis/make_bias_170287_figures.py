"""MO-CAPO-style curated figures for advanced Bias in Bios 1M runs.

This replaces the old single-job 170287 diagnostic script.  It summarizes the
prompt-only advanced Bias setup across multiple optimizers and seeds, using the
same result layout produced by jobs/bias_advanced.sbatch.

Default input root:
  results/tri_fair_bias_high_macro_f1/compact_full_1m

Expected runs:
  qwen-3-30b/bias_in_bios/{Tri-Fair,NSGAII-PO-Fair}/seed{42,43,44}/budget1000000

Output:
  analysis/output/curated_figures_bias_advanced_1m_mocapo_style
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
BUDGET = 1_000_000
TIER = "compact"
DEFAULT_RESULTS_ROOT = "results/tri_fair_bias_high_macro_f1/compact_full_1m"
DEFAULT_PREDICTION_ROOT = "analysis/output/bias_advanced_prediction_capture"
DEFAULT_OUT_DIR = "analysis/output/curated_figures_bias_advanced_1m_mocapo_style"
DEFAULT_OPTIMIZERS = ("Tri-Fair", "NSGAII-PO-Fair")
DEFAULT_SEEDS = (42, 43, 44)

N_PREFERENCES = 500
PREFERENCE_SEED = 2026
QWEN_INPUT_WEIGHT = 0.11
QWEN_OUTPUT_WEIGHT = 0.41

DISPLAY_NAME = {
    "Tri-Fair": "Tri-Fair",
    "NSGAII-PO-Fair": "NSGAII-PO-Fair",
}
COLORS = {
    "Tri-Fair": "black",
    "NSGAII-PO-Fair": "tab:blue",
}
MARKERS = {
    "Tri-Fair": "o",
    "NSGAII-PO-Fair": "s",
}


@dataclass(frozen=True)
class Bounds:
    cost_max: float

    def normalize(self, raw_minimize: np.ndarray) -> np.ndarray:
        out = raw_minimize.astype(float).copy()
        out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)  # 1 - quality
        out[:, 1] = np.clip(out[:, 1] / self.cost_max, 0.0, 1.1)
        out[:, 2] = np.clip(out[:, 2], 0.0, 1.0)  # unfairness
        return out


@dataclass
class RunData:
    optimizer: str
    seed: int
    output_dir: Path
    logging_dir: Path
    step_results: pd.DataFrame
    evals: pd.DataFrame
    run_summary: dict[str, object]
    pred_summary: pd.DataFrame | None = None
    per_profession: pd.DataFrame | None = None


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


def num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def save(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def require(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic warning only
        warnings.warn(f"Could not parse {path}: {exc}")
        return {}


def split_csv(raw: str, *, cast=str) -> list:
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


def safe_name(value: str) -> str:
    return value.replace("-", "_").replace("/", "_")


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
    path = path if path.is_absolute() else pointer.parent / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Resolved logging directory does not exist: {path}")
    return path


def stage_eval(evals: pd.DataFrame, budget: int) -> pd.DataFrame:
    out = evals.copy()
    if "budget_checkpoint" in out:
        stage = out[out["budget_checkpoint"] == budget].copy()
        if not stage.empty:
            out = stage
    if "test_fairness_ready" in out:
        ready = out["test_fairness_ready"].fillna(False).astype(bool)
        if ready.any():
            out = out[ready].copy()
    return out.reset_index(drop=True)


def step_tokens(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    if "total_tokens_downstream" in out:
        return out, "total_tokens_downstream"
    if "actual_budget_tokens" in out:
        return out, "actual_budget_tokens"
    if {"input_tokens_downstream", "output_tokens_downstream"}.issubset(out.columns):
        out["_total_tokens"] = num(out["input_tokens_downstream"]) + num(
            out["output_tokens_downstream"]
        )
        return out, "_total_tokens"
    raise ValueError(
        "step_results needs total_tokens_downstream or input/output token columns"
    )


def step_objectives(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            1.0 - num(df["quality"]),
            num(df["cost"]),
            num(df["fairness"]),
        ]
    )


def eval_objectives(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            1.0 - num(df["test_quality"]),
            num(df["test_cost"]),
            num(df["test_fairness"]),
        ]
    )


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(len(values), dtype=bool)
    for i in range(len(values)):
        dominates = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        dominates[i] = False
        keep[i] = not np.any(dominates)
    return keep


def preferences() -> np.ndarray:
    return np.random.default_rng(PREFERENCE_SEED).dirichlet(np.ones(3), size=N_PREFERENCES)


def r2(front: np.ndarray) -> float:
    if len(front) == 0:
        return float("nan")
    weights = preferences()
    utilities = np.max(weights[:, None, :] * front[None, :, :], axis=2)
    return float(np.mean(np.min(utilities, axis=1)))


def hv3(front: np.ndarray, ref: tuple[float, float, float] = (1.05, 1.05, 1.05)) -> float:
    """Exact grid-sweep hypervolume for small normalized 3D minimization fronts."""
    points = np.asarray(front, dtype=float)
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


def infer_global_bounds(runs: Sequence[RunData]) -> Bounds:
    vals: list[float] = []
    for run in runs:
        if "cost" in run.step_results:
            arr = num(run.step_results["cost"])
            if np.isfinite(arr).any():
                vals.append(float(np.nanmax(arr)))
        for col in ["test_cost", "dev_cost", "cost"]:
            if col in run.evals:
                arr = num(run.evals[col])
                if np.isfinite(arr).any():
                    vals.append(float(np.nanmax(arr)))
    finite = [v for v in vals if np.isfinite(v)]
    return Bounds(cost_max=max((max(finite) if finite else 1.0) * 1.05, 1.0))


def prediction_candidates(root: Path, optimizer: str, seed: int) -> list[Path]:
    safe = safe_name(optimizer)
    patterns = [
        f"**/summary_{safe}_seed{seed}.csv",
        "**/bias_in_bios_best_prompt_prediction_summary.csv",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    return sorted(set(paths), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def read_matching_summary(path: Path, optimizer: str, seed: int) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    if "optimizer" in df.columns:
        df = df[df["optimizer"].astype(str) == optimizer]
    if "seed" in df.columns:
        df = df[pd.to_numeric(df["seed"], errors="coerce").astype("Int64") == int(seed)]
    return df.reset_index(drop=True) if not df.empty else None


def load_prediction_summary(args: argparse.Namespace, optimizer: str, seed: int) -> pd.DataFrame | None:
    root = Path(args.prediction_root)
    if not root.exists():
        return None
    for path in prediction_candidates(root, optimizer, seed):
        match = read_matching_summary(path, optimizer, seed)
        if match is not None:
            return match
    return None


def load_per_profession(args: argparse.Namespace, optimizer: str, seed: int) -> pd.DataFrame | None:
    root = Path(args.prediction_root)
    if not root.exists():
        return None
    safe = safe_name(optimizer)
    candidates = sorted(
        root.glob(f"**/per_profession_f1_{safe}_seed{seed}.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty:
            return df
    return None


def load_run(args: argparse.Namespace, optimizer: str, seed: int) -> RunData:
    output_dir = resolve_output_dir(args, optimizer, seed)
    logging_dir = resolve_logging_dir(output_dir)
    step_path = logging_dir / "step_results.parquet"
    eval_path = logging_dir / "eval.parquet"
    if not step_path.is_file():
        raise FileNotFoundError(step_path)
    if not eval_path.is_file():
        raise FileNotFoundError(eval_path)
    return RunData(
        optimizer=optimizer,
        seed=seed,
        output_dir=output_dir,
        logging_dir=logging_dir,
        step_results=pd.read_parquet(step_path),
        evals=pd.read_parquet(eval_path),
        run_summary=read_json(logging_dir / "run_summary.json"),
        pred_summary=load_prediction_summary(args, optimizer, seed),
        per_profession=load_per_profession(args, optimizer, seed),
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
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing required runs:\n  - {joined}")
    if not runs:
        raise RuntimeError("No runs could be loaded")
    return runs


def build_step_metrics(run: RunData, bounds: Bounds) -> pd.DataFrame:
    require(run.step_results, ["step", "quality", "cost", "fairness"], "step_results.parquet")
    data, tok = step_tokens(run.step_results)
    if "fairness_ready" in data:
        ready = data["fairness_ready"].fillna(False).astype(bool)
        if ready.any():
            data = data[ready].copy()

    rows: list[dict[str, object]] = []
    for step, group in data.groupby("step", sort=True):
        group = group.copy()
        for col in ["quality", "cost", "fairness", tok]:
            group[col] = pd.to_numeric(group[col], errors="coerce")
        group = group.dropna(subset=["quality", "cost", "fairness", tok])
        if group.empty:
            continue

        raw = step_objectives(group)
        raw = raw[np.all(np.isfinite(raw), axis=1)]
        if len(raw) == 0:
            continue
        norm = bounds.normalize(raw)
        front = norm[pareto_mask_minimize(norm)]
        best = group.sort_values(
            ["quality", "fairness", "cost"], ascending=[False, True, True]
        ).iloc[0]

        rows.append(
            {
                "run_key": f"{run.optimizer}/seed{run.seed}",
                "model": MODEL,
                "dataset": DATASET,
                "optimizer": run.optimizer,
                "method": DISPLAY_NAME.get(run.optimizer, run.optimizer),
                "seed": int(run.seed),
                "step": int(step),
                "actual_budget_tokens": int(group[tok].max()),
                "dev_noisy_r2_proxy": r2(front),
                "hv_dev_3d_proxy": hv3(front),
                "dev_proxy_front_size": int(len(front)),
                "normalization_cost_max": float(bounds.cost_max),
                "step_best_quality": float(best["quality"]),
                "step_best_cost": float(best["cost"]),
                "step_best_fairness": float(best["fairness"]),
                "n_candidates": int(len(group)),
            }
        )

    if not rows:
        raise RuntimeError(f"No valid stepwise rows for {run.optimizer} seed{run.seed}")
    return pd.DataFrame(rows).sort_values("actual_budget_tokens").reset_index(drop=True)


def best_eval_for_run(run: RunData, budget: int) -> pd.Series:
    stage = stage_eval(run.evals, budget)
    require(stage, ["test_quality", "test_cost", "test_fairness"], "eval.parquet")
    return stage.sort_values(
        ["test_quality", "test_fairness", "test_cost"], ascending=[False, True, True]
    ).iloc[0]


def test_front_metrics(evals: pd.DataFrame, budget: int, bounds: Bounds) -> dict[str, float]:
    stage = stage_eval(evals, budget)
    raw = eval_objectives(stage)
    raw = raw[np.all(np.isfinite(raw), axis=1)]
    if len(raw) == 0:
        return {
            "test_noisy_r2_proxy": float("nan"),
            "hv_test_3d_proxy": float("nan"),
            "test_front_size": 0,
        }
    norm = bounds.normalize(raw)
    front = norm[pareto_mask_minimize(norm)]
    return {
        "test_noisy_r2_proxy": r2(front),
        "hv_test_3d_proxy": hv3(front),
        "test_front_size": int(len(front)),
    }


def output_cost_share(df: pd.DataFrame) -> np.ndarray:
    if {"test_input_tokens", "test_output_tokens"}.issubset(df.columns):
        inp = num(df["test_input_tokens"])
        out = num(df["test_output_tokens"])
    elif {"input_tokens", "output_tokens"}.issubset(df.columns):
        inp = num(df["input_tokens"])
        out = num(df["output_tokens"])
    else:
        return np.zeros(len(df), dtype=float)
    weighted_in = QWEN_INPUT_WEIGHT * inp
    weighted_out = QWEN_OUTPUT_WEIGHT * out
    denom = weighted_in + weighted_out
    return np.divide(weighted_out, denom, out=np.zeros_like(weighted_out), where=denom > 0)


def fewshot_count(value: object) -> int:
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


def all_final_eval_rows(runs: Sequence[RunData], budget: int) -> pd.DataFrame:
    frames = []
    for run in runs:
        frame = stage_eval(run.evals, budget).copy()
        frame["optimizer"] = run.optimizer
        frame["method"] = DISPLAY_NAME.get(run.optimizer, run.optimizer)
        frame["seed"] = run.seed
        if "few_shots_json" in frame:
            frame["fewshot_count"] = frame["few_shots_json"].apply(fewshot_count)
        else:
            frame["fewshot_count"] = 0
        frame["output_cost_share"] = output_cost_share(frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def build_summary(runs: Sequence[RunData], step_metrics: pd.DataFrame, bounds: Bounds, budget: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in runs:
        best = best_eval_for_run(run, budget)
        sm = step_metrics[(step_metrics["optimizer"] == run.optimizer) & (step_metrics["seed"] == run.seed)]
        final_dev = sm.sort_values("actual_budget_tokens").iloc[-1]
        test_metrics = test_front_metrics(run.evals, budget, bounds)
        actual_tokens = int(final_dev["actual_budget_tokens"])
        if "actual_tokens" in run.run_summary:
            actual_tokens = int(float(run.run_summary["actual_tokens"]))

        row: dict[str, object] = {
            "Method": DISPLAY_NAME.get(run.optimizer, run.optimizer),
            "Optimizer": run.optimizer,
            "Seed": int(run.seed),
            "Requested budget": int(budget),
            "Actual tokens": actual_tokens,
            "Tier": TIER,
            "Chosen step": best.get("chosen_step", np.nan),
            "Eval rows": len(stage_eval(run.evals, budget)),
            "Development nR2 proxy ↓": float(final_dev["dev_noisy_r2_proxy"]),
            "Test nR2 proxy ↓": test_metrics["test_noisy_r2_proxy"],
            "Development HV proxy ↑": float(final_dev["hv_dev_3d_proxy"]),
            "Test HV proxy ↑": test_metrics["hv_test_3d_proxy"],
            "nR2 Gap Proxy ↓": test_metrics["test_noisy_r2_proxy"] - float(final_dev["dev_noisy_r2_proxy"]),
            "Best Test Macro-F1 ↑": float(best["test_quality"]),
            "Best Test Cost ↓": float(best["test_cost"]),
            "Best Test Unfairness ↓": float(best["test_fairness"]),
            "Test Fairness Ready": best.get("test_fairness_ready", np.nan),
            "Logging dir": str(run.logging_dir),
        }
        if run.pred_summary is not None and len(run.pred_summary):
            pred = run.pred_summary.iloc[0]
            for src, dst in [
                ("recomputed_macro_f1_normalized", "Recomputed Macro-F1"),
                ("accuracy_normalized", "Accuracy"),
                ("invalid_rate_after_normalization", "Invalid Rate"),
                ("n_examples", "Prediction Examples"),
            ]:
                if src in pred:
                    row[dst] = pred[src]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Optimizer", "Seed"]).reset_index(drop=True)


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "Actual tokens",
        "Best Test Macro-F1 ↑",
        "Best Test Cost ↓",
        "Best Test Unfairness ↓",
        "Development nR2 proxy ↓",
        "Test nR2 proxy ↓",
        "Development HV proxy ↑",
        "Test HV proxy ↑",
        "nR2 Gap Proxy ↓",
    ]
    optional = ["Recomputed Macro-F1", "Accuracy", "Invalid Rate"]
    metric_cols.extend([col for col in optional if col in summary.columns])
    rows = []
    for method, group in summary.groupby("Method", sort=False):
        row: dict[str, object] = {"Method": method, "n_seeds": int(group["Seed"].nunique())}
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col} mean"] = float(values.mean())
            row[f"{col} std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def step_interpolate(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    out = np.full(len(grid), np.nan)
    for i, gx in enumerate(grid):
        idx = np.searchsorted(x, gx, side="right") - 1
        if idx >= 0:
            out[i] = y[idx]
    return out


def plot_step_metric(
    step_metrics: pd.DataFrame,
    outdir: Path,
    col: str,
    ylabel: str,
    title: str,
    stem: str,
    max_budget: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    grid = np.linspace(0, max_budget, 160)

    for optimizer, group in step_metrics.groupby("optimizer", sort=False):
        color = COLORS.get(optimizer, None)
        marker = MARKERS.get(optimizer, "o")
        curves = []
        for seed, seed_group in group.groupby("seed", sort=True):
            seed_group = seed_group.sort_values("actual_budget_tokens")
            x = num(seed_group["actual_budget_tokens"])
            y = num(seed_group[col])
            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.any():
                continue
            x = x[mask]
            y = y[mask]
            ax.step(
                x / 1_000_000,
                y,
                where="post",
                color=color,
                alpha=0.22,
                linewidth=1.1,
            )
            curves.append(step_interpolate(x, y, grid))

        if curves:
            arr = np.vstack(curves)
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            valid = np.isfinite(mean)
            ax.step(
                grid[valid] / 1_000_000,
                mean[valid],
                where="post",
                color=color,
                marker=marker,
                markevery=max(1, int(valid.sum() / 8)),
                linewidth=2.2,
                markersize=4,
                label=DISPLAY_NAME.get(optimizer, optimizer),
            )
            ax.fill_between(
                grid[valid] / 1_000_000,
                mean[valid] - std[valid],
                mean[valid] + std[valid],
                step="post",
                color=color,
                alpha=0.12,
            )

    ax.set_title(title)
    ax.set_xlabel("Token Budget [×10⁶]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    save(fig, outdir, stem)


def attained_y(front: pd.DataFrame, grid: np.ndarray, x_col: str) -> np.ndarray:
    x = num(front[x_col])
    y = num(front["test_quality"])
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    out = np.full(len(grid), np.nan)
    for i, gx in enumerate(grid):
        eligible = y[x <= gx]
        if len(eligible):
            out[i] = np.max(eligible)
    return out


def plot_attainment(final: pd.DataFrame, outdir: Path, x_col: str, xlabel: str, stem: str) -> None:
    require(final, [x_col, "test_quality", "test_cost", "test_fairness", "optimizer"], "final eval rows")
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)

    for optimizer, group in final.groupby("optimizer", sort=False):
        color = COLORS.get(optimizer, None)
        marker = MARKERS.get(optimizer, "o")
        x = num(group[x_col])
        y = num(group["test_quality"])
        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            alpha=0.35,
            s=36,
            label=f"{DISPLAY_NAME.get(optimizer, optimizer)} candidates",
        )

        obj = eval_objectives(group)
        mask = np.all(np.isfinite(obj), axis=1)
        front = group.loc[mask].iloc[pareto_mask_minimize(obj[mask])].copy()
        if front.empty:
            continue
        xmin = float(np.nanmin(num(front[x_col])))
        xmax = float(np.nanmax(num(front[x_col])))
        pad = 0.03 * max(xmax - xmin, 1e-6)
        grid = np.linspace(xmin - pad, xmax + pad, 400)
        curve = attained_y(front, grid, x_col)
        valid = np.isfinite(curve)
        ax.step(
            grid[valid],
            curve[valid],
            where="post",
            color=color,
            linewidth=2.2,
            label=f"{DISPLAY_NAME.get(optimizer, optimizer)} Pareto attainment",
        )

    ax.set_title("Bias in Bios — Qwen-3-30B at 1M")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test Macro-F1")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    save(fig, outdir, stem)


def plot_seed_summary(summary: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    methods = list(summary["Method"].drop_duplicates())
    seeds = sorted(summary["Seed"].unique())
    width = 0.34 if len(methods) <= 2 else 0.8 / len(methods)
    x = np.arange(len(seeds))

    for j, method in enumerate(methods):
        group = summary[summary["Method"] == method].set_index("Seed")
        vals = [float(group.loc[seed, "Best Test Macro-F1 ↑"]) if seed in group.index else np.nan for seed in seeds]
        offset = (j - (len(methods) - 1) / 2) * width
        optimizer = str(summary[summary["Method"] == method]["Optimizer"].iloc[0])
        ax.bar(x + offset, vals, width=width, color=COLORS.get(optimizer, None), alpha=0.85, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels([str(seed) for seed in seeds])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Best Test Macro-F1")
    ax.set_title("Bias in Bios — 1M best test Macro-F1 by seed")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save(fig, outdir, "bias_in_bios_qwen3_1m_seed_macro_f1_summary")


def plot_cost_fairness_summary(summary: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    for optimizer, group in summary.groupby("Optimizer", sort=False):
        ax.scatter(
            group["Best Test Cost ↓"],
            group["Best Test Unfairness ↓"],
            s=80,
            color=COLORS.get(optimizer, None),
            marker=MARKERS.get(optimizer, "o"),
            label=DISPLAY_NAME.get(optimizer, optimizer),
            alpha=0.9,
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"s{int(row['Seed'])}",
                (float(row["Best Test Cost ↓"]), float(row["Best Test Unfairness ↓"])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_title("Bias in Bios — 1M cost–unfairness selected prompts")
    ax.set_xlabel("Best Test Cost ↓")
    ax.set_ylabel("Best Test Unfairness ↓")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save(fig, outdir, "bias_in_bios_qwen3_1m_cost_unfairness_selected")


def plot_per_profession(per_prof: pd.DataFrame, outdir: Path) -> None:
    if per_prof.empty or not {"profession", "f1", "optimizer"}.issubset(per_prof.columns):
        return
    data = per_prof.copy()
    data["f1"] = pd.to_numeric(data["f1"], errors="coerce")
    data = data.dropna(subset=["f1"])
    if data.empty:
        return
    worst = data.groupby("profession")["f1"].mean().sort_values().head(15).index.tolist()
    plot_df = data[data["profession"].isin(worst)].copy()
    order = {name: i for i, name in enumerate(worst)}

    fig, ax = plt.subplots(figsize=(8.0, 5.6), constrained_layout=True)
    methods = list(plot_df["optimizer"].drop_duplicates())
    width = 0.35 if len(methods) <= 2 else 0.8 / len(methods)
    x = np.arange(len(worst))
    for j, optimizer in enumerate(methods):
        vals = (
            plot_df[plot_df["optimizer"] == optimizer]
            .groupby("profession")["f1"]
            .mean()
            .reindex(worst)
            .to_numpy()
        )
        offset = (j - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, color=COLORS.get(optimizer, None), alpha=0.85, label=DISPLAY_NAME.get(optimizer, optimizer))

    ax.set_xticks(x)
    ax.set_xticklabels(worst, rotation=45, ha="right")
    ax.set_ylabel("Mean F1")
    ax.set_title("Bias in Bios — hardest professions by mean F1")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save(fig, outdir, "bias_in_bios_qwen3_1m_worst_profession_f1")


def write_readme(outdir: Path, args: argparse.Namespace, runs: Sequence[RunData]) -> None:
    loaded = "\n".join(
        f"- {run.optimizer} seed{run.seed}: `{run.logging_dir}`" for run in runs
    )
    text = f"""# Bias in Bios / Qwen-3-30B advanced 1M curated figures

Generated by `analysis.make_bias_170287_figures` after upgrading the old single-run
170287 diagnostic script to the full prompt-only advanced 1M setup.

Results root: `{args.results_root}`
Prediction root searched: `{args.prediction_root}`
Budget: `{args.budget}`
Optimizers: `{args.optimizers}`
Seeds: `{args.seeds}`

Loaded runs:
{loaded}

Main files:
- `bias_in_bios_qwen3_1m_stepwise_dev_metrics.csv`
- `bias_in_bios_qwen3_1m_final_eval_rows.csv`
- `bias_in_bios_qwen3_1m_summary_table.csv`
- `bias_in_bios_qwen3_1m_method_summary.csv`
- `bias_in_bios_qwen3_1m_nR2_trajectory_stepwise.*`
- `bias_in_bios_qwen3_1m_hv_trajectory_stepwise.*`
- `bias_in_bios_qwen3_1m_attainment_macro_f1_cost.*`
- `bias_in_bios_qwen3_1m_attainment_macro_f1_unfairness.*`

Note: this script intentionally does not add Initial Instructions yet.  Add Initial
after the six optimized runs are complete and then regenerate the final Bias figure.
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--tier", default=TIER)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--prediction-root", default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--optimizers", default=",".join(DEFAULT_OPTIMIZERS))
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args)
    bounds = infer_global_bounds(runs)
    step_metrics = pd.concat(
        [build_step_metrics(run, bounds) for run in runs],
        ignore_index=True,
        sort=False,
    )
    final_rows = all_final_eval_rows(runs, args.budget)
    summary = build_summary(runs, step_metrics, bounds, args.budget)
    method_summary = aggregate_summary(summary)

    per_prof_frames = []
    for run in runs:
        if run.per_profession is not None and len(run.per_profession):
            frame = run.per_profession.copy()
            frame["optimizer"] = run.optimizer
            frame["method"] = DISPLAY_NAME.get(run.optimizer, run.optimizer)
            frame["seed"] = run.seed
            per_prof_frames.append(frame)
    per_prof_all = pd.concat(per_prof_frames, ignore_index=True, sort=False) if per_prof_frames else pd.DataFrame()

    step_metrics.to_csv(outdir / "bias_in_bios_qwen3_1m_stepwise_dev_metrics.csv", index=False)
    final_rows.to_csv(outdir / "bias_in_bios_qwen3_1m_final_eval_rows.csv", index=False)
    summary.to_csv(outdir / "bias_in_bios_qwen3_1m_summary_table.csv", index=False)
    method_summary.to_csv(outdir / "bias_in_bios_qwen3_1m_method_summary.csv", index=False)
    if not per_prof_all.empty:
        per_prof_all.to_csv(outdir / "bias_in_bios_qwen3_1m_per_profession_f1_source.csv", index=False)

    try:
        (outdir / "bias_in_bios_qwen3_1m_summary_table.md").write_text(
            summary.to_markdown(index=False, floatfmt=".6f") + "\n",
            encoding="utf-8",
        )
        (outdir / "bias_in_bios_qwen3_1m_method_summary.md").write_text(
            method_summary.to_markdown(index=False, floatfmt=".6f") + "\n",
            encoding="utf-8",
        )
    except Exception:
        (outdir / "bias_in_bios_qwen3_1m_summary_table.md").write_text(
            summary.to_csv(index=False), encoding="utf-8"
        )
        (outdir / "bias_in_bios_qwen3_1m_method_summary.md").write_text(
            method_summary.to_csv(index=False), encoding="utf-8"
        )

    max_budget = int(max(args.budget, step_metrics["actual_budget_tokens"].max()))
    plot_step_metric(
        step_metrics,
        outdir,
        "dev_noisy_r2_proxy",
        "Development nR2 Proxy ↓",
        "Bias in Bios — Qwen-3-30B: development nR2 proxy",
        "bias_in_bios_qwen3_1m_nR2_trajectory_stepwise",
        max_budget,
    )
    plot_step_metric(
        step_metrics,
        outdir,
        "hv_dev_3d_proxy",
        "Development HV Proxy ↑",
        "Bias in Bios — Qwen-3-30B: development hypervolume proxy",
        "bias_in_bios_qwen3_1m_hv_trajectory_stepwise",
        max_budget,
    )
    plot_attainment(
        final_rows,
        outdir,
        "test_cost",
        "Avg. Cost [$] per 1M Calls",
        "bias_in_bios_qwen3_1m_attainment_macro_f1_cost",
    )
    plot_attainment(
        final_rows,
        outdir,
        "test_fairness",
        "Test Unfairness",
        "bias_in_bios_qwen3_1m_attainment_macro_f1_unfairness",
    )
    plot_seed_summary(summary, outdir)
    plot_cost_fairness_summary(summary, outdir)
    if not per_prof_all.empty:
        plot_per_profession(per_prof_all, outdir)

    write_readme(outdir, args, runs)
    pd.DataFrame(
        [
            {"file": p.name, "bytes": p.stat().st_size}
            for p in sorted(outdir.iterdir())
            if p.is_file()
        ]
    ).to_csv(outdir / "manifest.csv", index=False)

    print("\nCurated advanced Bias 1M MO-CAPO-style figures written to:")
    print(outdir.resolve())
    print("\nPer-run summary")
    print(summary.to_string(index=False))
    print("\nMean by method")
    print(method_summary.to_string(index=False))


if __name__ == "__main__":
    main()
