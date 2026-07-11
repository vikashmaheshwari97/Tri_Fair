"""MO-CAPO-style curated figures for Bias in Bios advanced run 170287.

Creates the same style of figures as analysis/make_bias_1m_figures.py, but for
one temporary advanced run:
  results/tri_fair_bias_high_macro_f1/compact/qwen-3-30b/bias_in_bios/Tri-Fair/seed42/budget2000000

Output:
  analysis/output/curated_figures_bias_170287_mocapo_style
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL = "qwen-3-30b"
DATASET = "bias_in_bios"
OPTIMIZER = "Tri-Fair"
SEED = 42
JOB_ID = "170287"
BUDGET = 2_000_000
TIER = "compact"
N_PREFERENCES = 500
PREFERENCE_SEED = 2026

DEFAULT_RESULTS_ROOT = "results/tri_fair_bias_high_macro_f1/compact"
DEFAULT_PRED_DIR = f"analysis/output/bias_advanced_prediction_capture/job_{JOB_ID}_{TIER}"
DEFAULT_OUT_DIR = "analysis/output/curated_figures_bias_170287_mocapo_style"

QWEN_INPUT_WEIGHT = 0.11
QWEN_OUTPUT_WEIGHT = 0.41
DISPLAY_NAME = {OPTIMIZER: "Tri-Fair"}
COLORS = {OPTIMIZER: "black"}
MARKERS = {OPTIMIZER: "o"}


@dataclass(frozen=True)
class Bounds:
    cost_max: float

    def normalize(self, raw_minimize: np.ndarray) -> np.ndarray:
        out = raw_minimize.astype(float).copy()
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
    except Exception as exc:
        warnings.warn(f"Could not parse {path}: {exc}")
        return {}


def resolve_logging_dir(args: argparse.Namespace) -> Path:
    if args.logging_dir:
        path = Path(args.logging_dir).expanduser().resolve()
    else:
        pointer = (
            Path(args.results_root)
            / args.model
            / args.dataset
            / args.optimizer
            / f"seed{args.seed}"
            / f"budget{args.budget}"
            / "logging_dir.txt"
        )
        if not pointer.is_file():
            raise FileNotFoundError(f"Missing logging_dir.txt: {pointer}")
        raw = pointer.read_text(encoding="utf-8").strip()
        path = Path(raw).expanduser()
        path = (pointer.parent / path).resolve() if not path.is_absolute() else path.resolve()

    if not path.is_dir():
        raise FileNotFoundError(f"Resolved logging directory does not exist: {path}")
    return path


def load_run(args: argparse.Namespace) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    logdir = resolve_logging_dir(args)
    step_path = logdir / "step_results.parquet"
    eval_path = logdir / "eval.parquet"

    if not step_path.is_file() or not eval_path.is_file():
        raise FileNotFoundError(f"Missing {step_path} or {eval_path}")

    return (
        logdir,
        pd.read_parquet(step_path),
        pd.read_parquet(eval_path),
        read_json(logdir / "run_summary.json"),
    )


def load_prediction_outputs(
    pred_dir: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    summary = pred_dir / "bias_in_bios_best_prompt_prediction_summary.csv"
    per_prof = pred_dir / "bias_in_bios_best_prompt_per_profession_f1.csv"

    if not summary.is_file():
        found = sorted(pred_dir.glob("summary_*_seed*.csv"))
        summary = found[0] if found else summary

    if not per_prof.is_file():
        found = sorted(pred_dir.glob("per_profession_f1_*_seed*.csv"))
        per_prof = found[0] if found else per_prof

    confusion_files = sorted(pred_dir.glob("confusion_*_seed*.csv"))
    confusion = (
        confusion_files[0]
        if confusion_files
        else pred_dir / "confusion_Tri_Fair_seed42.csv"
    )

    return (
        pd.read_csv(summary) if summary.is_file() else None,
        pd.read_csv(per_prof) if per_prof.is_file() else None,
        pd.read_csv(confusion, index_col=0) if confusion.is_file() else None,
    )


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.zeros(0, dtype=bool)

    keep = np.ones(len(values), dtype=bool)
    for i in range(len(values)):
        dominates = np.all(values <= values[i], axis=1) & np.any(
            values < values[i], axis=1
        )
        dominates[i] = False
        keep[i] = not np.any(dominates)
    return keep


def preferences() -> np.ndarray:
    return np.random.default_rng(PREFERENCE_SEED).dirichlet(
        np.ones(3), size=N_PREFERENCES
    )


def r2(front: np.ndarray) -> float:
    if len(front) == 0:
        return float("nan")

    weights = preferences()
    utilities = np.max(weights[:, None, :] * front[None, :, :], axis=2)
    return float(np.mean(np.min(utilities, axis=1)))


def hv3(
    front: np.ndarray,
    ref: tuple[float, float, float] = (1.05, 1.05, 1.05),
) -> float:
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
    coords = [
        np.sort(np.unique(np.r_[points[:, d], ref_arr[d]])) for d in range(3)
    ]

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
        out["_total_tokens"] = num(out["input_tokens_downstream"]) + num(
            out["output_tokens_downstream"]
        )
        return out, "_total_tokens"

    raise ValueError(
        "step_results needs total_tokens_downstream or input/output token columns"
    )


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


def infer_bounds(step_results: pd.DataFrame, evals: pd.DataFrame) -> Bounds:
    vals: list[float] = []

    if "cost" in step_results:
        vals.append(float(np.nanmax(num(step_results["cost"]))))

    for col in ["test_cost", "dev_cost", "cost"]:
        if col in evals:
            vals.append(float(np.nanmax(num(evals[col]))))

    finite = [v for v in vals if np.isfinite(v)]
    return Bounds(cost_max=max((max(finite) if finite else 1.0) * 1.05, 1.0))


def build_step_metrics(step_results: pd.DataFrame, evals: pd.DataFrame) -> pd.DataFrame:
    require(step_results, ["step", "quality", "cost", "fairness"], "step_results.parquet")

    data, tok = step_tokens(step_results)

    if "fairness_ready" in data:
        ready = data["fairness_ready"].fillna(False).astype(bool)
        if ready.any():
            data = data[ready].copy()

    bounds = infer_bounds(data, evals)
    rows: list[dict[str, object]] = []

    for step, group in data.groupby("step", sort=True):
        group = group.copy()
        for col in ["quality", "cost", "fairness", tok]:
            group[col] = pd.to_numeric(group[col], errors="coerce")

        group = group.dropna(subset=["quality", "cost", "fairness", tok])
        if group.empty:
            continue

        norm = bounds.normalize(step_objectives(group))
        norm = norm[np.all(np.isfinite(norm), axis=1)]
        if len(norm) == 0:
            continue

        front = norm[pareto_mask_minimize(norm)]
        best = group.sort_values(
            ["quality", "fairness", "cost"],
            ascending=[False, True, True],
        ).iloc[0]

        rows.append(
            {
                "run_key": f"{MODEL}/{DATASET}/{OPTIMIZER}/seed{SEED}/budget{BUDGET}",
                "model": MODEL,
                "dataset": DATASET,
                "optimizer": OPTIMIZER,
                "seed": SEED,
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
        raise RuntimeError("No valid stepwise rows could be computed")

    return (
        pd.DataFrame(rows)
        .sort_values("actual_budget_tokens")
        .reset_index(drop=True)
    )


def best_eval(evals: pd.DataFrame, budget: int) -> pd.Series:
    stage = stage_eval(evals, budget)
    require(stage, ["test_quality", "test_cost", "test_fairness"], "eval.parquet")

    return stage.sort_values(
        ["test_quality", "test_fairness", "test_cost"],
        ascending=[False, True, True],
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


def step_curve_stats(
    df: pd.DataFrame,
    col: str,
    max_budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = df[df["actual_budget_tokens"] <= max_budget].sort_values(
        "actual_budget_tokens"
    )

    x = num(data["actual_budget_tokens"]).astype(int)
    y = num(data[col])
    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    return x, y, y, y


def plot_nR2(step_metrics: pd.DataFrame, outdir: Path, max_budget: int) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)

    x, y, lo, hi = step_curve_stats(step_metrics, "dev_noisy_r2_proxy", max_budget)

    ax.step(
        x / 1_000_000,
        y,
        where="post",
        color=COLORS[OPTIMIZER],
        marker=MARKERS[OPTIMIZER],
        linewidth=2,
        markersize=4,
        label=DISPLAY_NAME[OPTIMIZER],
    )
    ax.fill_between(
        x / 1_000_000,
        lo,
        hi,
        step="post",
        color=COLORS[OPTIMIZER],
        alpha=0.16,
    )

    ax.set_title("Bias in Bios — Qwen-3-30B")
    ax.set_xlabel("Token Budget [×10⁶]")
    ax.set_ylabel("Development nR2 Proxy ↓")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")

    ax.text(
        0.02,
        0.02,
        "Single-seed advanced run.\nStepwise proxy from development objectives.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        alpha=0.75,
    )

    save(fig, outdir, "bias_in_bios_qwen3_2m_nR2_trajectory_stepwise")


def plot_hv_gap(step_metrics: pd.DataFrame, summary: pd.DataFrame, outdir: Path, max_budget: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    x, y, lo, hi = step_curve_stats(step_metrics, "hv_dev_3d_proxy", max_budget)

    axes[0].step(
        x / 1_000_000,
        y,
        where="post",
        color=COLORS[OPTIMIZER],
        marker=MARKERS[OPTIMIZER],
        linewidth=2,
        markersize=4,
        label=DISPLAY_NAME[OPTIMIZER],
    )
    axes[0].fill_between(
        x / 1_000_000,
        lo,
        hi,
        step="post",
        color=COLORS[OPTIMIZER],
        alpha=0.16,
    )
    axes[0].set_title("Development Hypervolume Proxy ↑")
    axes[0].set_xlabel("Token Budget [×10⁶]")
    axes[0].set_ylabel("Development HV Proxy")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="best")

    gap = float(summary.iloc[0]["nR2 Gap Proxy ↓"])

    axes[1].step(
        [BUDGET / 1_000_000],
        [gap],
        where="post",
        color=COLORS[OPTIMIZER],
        marker=MARKERS[OPTIMIZER],
        linewidth=2,
        markersize=5,
        label=DISPLAY_NAME[OPTIMIZER],
    )
    axes[1].scatter(
        [BUDGET / 1_000_000],
        [gap],
        color=COLORS[OPTIMIZER],
        s=40,
        zorder=3,
    )
    axes[1].annotate(
        f"{gap:.4f}",
        xy=(BUDGET / 1_000_000, gap),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=8,
    )
    axes[1].set_title("Test-vs-Development nR2 Gap Proxy ↓")
    axes[1].set_xlabel("Evaluation-token checkpoint [×10⁶]")
    axes[1].set_ylabel("nR2 Gap Proxy")
    axes[1].set_xticks([BUDGET / 1_000_000])
    axes[1].set_xticklabels(["2M"])
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, loc="best")

    fig.suptitle("Bias in Bios — Qwen-3-30B: Trajectory diagnostics", y=1.03)

    save(fig, outdir, "bias_in_bios_qwen3_2m_hv_gap_trajectory_stepwise")


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


def plot_attainment(evals: pd.DataFrame, outdir: Path, x_col: str, xlabel: str, filename: str) -> None:
    final = stage_eval(evals, BUDGET)
    require(final, [x_col, "test_quality", "test_cost", "test_fairness"], "eval.parquet")

    xmin = float(np.nanmin(num(final[x_col])))
    xmax = float(np.nanmax(num(final[x_col])))
    pad = 0.03 * max(xmax - xmin, 1e-6)
    grid = np.linspace(xmin - pad, xmax + pad, 400)

    front = final.loc[pareto_mask_minimize(eval_objectives(final))].copy()
    curve = attained_y(front, grid, x_col)
    valid = np.isfinite(curve)

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)

    ax.step(
        grid[valid],
        curve[valid],
        where="post",
        color=COLORS[OPTIMIZER],
        marker=MARKERS[OPTIMIZER],
        markevery=max(1, int(valid.sum() / 8)),
        linewidth=2,
        markersize=4,
        label=DISPLAY_NAME[OPTIMIZER],
    )
    ax.fill_between(
        grid[valid],
        curve[valid],
        curve[valid],
        step="post",
        color=COLORS[OPTIMIZER],
        alpha=0.16,
    )

    ax.set_title("Bias in Bios — Qwen-3-30B at 2M")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test Macro-F1")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    save(fig, outdir, filename)


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

    return np.divide(
        weighted_out,
        denom,
        out=np.zeros_like(weighted_out),
        where=denom > 0,
    )


def final_candidates(evals: pd.DataFrame) -> pd.DataFrame:
    data = stage_eval(evals, BUDGET)

    if "is_incumbent" in data and data["is_incumbent"].notna().any():
        inc = data[data["is_incumbent"].fillna(False).astype(bool)].copy()
        if not inc.empty:
            data = inc

    data["fewshot_count"] = (
        data["few_shots_json"].apply(fewshot_count)
        if "few_shots_json" in data
        else 0
    )
    data["output_cost_share"] = output_cost_share(data)

    return data.reset_index(drop=True)


def plot_fewshot_scatter(evals: pd.DataFrame, outdir: Path, color_col: str, label: str, filename: str) -> None:
    data = final_candidates(evals)
    values = num(data[color_col])

    vmin = float(np.nanmin(values)) if np.isfinite(values).any() else 0.0
    vmax = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)

    sc = ax.scatter(
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

    ax.set_title("Tri-Fair on Bias in Bios — Qwen-3-30B at 2M")
    ax.set_xlabel("Avg. Cost [$] per 1M Calls")
    ax.set_ylabel("Test Macro-F1")
    ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(label)

    save(fig, outdir, filename)


def clean_label(x: object) -> str:
    return str(x).replace("true__", "").replace("pred__", "")


def confusion_labels(cm: pd.DataFrame, per_prof: pd.DataFrame | None, n: int = 12) -> list[str]:
    labels: list[str] = []

    if per_prof is not None and {"profession", "f1"}.issubset(per_prof.columns):
        labels.extend(
            per_prof.assign(f1=pd.to_numeric(per_prof["f1"], errors="coerce"))
            .dropna(subset=["f1"])
            .sort_values("f1")
            .head(8)["profession"]
            .astype(str)
            .tolist()
        )

    off = cm.copy()
    for lab in off.index.intersection(off.columns):
        off.loc[lab, lab] = 0

    for (true, pred), count in off.stack().sort_values(ascending=False).items():
        if count <= 0:
            break
        labels.extend([str(true), str(pred)])
        if len(dict.fromkeys(labels)) >= n:
            break

    return [x for x in dict.fromkeys(labels) if x in cm.index and x in cm.columns][:n]


def plot_confusion(confusion: pd.DataFrame, per_prof: pd.DataFrame | None, outdir: Path) -> None:
    cm = confusion.copy()
    cm.index = [clean_label(x) for x in cm.index]
    cm.columns = [clean_label(x) for x in cm.columns]
    cm = cm.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    labels = confusion_labels(cm, per_prof)
    if not labels:
        return

    sub = cm.loc[labels, labels]

    fig, ax = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    im = ax.imshow(sub.to_numpy(dtype=float), cmap="Blues")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Bias in Bios — compact confusion matrix for hardest labels")

    vals = sub.to_numpy(dtype=int)
    threshold = max(vals.max() * 0.55, 1)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if vals[i, j] > 0:
                ax.text(
                    j,
                    i,
                    str(vals[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if vals[i, j] >= threshold else "black",
                )

    fig.colorbar(im, ax=ax).set_label("Count")

    save(fig, outdir, "bias_in_bios_qwen3_2m_confusion_hard_labels")


def build_summary(
    args: argparse.Namespace,
    logdir: Path,
    step_metrics: pd.DataFrame,
    evals: pd.DataFrame,
    pred_summary: pd.DataFrame | None,
    run_summary: dict[str, object],
) -> pd.DataFrame:
    best = best_eval(evals, args.budget)
    final_dev = step_metrics.iloc[-1]
    bounds = Bounds(cost_max=float(final_dev["normalization_cost_max"]))
    test_metrics = test_front_metrics(evals, args.budget, bounds)

    actual_tokens = int(step_metrics["actual_budget_tokens"].max())
    if "actual_tokens" in run_summary:
        actual_tokens = int(float(run_summary["actual_tokens"]))

    row: dict[str, object] = {
        "Method": DISPLAY_NAME[args.optimizer],
        "Job ID": args.job_id,
        "Actual tokens": actual_tokens,
        "Requested budget": args.budget,
        "Seed": args.seed,
        "Tier": args.tier,
        "Chosen step": best.get("chosen_step", np.nan),
        "Eval rows": len(stage_eval(evals, args.budget)),
        "Development nR2 proxy ↓": float(final_dev["dev_noisy_r2_proxy"]),
        "Test nR2 proxy ↓": test_metrics["test_noisy_r2_proxy"],
        "Development HV proxy ↑": float(final_dev["hv_dev_3d_proxy"]),
        "Test HV proxy ↑": test_metrics["hv_test_3d_proxy"],
        "nR2 Gap Proxy ↓": test_metrics["test_noisy_r2_proxy"]
        - float(final_dev["dev_noisy_r2_proxy"]),
        "Best Test Macro-F1 ↑": float(best["test_quality"]),
        "Best Test Cost ↓": float(best["test_cost"]),
        "Best Test Unfairness ↓": float(best["test_fairness"]),
        "Test Max Abs TPR Gap ↓": best.get("test_max_abs_tpr_gap", np.nan),
        "Test Fairness Ready": best.get("test_fairness_ready", np.nan),
        "Logging dir": str(logdir),
    }

    if pred_summary is not None and len(pred_summary):
        pred = pred_summary.iloc[0]
        for src, dst in [
            ("recomputed_macro_f1_normalized", "Recomputed Macro-F1"),
            ("accuracy_normalized", "Accuracy"),
            ("invalid_rate_after_normalization", "Invalid Rate"),
            ("n_examples", "Prediction Examples"),
        ]:
            if src in pred:
                row[dst] = pred[src]

    return pd.DataFrame([row])


def write_readme(outdir: Path, args: argparse.Namespace, logdir: Path) -> None:
    text = f"""# Bias in Bios / Qwen-3-30B 2M curated MO-CAPO-style figures

Generated by `analysis.make_bias_170287_figures`.

This folder follows the same visual style as `analysis.make_bias_1m_figures`, but
it is for one temporary advanced run only, so nR2, HV, and Gap are proxy
diagnostics rather than final multi-seed paper metrics.

Run: job `{args.job_id}`, `{args.model}`, `{args.dataset}`, `{args.optimizer}`, seed `{args.seed}`, tier `{args.tier}`, budget `{args.budget}`.

Logging directory: `{logdir}`.

Prediction-capture directory: `{args.prediction_dir}`.

Figures:
- `bias_in_bios_qwen3_2m_nR2_trajectory_stepwise.*`
- `bias_in_bios_qwen3_2m_hv_gap_trajectory_stepwise.*`
- `bias_in_bios_qwen3_2m_attainment_macro_f1_cost.*`
- `bias_in_bios_qwen3_2m_attainment_macro_f1_unfairness.*`
- `bias_in_bios_qwen3_2m_trifair_fewshot_outputshare.*`
- `bias_in_bios_qwen3_2m_trifair_fewshot_unfairness.*`
- `bias_in_bios_qwen3_2m_confusion_hard_labels.*`
"""
    (outdir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default=JOB_ID)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--optimizer", default=OPTIMIZER)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--tier", default=TIER)
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--prediction-dir", default=DEFAULT_PRED_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--logging-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    logdir, step_results, evals, run_summary = load_run(args)
    pred_summary, per_prof, confusion = load_prediction_outputs(Path(args.prediction_dir))

    step_metrics = build_step_metrics(step_results, evals)
    summary = build_summary(
        args=args,
        logdir=logdir,
        step_metrics=step_metrics,
        evals=evals,
        pred_summary=pred_summary,
        run_summary=run_summary,
    )

    step_metrics.to_csv(
        outdir / "bias_in_bios_qwen3_2m_stepwise_dev_metrics.csv",
        index=False,
    )
    stage_eval(evals, args.budget).to_csv(
        outdir / "bias_in_bios_qwen3_2m_eval_rows.csv",
        index=False,
    )
    summary.to_csv(
        outdir / "bias_in_bios_qwen3_2m_summary_table.csv",
        index=False,
    )

    try:
        (outdir / "bias_in_bios_qwen3_2m_summary_table.md").write_text(
            summary.to_markdown(index=False, floatfmt=".6f") + "\n",
            encoding="utf-8",
        )
    except Exception:
        (outdir / "bias_in_bios_qwen3_2m_summary_table.md").write_text(
            summary.to_csv(index=False),
            encoding="utf-8",
        )

    max_budget = int(max(args.budget, step_metrics["actual_budget_tokens"].max()))

    plot_nR2(step_metrics, outdir, max_budget)
    plot_hv_gap(step_metrics, summary, outdir, max_budget)

    plot_attainment(
        evals,
        outdir,
        "test_cost",
        "Avg. Cost [$] per 1M Calls",
        "bias_in_bios_qwen3_2m_attainment_macro_f1_cost",
    )
    plot_attainment(
        evals,
        outdir,
        "test_fairness",
        "Test Unfairness",
        "bias_in_bios_qwen3_2m_attainment_macro_f1_unfairness",
    )

    plot_fewshot_scatter(
        evals,
        outdir,
        "output_cost_share",
        "Output Token Cost Share",
        "bias_in_bios_qwen3_2m_trifair_fewshot_outputshare",
    )
    plot_fewshot_scatter(
        evals,
        outdir,
        "test_fairness",
        "Test Unfairness ↓",
        "bias_in_bios_qwen3_2m_trifair_fewshot_unfairness",
    )

    if per_prof is not None:
        per_prof.to_csv(
            outdir / "bias_in_bios_qwen3_2m_per_profession_f1_source.csv",
            index=False,
        )

    if confusion is not None:
        confusion.to_csv(outdir / "bias_in_bios_qwen3_2m_confusion_source.csv")
        plot_confusion(confusion, per_prof, outdir)

    write_readme(outdir, args, logdir)

    pd.DataFrame(
        [
            {"file": p.name, "bytes": p.stat().st_size}
            for p in sorted(outdir.iterdir())
            if p.is_file()
        ]
    ).to_csv(outdir / "manifest.csv", index=False)

    print("\nCurated MO-CAPO-style Bias 170287 figures written to:")
    print(outdir.resolve())
    print("\n2M summary table")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()