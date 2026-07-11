"""MO-CAPO-style curated figures for Bias in Bios advanced run 170287.

This script is intentionally for the recent temporary advanced Bias run, not the
frozen multi-seed 1M main analysis.

Default input run:
  results/tri_fair_bias_high_macro_f1/compact/
    qwen-3-30b/bias_in_bios/Tri-Fair/seed42/budget2000000/logging_dir.txt

Default prediction-capture directory:
  analysis/output/bias_advanced_prediction_capture/job_170287_compact

Default output directory:
  analysis/output/curated_figures_bias_170287

Figures:
  1. Development incumbent trajectory over token budget.
  2. Test Macro-F1 vs test cost scatter, highlighting the selected prompt.
  3. Test Macro-F1 vs test unfairness scatter, highlighting the selected prompt.
  4. Test cost vs unfairness scatter, colored by Macro-F1.
  5. Prediction-capture per-profession F1 bar plot.
  6. Prediction-capture compact confusion heatmap for the hardest labels.

The figure style follows the earlier curated Bias MO-CAPO-style script, but this
script reads the single run's raw files directly instead of aggregated
analysis/output/run_metrics.csv and all_evaluations.parquet.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
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

DEFAULT_RESULTS_ROOT = "results/tri_fair_bias_high_macro_f1/compact"
DEFAULT_OUT_DIR = "analysis/output/curated_figures_bias_170287"
DEFAULT_PREDICTION_DIR = f"analysis/output/bias_advanced_prediction_capture/job_{JOB_ID}_{TIER}"

QWEN_INPUT_WEIGHT = 0.11
QWEN_OUTPUT_WEIGHT = 0.41

MAIN_COLOR = "black"
ALT_COLOR = "#E69F00"
ACCENT_COLOR = "#0072B2"
BAD_COLOR = "#D55E00"


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


def finite_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.warn(f"Could not parse JSON file {path}: {exc}")
        return {}


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    try:
        text = df.to_markdown(index=False, floatfmt=".6f") + "\n"
    except Exception:
        text = df.to_csv(index=False)
    path.write_text(text, encoding="utf-8")


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def require_columns(df: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def run_output_dir(args: argparse.Namespace) -> Path:
    return (
        Path(args.results_root)
        / args.model
        / args.dataset
        / args.optimizer
        / f"seed{args.seed}"
        / f"budget{args.budget}"
    )


def resolve_logging_dir(args: argparse.Namespace) -> Path:
    if args.logging_dir:
        logging_dir = Path(args.logging_dir).expanduser().resolve()
    else:
        pointer = run_output_dir(args) / "logging_dir.txt"
        if not pointer.is_file():
            raise FileNotFoundError(
                f"Missing logging_dir.txt: {pointer}\n"
                "Pass --logging-dir directly if this run used a different output root."
            )
        logging_dir = Path(pointer.read_text(encoding="utf-8").strip()).expanduser()
        if not logging_dir.is_absolute():
            logging_dir = (pointer.parent / logging_dir).resolve()
        else:
            logging_dir = logging_dir.resolve()

    if not logging_dir.is_dir():
        raise FileNotFoundError(f"Resolved logging directory does not exist: {logging_dir}")
    return logging_dir


def load_run_files(args: argparse.Namespace) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    logging_dir = resolve_logging_dir(args)
    step_path = logging_dir / "step_results.parquet"
    eval_path = logging_dir / "eval.parquet"
    summary_path = logging_dir / "run_summary.json"

    missing = [str(path) for path in [step_path, eval_path] if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required run files:\n" + "\n".join(missing))

    step_results = pd.read_parquet(step_path)
    evaluations = pd.read_parquet(eval_path)
    run_summary = read_json_if_exists(summary_path)

    return logging_dir, step_results, evaluations, run_summary


def load_prediction_files(pred_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Return prediction summary, per-profession F1, confusion matrix."""
    summary_path = pred_dir / "bias_in_bios_best_prompt_prediction_summary.csv"
    per_prof_path = pred_dir / "bias_in_bios_best_prompt_per_profession_f1.csv"

    if not summary_path.is_file():
        candidates = sorted(pred_dir.glob("summary_*_seed*.csv"))
        summary_path = candidates[0] if candidates else summary_path

    if not per_prof_path.is_file():
        candidates = sorted(pred_dir.glob("per_profession_f1_*_seed*.csv"))
        per_prof_path = candidates[0] if candidates else per_prof_path

    confusion_candidates = sorted(pred_dir.glob("confusion_*_seed*.csv"))
    confusion_path = confusion_candidates[0] if confusion_candidates else pred_dir / "confusion_Tri_Fair_seed42.csv"

    summary = pd.read_csv(summary_path) if summary_path.is_file() else None
    per_prof = pd.read_csv(per_prof_path) if per_prof_path.is_file() else None
    confusion = pd.read_csv(confusion_path, index_col=0) if confusion_path.is_file() else None

    if summary is None:
        warnings.warn(f"Prediction summary not found under {pred_dir}")
    if per_prof is None:
        warnings.warn(f"Per-profession F1 file not found under {pred_dir}")
    if confusion is None:
        warnings.warn(f"Confusion matrix file not found under {pred_dir}")

    return summary, per_prof, confusion


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
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


def eval_objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            1.0 - finite_numeric(frame["test_quality"]),
            finite_numeric(frame["test_cost"]),
            finite_numeric(frame["test_fairness"]),
        ]
    )


def evaluation_stage(evaluations: pd.DataFrame, budget: int) -> pd.DataFrame:
    if "budget_checkpoint" in evaluations.columns:
        stage = evaluations[evaluations["budget_checkpoint"] == budget].copy()
        if not stage.empty:
            return stage.reset_index(drop=True)
    return evaluations.copy().reset_index(drop=True)


def choose_best_eval_prompt(evaluations: pd.DataFrame, budget: int) -> pd.Series:
    stage = evaluation_stage(evaluations, budget)
    require_columns(stage, ["test_quality", "test_cost", "test_fairness"], name="eval.parquet")
    return stage.sort_values(
        ["test_quality", "test_fairness", "test_cost"],
        ascending=[False, True, True],
    ).iloc[0]


def step_token_column(step_results: pd.DataFrame) -> str:
    for column in ["total_tokens_downstream", "actual_budget_tokens"]:
        if column in step_results.columns:
            return column
    if {"input_tokens_downstream", "output_tokens_downstream"}.issubset(step_results.columns):
        return "__computed_total_tokens_downstream"
    raise ValueError(
        "step_results.parquet needs total_tokens_downstream or input/output token columns."
    )


def prepare_stepwise_incumbents(step_results: pd.DataFrame) -> pd.DataFrame:
    require_columns(step_results, ["step", "quality", "cost", "fairness"], name="step_results.parquet")

    data = step_results.copy()
    token_col = step_token_column(data)
    if token_col == "__computed_total_tokens_downstream":
        data[token_col] = finite_numeric(data["input_tokens_downstream"]) + finite_numeric(
            data["output_tokens_downstream"]
        )

    if "fairness_ready" in data.columns:
        ready = data["fairness_ready"].fillna(False).astype(bool)
        if ready.any():
            data = data[ready].copy()

    rows: list[dict[str, object]] = []
    best: pd.Series | None = None

    for step, group in data.groupby("step", sort=True):
        group = group.copy()
        group["quality"] = pd.to_numeric(group["quality"], errors="coerce")
        group["cost"] = pd.to_numeric(group["cost"], errors="coerce")
        group["fairness"] = pd.to_numeric(group["fairness"], errors="coerce")
        group[token_col] = pd.to_numeric(group[token_col], errors="coerce")
        group = group.dropna(subset=["quality", "cost", "fairness", token_col])
        if group.empty:
            continue

        step_best = group.sort_values(
            ["quality", "fairness", "cost"],
            ascending=[False, True, True],
        ).iloc[0]

        if best is None:
            best = step_best
        else:
            old_key = (float(best["quality"]), -float(best["fairness"]), -float(best["cost"]))
            new_key = (
                float(step_best["quality"]),
                -float(step_best["fairness"]),
                -float(step_best["cost"]),
            )
            if new_key > old_key:
                best = step_best

        rows.append(
            {
                "step": int(step),
                "actual_budget_tokens": int(group[token_col].max()),
                "step_best_quality": float(step_best["quality"]),
                "step_best_cost": float(step_best["cost"]),
                "step_best_fairness": float(step_best["fairness"]),
                "incumbent_quality": float(best["quality"]),
                "incumbent_cost": float(best["cost"]),
                "incumbent_fairness": float(best["fairness"]),
                "n_candidates": int(len(group)),
            }
        )

    if not rows:
        raise RuntimeError("No valid stepwise rows could be built from step_results.parquet")

    return pd.DataFrame(rows).sort_values("actual_budget_tokens").reset_index(drop=True)


def plot_stepwise_incumbent(stepwise: pd.DataFrame, outdir: Path) -> None:
    x = stepwise["actual_budget_tokens"].to_numpy(dtype=float) / 1_000_000.0

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)

    panels = [
        ("incumbent_quality", "Dev Macro-F1 ↑", MAIN_COLOR),
        ("incumbent_cost", "Dev Cost ↓", ALT_COLOR),
        ("incumbent_fairness", "Dev Unfairness ↓", BAD_COLOR),
    ]
    for ax, (column, ylabel, color) in zip(axes, panels):
        ax.step(x, stepwise[column], where="post", color=color, linewidth=2)
        ax.scatter(x, stepwise[column], color=color, s=24, zorder=3)
        ax.set_xlabel("Token Budget [×10⁶]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Bias in Bios — Qwen-3-30B — Tri-Fair 2M compact trajectory", y=1.04)
    save_figure(fig, outdir, "bias_170287_stepwise_incumbent_trajectory")


def annotate_selected(ax: plt.Axes, best: pd.Series, x_col: str, y_col: str) -> None:
    x = float(best[x_col])
    y = float(best[y_col])
    ax.scatter(
        [x],
        [y],
        marker="*",
        s=220,
        color=BAD_COLOR,
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
    )
    ax.annotate(
        "selected best\nMacro-F1",
        xy=(x, y),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )


def plot_eval_macro_f1_vs_cost(evaluations: pd.DataFrame, best: pd.Series, outdir: Path, budget: int) -> None:
    stage = evaluation_stage(evaluations, budget)
    stage = stage.dropna(subset=["test_quality", "test_cost", "test_fairness"]).copy()
    pareto = pareto_mask_minimize(eval_objective_matrix(stage))

    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    scatter = ax.scatter(
        stage["test_cost"],
        stage["test_quality"],
        c=stage["test_fairness"],
        cmap="viridis_r",
        s=72,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.92,
        label="evaluated prompts",
    )
    if pareto.any():
        front = stage.loc[pareto].sort_values("test_cost")
        ax.plot(
            front["test_cost"],
            front["test_quality"],
            color=MAIN_COLOR,
            linewidth=1.6,
            label="3D non-dominated front",
        )

    annotate_selected(ax, best, "test_cost", "test_quality")
    ax.set_title("Bias in Bios — Qwen-3-30B at 2M")
    ax.set_xlabel("Avg. Cost [$] per 1M Calls")
    ax.set_ylabel("Test Macro-F1")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Test Unfairness ↓")
    save_figure(fig, outdir, "bias_170287_test_macro_f1_vs_cost")


def plot_eval_macro_f1_vs_unfairness(evaluations: pd.DataFrame, best: pd.Series, outdir: Path, budget: int) -> None:
    stage = evaluation_stage(evaluations, budget)
    stage = stage.dropna(subset=["test_quality", "test_cost", "test_fairness"]).copy()
    pareto = pareto_mask_minimize(eval_objective_matrix(stage))

    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    scatter = ax.scatter(
        stage["test_fairness"],
        stage["test_quality"],
        c=stage["test_cost"],
        cmap="viridis",
        s=72,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.92,
        label="evaluated prompts",
    )
    if pareto.any():
        front = stage.loc[pareto].sort_values("test_fairness")
        ax.plot(
            front["test_fairness"],
            front["test_quality"],
            color=MAIN_COLOR,
            linewidth=1.6,
            label="3D non-dominated front",
        )

    annotate_selected(ax, best, "test_fairness", "test_quality")
    ax.set_title("Bias in Bios — Qwen-3-30B at 2M")
    ax.set_xlabel("Test Unfairness ↓")
    ax.set_ylabel("Test Macro-F1")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Avg. Cost [$] per 1M Calls ↓")
    save_figure(fig, outdir, "bias_170287_test_macro_f1_vs_unfairness")


def plot_cost_vs_unfairness(evaluations: pd.DataFrame, best: pd.Series, outdir: Path, budget: int) -> None:
    stage = evaluation_stage(evaluations, budget)
    stage = stage.dropna(subset=["test_quality", "test_cost", "test_fairness"]).copy()

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    scatter = ax.scatter(
        stage["test_cost"],
        stage["test_fairness"],
        c=stage["test_quality"],
        cmap="viridis",
        s=72,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.92,
    )
    annotate_selected(ax, best, "test_cost", "test_fairness")
    ax.set_title("Bias in Bios — Cost/Fairness trade-off at 2M")
    ax.set_xlabel("Avg. Cost [$] per 1M Calls ↓")
    ax.set_ylabel("Test Unfairness ↓")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Test Macro-F1 ↑")
    save_figure(fig, outdir, "bias_170287_test_cost_vs_unfairness")


def plot_per_profession_f1(per_prof: pd.DataFrame, outdir: Path, worst_n: int = 28) -> None:
    require_columns(per_prof, ["profession", "f1", "support"], name="per-profession F1 file")
    data = per_prof.copy()
    data["f1"] = pd.to_numeric(data["f1"], errors="coerce")
    data["support"] = pd.to_numeric(data["support"], errors="coerce")
    data = data.dropna(subset=["f1"]).sort_values("f1", ascending=True).head(worst_n)

    height = max(5.0, 0.28 * len(data) + 1.5)
    fig, ax = plt.subplots(figsize=(7.2, height), constrained_layout=True)
    y = np.arange(len(data))
    ax.barh(y, data["f1"], color=ACCENT_COLOR, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(data["profession"])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("F1")
    ax.set_title("Bias in Bios — Prediction-capture per-profession F1")
    ax.grid(True, axis="x", alpha=0.25)

    for yi, (_, row) in enumerate(data.iterrows()):
        support = int(row["support"]) if np.isfinite(row["support"]) else 0
        ax.text(
            float(row["f1"]) + 0.01,
            yi,
            f"{float(row['f1']):.3f}  n={support}",
            va="center",
            fontsize=8,
        )

    save_figure(fig, outdir, "bias_170287_per_profession_f1")


def clean_label_name(value: object) -> str:
    text = str(value)
    text = text.replace("true__", "").replace("pred__", "")
    return text


def most_confused_labels(confusion: pd.DataFrame, per_prof: pd.DataFrame | None, n_labels: int = 12) -> list[str]:
    cm = confusion.copy()
    cm.index = [clean_label_name(x) for x in cm.index]
    cm.columns = [clean_label_name(x) for x in cm.columns]

    labels: list[str] = []
    if per_prof is not None and {"profession", "f1"}.issubset(per_prof.columns):
        worst = (
            per_prof.assign(f1=pd.to_numeric(per_prof["f1"], errors="coerce"))
            .dropna(subset=["f1"])
            .sort_values("f1")
            .head(8)["profession"]
            .astype(str)
            .tolist()
        )
        labels.extend(worst)

    offdiag = cm.copy()
    for label in offdiag.index.intersection(offdiag.columns):
        offdiag.loc[label, label] = 0

    pairs = offdiag.stack().sort_values(ascending=False)
    for (true_label, pred_label), count in pairs.items():
        if count <= 0:
            break
        labels.extend([str(true_label), str(pred_label)])
        if len(dict.fromkeys(labels)) >= n_labels:
            break

    unique = list(dict.fromkeys(labels))
    return [label for label in unique if label in cm.index and label in cm.columns][:n_labels]


def plot_confusion_heatmap(confusion: pd.DataFrame, per_prof: pd.DataFrame | None, outdir: Path) -> None:
    cm = confusion.copy()
    cm.index = [clean_label_name(x) for x in cm.index]
    cm.columns = [clean_label_name(x) for x in cm.columns]
    cm = cm.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    labels = most_confused_labels(cm, per_prof, n_labels=12)
    if not labels:
        warnings.warn("Could not identify labels for compact confusion heatmap")
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

    values = sub.to_numpy(dtype=int)
    threshold = max(values.max() * 0.55, 1)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if value > 0:
                color = "white" if value >= threshold else "black"
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")
    save_figure(fig, outdir, "bias_170287_confusion_hard_labels")


def build_summary_table(
    args: argparse.Namespace,
    logging_dir: Path,
    evaluations: pd.DataFrame,
    best: pd.Series,
    pred_summary: pd.DataFrame | None,
    run_summary: dict[str, object],
) -> pd.DataFrame:
    stage = evaluation_stage(evaluations, args.budget)
    row: dict[str, object] = {
        "job_id": args.job_id,
        "model": args.model,
        "dataset": args.dataset,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "tier": args.tier,
        "requested_budget": args.budget,
        "logging_dir": str(logging_dir),
        "eval_rows_at_budget": len(stage),
        "best_prompt_id": best.get("prompt_id", "unknown"),
        "chosen_step": best.get("chosen_step", np.nan),
        "eval_macro_f1": float(best["test_quality"]),
        "eval_cost": float(best["test_cost"]),
        "eval_unfairness": float(best["test_fairness"]),
        "eval_max_abs_tpr_gap": best.get("test_max_abs_tpr_gap", np.nan),
        "test_fairness_ready": best.get("test_fairness_ready", np.nan),
    }

    if "actual_tokens" in run_summary:
        row["run_summary_actual_tokens"] = run_summary.get("actual_tokens")
    if "status" in run_summary:
        row["run_summary_status"] = run_summary.get("status")

    if pred_summary is not None and len(pred_summary):
        pred = pred_summary.iloc[0]
        for src, dst in [
            ("eval_parquet_test_quality", "prediction_summary_eval_macro_f1"),
            ("recomputed_macro_f1_raw", "recomputed_macro_f1_raw"),
            ("recomputed_macro_f1_normalized", "recomputed_macro_f1_normalized"),
            ("accuracy_raw", "accuracy_raw"),
            ("accuracy_normalized", "accuracy_normalized"),
            ("invalid_rate_raw", "invalid_rate_raw"),
            ("invalid_rate_after_normalization", "invalid_rate_after_normalization"),
            ("n_examples", "prediction_n_examples"),
        ]:
            if src in pred:
                row[dst] = pred[src]

    return pd.DataFrame([row])


def write_readme(outdir: Path, args: argparse.Namespace, logging_dir: Path) -> None:
    readme = f"""# Bias in Bios advanced compact run {args.job_id} curated figures

Generated by `analysis.make_bias_170287_figures`.

## Run

- Model: `{args.model}`
- Dataset: `{args.dataset}`
- Optimizer: `{args.optimizer}`
- Seed: `{args.seed}`
- Prompt tier: `{args.tier}`
- Requested budget: `{args.budget}`
- Logging directory: `{logging_dir}`
- Prediction-capture directory: `{args.prediction_dir}`

## Figures

- `bias_170287_stepwise_incumbent_trajectory.*`  
  Development-side incumbent Macro-F1, cost, and unfairness over token budget.

- `bias_170287_test_macro_f1_vs_cost.*`  
  Holdout Test Macro-F1 vs cost for evaluated prompts at the 2M checkpoint.

- `bias_170287_test_macro_f1_vs_unfairness.*`  
  Holdout Test Macro-F1 vs unfairness for evaluated prompts at the 2M checkpoint.

- `bias_170287_test_cost_vs_unfairness.*`  
  Cost/fairness trade-off colored by Test Macro-F1.

- `bias_170287_per_profession_f1.*`  
  Per-profession F1 from prediction capture for the selected best prompt.

- `bias_170287_confusion_hard_labels.*`  
  Compact confusion matrix focused on the hardest / most-confused labels.

## Tables

- `bias_170287_summary.csv` and `.md`
- `bias_170287_stepwise_incumbent_trajectory.csv`
- `bias_170287_eval_rows.csv`
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")


def write_manifest(outdir: Path) -> None:
    files = []
    for path in sorted(outdir.iterdir()):
        if path.is_file():
            files.append({"file": path.name, "bytes": path.stat().st_size})
    pd.DataFrame(files).to_csv(outdir / "manifest.csv", index=False)


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
    parser.add_argument("--prediction-dir", default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--logging-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    logging_dir, step_results, evaluations, run_summary = load_run_files(args)
    pred_dir = Path(args.prediction_dir)
    pred_summary, per_prof, confusion = load_prediction_files(pred_dir)

    best = choose_best_eval_prompt(evaluations, args.budget)
    stage = evaluation_stage(evaluations, args.budget)

    stepwise = prepare_stepwise_incumbents(step_results)
    stepwise.to_csv(outdir / "bias_170287_stepwise_incumbent_trajectory.csv", index=False)
    stage.to_csv(outdir / "bias_170287_eval_rows.csv", index=False)

    summary = build_summary_table(args, logging_dir, evaluations, best, pred_summary, run_summary)
    summary.to_csv(outdir / "bias_170287_summary.csv", index=False)
    write_markdown_table(summary, outdir / "bias_170287_summary.md")

    plot_stepwise_incumbent(stepwise, outdir)
    plot_eval_macro_f1_vs_cost(evaluations, best, outdir, args.budget)
    plot_eval_macro_f1_vs_unfairness(evaluations, best, outdir, args.budget)
    plot_cost_vs_unfairness(evaluations, best, outdir, args.budget)

    if per_prof is not None:
        per_prof.to_csv(outdir / "bias_170287_per_profession_f1_source.csv", index=False)
        plot_per_profession_f1(per_prof, outdir)

    if confusion is not None:
        confusion.to_csv(outdir / "bias_170287_confusion_source.csv")
        plot_confusion_heatmap(confusion, per_prof, outdir)

    write_readme(outdir, args, logging_dir)
    write_manifest(outdir)

    print("\nBias 170287 curated figures written to:")
    print(outdir.resolve())
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()