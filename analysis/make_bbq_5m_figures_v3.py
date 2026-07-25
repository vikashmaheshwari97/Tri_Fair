"""Create publication-style BBQ Qwen-3-30B 5M v3 figures.

This companion script generates the figure families that are intentionally not
covered by ``plot_v3_checkpoint_metrics.py``:

* accuracy-versus-cost empirical attainment;
* accuracy-versus-unfairness empirical attainment;
* Tri-Fair-v3 few-shot/output-token-share diagnostics;
* Tri-Fair-v3 few-shot/unfairness diagnostics;
* cost-versus-unfairness and three-objective Pareto projections.

Inputs are the real post-hoc ``eval_checkpoints.parquet`` files.  Main figures
retain only holdout rows with ``test_fairness_ready=True`` and BBQ diagnostic
``coverage_valid=True``.  The 5M label is nominal: rows are mapped to the
nearest real logged optimization state under the frozen v3 tolerance policy.
No interpolation or synthetic result is introduced.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.model_configs import ALL_MODELS

MODEL = "qwen-3-30b"
DATASET = "bbq"
FINAL_BUDGET = 5_000_000
EXPECTED_SEEDS = (42, 43, 44)
METHOD_ORDER = ("Tri-Fair-v3", "NSGAII-PO-Fair")
DISPLAY = {
    "Tri-Fair-v3": "Tri-Fair v3",
    "NSGAII-PO-Fair": "NSGA-II-PO-Fair",
}
COLORS = {
    "Tri-Fair-v3": "black",
    "NSGAII-PO-Fair": "#E69F00",
}
MARKERS = {
    "Tri-Fair-v3": "o",
    "NSGAII-PO-Fair": "s",
}
COST_AXIS_LABEL = "Weighted Mean-Token Cost ↓"
UNFAIRNESS_AXIS_LABEL = "Statistical BBQ Unfairness ↓"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="results/tri_fair_v3_qwen_5m",
        help="v3 result namespace containing qwen-3-30b/bbq and initial/.",
    )
    parser.add_argument(
        "--figure-dir",
        default="analysis/output/bbq_5m_v3/publication_figures",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--final-budget", type=int, default=FINAL_BUDGET)
    parser.add_argument(
        "--maximum-checkpoint-relative-error",
        type=float,
        default=0.12,
        help="Frozen nearest-logged-state token tolerance used for audit checks.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise when validity diagnostics or the complete six-run grid are missing.",
    )
    return parser.parse_args()


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


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def _json_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def finite_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_checkpoint_evaluations(
    results_root: Path,
    *,
    model: str,
    dataset: str,
) -> pd.DataFrame:
    root = results_root / model / dataset
    files = sorted(root.rglob("eval_checkpoints.parquet"))
    if not files:
        raise FileNotFoundError(f"No eval_checkpoints.parquet files beneath {root}")

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_parquet(path).copy()
        frame["source_file"] = str(path)
        frame["run_dir"] = str(path.parent)
        frames.append(frame)

    output = pd.concat(frames, ignore_index=True, sort=False)
    require_columns(
        output,
        [
            "optimizer",
            "seed",
            "budget_checkpoint",
            "test_quality",
            "test_cost",
            "test_fairness",
        ],
        "checkpoint evaluations",
    )
    return output


def attach_publication_validity(frame: pd.DataFrame, *, strict: bool) -> pd.DataFrame:
    output = frame.copy()
    diagnostics = "test_fairness_diagnostics_json"
    if diagnostics not in output:
        if strict:
            raise ValueError(f"Missing {diagnostics}")
        output["test_coverage_valid"] = True
    else:
        output["test_coverage_valid"] = output[diagnostics].map(
            lambda value: bool(_json_object(value).get("coverage_valid", False))
        )

    if "test_fairness_ready" not in output:
        if strict:
            raise ValueError("Missing test_fairness_ready")
        output["test_fairness_ready"] = True

    output["publication_valid"] = (
        output["test_fairness_ready"].fillna(False).astype(bool)
        & output["test_coverage_valid"].fillna(False).astype(bool)
    )
    return output


def deduplicate_final(frame: pd.DataFrame, *, final_budget: int) -> pd.DataFrame:
    output = frame.copy()
    output["budget_checkpoint"] = pd.to_numeric(
        output["budget_checkpoint"], errors="coerce"
    )
    output["seed"] = pd.to_numeric(output["seed"], errors="coerce")
    output = output[
        output["optimizer"].astype(str).isin(METHOD_ORDER)
        & output["seed"].isin(EXPECTED_SEEDS)
        & (output["budget_checkpoint"] == final_budget)
    ].copy()
    if output.empty:
        raise RuntimeError(f"No nominal {final_budget} checkpoint rows were found")

    output["seed"] = output["seed"].astype(int)
    if "evaluation_timestamp" in output:
        output = output.sort_values("evaluation_timestamp", kind="stable")

    dedup = [
        column
        for column in ("optimizer", "seed", "prompt_id", "budget_checkpoint")
        if column in output
    ]
    if dedup:
        output = output.drop_duplicates(dedup, keep="last")

    return output.reset_index(drop=True)


def validate_grid(frame: pd.DataFrame, *, tolerance: float, strict: bool) -> None:
    observed = {
        (str(row.optimizer), int(row.seed))
        for row in frame[["optimizer", "seed"]].drop_duplicates().itertuples(index=False)
    }
    expected = {
        (optimizer, seed)
        for optimizer in METHOD_ORDER
        for seed in EXPECTED_SEEDS
    }
    missing = sorted(expected - observed)
    if missing:
        message = "Missing final method/seed groups: " + ", ".join(
            f"{method}/seed{seed}" for method, seed in missing
        )
        if strict:
            raise RuntimeError(message)
        print("WARNING:", message)

    if "checkpoint_relative_error" in frame:
        error = pd.to_numeric(frame["checkpoint_relative_error"], errors="coerce")
        too_far = frame[error > tolerance]
        if not too_far.empty:
            message = (
                f"{len(too_far)} rows exceed checkpoint relative error {tolerance:.3f}"
            )
            if strict:
                raise RuntimeError(message)
            print("WARNING:", message)


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for index in range(len(values)):
        dominates = np.all(values <= values[index], axis=1) & np.any(
            values < values[index], axis=1
        )
        dominates[index] = False
        if np.any(dominates):
            keep[index] = False
    return keep


def objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            1.0 - finite_numeric(frame["test_quality"]),
            finite_numeric(frame["test_cost"]),
            finite_numeric(frame["test_fairness"]),
        ]
    )


def valid_pareto_front(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = objective_matrix(frame)
    valid = np.all(np.isfinite(matrix), axis=1)
    data = frame.loc[valid].reset_index(drop=True)
    matrix = matrix[valid]
    if data.empty:
        return data
    return data.loc[pareto_mask_minimize(matrix)].reset_index(drop=True)


def y_attained_at_x(
    data: pd.DataFrame,
    x_grid: np.ndarray,
    *,
    x_col: str,
) -> np.ndarray:
    x = finite_numeric(data[x_col])
    y = finite_numeric(data["test_quality"])
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    output = np.full(len(x_grid), np.nan)
    for index, value in enumerate(x_grid):
        eligible = y[x <= value]
        if len(eligible):
            output[index] = float(np.max(eligible))
    return output


def plot_empirical_attainment(
    final: pd.DataFrame,
    outdir: Path,
    *,
    x_col: str,
    xlabel: str,
    stem: str,
) -> None:
    x_values = finite_numeric(final[x_col])
    x_values = x_values[np.isfinite(x_values)]
    if len(x_values) == 0:
        raise RuntimeError(f"No finite {x_col} values")
    xmin = float(np.min(x_values))
    xmax = float(np.max(x_values))
    padding = 0.03 * max(xmax - xmin, 1e-6)
    x_grid = np.linspace(xmin - padding, xmax + padding, 400)

    fig, ax = plt.subplots(figsize=(6.7, 4.3), constrained_layout=True)
    for optimizer in METHOD_ORDER:
        curves: list[np.ndarray] = []
        data = final[final["optimizer"] == optimizer]
        for _, seed_data in data.groupby("seed", sort=True):
            front = valid_pareto_front(seed_data)
            if not front.empty:
                curves.append(y_attained_at_x(front, x_grid, x_col=x_col))
        if not curves:
            continue
        matrix = np.vstack(curves)
        with np.errstate(all="ignore"):
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
            label=DISPLAY[optimizer],
        )
        ax.fill_between(
            x_grid[valid],
            lower[valid],
            upper[valid],
            step="post",
            color=COLORS[optimizer],
            alpha=0.16,
        )

    ax.set_title("BBQ — Qwen-3-30B at nominal 5M")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Holdout Accuracy ↑")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.01,
        0.01,
        "Median and seed range; nearest real logged 5M state.",
        transform=ax.transAxes,
        fontsize=7,
        alpha=0.72,
    )
    save_figure(fig, outdir, stem)


def parse_fewshot_count(value: object) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0
    if isinstance(value, list):
        return len(value)
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0
    return len(parsed) if isinstance(parsed, list) else 0


def output_cost_share(
    frame: pd.DataFrame,
    *,
    input_weight: float,
    output_weight: float,
) -> np.ndarray:
    require_columns(
        frame,
        ["test_input_tokens", "test_output_tokens"],
        "checkpoint evaluations",
    )
    input_tokens = finite_numeric(frame["test_input_tokens"])
    output_tokens = finite_numeric(frame["test_output_tokens"])
    weighted_input = input_weight * input_tokens
    weighted_output = output_weight * output_tokens
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


def trifair_candidates(
    final: pd.DataFrame,
    *,
    input_weight: float,
    output_weight: float,
) -> pd.DataFrame:
    data = final[final["optimizer"] == "Tri-Fair-v3"].copy()
    if "is_incumbent" in data and data["is_incumbent"].notna().any():
        incumbents = data[data["is_incumbent"].fillna(False).astype(bool)]
        if not incumbents.empty:
            data = incumbents.copy()
    if data.empty:
        raise RuntimeError("No Tri-Fair-v3 final candidate rows")
    data["fewshot_count"] = (
        data["few_shots_json"].map(parse_fewshot_count)
        if "few_shots_json" in data
        else 0
    )
    data["output_cost_share"] = output_cost_share(
        data,
        input_weight=input_weight,
        output_weight=output_weight,
    )
    return data.reset_index(drop=True)


def plot_trifair_diagnostic(
    data: pd.DataFrame,
    outdir: Path,
    *,
    color_col: str,
    color_label: str,
    stem: str,
) -> None:
    values = finite_numeric(data[color_col])
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(6.7, 4.8), constrained_layout=True)
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
    ax.set_title("Tri-Fair v3 on BBQ — Qwen-3-30B at nominal 5M")
    ax.set_xlabel(COST_AXIS_LABEL)
    ax.set_ylabel("Holdout Accuracy ↑")
    ax.grid(True, alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(color_label)
    ax.text(
        0.01,
        0.01,
        "Numeric labels show few-shot example count.",
        transform=ax.transAxes,
        fontsize=7,
        alpha=0.72,
    )
    save_figure(fig, outdir, stem)


def plot_cost_unfairness(final: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 4.5), constrained_layout=True)
    for optimizer in METHOD_ORDER:
        pooled: list[pd.DataFrame] = []
        for _, seed_data in final[final["optimizer"] == optimizer].groupby(
            "seed", sort=True
        ):
            front = valid_pareto_front(seed_data)
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
        data = pd.concat(pooled, ignore_index=True)
        projection = np.column_stack(
            [finite_numeric(data["test_cost"]), finite_numeric(data["test_fairness"])]
        )
        valid = np.all(np.isfinite(projection), axis=1)
        data = data.loc[valid].reset_index(drop=True)
        projection = projection[valid]
        front = data.loc[pareto_mask_minimize(projection)].sort_values("test_cost")
        ax.plot(
            front["test_cost"],
            front["test_fairness"],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            linewidth=2,
            markersize=5,
            label=DISPLAY[optimizer],
        )
    ax.set_title("BBQ — nominal 5M cost versus holdout unfairness")
    ax.set_xlabel(COST_AXIS_LABEL)
    ax.set_ylabel(UNFAIRNESS_AXIS_LABEL)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, outdir, "bbq_qwen3_5m_v3_pareto_cost_unfairness")


def plot_three_objective_pareto(final: pd.DataFrame, outdir: Path) -> None:
    fig = plt.figure(figsize=(7.2, 5.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for optimizer in METHOD_ORDER:
        for seed, seed_data in final[final["optimizer"] == optimizer].groupby(
            "seed", sort=True
        ):
            front = valid_pareto_front(seed_data)
            if front.empty:
                continue
            ax.scatter(
                front["test_cost"],
                front["test_fairness"],
                front["test_quality"],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                alpha=0.78,
                s=34,
                label=DISPLAY[optimizer] if seed == EXPECTED_SEEDS[0] else None,
            )
    ax.set_title("BBQ — Qwen-3-30B nominal 5M holdout Pareto fronts")
    ax.set_xlabel("Cost Objective ↓")
    ax.set_ylabel(UNFAIRNESS_AXIS_LABEL)
    ax.set_zlabel("Accuracy ↑")
    ax.legend(frameon=False)
    save_figure(fig, outdir, "bbq_qwen3_5m_v3_test_pareto_3d")


def write_audit_tables(
    raw_final: pd.DataFrame,
    valid_final: pd.DataFrame,
    trifair: pd.DataFrame,
    outdir: Path,
) -> None:
    raw_final.to_parquet(
        outdir / "bbq_qwen3_5m_v3_raw_final_evaluations.parquet",
        index=False,
    )
    valid_final.to_parquet(
        outdir / "bbq_qwen3_5m_v3_publication_valid_evaluations.parquet",
        index=False,
    )
    trifair.to_csv(
        outdir / "bbq_qwen3_5m_v3_trifair_candidate_diagnostics.csv",
        index=False,
    )

    group_summary = (
        valid_final.groupby(["optimizer", "seed"], sort=True)
        .agg(
            valid_candidates=("test_quality", "size"),
            best_accuracy=("test_quality", "max"),
            lowest_cost=("test_cost", "min"),
            lowest_unfairness=("test_fairness", "min"),
            actual_budget_tokens=("actual_budget_tokens", "max"),
            checkpoint_relative_error=("checkpoint_relative_error", "max"),
        )
        .reset_index()
    )
    group_summary.to_csv(
        outdir / "bbq_qwen3_5m_v3_final_candidate_summary.csv",
        index=False,
    )

    readme = """# BBQ Qwen-3-30B 5M v3 publication figures

Main files:

- `bbq_qwen3_5m_v3_attainment_accuracy_cost.*`
- `bbq_qwen3_5m_v3_attainment_accuracy_unfairness.*`
- `bbq_qwen3_5m_v3_trifair_fewshot_outputshare.*`
- `bbq_qwen3_5m_v3_trifair_fewshot_unfairness.*`
- `bbq_qwen3_5m_v3_pareto_cost_unfairness.*`
- `bbq_qwen3_5m_v3_test_pareto_3d.*`

Rows in the main figures require `test_fairness_ready=True` and BBQ
`coverage_valid=True`.  The nominal 5M state is the nearest real logged
optimization step within the frozen 12% token tolerance.  No interpolated or
synthetic holdout result is used.
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()

    results_root = Path(args.results_root).expanduser().resolve()
    outdir = Path(args.figure_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_checkpoint_evaluations(
        results_root,
        model=str(args.model),
        dataset=str(args.dataset),
    )
    enriched = attach_publication_validity(raw, strict=bool(args.strict))
    raw_final = deduplicate_final(enriched, final_budget=int(args.final_budget))
    validate_grid(
        raw_final,
        tolerance=float(args.maximum_checkpoint_relative_error),
        strict=bool(args.strict),
    )
    final = raw_final[raw_final["publication_valid"]].copy()
    if final.empty:
        raise RuntimeError("No publication-valid nominal-5M rows remain")
    validate_grid(
        final,
        tolerance=float(args.maximum_checkpoint_relative_error),
        strict=bool(args.strict),
    )

    model_config = ALL_MODELS[str(args.model)]
    input_weight = float(model_config.input_costs)
    output_weight = float(model_config.output_costs)
    trifair = trifair_candidates(
        final,
        input_weight=input_weight,
        output_weight=output_weight,
    )

    plot_empirical_attainment(
        final,
        outdir,
        x_col="test_cost",
        xlabel=COST_AXIS_LABEL,
        stem="bbq_qwen3_5m_v3_attainment_accuracy_cost",
    )
    plot_empirical_attainment(
        final,
        outdir,
        x_col="test_fairness",
        xlabel=UNFAIRNESS_AXIS_LABEL,
        stem="bbq_qwen3_5m_v3_attainment_accuracy_unfairness",
    )
    plot_trifair_diagnostic(
        trifair,
        outdir,
        color_col="output_cost_share",
        color_label="Output Token Cost Share",
        stem="bbq_qwen3_5m_v3_trifair_fewshot_outputshare",
    )
    plot_trifair_diagnostic(
        trifair,
        outdir,
        color_col="test_fairness",
        color_label=UNFAIRNESS_AXIS_LABEL,
        stem="bbq_qwen3_5m_v3_trifair_fewshot_unfairness",
    )
    plot_cost_unfairness(final, outdir)
    plot_three_objective_pareto(final, outdir)
    write_audit_tables(raw_final, final, trifair, outdir)

    print("Generated BBQ v3 publication files:")
    for path in sorted(outdir.iterdir()):
        if path.is_file():
            print(" ", path.name)
    print(f"\nWritten to: {outdir}")


if __name__ == "__main__":
    main()
