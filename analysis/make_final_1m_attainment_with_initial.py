"""Fig. 3-style 1M empirical-attainment plots with Initial Instructions.

Inputs:
  analysis/output/all_evaluations.parquet
  analysis/output/initial_instructions_evaluations.parquet

Outputs:
  analysis/output/curated_figures_final_1m_with_initial/
    fairness_qwen3_1m_attainment_quality_cost_all_datasets.*
    fairness_qwen3_1m_attainment_quality_unfairness_all_datasets.*
    fairness_qwen3_1m_initial_plus_methods_summary.*
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL = "qwen-3-30b"
FINAL_BUDGET = 1_000_000

DATASET_ORDER = ["bbq", "bias_in_bios", "civil_comments"]
DATASET_TITLE = {
    "bbq": "BBQ",
    "bias_in_bios": "Bias in Bios",
    "civil_comments": "Civil Comments",
}
Y_LABEL = {
    "bbq": "Test Accuracy",
    "bias_in_bios": "Test Macro-F1",
    "civil_comments": "Test Accuracy",
}

METHOD_ORDER = ["Initial Instructions", "NSGAII-PO-Fair", "Tri-Fair"]
DISPLAY_NAME = {
    "Initial Instructions": "Initial Instructions",
    "NSGAII-PO-Fair": "NSGA-II-PO-Fair",
    "Tri-Fair": "Tri-Fair",
}
COLORS = {
    "Initial Instructions": "#8c8c8c",
    "NSGAII-PO-Fair": "#E69F00",
    "Tri-Fair": "black",
}
MARKERS = {
    "Initial Instructions": "x",
    "NSGAII-PO-Fair": "s",
    "Tri-Fair": "o",
}
LINESTYLES = {
    "Initial Instructions": "--",
    "NSGAII-PO-Fair": "-",
    "Tri-Fair": "-",
}


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
    out = Path("analysis/output/curated_figures_final_1m_with_initial")
    out.mkdir(parents=True, exist_ok=True)
    return out


def finite_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "run_dir" in out:
        out = out[
            ~out["run_dir"].astype(str).str.contains("pilot_reports", regex=False)
        ]
    if "run_key" in out:
        out = out[~out["run_key"].astype(str).str.endswith("/logging")]
    if "model" in out:
        out = out[out["model"] == MODEL]
    if "dataset" in out:
        out = out[out["dataset"].isin(DATASET_ORDER)]
    if "budget_checkpoint" in out:
        out = out[
            pd.to_numeric(out["budget_checkpoint"], errors="coerce").fillna(-1).astype(int)
            == FINAL_BUDGET
        ]
    if "test_fairness_ready" in out:
        ready = out["test_fairness_ready"].fillna(False).astype(bool)
        if ready.any():
            out = out[ready]
    return out.reset_index(drop=True)


def read_inputs() -> pd.DataFrame:
    root = Path("analysis/output")
    all_eval_path = root / "all_evaluations.parquet"
    initial_path = root / "initial_instructions_evaluations.parquet"

    missing = [str(path) for path in [all_eval_path, initial_path] if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n"
            + "\n".join(missing)
            + "\nRun analysis.analysis_pipeline and scripts.evaluate_initial_instructions first."
        )

    evaluations = clean_frame(pd.read_parquet(all_eval_path))
    initial = clean_frame(pd.read_parquet(initial_path))

    evaluations = evaluations[
        evaluations["optimizer"].isin(["Tri-Fair", "NSGAII-PO-Fair"])
    ].copy()

    data = pd.concat([initial, evaluations], ignore_index=True, sort=False)
    data = data[data["optimizer"].isin(METHOD_ORDER)].reset_index(drop=True)

    required = {"dataset", "optimizer", "seed", "test_quality", "test_cost", "test_fairness"}
    missing_cols = required - set(data.columns)
    if missing_cols:
        raise RuntimeError(f"Combined evaluations are missing columns: {sorted(missing_cols)}")

    return data


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


def test_objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            1.0 - finite_numeric(frame["test_quality"]),
            finite_numeric(frame["test_cost"]),
            finite_numeric(frame["test_fairness"]),
        ]
    )


def y_attained_at_x(data: pd.DataFrame, x_grid: np.ndarray, x_col: str) -> np.ndarray:
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


def method_seed_attainment(
    data: pd.DataFrame,
    *,
    method: str,
    x_grid: np.ndarray,
    x_col: str,
) -> np.ndarray | None:
    method_data = data[data["optimizer"] == method].copy()
    if method_data.empty:
        return None

    seed_curves = []
    for _, seed_group in method_data.groupby("seed", sort=True):
        if seed_group.empty:
            continue
        mask = pareto_mask_minimize(test_objective_matrix(seed_group))
        front = seed_group.loc[mask].copy()
        curve = y_attained_at_x(front, x_grid, x_col)
        seed_curves.append(curve)

    if not seed_curves:
        return None

    return np.vstack(seed_curves)


def plot_all_datasets(
    data: pd.DataFrame,
    *,
    x_col: str,
    xlabel: str,
    filename: str,
) -> None:
    outdir = output_dir()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3), constrained_layout=True)

    handles = []
    labels = []

    for ax, dataset in zip(axes, DATASET_ORDER):
        ds = data[data["dataset"] == dataset].copy()
        if ds.empty:
            ax.set_visible(False)
            continue

        xmin = float(np.nanmin(finite_numeric(ds[x_col])))
        xmax = float(np.nanmax(finite_numeric(ds[x_col])))
        padding = 0.04 * max(xmax - xmin, 1e-6)
        x_grid = np.linspace(xmin - padding, xmax + padding, 400)

        for method in METHOD_ORDER:
            matrix = method_seed_attainment(
                ds,
                method=method,
                x_grid=x_grid,
                x_col=x_col,
            )
            if matrix is None:
                continue

            median = np.nanmedian(matrix, axis=0)
            lower = np.nanmin(matrix, axis=0)
            upper = np.nanmax(matrix, axis=0)
            valid = np.isfinite(median)
            if not valid.any():
                continue

            line = ax.step(
                x_grid[valid],
                median[valid],
                where="post",
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                marker=MARKERS[method],
                markevery=max(1, int(valid.sum() / 8)),
                linewidth=2,
                markersize=4.5,
                label=DISPLAY_NAME[method],
            )[0]
            ax.fill_between(
                x_grid[valid],
                lower[valid],
                upper[valid],
                step="post",
                color=COLORS[method],
                alpha=0.14,
            )

            if DISPLAY_NAME[method] not in labels:
                handles.append(line)
                labels.append(DISPLAY_NAME[method])

        ax.set_title(f"{DATASET_TITLE[dataset]} — Qwen-3-30B at 1M")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(Y_LABEL[dataset])
        ax.grid(True, alpha=0.25)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )

    fig.savefig(outdir / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{filename}.png", bbox_inches="tight")
    plt.close(fig)


def best_quality_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dataset, method, seed), group in data.groupby(
        ["dataset", "optimizer", "seed"],
        sort=True,
    ):
        group = group.copy()
        group["test_quality"] = pd.to_numeric(group["test_quality"], errors="coerce")
        group["test_cost"] = pd.to_numeric(group["test_cost"], errors="coerce")
        group["test_fairness"] = pd.to_numeric(group["test_fairness"], errors="coerce")
        group = group.dropna(subset=["test_quality", "test_cost", "test_fairness"])
        if group.empty:
            continue

        best = group.sort_values(
            ["test_quality", "test_fairness", "test_cost"],
            ascending=[False, True, True],
        ).iloc[0]

        rows.append(
            {
                "dataset": dataset,
                "method": DISPLAY_NAME.get(method, method),
                "seed": int(seed),
                "best_test_quality": float(best["test_quality"]),
                "best_test_cost": float(best["test_cost"]),
                "best_test_unfairness": float(best["test_fairness"]),
                "worst_group_accuracy_if_applicable": (
                    1.0 - float(best["test_fairness"])
                    if dataset == "civil_comments"
                    else np.nan
                ),
                "prompt_id": best.get("prompt_id", ""),
            }
        )

    return pd.DataFrame(rows)


def write_summary(data: pd.DataFrame) -> None:
    outdir = output_dir()

    best = best_quality_rows(data)
    best.to_csv(outdir / "fairness_qwen3_1m_best_quality_by_seed.csv", index=False)

    summary = (
        best.groupby(["dataset", "method"])[
            ["best_test_quality", "best_test_cost", "best_test_unfairness"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]

    summary.to_csv(
        outdir / "fairness_qwen3_1m_initial_plus_methods_summary.csv",
        index=False,
    )

    md_path = outdir / "fairness_qwen3_1m_initial_plus_methods_summary.md"
    md_path.write_text(
        summary.to_markdown(index=False, floatfmt=".4f") + "\n",
        encoding="utf-8",
    )

    print("\nBest-quality summary")
    print(summary.to_string(index=False))


def write_readme() -> None:
    outdir = output_dir()
    readme = """# Final 1M Fig. 3-style attainment plots with Initial Instructions

Files generated by `analysis.make_final_1m_attainment_with_initial`.

Methods:
- Initial Instructions
- NSGA-II-PO-Fair
- Tri-Fair

Datasets:
- BBQ: y-axis is Test Accuracy.
- Bias in Bios: y-axis is Test Macro-F1.
- Civil Comments: y-axis is Test Accuracy.

Figures:
- `fairness_qwen3_1m_attainment_quality_cost_all_datasets.*`
  Empirical-attainment-style quality vs cost across all three datasets.

- `fairness_qwen3_1m_attainment_quality_unfairness_all_datasets.*`
  Empirical-attainment-style quality vs unfairness across all three datasets.

Notes:
- Lines show median attainment across three seeds.
- Shaded bands show min-to-max attainment across seeds.
- Initial Instructions are evaluated as fixed starting prompts, not as an optimizer.
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    configure_style()
    data = read_inputs()

    print("\nCounts")
    print(
        data.groupby(["dataset", "optimizer", "seed"])
        .size()
        .rename("n_rows")
        .reset_index()
        .to_string(index=False)
    )

    plot_all_datasets(
        data,
        x_col="test_cost",
        xlabel="Avg. Cost [$] per 1M Calls",
        filename="fairness_qwen3_1m_attainment_quality_cost_all_datasets",
    )

    plot_all_datasets(
        data,
        x_col="test_fairness",
        xlabel="Test Unfairness",
        filename="fairness_qwen3_1m_attainment_quality_unfairness_all_datasets",
    )

    write_summary(data)
    write_readme()

    print("\nFinal 1M attainment figures written to:")
    print(output_dir().resolve())


if __name__ == "__main__":
    main()