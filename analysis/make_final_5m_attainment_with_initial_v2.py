"""Final 5M v2 empirical-attainment figures with exact Initial Instructions.

This module combines the three completed Qwen-3-30B fairness studies:

* BBQ
* Bias-in-Bios
* Civil Comments

For each dataset and seed, the optimized sets are read from the exact holdout
checkpoint selected by ``eval_5000000.done.json``. The initial baseline is read
from the exact 12-prompt starting population evaluated by
``scripts.evaluate_initial_instructions_5m_v2``.

Publication filtering follows the v2 dataset rules:

* BBQ: statistically ready test fairness and valid disambiguated coverage.
* Bias-in-Bios: statistically ready development and test fairness. Macro-recall
  shortfalls remain in the set because they are already penalized in the native
  unfairness objective.
* Civil Comments: statistically ready development and test fairness. Class-
  recall shortfalls remain because they are already penalized in the native
  unfairness objective.

The optimized methods are nominal 5M runs. Initial Instructions are a budget-zero
baseline recovered from those runs' exact starting populations.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL = "qwen-3-30b"
FINAL_BUDGET = 5_000_000
EXPECTED_INITIAL_PROMPTS = 12

DATASET_ORDER = ("bbq", "bias_in_bios", "civil_comments")
DATASET_TITLE = {
    "bbq": "BBQ",
    "bias_in_bios": "Bias-in-Bios",
    "civil_comments": "Civil Comments",
}
QUALITY_LABEL = {
    "bbq": "Test Accuracy ↑",
    "bias_in_bios": "Test Macro-F1 ↑",
    "civil_comments": "Test Accuracy ↑",
}
UNFAIRNESS_LABEL = {
    "bbq": "Statistical BBQ Unfairness ↓",
    "bias_in_bios": "Statistical Gender TPR-Gap Unfairness ↓",
    "civil_comments": "Statistical Equalized-Odds Unfairness ↓",
}

OPTIMIZER_ORDER = ("Tri-Fair", "NSGAII-PO-Fair")
METHOD_ORDER = ("Initial Instructions", "NSGAII-PO-Fair", "Tri-Fair")
DISPLAY_NAME = {
    "Initial Instructions": "Initial Instructions",
    "NSGAII-PO-Fair": "NSGA-II-PO-Fair",
    "Tri-Fair": "Tri-Fair",
}
COLORS = {
    "Initial Instructions": "#7A7A7A",
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
DEFAULT_SEEDS = (42, 43, 44)
COST_LABEL = "Weighted Mean-Token Cost ↓"


def parse_csv(
    raw: str | Iterable[str],
    *,
    cast=str,
) -> tuple:
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in raw if str(part).strip()]
    values = tuple(cast(part) for part in parts)
    if not values:
        raise ValueError("At least one value is required")
    return values


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


def require_file(path: Path, *, nonempty: bool = True) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if nonempty and path.stat().st_size == 0:
        raise RuntimeError(f"Required file is empty: {path}")


def read_json(path: Path) -> dict[str, object]:
    require_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def json_object(value: object) -> dict[str, object]:
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
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def resolve_logging_dir(output_dir: Path) -> Path:
    pointer = output_dir / "logging_dir.txt"
    require_file(pointer)
    text = pointer.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"Empty logging-directory pointer: {pointer}")

    raw = Path(text).expanduser()
    candidates = (
        [raw.resolve()]
        if raw.is_absolute()
        else [
            (Path.cwd() / raw).resolve(),
            (output_dir / raw).resolve(),
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve logging directory from {pointer}; tried {candidates}"
    )


def stable_prompt_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    key = "prompt_id" if "prompt_id" in work else "prompt"
    if key not in work:
        raise RuntimeError("Evaluation table contains neither prompt_id nor prompt")

    sort_columns = [
        column
        for column in ("chosen_step", "step", "prompt_order")
        if column in work
    ]
    if sort_columns:
        work = work.sort_values(sort_columns, kind="stable")
    return work.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)


def optimized_output_dir(
    results_root: Path,
    *,
    model: str,
    dataset: str,
    optimizer: str,
    seed: int,
) -> Path:
    return results_root / model / dataset / optimizer / f"seed{seed}"


def load_optimized_exact(
    results_root: Path,
    *,
    model: str,
    dataset: str,
    optimizer: str,
    seed: int,
    budget: int,
) -> pd.DataFrame:
    output_dir = optimized_output_dir(
        results_root,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        seed=seed,
    )
    status_dir = output_dir / "status"
    done_path = status_dir / f"eval_{budget}.done.json"
    done = read_json(done_path)

    expected = {
        "model": model,
        "dataset": dataset,
        "optimizer": optimizer,
        "seed": seed,
        "requested_budget": budget,
    }
    for key, wanted in expected.items():
        observed = done.get(key)
        if observed is not None and str(observed) != str(wanted):
            raise RuntimeError(
                f"{done_path}: {key}={observed!r}, expected {wanted!r}"
            )

    for marker in (
        status_dir / f"stage_{budget}.failed.json",
        status_dir / f"stage_{budget}.running.json",
        status_dir / f"eval_{budget}.failed.json",
        status_dir / f"eval_{budget}.running.json",
    ):
        if marker.exists():
            raise RuntimeError(f"Incomplete/failure marker exists: {marker}")

    chosen_step = int(done["chosen_step"])
    actual_tokens = int(done["actual_tokens"])

    run_dir = resolve_logging_dir(output_dir)
    eval_path = run_dir / "eval.parquet"
    require_file(eval_path)
    frame = pd.read_parquet(eval_path)

    if "chosen_step" not in frame:
        raise RuntimeError(f"{eval_path} lacks chosen_step")

    steps = pd.to_numeric(frame["chosen_step"], errors="coerce")
    exact = frame[steps == chosen_step].copy()
    if exact.empty:
        observed = sorted(steps.dropna().astype(int).unique().tolist())
        raise RuntimeError(
            f"No exact chosen-step rows in {eval_path}: "
            f"chosen={chosen_step}, observed={observed}"
        )

    exact = stable_prompt_rows(exact)
    exact["optimizer"] = optimizer
    exact["dataset"] = dataset
    exact["model"] = model
    exact["seed"] = int(seed)
    exact["budget_checkpoint"] = int(budget)
    exact["comparison_budget"] = int(budget)
    exact["actual_budget_tokens"] = actual_tokens
    exact["source_kind"] = "optimized_5m"
    exact["run_dir"] = str(run_dir)
    exact["run_key"] = (
        f"optimized/{model}/{dataset}/{optimizer}/seed{seed}/budget{budget}"
    )
    return exact.reset_index(drop=True)


def initial_eval_path(
    results_root: Path,
    *,
    model: str,
    dataset: str,
    seed: int,
) -> Path:
    return (
        results_root
        / model
        / dataset
        / "init"
        / f"seed{seed}"
        / "initial"
        / "eval.parquet"
    )


def load_initial(
    results_root: Path,
    *,
    model: str,
    dataset: str,
    seed: int,
    budget: int,
    expected_count: int,
) -> pd.DataFrame:
    path = initial_eval_path(
        results_root,
        model=model,
        dataset=dataset,
        seed=seed,
    )
    require_file(path)
    frame = stable_prompt_rows(pd.read_parquet(path))

    required = {
        "test_quality",
        "test_cost",
        "test_fairness",
        "test_fairness_ready",
        "prompt_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} lacks required columns: {sorted(missing)}")

    if len(frame) != expected_count or frame["prompt_id"].astype(str).nunique() != expected_count:
        raise RuntimeError(
            f"{path} has {len(frame)} rows and "
            f"{frame['prompt_id'].astype(str).nunique()} unique prompts; "
            f"expected {expected_count}"
        )

    for column, wanted in {
        "model": model,
        "dataset": dataset,
        "seed": seed,
    }.items():
        if column in frame:
            observed = set(frame[column].astype(str))
            if observed != {str(wanted)}:
                raise RuntimeError(
                    f"{path}: {column}={sorted(observed)}, expected {wanted!r}"
                )

    if "comparison_budget" in frame:
        observed = set(
            pd.to_numeric(frame["comparison_budget"], errors="coerce")
            .dropna()
            .astype(int)
        )
        if observed != {int(budget)}:
            raise RuntimeError(
                f"{path}: comparison_budget={sorted(observed)}, expected {budget}"
            )

    frame["optimizer"] = "Initial Instructions"
    frame["dataset"] = dataset
    frame["model"] = model
    frame["seed"] = int(seed)
    frame["budget_checkpoint"] = 0
    frame["comparison_budget"] = int(budget)
    frame["actual_budget_tokens"] = 0
    frame["source_kind"] = "exact_initial_population"
    frame["run_dir"] = str(path.parent)
    frame["run_key"] = f"initial/{model}/{dataset}/seed{seed}"
    return frame.reset_index(drop=True)


def numeric_bool(
    frame: pd.DataFrame,
    column: str,
    *,
    default: bool = False,
) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].fillna(default).astype(bool)


def diagnostic_series(
    frame: pd.DataFrame,
    *,
    split: str,
    key: str,
    default: object = np.nan,
) -> pd.Series:
    column = f"{split}_fairness_diagnostics_json"
    if column not in frame:
        return pd.Series([default] * len(frame), index=frame.index)
    return frame[column].map(lambda value: json_object(value).get(key, default))


def attach_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    for split in ("dev", "test"):
        out[f"{split}_coverage_valid"] = diagnostic_series(
            out,
            split=split,
            key="coverage_valid",
            default=False,
        ).fillna(False).astype(bool)
        for key in (
            "disambig_coverage",
            "rms_abs_bias",
            "minimum_class_recall",
            "required_minimum_class_recall",
            "equalized_odds_rms",
            "macro_recall",
            "required_macro_recall",
            "rms_tpr_gap",
            "utility_penalty",
            "utility_valid_unfairness",
            "valid_identities",
            "valid_professions",
            "max_ci_width",
        ):
            out[f"{split}_{key}"] = pd.to_numeric(
                diagnostic_series(
                    out,
                    split=split,
                    key=key,
                    default=np.nan,
                ),
                errors="coerce",
            )

    return out


def publication_mask(frame: pd.DataFrame) -> pd.Series:
    required_numeric = (
        pd.to_numeric(frame["test_quality"], errors="coerce").notna()
        & pd.to_numeric(frame["test_cost"], errors="coerce").notna()
        & pd.to_numeric(frame["test_fairness"], errors="coerce").notna()
    )
    test_ready = numeric_bool(frame, "test_fairness_ready")
    dataset = frame["dataset"].astype(str)

    valid = required_numeric & test_ready

    bbq = dataset.eq("bbq")
    valid &= (~bbq) | frame["test_coverage_valid"].fillna(False).astype(bool)

    needs_dev = dataset.isin(("bias_in_bios", "civil_comments"))
    dev_ready = numeric_bool(frame, "dev_fairness_ready")
    valid &= (~needs_dev) | dev_ready

    return valid.astype(bool)


def validate_manifest_consistency(
    raw: pd.DataFrame,
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    strict: bool,
) -> None:
    if "manifest_sha256" not in raw:
        if strict:
            raise RuntimeError("Combined rows lack manifest_sha256")
        warnings.warn("Manifest hashes are unavailable; consistency was not verified")
        return

    failures: list[str] = []
    for dataset in datasets:
        for seed in seeds:
            group = raw[
                (raw["dataset"] == dataset)
                & (pd.to_numeric(raw["seed"], errors="coerce") == int(seed))
            ]
            hashes = set(group["manifest_sha256"].dropna().astype(str))
            if len(hashes) != 1:
                failures.append(
                    f"{dataset}/seed{seed}: observed manifest hashes {sorted(hashes)}"
                )
    if failures:
        message = "Manifest consistency check failed:\n  - " + "\n  - ".join(failures)
        if strict:
            raise RuntimeError(message)
        warnings.warn(message)


def load_all_rows(
    *,
    results_root: Path,
    model: str,
    datasets: Sequence[str],
    optimizers: Sequence[str],
    seeds: Sequence[int],
    budget: int,
    expected_initial_prompts: int,
    strict: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for dataset in datasets:
        for optimizer in optimizers:
            for seed in seeds:
                try:
                    frames.append(
                        load_optimized_exact(
                            results_root,
                            model=model,
                            dataset=dataset,
                            optimizer=optimizer,
                            seed=seed,
                            budget=budget,
                        )
                    )
                except Exception as error:
                    errors.append(
                        f"optimized {dataset}/{optimizer}/seed{seed}: {error}"
                    )

        for seed in seeds:
            try:
                frames.append(
                    load_initial(
                        results_root,
                        model=model,
                        dataset=dataset,
                        seed=seed,
                        budget=budget,
                        expected_count=expected_initial_prompts,
                    )
                )
            except Exception as error:
                errors.append(f"initial {dataset}/seed{seed}: {error}")

    if errors:
        message = "Could not load the complete 5M v2 + initial grid:\n  - " + "\n  - ".join(errors)
        if strict:
            raise RuntimeError(message)
        warnings.warn(message)

    if not frames:
        raise RuntimeError("No optimized or initial evaluation rows were loaded")

    raw = attach_diagnostics(
        pd.concat(frames, ignore_index=True, sort=False)
    )
    raw["publication_valid"] = publication_mask(raw)
    valid = raw[raw["publication_valid"]].copy().reset_index(drop=True)

    if valid.empty:
        raise RuntimeError("No publication-valid rows remain")

    validate_manifest_consistency(
        raw,
        datasets=datasets,
        seeds=seeds,
        strict=strict,
    )

    if strict:
        expected_groups = {
            (dataset, method, int(seed))
            for dataset in datasets
            for method in METHOD_ORDER
            for seed in seeds
        }
        observed_raw = {
            (str(row.dataset), str(row.optimizer), int(row.seed))
            for row in raw[["dataset", "optimizer", "seed"]]
            .dropna()
            .itertuples(index=False)
        }
        missing_raw = sorted(expected_groups - observed_raw)
        if missing_raw:
            raise RuntimeError(f"Missing raw method/seed groups: {missing_raw}")

        observed_valid = {
            (str(row.dataset), str(row.optimizer), int(row.seed))
            for row in valid[["dataset", "optimizer", "seed"]]
            .dropna()
            .itertuples(index=False)
        }
        missing_valid = sorted(expected_groups - observed_valid)
        if missing_valid:
            raise RuntimeError(
                "At least one method/seed group has no publication-valid candidate: "
                f"{missing_valid}"
            )

    return raw.reset_index(drop=True), valid


def finite_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.zeros(0, dtype=bool)

    keep = np.ones(len(values), dtype=bool)
    for index in range(len(values)):
        dominates = np.all(values <= values[index], axis=1) & np.any(
            values < values[index],
            axis=1,
        )
        dominates[index] = False
        if np.any(dominates):
            keep[index] = False
    return keep


def test_objectives(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        (
            1.0 - finite_numeric(frame["test_quality"]),
            finite_numeric(frame["test_cost"]),
            finite_numeric(frame["test_fairness"]),
        )
    )


def attained_quality(
    frame: pd.DataFrame,
    *,
    x_grid: np.ndarray,
    x_column: str,
) -> np.ndarray:
    x = finite_numeric(frame[x_column])
    quality = finite_numeric(frame["test_quality"])
    finite = np.isfinite(x) & np.isfinite(quality)
    x = x[finite]
    quality = quality[finite]

    output = np.full(len(x_grid), np.nan)
    for index, threshold in enumerate(x_grid):
        eligible = quality[x <= threshold]
        if len(eligible):
            output[index] = float(np.max(eligible))
    return output


def seed_attainment(
    data: pd.DataFrame,
    *,
    method: str,
    seeds: Sequence[int],
    x_grid: np.ndarray,
    x_column: str,
) -> np.ndarray | None:
    method_rows = data[data["optimizer"] == method]
    curves: list[np.ndarray] = []

    for seed in seeds:
        group = method_rows[
            pd.to_numeric(method_rows["seed"], errors="coerce") == int(seed)
        ].copy()
        if group.empty:
            continue

        objectives = test_objectives(group)
        finite = np.all(np.isfinite(objectives), axis=1)
        group = group.loc[finite].reset_index(drop=True)
        objectives = objectives[finite]
        if group.empty:
            continue

        front = group.loc[pareto_mask_minimize(objectives)].reset_index(drop=True)
        curves.append(
            attained_quality(
                front,
                x_grid=x_grid,
                x_column=x_column,
            )
        )

    if not curves:
        return None
    return np.vstack(curves)


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_all_datasets(
    data: pd.DataFrame,
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    outdir: Path,
    x_column: str,
    x_label: str,
    filename: str,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.0 * len(datasets), 4.4),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    handles: dict[str, object] = {}

    for ax, dataset in zip(axes_flat, datasets):
        subset = data[data["dataset"] == dataset].copy()
        if subset.empty:
            ax.set_visible(False)
            continue

        x = finite_numeric(subset[x_column])
        x = x[np.isfinite(x)]
        if not len(x):
            ax.set_visible(False)
            continue

        minimum = float(np.min(x))
        maximum = float(np.max(x))
        span = max(maximum - minimum, 1e-6)
        padding = 0.04 * span
        x_grid = np.linspace(minimum - padding, maximum + padding, 500)

        for method in METHOD_ORDER:
            matrix = seed_attainment(
                subset,
                method=method,
                seeds=seeds,
                x_grid=x_grid,
                x_column=x_column,
            )
            if matrix is None:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                median = np.nanmedian(matrix, axis=0)
                lower = np.nanmin(matrix, axis=0)
                upper = np.nanmax(matrix, axis=0)

            finite_curve = np.isfinite(median)
            if not finite_curve.any():
                continue

            line = ax.step(
                x_grid[finite_curve],
                median[finite_curve],
                where="post",
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                marker=MARKERS[method],
                markevery=max(1, int(finite_curve.sum() / 8)),
                linewidth=2,
                markersize=4.5,
                label=DISPLAY_NAME[method],
            )[0]
            ax.fill_between(
                x_grid[finite_curve],
                lower[finite_curve],
                upper[finite_curve],
                step="post",
                color=COLORS[method],
                alpha=0.14,
            )
            handles.setdefault(method, line)

        ax.set_title(f"{DATASET_TITLE[dataset]} — Initial vs 5M")
        ax.set_xlabel(x_label)
        ax.set_ylabel(QUALITY_LABEL[dataset])
        ax.grid(True, alpha=0.25)

        if x_column == "test_fairness":
            ax.text(
                0.98,
                0.02,
                UNFAIRNESS_LABEL[dataset],
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                alpha=0.70,
            )

    ordered_methods = [method for method in METHOD_ORDER if method in handles]
    if ordered_methods:
        fig.legend(
            [handles[method] for method in ordered_methods],
            [DISPLAY_NAME[method] for method in ordered_methods],
            loc="upper center",
            ncol=len(ordered_methods),
            frameon=False,
            bbox_to_anchor=(0.5, 1.08),
        )

    fig.text(
        0.5,
        -0.01,
        "Lines: median attainment across seeds; bands: seedwise minimum to maximum. "
        "Initial Instructions are the exact budget-zero starting populations.",
        ha="center",
        va="top",
        fontsize=8,
    )
    save_figure(fig, outdir, filename)


def policy_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (dataset, method, seed), group in data.groupby(
        ["dataset", "optimizer", "seed"],
        sort=True,
    ):
        work = group.copy()
        for column in ("test_quality", "test_cost", "test_fairness"):
            work[column] = pd.to_numeric(work[column], errors="coerce")
        work = work.dropna(
            subset=("test_quality", "test_cost", "test_fairness")
        )
        if work.empty:
            continue

        best = work.sort_values(
            ["test_quality", "test_fairness", "test_cost"],
            ascending=[False, True, True],
            kind="mergesort",
        ).iloc[0]

        quality_floor = 0.90 * float(work["test_quality"].max())
        robust = (
            work[work["test_quality"] >= quality_floor]
            .sort_values(
                ["test_fairness", "test_cost", "test_quality"],
                ascending=[True, True, False],
                kind="mergesort",
            )
            .iloc[0]
        )

        for policy, selected, floor in (
            ("best_quality", best, np.nan),
            ("lowest_unfairness_within_90pct_best_quality", robust, quality_floor),
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "method_display": DISPLAY_NAME.get(method, method),
                    "seed": int(seed),
                    "policy": policy,
                    "quality_floor": floor,
                    "test_quality": float(selected["test_quality"]),
                    "test_cost": float(selected["test_cost"]),
                    "test_unfairness": float(selected["test_fairness"]),
                    "prompt_id": selected.get("prompt_id", ""),
                    "source_kind": selected.get("source_kind", ""),
                    "actual_budget_tokens": selected.get(
                        "actual_budget_tokens",
                        np.nan,
                    ),
                }
            )

    return pd.DataFrame(rows)


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in out.columns
    ]
    return out


def write_tables(
    *,
    raw: pd.DataFrame,
    valid: pd.DataFrame,
    outdir: Path,
) -> None:
    raw.to_parquet(
        outdir / "fairness_qwen3_5m_v2_with_initial_all_raw_rows.parquet",
        index=False,
    )
    raw.to_csv(
        outdir / "fairness_qwen3_5m_v2_with_initial_all_raw_rows.csv",
        index=False,
    )
    valid.to_parquet(
        outdir / "fairness_qwen3_5m_v2_with_initial_publication_rows.parquet",
        index=False,
    )
    valid.to_csv(
        outdir / "fairness_qwen3_5m_v2_with_initial_publication_rows.csv",
        index=False,
    )

    counts = (
        raw.groupby(["dataset", "optimizer", "seed"], sort=True)
        .agg(
            raw_rows=("publication_valid", "size"),
            publication_valid_rows=("publication_valid", "sum"),
        )
        .reset_index()
    )
    counts["invalid_rows"] = (
        counts["raw_rows"] - counts["publication_valid_rows"]
    )
    counts.to_csv(
        outdir / "fairness_qwen3_5m_v2_with_initial_row_counts.csv",
        index=False,
    )
    (
        outdir / "fairness_qwen3_5m_v2_with_initial_row_counts.md"
    ).write_text(
        counts.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )

    selected = policy_rows(valid)
    selected.to_csv(
        outdir / "fairness_qwen3_5m_v2_with_initial_operating_points_by_seed.csv",
        index=False,
    )
    (
        outdir / "fairness_qwen3_5m_v2_with_initial_operating_points_by_seed.md"
    ).write_text(
        selected.to_markdown(index=False, floatfmt=".5f") + "\n",
        encoding="utf-8",
    )

    summary = flatten_columns(
        selected.groupby(
            ["dataset", "method_display", "policy"],
            sort=True,
        )[["test_quality", "test_cost", "test_unfairness"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(
        outdir / "fairness_qwen3_5m_v2_initial_plus_methods_summary.csv",
        index=False,
    )
    (
        outdir / "fairness_qwen3_5m_v2_initial_plus_methods_summary.md"
    ).write_text(
        summary.to_markdown(index=False, floatfmt=".5f") + "\n",
        encoding="utf-8",
    )

    print("\nPublication-valid row counts")
    print(counts.to_string(index=False))
    print("\nInitial plus 5M method summary")
    print(summary.to_string(index=False))


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def write_manifest(
    *,
    outdir: Path,
    results_root: Path,
    model: str,
    datasets: Sequence[str],
    optimizers: Sequence[str],
    seeds: Sequence[int],
    budget: int,
    expected_initial_prompts: int,
    raw_rows: int,
    valid_rows: int,
) -> None:
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "results_root": str(results_root.resolve()),
        "model": model,
        "datasets": list(datasets),
        "optimizers": list(optimizers),
        "methods": list(METHOD_ORDER),
        "seeds": list(map(int, seeds)),
        "optimized_budget": int(budget),
        "initial_budget": 0,
        "expected_initial_prompts_per_dataset_seed": int(
            expected_initial_prompts
        ),
        "raw_rows": int(raw_rows),
        "publication_valid_rows": int(valid_rows),
        "validity_rules": {
            "bbq": "test_fairness_ready and test coverage_valid",
            "bias_in_bios": "dev_fairness_ready and test_fairness_ready",
            "civil_comments": "dev_fairness_ready and test_fairness_ready",
        },
        "cost_definition": (
            "0.11 * mean input tokens + 0.41 * mean output tokens; "
            "not monetary cost"
        ),
    }
    (outdir / "analysis_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_readme(
    *,
    outdir: Path,
    results_root: Path,
) -> None:
    content = f"""# Final 5M v2 empirical attainment with Initial Instructions

Generated by:

```bash
python -m analysis.make_final_5m_attainment_with_initial_v2 --strict
```

Results root: `{results_root}`

## Methods

- Initial Instructions: the exact 12-prompt budget-zero population sampled for
  each dataset/seed in the final 5M runs.
- NSGA-II-PO-Fair: exact nominal-5M holdout candidate set.
- Tri-Fair: exact nominal-5M holdout candidate set.

## Figures

- `fairness_qwen3_5m_v2_attainment_quality_cost_all_datasets.*`
- `fairness_qwen3_5m_v2_attainment_quality_unfairness_all_datasets.*`

Lines are median empirical-attainment curves across seeds 42, 43, and 44.
Bands show the seedwise minimum-to-maximum range.

## Scientific rules

- Optimized rows are loaded only from the step selected by
  `eval_5000000.done.json`.
- BBQ requires statistically ready test fairness and valid disambiguated
  coverage.
- Bias-in-Bios and Civil Comments require statistically ready development and
  test fairness.
- Native utility penalties remain in the unfairness objectives; penalized
  candidates are not removed after optimization.
- Cost is the weighted mean-token objective, not a dollar or GPU charge.
"""
    (outdir / "README.md").write_text(content, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="results/tri_fair_v2_2",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis/output/final_5m_v2_with_initial",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASET_ORDER),
    )
    parser.add_argument(
        "--optimizers",
        default=",".join(OPTIMIZER_ORDER),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(map(str, DEFAULT_SEEDS)),
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=FINAL_BUDGET,
    )
    parser.add_argument(
        "--expected-initial-prompts",
        type=int,
        default=EXPECTED_INITIAL_PROMPTS,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require the complete 27-group method/dataset/seed grid.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_style()

    results_root = Path(args.results_root).expanduser().resolve()
    outdir = Path(args.out_dir).expanduser()
    datasets = parse_csv(args.datasets, cast=str)
    optimizers = parse_csv(args.optimizers, cast=str)
    seeds = parse_csv(args.seeds, cast=int)

    unknown_datasets = sorted(set(datasets) - set(DATASET_ORDER))
    if unknown_datasets:
        raise ValueError(f"Unsupported datasets: {unknown_datasets}")
    unknown_optimizers = sorted(set(optimizers) - set(OPTIMIZER_ORDER))
    if unknown_optimizers:
        raise ValueError(f"Unsupported optimizers: {unknown_optimizers}")
    if args.budget <= 0:
        raise ValueError("--budget must be positive")
    if args.expected_initial_prompts <= 0:
        raise ValueError("--expected-initial-prompts must be positive")

    raw, valid = load_all_rows(
        results_root=results_root,
        model=args.model,
        datasets=datasets,
        optimizers=optimizers,
        seeds=seeds,
        budget=args.budget,
        expected_initial_prompts=args.expected_initial_prompts,
        strict=bool(args.strict),
    )

    # Do not create a success-looking directory until the complete grid and
    # publication-valid rows have passed validation.
    outdir.mkdir(parents=True, exist_ok=True)

    plot_all_datasets(
        valid,
        datasets=datasets,
        seeds=seeds,
        outdir=outdir,
        x_column="test_cost",
        x_label=COST_LABEL,
        filename="fairness_qwen3_5m_v2_attainment_quality_cost_all_datasets",
    )
    plot_all_datasets(
        valid,
        datasets=datasets,
        seeds=seeds,
        outdir=outdir,
        x_column="test_fairness",
        x_label="Test Unfairness ↓",
        filename="fairness_qwen3_5m_v2_attainment_quality_unfairness_all_datasets",
    )

    write_tables(
        raw=raw,
        valid=valid,
        outdir=outdir,
    )
    write_manifest(
        outdir=outdir,
        results_root=results_root,
        model=args.model,
        datasets=datasets,
        optimizers=optimizers,
        seeds=seeds,
        budget=args.budget,
        expected_initial_prompts=args.expected_initial_prompts,
        raw_rows=len(raw),
        valid_rows=len(valid),
    )
    write_readme(
        outdir=outdir,
        results_root=results_root,
    )

    generated = sorted(path.name for path in outdir.iterdir() if path.is_file())
    print("\nGenerated files")
    for name in generated:
        print(f"  {name}")
    print(f"\nOutput directory:\n{outdir.resolve()}")


if __name__ == "__main__":
    main()
