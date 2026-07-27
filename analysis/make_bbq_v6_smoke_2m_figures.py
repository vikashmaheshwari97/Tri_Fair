"""Generate BBQ Tri-Fair-v6 2M smoke-run publication figures.

The script compares:

* Tri-Fair-v6
* NSGA-II-PO-Fair

It uses the completed development histories and all-step held-out evaluations
from the matched 2M BBQ smoke run.  It does not use balanced operating points.

All main scalar policies are selected on development data:

* accuracy-first: highest development accuracy;
* cost-first: lowest development cost;
* fairness-first: lowest development unfairness;
* high-accuracy cost/fairness: lowest cost or unfairness subject to a
  development-accuracy threshold.

The corresponding held-out values are then reported for the same prompts.
Independent held-out archive extremes are generated only as clearly labelled
diagnostics.

Every figure uses the term ``Accuracy`` rather than ``Test Quality``.

Default Rocket command
----------------------

    PYTHONPATH="$PWD" python -m analysis.make_bbq_v6_smoke_2m_figures --strict

Explicit command
----------------

    PYTHONPATH="$PWD" python -m analysis.make_bbq_v6_smoke_2m_figures \
      --results-root \
        results/tri_fair_v6_smoke_2m_matched1/qwen-3-30b/bbq \
      --output-dir \
        analysis/output/tri_fair_v6_smoke_2m/publication_figures/bbq \
      --accuracy-threshold 0.90 \
      --strict

The output directory contains PNG, PDF and SVG figures plus CSV/Markdown
tables and a provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL = "qwen-3-30b"
DATASET = "bbq"
SEED = 62
BUDGET = 2_000_000
COST_BOUND = 100.0
REFERENCE_POINT = np.asarray((1.1, 1.1, 1.1), dtype=float)

METHOD_ORDER = ("Tri-Fair-v6", "NSGAII-PO-Fair")
DISPLAY_NAME = {
    "Tri-Fair-v6": "Tri-Fair v6",
    "NSGAII-PO-Fair": "NSGA-II-PO-Fair",
}
COLORS = {
    "Tri-Fair-v6": "black",
    "NSGAII-PO-Fair": "#E69F00",
}
MARKERS = {
    "Tri-Fair-v6": "o",
    "NSGAII-PO-Fair": "s",
}
LINESTYLES = {
    "Tri-Fair-v6": "-",
    "NSGAII-PO-Fair": "--",
}

POLICY_DISPLAY = {
    "accuracy_first": "Accuracy-first",
    "cost_first": "Cost-first",
    "fairness_first": "Fairness-first",
    "high_accuracy_cost": "Lowest cost at accuracy threshold",
    "high_accuracy_fairness": "Lowest unfairness at accuracy threshold",
}


@dataclass(frozen=True)
class RunFiles:
    optimizer: str
    run_dir: Path
    evaluation_path: Path
    step_path: Path
    summary_path: Path
    budget_path: Path
    args_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=(
            "results/tri_fair_v6_smoke_2m_matched1/"
            "qwen-3-30b/bbq"
        ),
        help="Directory containing the two optimizer run trees.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "analysis/output/tri_fair_v6_smoke_2m/"
            "publication_figures/bbq"
        ),
        help="Destination for figures and tables.",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.90,
        help="Development-accuracy constraint for high-accuracy policies.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf,svg",
        help="Comma-separated output formats.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on missing files, invalid run matrices or empty final fronts.",
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


def parse_formats(raw: str) -> tuple[str, ...]:
    formats = tuple(
        dict.fromkeys(
            piece.strip().lower()
            for piece in str(raw).split(",")
            if piece.strip()
        )
    )
    invalid = sorted(set(formats) - {"png", "pdf", "svg"})
    if invalid:
        raise ValueError(f"Unsupported output formats: {invalid}")
    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_markdown(
    frame: pd.DataFrame,
    path: Path,
    *,
    floatfmt: str = ".4f",
) -> None:
    try:
        text = frame.to_markdown(index=False, floatfmt=floatfmt) + "\n"
    except ImportError:
        text = "```text\n" + frame.to_string(index=False) + "\n```\n"
    path.write_text(text, encoding="utf-8")


def locate_run_files(
    results_root: Path,
    optimizer: str,
    *,
    strict: bool,
) -> RunFiles:
    method_root = results_root / optimizer / f"seed{SEED}"
    evaluation_files = sorted(method_root.rglob("eval_checkpoints.parquet"))
    step_files = sorted(method_root.rglob("step_results.parquet"))

    if len(evaluation_files) != 1 or len(step_files) != 1:
        message = (
            f"{optimizer}: expected one eval_checkpoints.parquet and one "
            f"step_results.parquet beneath {method_root}; found "
            f"{len(evaluation_files)} and {len(step_files)}"
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message)

    if not evaluation_files or not step_files:
        raise FileNotFoundError(message)

    run_dir = evaluation_files[-1].parent
    required = {
        "run_summary.json": run_dir / "run_summary.json",
        "budget_summary.json": run_dir / "budget_summary.json",
        "args.json": run_dir / "args.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        message = f"{optimizer}: missing run files: {missing}"
        if strict:
            raise FileNotFoundError(message)
        warnings.warn(message)

    return RunFiles(
        optimizer=optimizer,
        run_dir=run_dir,
        evaluation_path=evaluation_files[-1],
        step_path=step_files[-1],
        summary_path=required["run_summary.json"],
        budget_path=required["budget_summary.json"],
        args_path=required["args.json"],
    )


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


def attach_publication_validity(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()

    dev_ready = (
        output["dev_fairness_ready"].fillna(False).astype(bool)
        if "dev_fairness_ready" in output
        else pd.Series(True, index=output.index)
    )
    heldout_ready = (
        output["test_fairness_ready"].fillna(False).astype(bool)
        if "test_fairness_ready" in output
        else pd.Series(True, index=output.index)
    )

    output["dev_coverage_valid"] = True
    output["heldout_coverage_valid"] = True

    dev_diag = "dev_fairness_diagnostics_json"
    test_diag = "test_fairness_diagnostics_json"

    if dev_diag in output:
        output["dev_coverage_valid"] = output[dev_diag].map(
            lambda value: bool(
                _json_object(value).get("coverage_valid", True)
            )
        )
    if test_diag in output:
        output["heldout_coverage_valid"] = output[test_diag].map(
            lambda value: bool(
                _json_object(value).get("coverage_valid", True)
            )
        )

    output["publication_valid"] = (
        dev_ready
        & heldout_ready
        & output["dev_coverage_valid"].astype(bool)
        & output["heldout_coverage_valid"].astype(bool)
    )
    return output


def load_evaluations(run: RunFiles) -> pd.DataFrame:
    frame = pd.read_parquet(run.evaluation_path).copy()
    require_columns(
        frame,
        (
            "budget_checkpoint",
            "actual_budget_tokens",
            "chosen_step",
            "dev_quality",
            "dev_cost",
            "dev_fairness",
            "test_quality",
            "test_cost",
            "test_fairness",
        ),
        str(run.evaluation_path),
    )

    frame["optimizer"] = run.optimizer
    frame["seed"] = SEED
    frame["model"] = MODEL
    frame["dataset"] = DATASET
    frame["source_file"] = str(run.evaluation_path.resolve())

    for column in (
        "budget_checkpoint",
        "actual_budget_tokens",
        "chosen_step",
        "dev_quality",
        "dev_cost",
        "dev_fairness",
        "test_quality",
        "test_cost",
        "test_fairness",
        "test_input_tokens",
        "test_output_tokens",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return attach_publication_validity(frame)


def load_all_runs(
    results_root: Path,
    *,
    strict: bool,
) -> tuple[pd.DataFrame, dict[str, RunFiles], dict[str, dict[str, object]]]:
    run_files: dict[str, RunFiles] = {}
    summaries: dict[str, dict[str, object]] = {}
    frames: list[pd.DataFrame] = []

    for optimizer in METHOD_ORDER:
        run = locate_run_files(results_root, optimizer, strict=strict)
        run_files[optimizer] = run
        frames.append(load_evaluations(run))
        summaries[optimizer] = {
            "run": read_json(run.summary_path),
            "budget": read_json(run.budget_path),
            "args": read_json(run.args_path),
        }

    combined = pd.concat(frames, ignore_index=True, sort=False)
    observed = set(combined["optimizer"].astype(str))
    missing = sorted(set(METHOD_ORDER) - observed)
    if missing:
        raise RuntimeError(f"Missing optimizers from evaluation table: {missing}")

    return combined, run_files, summaries


def final_checkpoint_rows(
    frame: pd.DataFrame,
    optimizer: str,
) -> pd.DataFrame:
    method = frame[frame["optimizer"] == optimizer].copy()
    if method.empty:
        return method
    actual = pd.to_numeric(method["actual_budget_tokens"], errors="coerce")
    final_tokens = int(actual.max())
    return method[actual == final_tokens].copy().reset_index(drop=True)


def _sort_policy(
    frame: pd.DataFrame,
    policy: str,
    *,
    accuracy_threshold: float,
) -> tuple[pd.DataFrame, bool]:
    work = frame.copy()
    threshold_met = True

    if policy == "accuracy_first":
        columns = ["dev_quality", "dev_cost", "dev_fairness"]
        ascending = [False, True, True]
    elif policy == "cost_first":
        columns = ["dev_cost", "dev_fairness", "dev_quality"]
        ascending = [True, True, False]
    elif policy == "fairness_first":
        columns = ["dev_fairness", "dev_cost", "dev_quality"]
        ascending = [True, True, False]
    elif policy in {"high_accuracy_cost", "high_accuracy_fairness"}:
        eligible = work[work["dev_quality"] >= accuracy_threshold].copy()
        threshold_met = not eligible.empty
        if threshold_met:
            work = eligible
        if policy == "high_accuracy_cost":
            columns = ["dev_cost", "dev_fairness", "dev_quality"]
            ascending = [True, True, False]
        else:
            columns = ["dev_fairness", "dev_cost", "dev_quality"]
            ascending = [True, True, False]
    else:
        raise ValueError(f"Unknown policy: {policy}")

    return (
        work.sort_values(
            columns,
            ascending=ascending,
            kind="mergesort",
        ),
        threshold_met,
    )


def select_development_policy(
    frame: pd.DataFrame,
    policy: str,
    *,
    accuracy_threshold: float,
) -> pd.Series:
    valid = frame[frame["publication_valid"]].copy()
    valid = valid.dropna(
        subset=(
            "dev_quality",
            "dev_cost",
            "dev_fairness",
            "test_quality",
            "test_cost",
            "test_fairness",
        )
    )
    if valid.empty:
        raise RuntimeError(
            f"No valid candidates are available for policy {policy}"
        )

    ordered, threshold_met = _sort_policy(
        valid,
        policy,
        accuracy_threshold=accuracy_threshold,
    )
    selected = ordered.iloc[0].copy()
    selected["policy"] = policy
    selected["accuracy_threshold"] = float(accuracy_threshold)
    selected["threshold_met"] = bool(threshold_met)
    return selected


def build_policy_trajectory(
    evaluations: pd.DataFrame,
    *,
    accuracy_threshold: float,
) -> pd.DataFrame:
    policies = (
        "accuracy_first",
        "cost_first",
        "fairness_first",
        "high_accuracy_cost",
        "high_accuracy_fairness",
    )
    rows: list[dict[str, object]] = []

    for optimizer in METHOD_ORDER:
        method = evaluations[evaluations["optimizer"] == optimizer].copy()
        checkpoints = (
            method[
                ["budget_checkpoint", "actual_budget_tokens", "chosen_step"]
            ]
            .drop_duplicates()
            .sort_values(
                ["actual_budget_tokens", "budget_checkpoint", "chosen_step"],
                kind="mergesort",
            )
        )

        for state in checkpoints.itertuples(index=False):
            candidates = method[
                (method["budget_checkpoint"] == state.budget_checkpoint)
                & (method["chosen_step"] == state.chosen_step)
            ].copy()
            for policy in policies:
                try:
                    selected = select_development_policy(
                        candidates,
                        policy,
                        accuracy_threshold=accuracy_threshold,
                    )
                except RuntimeError:
                    continue

                row = {
                    "optimizer": optimizer,
                    "method": DISPLAY_NAME[optimizer],
                    "policy": policy,
                    "policy_name": POLICY_DISPLAY[policy],
                    "budget_checkpoint": int(state.budget_checkpoint),
                    "actual_budget_tokens": int(state.actual_budget_tokens),
                    "chosen_step": int(state.chosen_step),
                    "prompt_id": selected.get("prompt_id", ""),
                    "threshold_met": bool(selected["threshold_met"]),
                    "accuracy_threshold": float(accuracy_threshold),
                }
                for split in ("dev", "test"):
                    for metric in ("quality", "cost", "fairness"):
                        row[f"{split}_{metric}"] = float(
                            selected[f"{split}_{metric}"]
                        )
                rows.append(row)

    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError("No development-selected trajectory rows were built")
    return output


def objective_matrix(
    frame: pd.DataFrame,
    *,
    split: str,
) -> np.ndarray:
    quality = pd.to_numeric(
        frame[f"{split}_quality"], errors="coerce"
    ).to_numpy(dtype=float)
    cost = pd.to_numeric(
        frame[f"{split}_cost"], errors="coerce"
    ).to_numpy(dtype=float)
    fairness = pd.to_numeric(
        frame[f"{split}_fairness"], errors="coerce"
    ).to_numpy(dtype=float)

    return np.column_stack(
        (
            1.0 - quality,
            cost / COST_BOUND,
            fairness,
        )
    )


def pareto_mask_minimise(values: np.ndarray) -> np.ndarray:
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


def pareto_rows(
    frame: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    matrix = objective_matrix(frame, split=split)
    valid = np.all(np.isfinite(matrix), axis=1)
    subset = frame.loc[valid].reset_index(drop=True)
    matrix = matrix[valid]
    if subset.empty:
        return subset
    return subset.loc[pareto_mask_minimise(matrix)].reset_index(drop=True)


def exact_hypervolume_3d(
    values: np.ndarray,
    *,
    reference: np.ndarray = REFERENCE_POINT,
) -> float:
    """Exact cell-decomposition hypervolume for small minimisation fronts."""
    points = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)

    valid = (
        np.all(np.isfinite(points), axis=1)
        & np.all(points <= reference, axis=1)
    )
    points = points[valid]
    if not len(points):
        return 0.0

    points = points[pareto_mask_minimise(points)]
    coordinates = [
        np.unique(np.append(points[:, dimension], reference[dimension]))
        for dimension in range(3)
    ]

    volume = 0.0
    for x0, x1 in zip(coordinates[0][:-1], coordinates[0][1:]):
        if x1 <= x0:
            continue
        for y0, y1 in zip(coordinates[1][:-1], coordinates[1][1:]):
            if y1 <= y0:
                continue
            for z0, z1 in zip(coordinates[2][:-1], coordinates[2][1:]):
                if z1 <= z0:
                    continue
                lower = np.asarray((x0, y0, z0), dtype=float)
                if np.any(np.all(points <= lower, axis=1)):
                    volume += float((x1 - x0) * (y1 - y0) * (z1 - z0))
    return volume


def simplex_weights(resolution: int = 20) -> np.ndarray:
    weights: list[tuple[float, float, float]] = []
    for first in range(resolution + 1):
        for second in range(resolution + 1 - first):
            third = resolution - first - second
            weights.append(
                (
                    first / resolution,
                    second / resolution,
                    third / resolution,
                )
            )
    return np.asarray(weights, dtype=float)


def noisy_r2(values: np.ndarray) -> float:
    """Deterministic normalised Tchebycheff R2; lower is better."""
    points = np.asarray(values, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        return float("nan")

    points = points[pareto_mask_minimise(points)]
    ideal = np.zeros(points.shape[1], dtype=float)
    weights = simplex_weights(20)
    utilities = []

    for weight in weights:
        safe_weight = np.maximum(weight, 1e-6)
        per_point = np.max(
            safe_weight[None, :] * np.abs(points - ideal),
            axis=1,
        )
        utilities.append(float(np.min(per_point)))

    return float(np.mean(utilities))


def build_mo_trajectory(evaluations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for optimizer in METHOD_ORDER:
        method = evaluations[evaluations["optimizer"] == optimizer].copy()
        groups = method.groupby(
            ["budget_checkpoint", "actual_budget_tokens", "chosen_step"],
            sort=True,
        )

        for (checkpoint, actual, step), candidates in groups:
            valid = candidates[candidates["publication_valid"]].copy()
            valid = valid.dropna(
                subset=(
                    "dev_quality",
                    "dev_cost",
                    "dev_fairness",
                    "test_quality",
                    "test_cost",
                    "test_fairness",
                )
            )
            if valid.empty:
                continue

            dev_values = objective_matrix(valid, split="dev")
            heldout_values = objective_matrix(valid, split="test")
            dev_hv = exact_hypervolume_3d(dev_values)
            heldout_hv = exact_hypervolume_3d(heldout_values)

            rows.append(
                {
                    "optimizer": optimizer,
                    "method": DISPLAY_NAME[optimizer],
                    "budget_checkpoint": int(checkpoint),
                    "actual_budget_tokens": int(actual),
                    "chosen_step": int(step),
                    "n_candidates": int(len(valid)),
                    "dev_nr2": noisy_r2(dev_values),
                    "heldout_nr2": noisy_r2(heldout_values),
                    "dev_hv": dev_hv,
                    "heldout_hv": heldout_hv,
                    "hv_generalization_gap": abs(dev_hv - heldout_hv),
                }
            )

    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError("No multi-objective trajectory rows were built")
    return output


def count_few_shots(value: object) -> int:
    if isinstance(value, list):
        return len(value)
    parsed = _json_object(value)
    if parsed:
        return len(parsed)
    try:
        loaded = json.loads(str(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0
    return len(loaded) if isinstance(loaded, list) else 0


def infer_model_token_costs() -> tuple[float, float]:
    try:
        from src.config.model_configs import ALL_MODELS

        config = ALL_MODELS[MODEL]
        return float(config.input_costs), float(config.output_costs)
    except Exception:
        warnings.warn(
            "Could not import model token weights; using equal input/output weights "
            "for the few-shot output-cost-share diagnostic"
        )
        return 1.0, 1.0


def attach_few_shot_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "few_shots_json" in output:
        output["few_shot_count"] = output["few_shots_json"].map(
            count_few_shots
        )
    elif "few_shots" in output:
        output["few_shot_count"] = output["few_shots"].map(
            count_few_shots
        )
    else:
        output["few_shot_count"] = 0

    input_weight, output_weight = infer_model_token_costs()
    if {
        "test_input_tokens",
        "test_output_tokens",
    }.issubset(output.columns):
        input_component = (
            input_weight
            * pd.to_numeric(
                output["test_input_tokens"], errors="coerce"
            )
        )
        output_component = (
            output_weight
            * pd.to_numeric(
                output["test_output_tokens"], errors="coerce"
            )
        )
        total = input_component + output_component
        output["heldout_output_cost_share"] = np.divide(
            output_component,
            total,
            out=np.full(len(output), np.nan, dtype=float),
            where=total.to_numpy(dtype=float) > 0,
        )
    else:
        output["heldout_output_cost_share"] = np.nan

    return output


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
) -> None:
    for extension in formats:
        kwargs: dict[str, object] = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = 400
        figure.savefig(output_dir / f"{stem}.{extension}", **kwargs)
    plt.close(figure)


def plot_accuracy_cost_unfairness_trajectory(
    trajectory: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 7.2),
        constrained_layout=True,
        sharex="col",
    )

    specifications = (
        (
            "accuracy_first",
            "dev_quality",
            "test_quality",
            "Best Development Accuracy ↑",
            "Held-out Accuracy of\nDevelopment Accuracy Champion ↑",
            "Accuracy",
        ),
        (
            "cost_first",
            "dev_cost",
            "test_cost",
            "Lowest Development Cost ↓",
            "Held-out Cost of\nDevelopment Cost Champion ↓",
            "Weighted Mean-Token Cost",
        ),
        (
            "fairness_first",
            "dev_fairness",
            "test_fairness",
            "Lowest Development Unfairness ↓",
            "Held-out Unfairness of\nDevelopment Fairness Champion ↓",
            "Unfairness",
        ),
    )

    for column_index, (
        policy,
        dev_column,
        heldout_column,
        dev_title,
        heldout_title,
        ylabel,
    ) in enumerate(specifications):
        for optimizer in METHOD_ORDER:
            group = trajectory[
                (trajectory["optimizer"] == optimizer)
                & (trajectory["policy"] == policy)
            ].sort_values("actual_budget_tokens")

            x = group["actual_budget_tokens"].to_numpy(dtype=float) / 1_000_000.0
            axes[0, column_index].plot(
                x,
                group[dev_column],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                linestyle=LINESTYLES[optimizer],
                linewidth=2.0,
                markersize=5,
                label=DISPLAY_NAME[optimizer],
            )
            axes[1, column_index].plot(
                x,
                group[heldout_column],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                linestyle=LINESTYLES[optimizer],
                linewidth=2.0,
                markersize=5,
                label=DISPLAY_NAME[optimizer],
            )

        axes[0, column_index].set_title(dev_title)
        axes[1, column_index].set_title(heldout_title)
        axes[0, column_index].set_ylabel(ylabel)
        axes[1, column_index].set_ylabel(ylabel)
        axes[1, column_index].set_xlabel(
            "Cumulative Downstream Tokens [×10⁶]"
        )
        axes[0, column_index].grid(True, alpha=0.25)
        axes[1, column_index].grid(True, alpha=0.25)
        axes[0, column_index].set_xlim(0.0, BUDGET / 1_000_000.0)
        axes[1, column_index].set_xlim(0.0, BUDGET / 1_000_000.0)

    axes[0, 0].legend(frameon=False, loc="best")
    figure.suptitle(
        "BBQ — Qwen-3-30B Tri-Fair v6 Matched 2M Smoke Run"
    )
    save_figure(
        figure,
        output_dir,
        "bbq_v6_smoke_2m_accuracy_cost_unfairness_trajectory",
        formats,
    )


def plot_final_development_heldout_policies(
    trajectory: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    final_rows = (
        trajectory.sort_values("actual_budget_tokens")
        .groupby(["optimizer", "policy"], as_index=False)
        .tail(1)
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.2),
        constrained_layout=True,
    )
    specifications = (
        (
            "accuracy_first",
            "dev_quality",
            "test_quality",
            "Accuracy-first Prompt",
            "Accuracy",
        ),
        (
            "cost_first",
            "dev_cost",
            "test_cost",
            "Cost-first Prompt",
            "Weighted Mean-Token Cost",
        ),
        (
            "fairness_first",
            "dev_fairness",
            "test_fairness",
            "Fairness-first Prompt",
            "Unfairness",
        ),
    )

    x = np.arange(len(METHOD_ORDER), dtype=float)
    width = 0.32

    for axis, (
        policy,
        dev_column,
        heldout_column,
        title,
        ylabel,
    ) in zip(axes, specifications):
        selected = final_rows[final_rows["policy"] == policy]
        dev_values = [
            float(
                selected.loc[
                    selected["optimizer"] == optimizer,
                    dev_column,
                ].iloc[0]
            )
            for optimizer in METHOD_ORDER
        ]
        heldout_values = [
            float(
                selected.loc[
                    selected["optimizer"] == optimizer,
                    heldout_column,
                ].iloc[0]
            )
            for optimizer in METHOD_ORDER
        ]

        axis.bar(
            x - width / 2,
            dev_values,
            width,
            color=[COLORS[optimizer] for optimizer in METHOD_ORDER],
            alpha=0.45,
            label="Development",
        )
        axis.bar(
            x + width / 2,
            heldout_values,
            width,
            color=[COLORS[optimizer] for optimizer in METHOD_ORDER],
            alpha=0.92,
            hatch="//",
            label="Held-out",
        )
        axis.set_xticks(
            x,
            [DISPLAY_NAME[optimizer] for optimizer in METHOD_ORDER],
            rotation=7,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)

    axes[0].legend(frameon=False)
    figure.suptitle(
        "BBQ — Final Development-selected Prompts at the 2M Real States"
    )
    save_figure(
        figure,
        output_dir,
        "bbq_v6_smoke_2m_final_development_vs_heldout",
        formats,
    )


def plot_nr2(
    mo_trajectory: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.2),
        constrained_layout=True,
    )

    for optimizer in METHOD_ORDER:
        group = mo_trajectory[
            mo_trajectory["optimizer"] == optimizer
        ].sort_values("actual_budget_tokens")
        x = group["actual_budget_tokens"].to_numpy(dtype=float) / 1_000_000.0

        axes[0].plot(
            x,
            group["dev_nr2"],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            linestyle=LINESTYLES[optimizer],
            linewidth=2,
            markersize=5,
            label=DISPLAY_NAME[optimizer],
        )

        final = group.iloc[-1]
        axes[1].scatter(
            [DISPLAY_NAME[optimizer]],
            [final["heldout_nr2"]],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            s=90,
            label=DISPLAY_NAME[optimizer],
        )

    axes[0].set_title("A. Development nR2 Proxy ↓")
    axes[0].set_xlabel("Cumulative Downstream Tokens [×10⁶]")
    axes[0].set_ylabel("Development nR2 Proxy")
    axes[0].set_xlim(0.0, BUDGET / 1_000_000.0)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].set_title("B. Exact Final Held-out nR2 ↓")
    axes[1].set_ylabel("Held-out nR2")
    axes[1].grid(True, axis="y", alpha=0.25)

    figure.suptitle(
        "BBQ — Qwen-3-30B Tri-Fair v6 Matched 2M nR2"
    )
    save_figure(
        figure,
        output_dir,
        "bbq_v6_smoke_2m_nr2_development_heldout",
        formats,
    )


def plot_hypervolume_gap(
    mo_trajectory: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.4, 4.1),
        constrained_layout=True,
    )
    specifications = (
        ("dev_hv", "Development Hypervolume ↑", "Hypervolume"),
        ("heldout_hv", "Held-out Hypervolume ↑", "Hypervolume"),
        (
            "hv_generalization_gap",
            "Absolute HV Generalization Gap ↓",
            "Absolute Gap",
        ),
    )

    for axis, (metric, title, ylabel) in zip(axes, specifications):
        for optimizer in METHOD_ORDER:
            group = mo_trajectory[
                mo_trajectory["optimizer"] == optimizer
            ].sort_values("actual_budget_tokens")
            axis.plot(
                group["actual_budget_tokens"].to_numpy(dtype=float)
                / 1_000_000.0,
                group[metric],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                linestyle=LINESTYLES[optimizer],
                linewidth=2,
                markersize=5,
                label=DISPLAY_NAME[optimizer],
            )

        axis.set_title(title)
        axis.set_xlabel("Cumulative Downstream Tokens [×10⁶]")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.0, BUDGET / 1_000_000.0)
        axis.grid(True, alpha=0.25)

    axes[0].legend(frameon=False)
    figure.suptitle(
        "BBQ — Development and Held-out Pareto-set Generalization"
    )
    save_figure(
        figure,
        output_dir,
        "bbq_v6_smoke_2m_hypervolume_gap_trajectory",
        formats,
    )


def y_attained_at_x(
    frame: pd.DataFrame,
    x_grid: np.ndarray,
    x_column: str,
) -> np.ndarray:
    x = pd.to_numeric(frame[x_column], errors="coerce").to_numpy(dtype=float)
    accuracy = pd.to_numeric(
        frame["test_quality"], errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(accuracy)
    x = x[valid]
    accuracy = accuracy[valid]

    output = np.full(len(x_grid), np.nan)
    for index, target in enumerate(x_grid):
        eligible = accuracy[x <= target]
        if len(eligible):
            output[index] = float(np.max(eligible))
    return output


def plot_attainment(
    final_valid: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    *,
    x_column: str,
    xlabel: str,
    suffix: str,
) -> None:
    values = pd.to_numeric(
        final_valid[x_column], errors="coerce"
    ).dropna()
    if values.empty:
        return

    minimum = float(values.min())
    maximum = float(values.max())
    padding = 0.03 * max(maximum - minimum, 1e-6)
    x_grid = np.linspace(
        minimum - padding,
        maximum + padding,
        500,
    )

    figure, axis = plt.subplots(
        figsize=(6.8, 4.5),
        constrained_layout=True,
    )

    for optimizer in METHOD_ORDER:
        method = final_valid[
            final_valid["optimizer"] == optimizer
        ].copy()
        front = pareto_rows(method, split="test")
        curve = y_attained_at_x(front, x_grid, x_column)
        valid = np.isfinite(curve)

        axis.step(
            x_grid[valid],
            curve[valid],
            where="post",
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            linestyle=LINESTYLES[optimizer],
            markevery=max(1, int(valid.sum() / 8)),
            linewidth=2.1,
            markersize=4.5,
            label=DISPLAY_NAME[optimizer],
        )

    axis.set_title("BBQ — Final Held-out Empirical Attainment")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Held-out Accuracy ↑")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, loc="lower right")

    save_figure(
        figure,
        output_dir,
        f"bbq_v6_smoke_2m_attainment_accuracy_{suffix}",
        formats,
    )


def plot_cost_unfairness_pareto(
    final_valid: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    figure, axis = plt.subplots(
        figsize=(6.8, 4.7),
        constrained_layout=True,
    )

    for optimizer in METHOD_ORDER:
        method = final_valid[
            final_valid["optimizer"] == optimizer
        ].copy()
        front = pareto_rows(method, split="test")
        axis.scatter(
            front["test_cost"],
            front["test_fairness"],
            c=front["test_quality"],
            cmap="viridis",
            marker=MARKERS[optimizer],
            edgecolor=COLORS[optimizer],
            linewidth=1.0,
            alpha=0.82,
            s=64,
            label=DISPLAY_NAME[optimizer],
        )

        projection = np.column_stack(
            (
                pd.to_numeric(
                    front["test_cost"], errors="coerce"
                ).to_numpy(dtype=float),
                pd.to_numeric(
                    front["test_fairness"], errors="coerce"
                ).to_numpy(dtype=float),
            )
        )
        valid = np.all(np.isfinite(projection), axis=1)
        projected = front.loc[valid].reset_index(drop=True)
        projection = projection[valid]
        if len(projected):
            projected = projected.loc[
                pareto_mask_minimise(projection)
            ].sort_values("test_cost")
            axis.plot(
                projected["test_cost"],
                projected["test_fairness"],
                color=COLORS[optimizer],
                linestyle=LINESTYLES[optimizer],
                linewidth=2,
            )

    axis.set_title("BBQ — Final Held-out Cost–Unfairness Pareto Front")
    axis.set_xlabel("Weighted Mean-Token Cost ↓")
    axis.set_ylabel("Held-out Unfairness ↓")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)

    save_figure(
        figure,
        output_dir,
        "bbq_v6_smoke_2m_pareto_cost_unfairness",
        formats,
    )


def plot_three_dimensional_pareto(
    final_valid: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    figure = plt.figure(
        figsize=(7.5, 5.8),
        constrained_layout=True,
    )
    axis = figure.add_subplot(111, projection="3d")

    for optimizer in METHOD_ORDER:
        method = final_valid[
            final_valid["optimizer"] == optimizer
        ].copy()
        front = pareto_rows(method, split="test")
        axis.scatter(
            front["test_cost"],
            front["test_fairness"],
            front["test_quality"],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            alpha=0.82,
            s=48,
            label=DISPLAY_NAME[optimizer],
        )

    axis.set_title(
        "BBQ — Qwen-3-30B Final Held-out Pareto Fronts at 2M"
    )
    axis.set_xlabel("Weighted Mean-Token Cost ↓")
    axis.set_ylabel("Held-out Unfairness ↓")
    axis.set_zlabel("Held-out Accuracy ↑")
    axis.legend(frameon=False)

    save_figure(
        figure,
        output_dir,
        "bbq_v6_smoke_2m_heldout_pareto_3d",
        formats,
    )


def plot_few_shot_diagnostics(
    final_valid: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    data = attach_few_shot_diagnostics(final_valid)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.3),
        constrained_layout=True,
    )

    for optimizer in METHOD_ORDER:
        group = data[data["optimizer"] == optimizer]
        axes[0].scatter(
            group["few_shot_count"],
            100.0 * group["heldout_output_cost_share"],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            s=60,
            alpha=0.78,
            label=DISPLAY_NAME[optimizer],
        )
        axes[1].scatter(
            group["few_shot_count"],
            group["test_fairness"],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            s=60,
            alpha=0.78,
            label=DISPLAY_NAME[optimizer],
        )

    axes[0].set_title("A. Few-shot Count and Output-Cost Share")
    axes[0].set_xlabel("Number of Few-shot Examples")
    axes[0].set_ylabel("Held-out Output-Cost Share [%]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].set_title("B. Few-shot Count and Held-out Unfairness")
    axes[1].set_xlabel("Number of Few-shot Examples")
    axes[1].set_ylabel("Held-out Unfairness ↓")
    axes[1].grid(True, alpha=0.25)

    figure.suptitle(
        "BBQ — Final 2M Few-shot Diagnostics"
    )
    save_figure(
        figure,
        output_dir,
        "bbq_v6_smoke_2m_few_shot_diagnostics",
        formats,
    )


def build_final_summary(
    evaluations: pd.DataFrame,
    policy_trajectory: pd.DataFrame,
    mo_trajectory: pd.DataFrame,
    summaries: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for optimizer in METHOD_ORDER:
        final = final_checkpoint_rows(evaluations, optimizer)
        valid = final[final["publication_valid"]].copy()
        final_policies = (
            policy_trajectory[
                policy_trajectory["optimizer"] == optimizer
            ]
            .sort_values("actual_budget_tokens")
            .groupby("policy", as_index=False)
            .tail(1)
        )
        final_mo = (
            mo_trajectory[
                mo_trajectory["optimizer"] == optimizer
            ]
            .sort_values("actual_budget_tokens")
            .iloc[-1]
        )
        budget = summaries[optimizer]["budget"]

        row: dict[str, object] = {
            "Method": DISPLAY_NAME[optimizer],
            "Final real tokens": int(
                float(budget.get("actual_downstream_tokens", np.nan))
            ),
            "Budget utilization": float(
                budget.get("budget_utilization", np.nan)
            ),
            "Valid final candidates": int(len(valid)),
            "Independent best development accuracy ↑": float(
                valid["dev_quality"].max()
            ),
            "Independent lowest development cost ↓": float(
                valid["dev_cost"].min()
            ),
            "Independent lowest development unfairness ↓": float(
                valid["dev_fairness"].min()
            ),
            "Independent best held-out accuracy ↑": float(
                valid["test_quality"].max()
            ),
            "Independent lowest held-out cost ↓": float(
                valid["test_cost"].min()
            ),
            "Independent lowest held-out unfairness ↓": float(
                valid["test_fairness"].min()
            ),
            "Development nR2 ↓": float(final_mo["dev_nr2"]),
            "Held-out nR2 ↓": float(final_mo["heldout_nr2"]),
            "Development hypervolume ↑": float(final_mo["dev_hv"]),
            "Held-out hypervolume ↑": float(final_mo["heldout_hv"]),
            "Absolute HV gap ↓": float(
                final_mo["hv_generalization_gap"]
            ),
        }

        for policy in (
            "accuracy_first",
            "cost_first",
            "fairness_first",
            "high_accuracy_cost",
            "high_accuracy_fairness",
        ):
            selected = final_policies[
                final_policies["policy"] == policy
            ]
            if selected.empty:
                continue
            selected = selected.iloc[0]
            prefix = POLICY_DISPLAY[policy]
            row[f"{prefix}: development accuracy"] = float(
                selected["dev_quality"]
            )
            row[f"{prefix}: held-out accuracy"] = float(
                selected["test_quality"]
            )
            row[f"{prefix}: held-out cost"] = float(
                selected["test_cost"]
            )
            row[f"{prefix}: held-out unfairness"] = float(
                selected["test_fairness"]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def shared_prompt_audit(
    final_valid: pd.DataFrame,
) -> pd.DataFrame:
    if "prompt_id" not in final_valid:
        return pd.DataFrame()

    duplicated = final_valid[
        final_valid["prompt_id"].astype(str).duplicated(keep=False)
    ].copy()
    if duplicated.empty:
        return duplicated

    counts = duplicated.groupby("prompt_id")["optimizer"].nunique()
    shared_ids = set(counts[counts > 1].index.astype(str))
    return duplicated[
        duplicated["prompt_id"].astype(str).isin(shared_ids)
    ].sort_values(["prompt_id", "optimizer"])


def write_readme(
    output_dir: Path,
    *,
    accuracy_threshold: float,
    run_files: Mapping[str, RunFiles],
) -> None:
    source_lines = "\n".join(
        f"- {DISPLAY_NAME[optimizer]}: `{run.evaluation_path}`"
        for optimizer, run in run_files.items()
    )
    content = f"""# BBQ Tri-Fair v6 matched 2M smoke-run figures

## Sources

{source_lines}

## Main policy

All headline prompts are selected on development data and then evaluated on
the held-out split.  Balanced points are not used.

- accuracy-first: maximum development accuracy;
- cost-first: minimum development cost;
- fairness-first: minimum development unfairness;
- high-accuracy threshold: {accuracy_threshold:.2f}.

## Labels

BBQ quality is accuracy, so every figure uses `Accuracy` rather than
`Test Quality`.

## Multi-objective diagnostics

The script uses minimisation objectives:

1. `1 - accuracy`;
2. `cost / {COST_BOUND:g}`;
3. `unfairness`.

Hypervolume uses reference point {tuple(REFERENCE_POINT.tolist())}.  nR2 is a
deterministic simplex-weight Tchebycheff indicator; lower is better.

The figures contain one seed and therefore do not show standard-deviation
bands.  The full study should show all seed points plus mean and sample SD.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def write_manifest(
    output_dir: Path,
    *,
    run_files: Mapping[str, RunFiles],
    accuracy_threshold: float,
) -> None:
    files = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file()
        and path.name != "figure_manifest.json"
    )
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "dataset": DATASET,
        "seed": SEED,
        "budget": BUDGET,
        "methods": list(METHOD_ORDER),
        "accuracy_threshold": float(accuracy_threshold),
        "policy": "development-selected prompts evaluated on held-out data",
        "balanced_points_used": False,
        "sources": {
            optimizer: {
                "run_dir": str(run.run_dir.resolve()),
                "evaluation_sha256": sha256(run.evaluation_path),
                "step_sha256": sha256(run.step_path),
            }
            for optimizer, run in run_files.items()
        },
        "outputs": {
            path.name: {
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256(path),
            }
            for path in files
        },
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_style()

    if not 0.0 <= args.accuracy_threshold <= 1.0:
        raise ValueError("--accuracy-threshold must lie in [0, 1]")

    formats = parse_formats(args.formats)
    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluations, run_files, summaries = load_all_runs(
        results_root,
        strict=bool(args.strict),
    )

    policy_trajectory = build_policy_trajectory(
        evaluations,
        accuracy_threshold=float(args.accuracy_threshold),
    )
    mo_trajectory = build_mo_trajectory(evaluations)

    final_frames = [
        final_checkpoint_rows(evaluations, optimizer)
        for optimizer in METHOD_ORDER
    ]
    final = pd.concat(final_frames, ignore_index=True, sort=False)
    final_valid = final[final["publication_valid"]].copy()
    if final_valid.empty:
        raise RuntimeError("The final valid candidate table is empty")

    evaluations.to_parquet(
        output_dir / "bbq_v6_smoke_2m_all_evaluations_with_validity.parquet",
        index=False,
    )
    final_valid.to_parquet(
        output_dir / "bbq_v6_smoke_2m_final_valid_candidates.parquet",
        index=False,
    )
    policy_trajectory.to_csv(
        output_dir / "bbq_v6_smoke_2m_development_selected_trajectory.csv",
        index=False,
    )
    mo_trajectory.to_csv(
        output_dir / "bbq_v6_smoke_2m_multiobjective_trajectory.csv",
        index=False,
    )

    final_summary = build_final_summary(
        evaluations,
        policy_trajectory,
        mo_trajectory,
        summaries,
    )
    final_summary.to_csv(
        output_dir / "bbq_v6_smoke_2m_summary.csv",
        index=False,
    )
    safe_markdown(
        final_summary,
        output_dir / "bbq_v6_smoke_2m_summary.md",
    )

    audit = shared_prompt_audit(final_valid)
    audit.to_csv(
        output_dir / "bbq_v6_smoke_2m_shared_prompt_audit.csv",
        index=False,
    )

    plot_accuracy_cost_unfairness_trajectory(
        policy_trajectory,
        output_dir,
        formats,
    )
    plot_final_development_heldout_policies(
        policy_trajectory,
        output_dir,
        formats,
    )
    plot_nr2(
        mo_trajectory,
        output_dir,
        formats,
    )
    plot_hypervolume_gap(
        mo_trajectory,
        output_dir,
        formats,
    )
    plot_attainment(
        final_valid,
        output_dir,
        formats,
        x_column="test_cost",
        xlabel="Held-out Weighted Mean-Token Cost ↓",
        suffix="cost",
    )
    plot_attainment(
        final_valid,
        output_dir,
        formats,
        x_column="test_fairness",
        xlabel="Held-out Unfairness ↓",
        suffix="unfairness",
    )
    plot_cost_unfairness_pareto(
        final_valid,
        output_dir,
        formats,
    )
    plot_three_dimensional_pareto(
        final_valid,
        output_dir,
        formats,
    )
    plot_few_shot_diagnostics(
        final_valid,
        output_dir,
        formats,
    )

    write_readme(
        output_dir,
        accuracy_threshold=float(args.accuracy_threshold),
        run_files=run_files,
    )
    write_manifest(
        output_dir,
        run_files=run_files,
        accuracy_threshold=float(args.accuracy_threshold),
    )

    generated = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
    )
    print(f"Generated {len(generated)} files:")
    for name in generated:
        print(f"  {name}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
