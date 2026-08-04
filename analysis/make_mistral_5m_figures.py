"""Create publication-style 5M figures for the Mistral Tri-Fair study.

This module is the Mistral counterpart of ``make_bbq_5m_figures_gptoss.py``.
It supports all three fairness datasets in the completed direct-to-5M namespace:

* BBQ (accuracy; statistical BBQ unfairness)
* Bias-in-Bios (macro-F1; gender TPR-gap unfairness)
* Civil Comments (accuracy; worst-group unfairness)

The script discovers exactly six runs per dataset (two methods x three seeds),
validates strict-budget final stops, reads final holdout ``eval.parquet`` and
all development ``step_results.parquet`` files, and generates:

1. development accuracy/Macro-F1, cost and unfairness trajectories;
2. final per-seed extrema and three-seed mean +/- sample SD;
3. exact normalized nR2, hypervolume and approximation-gap comparisons;
4. accuracy-cost and accuracy-unfairness empirical attainment curves;
5. cost-unfairness and three-objective Pareto figures;
6. high-quality operating-point comparisons;
7. Tri-Fair few-shot diagnostics showing few-shot count, output-cost share,
   quality, cost and unfairness;
8. CSV and Markdown audit/summary tables.

Default usage on Rocket
-----------------------
Generate every dataset::

    python -m analysis.make_mistral_5m_figures --dataset all --strict

Generate one dataset::

    python -m analysis.make_mistral_5m_figures --dataset bbq --strict

The three thin wrapper modules supplied with this file provide the familiar
per-dataset command names.

Important interpretation
------------------------
``test_cost`` is the configured Mistral weighted mean-token objective:
``0.08 * mean input tokens + 0.32 * mean output tokens``. It is not a GPU bill.
Maximum quality, minimum cost and minimum unfairness are independent extrema;
they need not belong to one prompt.
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

MODEL = "mistral-3-24b"
MODEL_DISPLAY = "Mistral-Small-3.2-24B"
FINAL_BUDGET = 5_000_000
EXPECTED_SEEDS = (42, 43, 44)
OPTIMIZER_ORDER = ("Tri-Fair", "NSGAII-PO-Fair")
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
MISTRAL_INPUT_WEIGHT = 0.08
MISTRAL_OUTPUT_WEIGHT = 0.32
DEFAULT_COST_UPPER_BOUND = 100.0
REFERENCE_POINT = np.asarray([1.1, 1.1, 1.1], dtype=float)
PREFERENCE_SEED = 2026
N_PREFERENCES = 1_000
STRICT_FINAL_STOP_REASONS = {
    "next_atomic_operation_exceeds_budget",
    "next_complete_candidate_exceeds_budget",
}
MIN_STRICT_UTILIZATION = 0.95


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display: str
    quality_label: str
    quality_short: str
    fairness_label: str
    threshold: float
    require_bbq_coverage: bool = False


DATASETS: dict[str, DatasetSpec] = {
    "bbq": DatasetSpec(
        key="bbq",
        display="BBQ",
        quality_label="Test Accuracy ↑",
        quality_short="Accuracy",
        fairness_label="Statistical BBQ Unfairness ↓",
        threshold=0.90,
        require_bbq_coverage=True,
    ),
    "bias_in_bios": DatasetSpec(
        key="bias_in_bios",
        display="Bias-in-Bios",
        quality_label="Test Macro-F1 ↑",
        quality_short="Macro-F1",
        fairness_label="Gender TPR-Gap Unfairness ↓",
        threshold=0.75,
    ),
    "civil_comments": DatasetSpec(
        key="civil_comments",
        display="Civil Comments",
        quality_label="Test Accuracy ↑",
        quality_short="Accuracy",
        fairness_label="Worst-Group Unfairness ↓",
        threshold=0.82,
    ),
}


@dataclass(frozen=True)
class RunArtifacts:
    dataset: str
    optimizer: str
    seed: int
    run_dir: Path
    eval_path: Path
    step_path: Path
    summary_path: Path
    actual_tokens: int
    utilization: float
    stopping_reason: str

    @property
    def run_key(self) -> str:
        return f"{self.dataset}/{self.optimizer}/seed{self.seed}/{self.run_dir.name}"


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="results/tri_fair_mistral32_5m",
        help="Root of the completed Mistral direct-to-5M namespace.",
    )
    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS.keys()),
        default="all",
        help="Dataset to generate, or all three.",
    )
    parser.add_argument(
        "--output-root",
        default="analysis/output/mistral32_5m",
        help="Root for generated tables and publication figures.",
    )
    parser.add_argument(
        "--cost-upper-bound",
        type=float,
        default=DEFAULT_COST_UPPER_BOUND,
        help="Fixed normalization upper bound for the cost objective.",
    )
    parser.add_argument(
        "--n-preferences", type=int, default=N_PREFERENCES
    )
    parser.add_argument(
        "--preference-seed", type=int, default=PREFERENCE_SEED
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing runs, invalid strict-budget stops or missing fairness fields.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing dataset output directory.",
    )
    return parser.parse_args(argv)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def json_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        if pd.isna(value):
            return {}
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def bool_series(frame: pd.DataFrame, column: str, *, default: bool) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    normalized = values.astype(str).str.strip().str.casefold()
    return normalized.isin({"1", "true", "yes", "y", "on"})


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def read_run_summary(path: Path) -> tuple[int, float, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing run summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    controller = summary.get("budget_controller") or {}
    if not isinstance(controller, dict):
        raise ValueError(f"budget_controller is invalid in {path}")
    tokens = int(controller.get("actual_downstream_tokens", 0) or 0)
    if tokens <= 0:
        tokens = int(
            (summary.get("final_downstream_tokens") or {}).get("total_tokens", 0)
            or 0
        )
    utilization = float(controller.get("budget_utilization", 0.0) or 0.0)
    if not np.isfinite(utilization) or utilization <= 0:
        utilization = tokens / FINAL_BUDGET if tokens > 0 else 0.0
    reason = str(controller.get("stopping_reason") or "")
    return tokens, utilization, reason


def valid_final_stop(tokens: int, utilization: float, reason: str) -> bool:
    if tokens >= FINAL_BUDGET:
        return True
    return (
        0 < tokens <= FINAL_BUDGET
        and MIN_STRICT_UTILIZATION <= utilization <= 1.0 + 1e-12
        and reason in STRICT_FINAL_STOP_REASONS
    )


def infer_run_metadata(results_root: Path, eval_path: Path) -> tuple[str, str, int]:
    relative = eval_path.relative_to(results_root)
    parts = relative.parts
    # Expected: MODEL / DATASET / OPTIMIZER / seedNN / hash / eval.parquet
    if len(parts) < 6:
        raise ValueError(f"Unexpected evaluation path: {eval_path}")
    model, dataset, optimizer, seed_part = parts[:4]
    if model != MODEL:
        raise ValueError(f"Expected model {MODEL}, found {model} in {eval_path}")
    if not seed_part.startswith("seed"):
        raise ValueError(f"Could not parse seed from {eval_path}")
    return dataset, optimizer, int(seed_part.removeprefix("seed"))


def discover_runs(
    results_root: Path,
    spec: DatasetSpec,
    *,
    strict: bool,
) -> list[RunArtifacts]:
    model_root = results_root / MODEL / spec.key
    if not model_root.is_dir():
        raise FileNotFoundError(f"Dataset result directory does not exist: {model_root}")

    artifacts: list[RunArtifacts] = []
    errors: list[str] = []
    for eval_path in sorted(model_root.rglob("eval.parquet")):
        try:
            dataset, optimizer, seed = infer_run_metadata(results_root, eval_path)
            if dataset != spec.key:
                continue
            if optimizer not in OPTIMIZER_ORDER or seed not in EXPECTED_SEEDS:
                continue
            run_dir = eval_path.parent
            step_path = run_dir / "step_results.parquet"
            summary_path = run_dir / "run_summary.json"
            if not step_path.is_file():
                raise FileNotFoundError(f"Missing {step_path}")
            tokens, utilization, reason = read_run_summary(summary_path)
            if not valid_final_stop(tokens, utilization, reason):
                raise RuntimeError(
                    f"Invalid final stop for {run_dir}: tokens={tokens}, "
                    f"utilization={utilization:.4f}, reason={reason!r}"
                )
            artifacts.append(
                RunArtifacts(
                    dataset=dataset,
                    optimizer=optimizer,
                    seed=seed,
                    run_dir=run_dir,
                    eval_path=eval_path,
                    step_path=step_path,
                    summary_path=summary_path,
                    actual_tokens=tokens,
                    utilization=utilization,
                    stopping_reason=reason,
                )
            )
        except Exception as error:  # noqa: BLE001 - aggregate audit failures
            errors.append(str(error))

    if errors:
        message = "Run-discovery problems:\n" + "\n".join(f"- {x}" for x in errors)
        if strict:
            raise RuntimeError(message)
        warnings.warn(message)

    expected = {
        (optimizer, seed)
        for optimizer in OPTIMIZER_ORDER
        for seed in EXPECTED_SEEDS
    }
    by_pair: dict[tuple[str, int], list[RunArtifacts]] = {}
    for run in artifacts:
        by_pair.setdefault((run.optimizer, run.seed), []).append(run)

    selected: list[RunArtifacts] = []
    duplicates: list[str] = []
    for pair in sorted(expected):
        candidates = sorted(
            by_pair.get(pair, []),
            key=lambda item: (item.actual_tokens, str(item.run_dir)),
            reverse=True,
        )
        if not candidates:
            continue
        selected.append(candidates[0])
        if len(candidates) > 1:
            duplicates.append(
                f"{pair[0]}/seed{pair[1]}: selected {candidates[0].run_dir} "
                f"from {len(candidates)} candidates"
            )

    observed = {(run.optimizer, run.seed) for run in selected}
    missing = sorted(expected - observed)
    if missing:
        message = "Missing final runs: " + ", ".join(
            f"{method}/seed{seed}" for method, seed in missing
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message)
    if duplicates:
        warnings.warn("Duplicate runs found:\n" + "\n".join(duplicates))
    return sorted(selected, key=lambda run: (run.optimizer, run.seed))


def attach_metadata(frame: pd.DataFrame, run: RunArtifacts) -> pd.DataFrame:
    out = frame.copy()
    out["model"] = MODEL
    out["dataset"] = run.dataset
    out["optimizer"] = run.optimizer
    out["seed"] = run.seed
    out["run_key"] = run.run_key
    out["run_dir"] = str(run.run_dir)
    out["configured_budget"] = FINAL_BUDGET
    out["actual_budget_tokens"] = run.actual_tokens
    out["strict_final_utilization"] = run.utilization
    out["strict_final_stopping_reason"] = run.stopping_reason
    return out


def evaluation_validity(frame: pd.DataFrame, spec: DatasetSpec, *, strict: bool) -> pd.Series:
    require_columns(frame, ["test_quality", "test_cost", "test_fairness"], "eval.parquet")
    finite = (
        numeric(frame, "test_quality").notna()
        & numeric(frame, "test_cost").notna()
        & numeric(frame, "test_fairness").notna()
    )
    ready = bool_series(frame, "test_fairness_ready", default=not strict)
    validity = finite & ready

    if spec.require_bbq_coverage:
        if "test_fairness_diagnostics_json" not in frame:
            if strict:
                raise ValueError(
                    "BBQ eval.parquet lacks test_fairness_diagnostics_json"
                )
            return validity
        coverage = frame["test_fairness_diagnostics_json"].map(
            lambda value: bool(json_object(value).get("coverage_valid", False))
        )
        validity &= coverage
    return validity


def load_evaluations(
    runs: list[RunArtifacts], spec: DatasetSpec, *, strict: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_frames: list[pd.DataFrame] = []
    valid_frames: list[pd.DataFrame] = []
    for run in runs:
        frame = attach_metadata(pd.read_parquet(run.eval_path), run)
        mask = evaluation_validity(frame, spec, strict=strict)
        frame["publication_valid"] = mask
        raw_frames.append(frame)
        valid_frames.append(frame.loc[mask].copy())
    raw = pd.concat(raw_frames, ignore_index=True, sort=False)
    valid = pd.concat(valid_frames, ignore_index=True, sort=False)
    if valid.empty:
        raise RuntimeError(f"No publication-valid final rows for {spec.display}")
    return raw, valid


def load_steps(runs: list[RunArtifacts], *, strict: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        frame = attach_metadata(pd.read_parquet(run.step_path), run)
        require_columns(frame, ["step", "quality", "cost", "fairness"], str(run.step_path))
        ready = bool_series(frame, "fairness_ready", default=not strict)
        finite = (
            numeric(frame, "quality").notna()
            & numeric(frame, "cost").notna()
            & numeric(frame, "fairness").notna()
        )
        frame = frame.loc[ready & finite].copy()
        if frame.empty:
            message = f"No fairness-ready development rows in {run.step_path}"
            if strict:
                raise RuntimeError(message)
            warnings.warn(message)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Pareto values must be a 2-D matrix")
    keep = np.ones(len(values), dtype=bool)
    for index in range(len(values)):
        dominates = np.all(values <= values[index], axis=1) & np.any(
            values < values[index], axis=1
        )
        dominates[index] = False
        if np.any(dominates):
            keep[index] = False
    return keep


def normalize_objectives(frame: pd.DataFrame, cost_upper_bound: float) -> np.ndarray:
    quality = numeric(frame, "test_quality").to_numpy(dtype=float)
    cost = numeric(frame, "test_cost").to_numpy(dtype=float)
    fairness = numeric(frame, "test_fairness").to_numpy(dtype=float)
    matrix = np.column_stack(
        [
            np.clip(1.0 - quality, 0.0, 1.1),
            np.clip(cost / cost_upper_bound, 0.0, 1.1),
            np.clip(fairness, 0.0, 1.1),
        ]
    )
    return matrix


def hypervolume_2d_anchored(points: np.ndarray, ref_y: float, ref_z: float) -> float:
    """Union area of [y, ref_y] x [z, ref_z] rectangles."""
    if len(points) == 0:
        return 0.0
    points = np.asarray(points, dtype=float)
    points = points[(points[:, 0] < ref_y) & (points[:, 1] < ref_z)]
    if len(points) == 0:
        return 0.0
    y_values = np.unique(np.concatenate([points[:, 0], [ref_y]]))
    y_values.sort()
    area = 0.0
    for left, right in zip(y_values[:-1], y_values[1:]):
        active = points[points[:, 0] <= left]
        if len(active) == 0:
            continue
        min_z = float(np.min(active[:, 1]))
        area += max(0.0, right - left) * max(0.0, ref_z - min_z)
    return float(area)


def hypervolume_3d(points: np.ndarray, reference: np.ndarray = REFERENCE_POINT) -> float:
    """Exact 3-D hypervolume for minimization boxes anchored at one reference."""
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(points) == 0:
        return 0.0
    points = points[np.all(np.isfinite(points), axis=1)]
    points = points[np.all(points < reference, axis=1)]
    if len(points) == 0:
        return 0.0
    points = points[pareto_mask_minimize(points)]
    x_values = np.unique(np.concatenate([points[:, 0], [reference[0]]]))
    x_values.sort()
    volume = 0.0
    for left, right in zip(x_values[:-1], x_values[1:]):
        active = points[points[:, 0] <= left]
        if len(active) == 0:
            continue
        area = hypervolume_2d_anchored(
            active[:, 1:], float(reference[1]), float(reference[2])
        )
        volume += max(0.0, right - left) * area
    return float(volume)


def make_preferences(n: int, seed: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n_preferences must be positive")
    return np.random.default_rng(seed).dirichlet(np.ones(3), size=n)


def noisy_r2(front: np.ndarray, weights: np.ndarray) -> float:
    if len(front) == 0:
        return float("nan")
    utility = np.max(weights[:, None, :] * front[None, :, :], axis=2)
    return float(np.mean(np.min(utility, axis=1)))


def approximation_gap(front: np.ndarray, reference_front: np.ndarray) -> float:
    """Mean nearest L-infinity distance from pooled reference front to a run front."""
    if len(front) == 0 or len(reference_front) == 0:
        return float("nan")
    distances = np.max(
        np.abs(reference_front[:, None, :] - front[None, :, :]), axis=2
    )
    return float(np.mean(np.min(distances, axis=1)))


def select_extreme(group: pd.DataFrame, column: str, *, maximize: bool) -> pd.Series:
    work = group.copy()
    work["_metric"] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["_metric"])
    if work.empty:
        raise RuntimeError(f"No finite values for {column}")
    tie_columns = ["_metric"]
    ascending = [not maximize]
    if column == "test_quality":
        work["_fairness"] = numeric(work, "test_fairness")
        work["_cost"] = numeric(work, "test_cost")
        tie_columns += ["_fairness", "_cost"]
        ascending += [True, True]
    return work.sort_values(tie_columns, ascending=ascending).iloc[0]


def compute_run_tables(
    evaluations: pd.DataFrame,
    runs: list[RunArtifacts],
    *,
    cost_upper_bound: float,
    n_preferences: int,
    preference_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = make_preferences(n_preferences, preference_seed)

    all_normalized = normalize_objectives(evaluations, cost_upper_bound)
    pooled_front = all_normalized[pareto_mask_minimize(all_normalized)]

    metric_rows: list[dict[str, object]] = []
    operating_rows: list[dict[str, object]] = []
    run_map = {run.run_key: run for run in runs}

    for run_key, group in evaluations.groupby("run_key", sort=True):
        run = run_map[str(run_key)]
        normalized = normalize_objectives(group, cost_upper_bound)
        front = normalized[pareto_mask_minimize(normalized)]

        quality_row = select_extreme(group, "test_quality", maximize=True)
        cost_row = select_extreme(group, "test_cost", maximize=False)
        fairness_row = select_extreme(group, "test_fairness", maximize=False)

        metric_rows.append(
            {
                "model": MODEL,
                "dataset": run.dataset,
                "optimizer": run.optimizer,
                "method": DISPLAY_NAME[run.optimizer],
                "seed": run.seed,
                "run_key": run.run_key,
                "run_dir": str(run.run_dir),
                "actual_budget_tokens": run.actual_tokens,
                "budget_utilization": run.utilization,
                "candidate_count": int(len(group)),
                "pareto_front_size": int(len(front)),
                "max_test_quality": float(quality_row["test_quality"]),
                "quality_point_cost": float(quality_row["test_cost"]),
                "quality_point_unfairness": float(quality_row["test_fairness"]),
                "min_test_cost": float(cost_row["test_cost"]),
                "cost_point_quality": float(cost_row["test_quality"]),
                "cost_point_unfairness": float(cost_row["test_fairness"]),
                "min_test_unfairness": float(fairness_row["test_fairness"]),
                "fairness_point_quality": float(fairness_row["test_quality"]),
                "fairness_point_cost": float(fairness_row["test_cost"]),
                "noisy_r2_3d": noisy_r2(front, weights),
                "hypervolume_3d": hypervolume_3d(front),
                "approximation_gap_3d": approximation_gap(front, pooled_front),
            }
        )

    return pd.DataFrame(metric_rows), pd.DataFrame(operating_rows)


def high_quality_points(
    evaluations: pd.DataFrame, spec: DatasetSpec
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (optimizer, seed), group in evaluations.groupby(
        ["optimizer", "seed"], sort=True
    ):
        work = group.copy()
        work["_quality"] = numeric(work, "test_quality")
        work["_cost"] = numeric(work, "test_cost")
        work["_fairness"] = numeric(work, "test_fairness")
        work = work.dropna(subset=["_quality", "_cost", "_fairness"])
        eligible = work[work["_quality"] >= spec.threshold].copy()
        threshold_met = not eligible.empty
        candidates = eligible if threshold_met else work
        candidates = candidates.sort_values(
            ["_fairness", "_cost", "_quality"],
            ascending=[True, True, False],
        )
        if candidates.empty:
            continue
        row = candidates.iloc[0]
        rows.append(
            {
                "dataset": spec.key,
                "optimizer": optimizer,
                "method": DISPLAY_NAME[str(optimizer)],
                "seed": int(seed),
                "threshold": spec.threshold,
                "threshold_met": threshold_met,
                "test_quality": float(row["_quality"]),
                "test_cost": float(row["_cost"]),
                "test_fairness": float(row["_fairness"]),
                "prompt_id": row.get("prompt_id", ""),
                "prompt": row.get("prompt", ""),
            }
        )
    return pd.DataFrame(rows)


def summarize_run_metrics(run_metrics: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    columns = [
        ("max_test_quality", f"Max {spec.quality_short} ↑"),
        ("min_test_cost", "Min Cost ↓"),
        ("min_test_unfairness", "Min Unfairness ↓"),
        ("noisy_r2_3d", "nR2 ↓"),
        ("hypervolume_3d", "Hypervolume ↑"),
        ("approximation_gap_3d", "Approximation Gap ↓"),
    ]
    rows: list[dict[str, object]] = []
    for optimizer in OPTIMIZER_ORDER:
        group = run_metrics[run_metrics["optimizer"] == optimizer]
        row: dict[str, object] = {
            "Dataset": spec.display,
            "Method": DISPLAY_NAME[optimizer],
            "Seeds": int(group["seed"].nunique()),
        }
        for source, label in columns:
            values = pd.to_numeric(group[source], errors="coerce").dropna()
            row[f"{label} Mean"] = float(values.mean())
            row[f"{label} SD"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[label] = (
                f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
                if len(values) > 1
                else f"{values.mean():.4f} ± 0.0000"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def step_token_column(frame: pd.DataFrame) -> pd.Series:
    if "total_tokens_downstream" in frame:
        return numeric(frame, "total_tokens_downstream")
    input_tokens = (
        numeric(frame, "input_tokens_downstream")
        if "input_tokens_downstream" in frame
        else pd.Series(0.0, index=frame.index)
    )
    output_tokens = (
        numeric(frame, "output_tokens_downstream")
        if "output_tokens_downstream" in frame
        else pd.Series(0.0, index=frame.index)
    )
    return input_tokens + output_tokens


def development_step_metrics(steps: pd.DataFrame, cost_upper_bound: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    weights = make_preferences(N_PREFERENCES, PREFERENCE_SEED)
    group_columns = ["run_key", "optimizer", "seed", "step"]
    for keys, group in steps.groupby(group_columns, sort=True, dropna=False):
        run_key, optimizer, seed, step = keys
        quality = numeric(group, "quality")
        cost = numeric(group, "cost")
        fairness = numeric(group, "fairness")
        valid = quality.notna() & cost.notna() & fairness.notna()
        if not valid.any():
            continue
        quality_values = quality[valid].to_numpy(dtype=float)
        cost_values = cost[valid].to_numpy(dtype=float)
        fairness_values = fairness[valid].to_numpy(dtype=float)
        normalized = np.column_stack(
            [
                np.clip(1.0 - quality_values, 0.0, 1.1),
                np.clip(cost_values / cost_upper_bound, 0.0, 1.1),
                np.clip(fairness_values, 0.0, 1.1),
            ]
        )
        front = normalized[pareto_mask_minimize(normalized)]
        tokens = int(np.nanmax(step_token_column(group).to_numpy(dtype=float)))
        rows.append(
            {
                "run_key": run_key,
                "optimizer": optimizer,
                "seed": int(seed),
                "step": int(step),
                "actual_budget_tokens": tokens,
                "best_quality": float(np.max(quality_values)),
                "best_cost": float(np.min(cost_values)),
                "best_unfairness": float(np.min(fairness_values)),
                "dev_nR2": noisy_r2(front, weights),
                "dev_hypervolume": hypervolume_3d(front),
                "front_size": int(len(front)),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("Could not derive development trajectory metrics")

    # Convert independent step extrema into anytime incumbent trajectories.
    result = result.sort_values(["run_key", "actual_budget_tokens", "step"])
    result["anytime_quality"] = result.groupby("run_key")["best_quality"].cummax()
    result["anytime_cost"] = result.groupby("run_key")["best_cost"].cummin()
    result["anytime_unfairness"] = result.groupby("run_key")[
        "best_unfairness"
    ].cummin()
    result["anytime_hypervolume"] = result.groupby("run_key")[
        "dev_hypervolume"
    ].cummax()
    result["anytime_nR2"] = result.groupby("run_key")["dev_nR2"].cummin()
    return result.reset_index(drop=True)


def staircase_grid(frame: pd.DataFrame) -> np.ndarray:
    tokens = pd.to_numeric(frame["actual_budget_tokens"], errors="coerce").dropna()
    return np.sort(tokens.astype(int).unique())


def staircase_values(run: pd.DataFrame, grid: np.ndarray, column: str) -> np.ndarray:
    work = run[["actual_budget_tokens", column]].copy()
    work["actual_budget_tokens"] = pd.to_numeric(
        work["actual_budget_tokens"], errors="coerce"
    )
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna().sort_values("actual_budget_tokens")
    work = work.groupby("actual_budget_tokens", as_index=False)[column].last()
    x = work["actual_budget_tokens"].to_numpy(dtype=int)
    y = work[column].to_numpy(dtype=float)
    result = np.full(len(grid), np.nan)
    for index, value in enumerate(grid):
        position = np.searchsorted(x, value, side="right") - 1
        if position >= 0:
            result[index] = y[position]
    return result


def trajectory_statistics(
    frame: pd.DataFrame, column: str
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    output: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for optimizer in OPTIMIZER_ORDER:
        method = frame[frame["optimizer"] == optimizer]
        if method.empty:
            continue
        grid = staircase_grid(method)
        matrix = np.vstack(
            [
                staircase_values(group, grid, column)
                for _, group in method.groupby("seed", sort=True)
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(matrix, axis=0)
            std = np.nanstd(matrix, axis=0, ddof=1)
        valid = np.isfinite(mean)
        output[optimizer] = (grid[valid], mean[valid], std[valid])
    return output


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_development_objectives(
    trajectory: pd.DataFrame, spec: DatasetSpec, outdir: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.0), constrained_layout=True)
    plots = [
        ("anytime_quality", f"Development {spec.quality_short} ↑", spec.quality_label),
        ("anytime_cost", "Development Cost ↓", "Weighted Mean-Token Cost ↓"),
        ("anytime_unfairness", "Development Unfairness ↓", spec.fairness_label),
    ]
    for axis, (column, title, ylabel) in zip(axes, plots):
        stats = trajectory_statistics(trajectory, column)
        for optimizer in OPTIMIZER_ORDER:
            if optimizer not in stats:
                continue
            grid, mean, std = stats[optimizer]
            x = grid / 1_000_000.0
            axis.step(
                x,
                mean,
                where="post",
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                markevery=max(1, len(x) // 10),
                linewidth=2,
                markersize=4,
                label=DISPLAY_NAME[optimizer],
            )
            axis.fill_between(
                x,
                mean - std,
                mean + std,
                step="post",
                color=COLORS[optimizer],
                alpha=0.16,
            )
        axis.set_title(title)
        axis.set_xlabel("Cumulative Downstream Tokens [×10⁶]")
        axis.set_ylabel(ylabel)
        axis.set_xlim(left=0.0)
        axis.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle(f"{spec.display} — {MODEL_DISPLAY}: 5M Development Trajectories")
    save_figure(figure, outdir, f"{spec.key}_mistral32_5m_development_objectives")


def plot_development_mo_metrics(
    trajectory: pd.DataFrame, spec: DatasetSpec, outdir: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
    plots = [
        ("anytime_nR2", "Development nR2 ↓", "nR2"),
        ("anytime_hypervolume", "Development Hypervolume ↑", "Hypervolume"),
    ]
    for axis, (column, title, ylabel) in zip(axes, plots):
        stats = trajectory_statistics(trajectory, column)
        for optimizer in OPTIMIZER_ORDER:
            if optimizer not in stats:
                continue
            grid, mean, std = stats[optimizer]
            x = grid / 1_000_000.0
            axis.step(
                x,
                mean,
                where="post",
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                markevery=max(1, len(x) // 10),
                linewidth=2,
                markersize=4,
                label=DISPLAY_NAME[optimizer],
            )
            axis.fill_between(
                x,
                mean - std,
                mean + std,
                step="post",
                color=COLORS[optimizer],
                alpha=0.16,
            )
        axis.set_title(title)
        axis.set_xlabel("Cumulative Downstream Tokens [×10⁶]")
        axis.set_ylabel(ylabel)
        axis.set_xlim(left=0.0)
        axis.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle(f"{spec.display} — {MODEL_DISPLAY}: Multi-objective Dynamics")
    save_figure(figure, outdir, f"{spec.key}_mistral32_5m_development_nr2_hv")


def plot_final_extrema(run_metrics: pd.DataFrame, spec: DatasetSpec, outdir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    plots = [
        ("max_test_quality", f"Maximum {spec.quality_short} ↑", spec.quality_label),
        ("min_test_cost", "Minimum Cost ↓", "Weighted Mean-Token Cost ↓"),
        ("min_test_unfairness", "Minimum Unfairness ↓", spec.fairness_label),
    ]
    for axis, (column, title, ylabel) in zip(axes, plots):
        positions = np.arange(len(OPTIMIZER_ORDER))
        means: list[float] = []
        stds: list[float] = []
        for optimizer in OPTIMIZER_ORDER:
            values = pd.to_numeric(
                run_metrics.loc[run_metrics["optimizer"] == optimizer, column],
                errors="coerce",
            ).dropna()
            means.append(float(values.mean()))
            stds.append(float(values.std(ddof=1)))
        bars = axis.bar(
            positions,
            means,
            yerr=stds,
            capsize=5,
            color=[COLORS[x] for x in OPTIMIZER_ORDER],
            alpha=0.84,
        )
        axis.set_xticks(
            positions, [DISPLAY_NAME[x] for x in OPTIMIZER_ORDER], rotation=7
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)
        for bar, mean, std in zip(bars, means, stds):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{mean:.4f}\n± {std:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle(f"{spec.display} — {MODEL_DISPLAY} at 5M (mean ± SD, 3 seeds)")
    save_figure(figure, outdir, f"{spec.key}_mistral32_5m_final_accuracy_cost_unfairness")


def plot_final_mo_metrics(run_metrics: pd.DataFrame, spec: DatasetSpec, outdir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    plots = [
        ("noisy_r2_3d", "Exact Holdout nR2 ↓", "nR2"),
        ("hypervolume_3d", "Exact Holdout Hypervolume ↑", "Hypervolume"),
        ("approximation_gap_3d", "Approximation Gap ↓", "Normalized Gap"),
    ]
    for axis, (column, title, ylabel) in zip(axes, plots):
        positions = np.arange(len(OPTIMIZER_ORDER))
        means: list[float] = []
        stds: list[float] = []
        for optimizer in OPTIMIZER_ORDER:
            values = pd.to_numeric(
                run_metrics.loc[run_metrics["optimizer"] == optimizer, column],
                errors="coerce",
            ).dropna()
            means.append(float(values.mean()))
            stds.append(float(values.std(ddof=1)))
        bars = axis.bar(
            positions,
            means,
            yerr=stds,
            capsize=5,
            color=[COLORS[x] for x in OPTIMIZER_ORDER],
            alpha=0.84,
        )
        axis.set_xticks(
            positions, [DISPLAY_NAME[x] for x in OPTIMIZER_ORDER], rotation=7
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle(f"{spec.display} — {MODEL_DISPLAY}: Final Multi-objective Metrics")
    save_figure(figure, outdir, f"{spec.key}_mistral32_5m_final_nr2_hv_gap")


def objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            1.0 - numeric(frame, "test_quality").to_numpy(dtype=float),
            numeric(frame, "test_cost").to_numpy(dtype=float),
            numeric(frame, "test_fairness").to_numpy(dtype=float),
        ]
    )


def attained_quality(frame: pd.DataFrame, grid: np.ndarray, x_column: str) -> np.ndarray:
    x = numeric(frame, x_column).to_numpy(dtype=float)
    quality = numeric(frame, "test_quality").to_numpy(dtype=float)
    result = np.full(len(grid), np.nan)
    for index, value in enumerate(grid):
        eligible = quality[x <= value]
        if len(eligible):
            result[index] = np.max(eligible)
    return result


def plot_attainment(
    evaluations: pd.DataFrame,
    spec: DatasetSpec,
    outdir: Path,
    *,
    x_column: str,
    x_label: str,
    suffix: str,
) -> None:
    x_values = numeric(evaluations, x_column).dropna().to_numpy(dtype=float)
    padding = 0.03 * max(float(np.ptp(x_values)), 1e-6)
    grid = np.linspace(float(np.min(x_values) - padding), float(np.max(x_values) + padding), 400)
    figure, axis = plt.subplots(figsize=(6.6, 4.3), constrained_layout=True)
    for optimizer in OPTIMIZER_ORDER:
        curves: list[np.ndarray] = []
        method = evaluations[evaluations["optimizer"] == optimizer]
        for _, group in method.groupby("seed", sort=True):
            matrix = objective_matrix(group)
            finite = np.all(np.isfinite(matrix), axis=1)
            group = group.loc[finite].reset_index(drop=True)
            matrix = matrix[finite]
            front = group.loc[pareto_mask_minimize(matrix)]
            curves.append(attained_quality(front, grid, x_column))
        matrix = np.vstack(curves)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(matrix, axis=0)
            lower = np.nanmin(matrix, axis=0)
            upper = np.nanmax(matrix, axis=0)
        valid = np.isfinite(median)
        axis.step(
            grid[valid],
            median[valid],
            where="post",
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            markevery=max(1, int(np.sum(valid)) // 9),
            linewidth=2,
            markersize=4,
            label=DISPLAY_NAME[optimizer],
        )
        axis.fill_between(
            grid[valid],
            lower[valid],
            upper[valid],
            step="post",
            color=COLORS[optimizer],
            alpha=0.16,
        )
    axis.set_title(f"{spec.display} — {MODEL_DISPLAY} at 5M")
    axis.set_xlabel(x_label)
    axis.set_ylabel(spec.quality_label)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, loc="best")
    save_figure(figure, outdir, f"{spec.key}_mistral32_5m_attainment_{suffix}")


def plot_cost_unfairness(evaluations: pd.DataFrame, spec: DatasetSpec, outdir: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    for optimizer in OPTIMIZER_ORDER:
        pooled: list[pd.DataFrame] = []
        method = evaluations[evaluations["optimizer"] == optimizer]
        for _, group in method.groupby("seed", sort=True):
            matrix = objective_matrix(group)
            finite = np.all(np.isfinite(matrix), axis=1)
            group = group.loc[finite].reset_index(drop=True)
            matrix = matrix[finite]
            front = group.loc[pareto_mask_minimize(matrix)].copy()
            pooled.append(front)
            axis.scatter(
                front["test_cost"],
                front["test_fairness"],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                alpha=0.24,
                s=26,
            )
        combined = pd.concat(pooled, ignore_index=True)
        projection = np.column_stack(
            [numeric(combined, "test_cost"), numeric(combined, "test_fairness")]
        )
        front = combined.loc[pareto_mask_minimize(projection)].sort_values("test_cost")
        axis.plot(
            front["test_cost"],
            front["test_fairness"],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            linewidth=2,
            markersize=5,
            label=DISPLAY_NAME[optimizer],
        )
    axis.set_title(f"{spec.display} — 5M Cost vs Unfairness")
    axis.set_xlabel("Weighted Mean-Token Cost ↓")
    axis.set_ylabel(spec.fairness_label)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, outdir, f"{spec.key}_mistral32_5m_pareto_cost_unfairness")


def plot_three_objective(evaluations: pd.DataFrame, spec: DatasetSpec, outdir: Path) -> None:
    figure = plt.figure(figsize=(7.3, 5.6), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    for optimizer in OPTIMIZER_ORDER:
        method = evaluations[evaluations["optimizer"] == optimizer]
        for seed, group in method.groupby("seed", sort=True):
            matrix = objective_matrix(group)
            finite = np.all(np.isfinite(matrix), axis=1)
            group = group.loc[finite].reset_index(drop=True)
            matrix = matrix[finite]
            front = group.loc[pareto_mask_minimize(matrix)]
            axis.scatter(
                front["test_cost"],
                front["test_fairness"],
                front["test_quality"],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                s=36,
                alpha=0.78,
                label=DISPLAY_NAME[optimizer] if int(seed) == EXPECTED_SEEDS[0] else None,
            )
    axis.set_title(f"{spec.display} — {MODEL_DISPLAY} 5M Test Pareto Fronts")
    axis.set_xlabel("Weighted Mean-Token Cost ↓")
    axis.set_ylabel(spec.fairness_label)
    axis.set_zlabel(spec.quality_label)
    axis.legend(frameon=False)
    save_figure(figure, outdir, f"{spec.key}_mistral32_5m_test_pareto_3d")


def plot_high_quality(points: pd.DataFrame, spec: DatasetSpec, outdir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    plots = [
        ("test_quality", spec.quality_label),
        ("test_cost", "Weighted Mean-Token Cost ↓"),
        ("test_fairness", spec.fairness_label),
    ]
    for axis, (column, ylabel) in zip(axes, plots):
        for optimizer in OPTIMIZER_ORDER:
            group = points[points["optimizer"] == optimizer].sort_values("seed")
            axis.plot(
                group["seed"],
                group[column],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                linewidth=2,
                markersize=6,
                label=DISPLAY_NAME[optimizer],
            )
        axis.set_xticks(EXPECTED_SEEDS)
        axis.set_xlabel("Seed")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[0].axhline(spec.threshold, color="0.5", linestyle="--", linewidth=1)
    axes[0].legend(frameon=False)
    figure.suptitle(
        f"{spec.display} — Lowest Unfairness with {spec.quality_short} ≥ {spec.threshold:.2f}"
    )
    save_figure(figure, outdir, f"{spec.key}_mistral32_5m_high_quality_operating_points")



def parse_fewshot_count(value: object) -> int:
    """Return the number of few-shot examples encoded in one evaluation row."""
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, dict):
        for key in ("examples", "few_shots", "fewshots", "items"):
            nested = value.get(key)
            if isinstance(nested, (list, tuple)):
                return len(nested)
        return 0

    text = str(value).strip()
    if not text:
        return 0
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0
    return parse_fewshot_count(parsed)


def resolve_fewshot_column(frame: pd.DataFrame) -> str | None:
    """Find the few-shot payload column used by the current result schema."""
    preferred = (
        "few_shots_json",
        "fewshots_json",
        "few_shot_examples_json",
        "few_shots",
        "fewshots",
    )
    for column in preferred:
        if column in frame.columns:
            return column
    for column in frame.columns:
        normalized = str(column).casefold().replace("-", "_")
        if "few" in normalized and "shot" in normalized:
            return str(column)
    return None


def resolve_token_columns(frame: pd.DataFrame) -> tuple[str, str] | None:
    """Find mean input/output token columns used by holdout evaluation rows."""
    candidate_pairs = (
        ("test_input_tokens", "test_output_tokens"),
        ("mean_test_input_tokens", "mean_test_output_tokens"),
        ("input_tokens", "output_tokens"),
        ("mean_input_tokens", "mean_output_tokens"),
    )
    for input_column, output_column in candidate_pairs:
        if input_column in frame.columns and output_column in frame.columns:
            return input_column, output_column
    return None


def mistral_output_cost_share(
    frame: pd.DataFrame,
    *,
    strict: bool,
) -> np.ndarray:
    """Compute the output-token fraction of the Mistral weighted cost objective."""
    resolved = resolve_token_columns(frame)
    if resolved is None:
        message = (
            "Few-shot output-share diagnostics require input/output token columns. "
            f"Available columns: {sorted(map(str, frame.columns))}"
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message)
        return np.full(len(frame), np.nan, dtype=float)

    input_column, output_column = resolved
    input_tokens = pd.to_numeric(frame[input_column], errors="coerce").to_numpy(dtype=float)
    output_tokens = pd.to_numeric(frame[output_column], errors="coerce").to_numpy(dtype=float)
    weighted_input = MISTRAL_INPUT_WEIGHT * input_tokens
    weighted_output = MISTRAL_OUTPUT_WEIGHT * output_tokens
    denominator = weighted_input + weighted_output
    return np.clip(
        np.divide(
            weighted_output,
            denominator,
            out=np.full_like(weighted_output, np.nan, dtype=float),
            where=np.isfinite(denominator) & (denominator > 0),
        ),
        0.0,
        1.0,
    )


def trifair_fewshot_candidates(
    evaluations: pd.DataFrame,
    spec: DatasetSpec,
    *,
    strict: bool,
) -> pd.DataFrame:
    """Prepare final Tri-Fair candidates for few-shot diagnostic figures."""
    data = evaluations[evaluations["optimizer"] == "Tri-Fair"].copy()
    if data.empty:
        raise RuntimeError(
            f"No Tri-Fair final candidates are available for {spec.display}"
        )

    # Match the Qwen publication diagnostics: prefer the retained incumbent
    # archive when the evaluator exposes an incumbent flag.
    if "is_incumbent" in data.columns and data["is_incumbent"].notna().any():
        incumbent_mask = bool_series(data, "is_incumbent", default=False)
        incumbents = data.loc[incumbent_mask].copy()
        if not incumbents.empty:
            data = incumbents

    fewshot_column = resolve_fewshot_column(data)
    if fewshot_column is None:
        message = (
            f"No few-shot payload column was found for {spec.display}. "
            f"Available columns: {sorted(map(str, data.columns))}"
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message)
        data["fewshot_count"] = 0
        data["fewshot_source_column"] = ""
    else:
        data["fewshot_count"] = data[fewshot_column].map(parse_fewshot_count)
        data["fewshot_source_column"] = fewshot_column

    data["output_cost_share"] = mistral_output_cost_share(data, strict=strict)
    for column in ("test_quality", "test_cost", "test_fairness"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["test_quality", "test_cost", "test_fairness"])
    if data.empty:
        raise RuntimeError(
            f"No finite Tri-Fair few-shot diagnostic candidates remain for {spec.display}"
        )

    data["fewshot_count"] = (
        pd.to_numeric(data["fewshot_count"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .astype(int)
    )
    return data.reset_index(drop=True)


def plot_trifair_fewshot_diagnostic(
    data: pd.DataFrame,
    spec: DatasetSpec,
    outdir: Path,
    *,
    color_column: str,
    color_label: str,
    suffix: str,
) -> None:
    """Plot Tri-Fair quality-cost candidates with few-shot counts as labels."""
    require_columns(
        data,
        ["test_cost", "test_quality", "test_fairness", "fewshot_count", color_column],
        "Tri-Fair few-shot candidates",
    )
    values = pd.to_numeric(data[color_column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        warnings.warn(
            f"Skipping {spec.display} few-shot {suffix}: {color_column} has no finite values"
        )
        return

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    figure, axis = plt.subplots(figsize=(6.8, 4.9), constrained_layout=True)
    scatter = axis.scatter(
        data["test_cost"],
        data["test_quality"],
        c=values,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolor="black",
        linewidth=0.55,
        s=76,
        alpha=0.94,
    )

    quality_span = float(
        pd.to_numeric(data["test_quality"], errors="coerce").max()
        - pd.to_numeric(data["test_quality"], errors="coerce").min()
    )
    label_offset = max(0.002, 0.018 * max(quality_span, 0.01))
    for row in data.itertuples(index=False):
        axis.text(
            float(row.test_cost),
            float(row.test_quality) + label_offset,
            str(int(row.fewshot_count)),
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="semibold",
        )

    axis.set_title(f"Tri-Fair on {spec.display} — {MODEL_DISPLAY} at 5M")
    axis.set_xlabel("Weighted Mean-Token Cost ↓")
    axis.set_ylabel(spec.quality_label)
    axis.grid(True, alpha=0.25)
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label(color_label)
    axis.text(
        0.02,
        0.02,
        "Point labels show the number of few-shot examples.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="0.3",
    )
    save_figure(
        figure,
        outdir,
        f"{spec.key}_mistral32_5m_trifair_fewshot_{suffix}",
    )

def write_markdown(frame: pd.DataFrame, path: Path, *, floatfmt: str = ".4f") -> None:
    try:
        text = frame.to_markdown(index=False, floatfmt=floatfmt) + "\n"
    except ImportError:
        text = frame.to_csv(index=False)
    path.write_text(text, encoding="utf-8")


def write_outputs(
    outdir: Path,
    spec: DatasetSpec,
    runs: list[RunArtifacts],
    raw: pd.DataFrame,
    evaluations: pd.DataFrame,
    steps: pd.DataFrame,
    trajectory: pd.DataFrame,
    run_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    high_points: pd.DataFrame,
    fewshot_candidates: pd.DataFrame,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    run_manifest = pd.DataFrame(
        [
            {
                "dataset": run.dataset,
                "optimizer": run.optimizer,
                "method": DISPLAY_NAME[run.optimizer],
                "seed": run.seed,
                "run_key": run.run_key,
                "run_dir": str(run.run_dir),
                "actual_tokens": run.actual_tokens,
                "utilization": run.utilization,
                "stopping_reason": run.stopping_reason,
            }
            for run in runs
        ]
    )
    run_manifest.to_csv(outdir / "selected_runs.csv", index=False)
    run_metrics.to_csv(outdir / "run_metrics.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    high_points.to_csv(outdir / "high_quality_operating_points.csv", index=False)
    fewshot_candidates.to_csv(
        outdir / "trifair_fewshot_candidates.csv",
        index=False,
    )
    trajectory.to_csv(outdir / "trajectory_metrics.csv", index=False)
    raw.to_parquet(outdir / "all_evaluations_raw.parquet", index=False)
    evaluations.to_parquet(outdir / "all_evaluations_valid.parquet", index=False)
    steps.to_parquet(outdir / "all_step_results_valid.parquet", index=False)

    per_seed = run_metrics[
        [
            "dataset",
            "method",
            "seed",
            "max_test_quality",
            "min_test_cost",
            "min_test_unfairness",
            "noisy_r2_3d",
            "hypervolume_3d",
            "approximation_gap_3d",
        ]
    ].copy()
    per_seed.columns = [
        "Dataset",
        "Method",
        "Seed",
        f"Max {spec.quality_short} ↑",
        "Min Cost ↓",
        "Min Unfairness ↓",
        "nR2 ↓",
        "Hypervolume ↑",
        "Approximation Gap ↓",
    ]
    write_markdown(per_seed, outdir / "per_seed_metrics.md")

    compact_summary = summary[
        [
            "Dataset",
            "Method",
            f"Max {spec.quality_short} ↑",
            "Min Cost ↓",
            "Min Unfairness ↓",
            "nR2 ↓",
            "Hypervolume ↑",
            "Approximation Gap ↓",
        ]
    ]
    write_markdown(compact_summary, outdir / "summary.md")

    readme = f"""# {spec.display} / {MODEL_DISPLAY} / direct 5M figures

- Results model: `{MODEL}`
- Final configured budget: `{FINAL_BUDGET:,}` downstream tokens
- Seeds: {', '.join(map(str, EXPECTED_SEEDS))}
- Methods: Tri-Fair and NSGA-II-PO-Fair
- Quality: {spec.quality_short}
- Fairness: {spec.fairness_label}
- Cost objective: `{MISTRAL_INPUT_WEIGHT} × mean input tokens + {MISTRAL_OUTPUT_WEIGHT} × mean output tokens`
- High-quality threshold: `{spec.threshold:.2f}`

## Tri-Fair few-shot diagnostics

- `trifair_fewshot_candidates.csv` records the final Tri-Fair candidates used.
- `{spec.key}_mistral32_5m_trifair_fewshot_outputshare.*` colors candidates by
  the output-token share of the weighted Mistral cost objective.
- `{spec.key}_mistral32_5m_trifair_fewshot_unfairness.*` colors candidates by
  holdout unfairness.
- Numeric labels beside points are the number of few-shot examples.
- When `is_incumbent` is available, the figures use the retained Tri-Fair
  incumbent archive, matching the Qwen diagnostic convention.

`max_test_quality`, `min_test_cost`, and `min_test_unfairness` are independent
per-run extrema and may correspond to different prompt candidates.
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")


def generate_dataset(
    results_root: Path,
    output_root: Path,
    spec: DatasetSpec,
    *,
    cost_upper_bound: float,
    n_preferences: int,
    preference_seed: int,
    strict: bool,
    overwrite: bool,
) -> Path:
    outdir = output_root / spec.key / "publication_figures"
    if outdir.exists() and any(outdir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {outdir}. Pass --overwrite deliberately."
        )
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {spec.display} ===")
    runs = discover_runs(results_root, spec, strict=strict)
    if len(runs) != 6:
        message = f"Expected six runs for {spec.display}, found {len(runs)}"
        if strict:
            raise RuntimeError(message)
        warnings.warn(message)
    print("Selected runs")
    for run in runs:
        print(
            f"  {run.optimizer:18s} seed={run.seed} tokens={run.actual_tokens:,} "
            f"utilization={run.utilization:.4f} dir={run.run_dir}"
        )

    raw, evaluations = load_evaluations(runs, spec, strict=strict)
    steps = load_steps(runs, strict=strict)
    trajectory = development_step_metrics(steps, cost_upper_bound)
    run_metrics, _ = compute_run_tables(
        evaluations,
        runs,
        cost_upper_bound=cost_upper_bound,
        n_preferences=n_preferences,
        preference_seed=preference_seed,
    )
    summary = summarize_run_metrics(run_metrics, spec)
    high_points = high_quality_points(evaluations, spec)
    fewshot_candidates = trifair_fewshot_candidates(
        evaluations,
        spec,
        strict=strict,
    )

    write_outputs(
        outdir,
        spec,
        runs,
        raw,
        evaluations,
        steps,
        trajectory,
        run_metrics,
        summary,
        high_points,
        fewshot_candidates,
    )

    plot_development_objectives(trajectory, spec, outdir)
    plot_development_mo_metrics(trajectory, spec, outdir)
    plot_final_extrema(run_metrics, spec, outdir)
    plot_final_mo_metrics(run_metrics, spec, outdir)
    plot_attainment(
        evaluations,
        spec,
        outdir,
        x_column="test_cost",
        x_label="Weighted Mean-Token Cost ↓",
        suffix="quality_cost",
    )
    plot_attainment(
        evaluations,
        spec,
        outdir,
        x_column="test_fairness",
        x_label=spec.fairness_label,
        suffix="quality_unfairness",
    )
    plot_cost_unfairness(evaluations, spec, outdir)
    plot_three_objective(evaluations, spec, outdir)
    plot_high_quality(high_points, spec, outdir)
    plot_trifair_fewshot_diagnostic(
        fewshot_candidates,
        spec,
        outdir,
        color_column="output_cost_share",
        color_label="Output Token Cost Share",
        suffix="outputshare",
    )
    plot_trifair_fewshot_diagnostic(
        fewshot_candidates,
        spec,
        outdir,
        color_column="test_fairness",
        color_label=spec.fairness_label,
        suffix="unfairness",
    )

    print("\nTri-Fair few-shot count distribution")
    print(
        fewshot_candidates["fewshot_count"]
        .value_counts()
        .sort_index()
        .rename_axis("fewshot_count")
        .rename("candidate_count")
        .to_string()
    )

    print("\nThree-seed summary")
    print(
        summary[
            [
                "Dataset",
                "Method",
                f"Max {spec.quality_short} ↑",
                "Min Cost ↓",
                "Min Unfairness ↓",
                "nR2 ↓",
                "Hypervolume ↑",
                "Approximation Gap ↓",
            ]
        ].to_string(index=False)
    )
    print(f"Figures written to: {outdir.resolve()}")
    return outdir


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_style()
    results_root = Path(args.results_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    if not np.isfinite(args.cost_upper_bound) or args.cost_upper_bound <= 0:
        raise ValueError("--cost-upper-bound must be positive and finite")

    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    generated: list[Path] = []
    for key in keys:
        generated.append(
            generate_dataset(
                results_root,
                output_root,
                DATASETS[key],
                cost_upper_bound=float(args.cost_upper_bound),
                n_preferences=int(args.n_preferences),
                preference_seed=int(args.preference_seed),
                strict=bool(args.strict),
                overwrite=bool(args.overwrite),
            )
        )

    print("\nGenerated dataset directories")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
