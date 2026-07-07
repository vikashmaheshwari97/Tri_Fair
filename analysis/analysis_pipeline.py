"""End-to-end analysis pipeline for Tri-Fair experiments.

Outputs
-------
``run_metrics.csv``
    One row per model/dataset/optimizer/seed/evaluated checkpoint.
``summary.csv``
    Mean and standard deviation across independent seeds.
``trajectory_metrics.csv``
    Development-side anytime metrics for every recorded optimizer step.
``normalization_bounds.json``
    Frozen three-objective bounds. Create at 1M, then reuse at 5M/7.5M.
``plots/``
    Budget curves and optional per-run Pareto projections.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_BUDGETS, parse_csv_ints
from .io import load_all_evaluations, load_all_step_results
from .metrics import (
    aggregate_run_metrics,
    analyse_evaluated_run,
    hypervolume_3d,
)
from .objectives import (
    BoundsStore,
    build_development_bounds,
    pareto_mask,
)
from .plots import plot_budget_metric, plot_test_front_projections

logger = logging.getLogger(__name__)


RUN_GROUP_COLUMNS = [
    "run_key",
    "run_dir",
    "model",
    "dataset",
    "optimizer",
    "seed",
    "chosen_step",
    "actual_budget_tokens",
    "budget_checkpoint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/tri_fair")
    parser.add_argument("--output-dir", default="analysis/output")
    parser.add_argument(
        "--bounds-file",
        default="analysis/normalization_bounds.json",
        help="Create this file once at 1M and reuse it for later stages.",
    )
    parser.add_argument(
        "--bounds-budget",
        type=int,
        default=1_000_000,
        help="Only this checkpoint is used when initially estimating cost bounds.",
    )
    parser.add_argument(
        "--rebuild-bounds",
        action="store_true",
        help="Overwrite frozen bounds. Avoid this after reporting 1M results.",
    )
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in DEFAULT_BUDGETS),
    )
    parser.add_argument("--n-preferences", type=int, default=1_000)
    parser.add_argument("--preference-seed", type=int, default=2026)
    parser.add_argument(
        "--reference-point",
        default="1.1,1.1,1.1",
        help="Normalized minimize-all hypervolume reference point.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--plot-fronts",
        action="store_true",
        help="Create 3-D and pairwise PDF plots for every evaluated run/checkpoint.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def parse_reference_point(raw: str) -> np.ndarray:
    values = np.asarray([float(part.strip()) for part in raw.split(",")], dtype=float)
    if values.shape != (3,) or np.any(~np.isfinite(values)):
        raise ValueError("--reference-point must contain three finite numbers")
    return values


def _bounds_source_frame(evaluations: pd.DataFrame, bounds_budget: int) -> pd.DataFrame:
    exact = evaluations[evaluations["budget_checkpoint"] == int(bounds_budget)]
    if not exact.empty:
        return exact
    prior = evaluations[
        (evaluations["budget_checkpoint"] > 0)
        & (evaluations["budget_checkpoint"] <= int(bounds_budget))
    ]
    if not prior.empty:
        logger.warning(
            "No exact %s checkpoint evaluations; constructing bounds from available <= target rows",
            bounds_budget,
        )
        return prior
    logger.warning(
        "No checkpoint labels available; constructing bounds from all development rows"
    )
    return evaluations


def load_or_create_bounds(
    evaluations: pd.DataFrame,
    path: str | Path,
    *,
    bounds_budget: int,
    rebuild: bool,
) -> BoundsStore:
    target = Path(path)
    if target.exists() and not rebuild:
        logger.info("Loading frozen normalization bounds from %s", target)
        return BoundsStore.load(target)
    source = _bounds_source_frame(evaluations, bounds_budget)
    bounds = build_development_bounds(source)
    bounds.save(target)
    logger.warning(
        "Created normalization bounds at %s. Freeze and reuse this file for 5M and 7.5M analyses.",
        target,
    )
    return bounds


def compute_run_metrics(
    evaluations: pd.DataFrame,
    bounds: BoundsStore,
    *,
    n_preferences: int,
    preference_seed: int,
    reference_point: Sequence[float],
    strict: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in evaluations.groupby(RUN_GROUP_COLUMNS, sort=True, dropna=False):
        metadata = dict(zip(RUN_GROUP_COLUMNS, keys))
        if int(metadata["budget_checkpoint"]) <= 0:
            logger.warning(
                "Skipping unlabelled checkpoint for %s step %s",
                metadata["run_key"],
                metadata["chosen_step"],
            )
            continue
        try:
            metrics = analyse_evaluated_run(
                group,
                bounds[(str(metadata["dataset"]), str(metadata["model"]))],
                n_preferences=n_preferences,
                preference_seed=(
                    int(preference_seed)
                    + int(metadata["seed"]) * 10_000
                    + int(metadata["budget_checkpoint"])
                ),
                reference_point=reference_point,
            )
        except Exception:
            if strict:
                raise
            logger.exception(
                "Could not analyse %s at step %s",
                metadata["run_key"],
                metadata["chosen_step"],
            )
            continue
        rows.append({**metadata, **metrics})
    if not rows:
        raise RuntimeError("No run/checkpoint metrics could be computed")
    return pd.DataFrame(rows)


def _step_objective_matrix(step_group: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    required = {"quality", "cost", "fairness"}
    missing = required - set(step_group)
    if missing:
        raise ValueError(f"Step log is missing {sorted(missing)}")
    quality = pd.to_numeric(step_group["quality"], errors="coerce").to_numpy(
        dtype=float
    )
    cost = pd.to_numeric(step_group["cost"], errors="coerce").to_numpy(dtype=float)
    fairness = pd.to_numeric(step_group["fairness"], errors="coerce").to_numpy(
        dtype=float
    )
    mask = np.isfinite(quality) & np.isfinite(cost) & np.isfinite(fairness)
    if "fairness_ready" in step_group:
        mask &= step_group["fairness_ready"].fillna(False).astype(bool).to_numpy()
    valid = step_group.loc[mask].reset_index(drop=True)
    return np.column_stack([1.0 - quality[mask], cost[mask], fairness[mask]]), valid


def compute_trajectory_metrics(
    step_results: pd.DataFrame,
    bounds: BoundsStore,
    *,
    budgets: Sequence[int],
    reference_point: Sequence[float],
    strict: bool,
) -> pd.DataFrame:
    group_columns = [
        "run_key",
        "run_dir",
        "model",
        "dataset",
        "optimizer",
        "seed",
        "step",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in step_results.groupby(group_columns, sort=True, dropna=False):
        metadata = dict(zip(group_columns, keys))
        try:
            raw, valid = _step_objective_matrix(group)
            if not len(raw):
                continue
            normalized = bounds[
                (str(metadata["dataset"]), str(metadata["model"]))
            ].normalize(raw)
            front = pareto_mask(normalized)
            if "total_tokens_downstream" in group:
                total_tokens = int(
                    pd.to_numeric(
                        group["total_tokens_downstream"], errors="coerce"
                    ).max()
                )
            else:
                input_tokens = (
                    pd.to_numeric(
                        group["input_tokens_downstream"], errors="coerce"
                    ).max()
                    if "input_tokens_downstream" in group
                    else 0
                )
                output_tokens = (
                    pd.to_numeric(
                        group["output_tokens_downstream"], errors="coerce"
                    ).max()
                    if "output_tokens_downstream" in group
                    else 0
                )
                total_tokens = int(input_tokens + output_tokens)
            crossed = [int(value) for value in budgets if int(value) <= total_tokens]
            checkpoint = max(crossed) if crossed else 0
            row = {
                **metadata,
                "actual_budget_tokens": total_tokens,
                "budget_checkpoint": checkpoint,
                "population_size_fairness_ready": int(len(valid)),
                "pareto_size": int(front.sum()),
                "hv_dev_3d": hypervolume_3d(normalized[front], reference_point),
                "best_quality": float(valid["quality"].max()),
                "minimum_cost": float(valid["cost"].min()),
                "minimum_fairness": float(valid["fairness"].min()),
                "mean_front_quality": float(valid.loc[front, "quality"].mean()),
                "mean_front_cost": float(valid.loc[front, "cost"].mean()),
                "mean_front_fairness": float(valid.loc[front, "fairness"].mean()),
                "wall_time_timestamp": (
                    float(pd.to_numeric(group["time"], errors="coerce").max())
                    if "time" in group
                    else float("nan")
                ),
            }
            rows.append(row)
        except Exception:
            if strict:
                raise
            logger.exception(
                "Could not analyse trajectory for %s step %s",
                metadata["run_key"],
                metadata["step"],
            )
    return pd.DataFrame(rows)


def _write_outputs(
    output_dir: Path,
    evaluations: pd.DataFrame,
    run_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    trajectory: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations.to_parquet(output_dir / "all_evaluations.parquet", index=False)
    run_metrics.to_csv(output_dir / "run_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    trajectory.to_csv(output_dir / "trajectory_metrics.csv", index=False)


def _make_plots(
    evaluations: pd.DataFrame,
    summary: pd.DataFrame,
    bounds: BoundsStore,
    output_dir: Path,
    *,
    plot_fronts: bool,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "hv_test_pessimistic_3d": "Pessimistic test hypervolume",
        "noisy_r2_3d": "Noisy R2",
        "approximation_gap_3d": "Approximation gap",
        "balanced_test_fairness": "Balanced-prompt test unfairness",
        "balanced_test_quality": "Balanced-prompt test quality",
    }
    for (dataset, model), _ in summary.groupby(["dataset", "model"], sort=True):
        for metric, label in metrics.items():
            if f"{metric}_mean" not in summary:
                continue
            plot_budget_metric(
                summary,
                dataset=str(dataset),
                model=str(model),
                metric=metric,
                ylabel=label,
                output_path=plot_dir / f"budget_{dataset}_{model}_{metric}.pdf",
            )

    if not plot_fronts:
        return
    for keys, group in evaluations.groupby(RUN_GROUP_COLUMNS, sort=True, dropna=False):
        metadata = dict(zip(RUN_GROUP_COLUMNS, keys))
        if int(metadata["budget_checkpoint"]) <= 0:
            continue
        title = (
            f"{metadata['optimizer']} | {metadata['dataset']} | {metadata['model']} | "
            f"seed {metadata['seed']} | {int(metadata['budget_checkpoint']):,} tokens"
        )
        stem = (
            f"{metadata['dataset']}_{metadata['model']}_{metadata['optimizer']}_"
            f"seed{metadata['seed']}_budget{metadata['budget_checkpoint']}"
        )
        plot_test_front_projections(
            group,
            bounds[(str(metadata["dataset"]), str(metadata["model"]))],
            plot_dir / "fronts",
            title=title,
            file_stem=stem,
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    budgets = parse_csv_ints(args.budgets)
    reference_point = parse_reference_point(args.reference_point)
    output_dir = Path(args.output_dir)

    evaluations = load_all_evaluations(
        args.results_root,
        budget_checkpoints=budgets,
        strict=args.strict,
    )
    bounds = load_or_create_bounds(
        evaluations,
        args.bounds_file,
        bounds_budget=args.bounds_budget,
        rebuild=args.rebuild_bounds,
    )
    run_metrics = compute_run_metrics(
        evaluations,
        bounds,
        n_preferences=args.n_preferences,
        preference_seed=args.preference_seed,
        reference_point=reference_point,
        strict=args.strict,
    )
    summary = aggregate_run_metrics(run_metrics)

    try:
        steps = load_all_step_results(args.results_root, strict=args.strict)
        trajectory = compute_trajectory_metrics(
            steps,
            bounds,
            budgets=budgets,
            reference_point=reference_point,
            strict=args.strict,
        )
    except FileNotFoundError:
        logger.warning("No step logs found; trajectory_metrics.csv will be empty")
        trajectory = pd.DataFrame()

    _write_outputs(output_dir, evaluations, run_metrics, summary, trajectory)
    if not args.skip_plots:
        _make_plots(
            evaluations,
            summary,
            bounds,
            output_dir,
            plot_fronts=args.plot_fronts,
        )
    logger.info("Tri-Fair analysis written to %s", output_dir)


if __name__ == "__main__":
    main()
