"""Build exact Tri-Fair v3 checkpoint and initial-baseline metric tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.metrics import aggregate_run_metrics, analyse_evaluated_run
from analysis.objectives import Bounds
from src.config.dataset_configs import ALL_DATASETS
from src.config.v3_profiles import apply_v3_dataset_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--initial-root", default=None)
    parser.add_argument("--output-dir", default="analysis/output/v3")
    parser.add_argument("--n-preferences", type=int, default=1000)
    parser.add_argument("--preference-seed", type=int, default=2026)
    parser.add_argument("--reference-point", default="1.1,1.1,1.1")
    return parser.parse_args()


def _json_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _coverage_valid(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or str(frame["dataset"].iloc[0]) != "bbq":
        return frame
    keep = np.ones(len(frame), dtype=bool)
    for split in ("dev", "test"):
        column = f"{split}_fairness_diagnostics_json"
        if column not in frame:
            keep &= False
            continue
        keep &= frame[column].map(
            lambda value: bool(_json_object(value).get("coverage_valid", False))
        ).to_numpy(dtype=bool)
    return frame.loc[keep].reset_index(drop=True)


def _load(root: Path, filename: str) -> pd.DataFrame:
    files = sorted(root.rglob(filename)) if root.is_dir() else []
    frames = [pd.read_parquet(path).assign(source_file=str(path)) for path in files]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _bounds(dataset: str) -> Bounds:
    config = apply_v3_dataset_profile(ALL_DATASETS[dataset])
    assert config.fairness is not None
    cost_max = float(
        config.fairness.fairness_kwargs.get("normalization_cost_upper_bound", 100.0)
    )
    return Bounds(
        minimum=np.asarray([0.0, 0.0, 0.0]),
        maximum=np.asarray([1.0, cost_max, 1.0]),
        source="tri_fair_v3_fixed_semantic_bounds",
    )



def _numeric_max(
    frame: pd.DataFrame, column: str, *, default: int | float
) -> float:
    if column not in frame:
        return float(default)
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.max()) if not values.empty else float(default)


def _run_metrics(
    evaluations: pd.DataFrame,
    *,
    n_preferences: int,
    preference_seed: int,
    reference_point: np.ndarray,
) -> pd.DataFrame:
    if evaluations.empty:
        return pd.DataFrame()
    identifiers = ["model", "dataset", "optimizer", "seed", "budget_checkpoint"]
    missing = set(identifiers) - set(evaluations.columns)
    if missing:
        raise ValueError(f"Evaluation table is missing {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for keys, group in evaluations.groupby(identifiers, sort=True, dropna=False):
        model, dataset, optimizer, seed, budget = keys
        valid = _coverage_valid(group.copy())
        if valid.empty:
            continue
        try:
            metrics = analyse_evaluated_run(
                valid,
                _bounds(str(dataset)),
                n_preferences=n_preferences,
                preference_seed=preference_seed,
                reference_point=reference_point,
            )
        except ValueError:
            continue
        rows.append(
            {
                "model": str(model),
                "dataset": str(dataset),
                "optimizer": str(optimizer),
                "seed": int(seed),
                "budget_checkpoint": int(budget),
                "chosen_step": int(
                    _numeric_max(group, "chosen_step", default=0)
                ),
                "actual_budget_tokens": int(
                    _numeric_max(
                        group, "actual_budget_tokens", default=int(budget)
                    )
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    reference = np.asarray(
        [float(value.strip()) for value in args.reference_point.split(",")],
        dtype=float,
    )
    if reference.shape != (3,):
        raise ValueError("--reference-point must contain three comma-separated values")

    results_root = Path(args.results_root).expanduser().resolve()
    checkpoint_evaluations = _load(results_root, "eval_checkpoints.parquet")
    initial_root = (
        Path(args.initial_root).expanduser().resolve()
        if args.initial_root
        else results_root
    )
    initial_evaluations = _load(initial_root, "initial_eval.parquet")
    combined = pd.concat(
        [checkpoint_evaluations, initial_evaluations],
        ignore_index=True,
        sort=False,
    )
    metrics = _run_metrics(
        combined,
        n_preferences=args.n_preferences,
        preference_seed=args.preference_seed,
        reference_point=reference,
    )
    if metrics.empty:
        raise RuntimeError("No valid v3 checkpoint or initial metrics were computed")

    summary = aggregate_run_metrics(metrics)
    if "n_runs" in summary and "n_seeds" not in summary:
        summary["n_seeds"] = summary["n_runs"]
    compact_columns = [
        "model",
        "dataset",
        "optimizer",
        "budget_checkpoint",
        "noisy_r2_3d_mean",
        "noisy_r2_3d_std",
        "hv_test_optimistic_3d_mean",
        "hv_test_optimistic_3d_std",
        "hv_test_pessimistic_3d_mean",
        "hv_test_pessimistic_3d_std",
        "approximation_gap_3d_mean",
        "approximation_gap_3d_std",
        "n_seeds",
    ]
    compact = summary[[column for column in compact_columns if column in summary]].copy()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "v3_checkpoint_run_metrics.csv", index=False)
    summary.to_csv(output_dir / "v3_checkpoint_summary.csv", index=False)
    compact.to_csv(output_dir / "v3_nr2_hv_gap_table.csv", index=False)
    markdown_path = output_dir / "v3_nr2_hv_gap_table.md"
    try:
        compact.to_markdown(markdown_path, index=False)
    except ImportError:
        markdown_path.write_text(
            "```text\n" + compact.to_string(index=False) + "\n```\n",
            encoding="utf-8",
        )
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
