"""Create publication-style BBQ 5M v2 figures for Tri-Fair and NSGA-II-PO-Fair.

This script is specific to the upgraded BBQ study stored under
``results/tri_fair_v2_2``.  It deliberately avoids the legacy 5M analysis
outputs and applies the v2 validity rules before computing publication metrics.

Key safeguards
--------------
1. Only the six intended Qwen-3-30B / BBQ / 5M runs are accepted:
   two optimizers × seeds 42, 43, and 44.
2. The old 1M evaluation rows retained in Tri-Fair seed 42's ``eval.parquet``
   are mapped below the configured 2M/3M/4M/5M checkpoint ladder and therefore
   cannot enter the exact 5M comparison.
3. Main holdout figures and multi-objective metrics use only rows satisfying
   ``test_fairness_ready`` and the BBQ diagnostic ``coverage_valid`` flag.
   Coverage-invalid candidates are retained only for a dedicated diagnostic
   figure and audit tables.
4. Objective normalization is deterministic and matches the v2 BBQ objective
   bounds: quality loss in [0, 1], weighted mean-token cost in [0, 100], and
   unfairness in [0, 1].  No legacy pre-v2 normalization file is reused.
5. Development trajectories are built only from the six selected 5M run
   histories; archived 1M or pilot runs are never mixed into the curves.
6. Cost labels describe the configured objective
   ``0.11 × mean input tokens + 0.41 × mean output tokens``.  They are not
   Rocket GPU charges and are not dollar costs.

Default usage on Rocket::

    python -m analysis.make_bbq_5m_figures_v2 --rebuild-analysis --strict

Generated outputs include development anytime trajectories, exact 5M
multi-objective comparisons, coverage-valid Pareto and attainment figures,
coverage-validity diagnostics, high-accuracy operating-point comparisons,
CSV/Markdown tables, and a selected-run manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_SCHEMA_VERSION = 2
MODEL = "qwen-3-30b"
DATASET = "bbq"
FINAL_BUDGET = 5_000_000
# The v2 direct-to-5M runs use the 2M/3M/4M/5M ladder.  Excluding 1M here
# prevents the earlier seed-42 evaluation from being treated as part of the
# exact v2 comparison.
DEFAULT_BUDGETS = (2_000_000, 3_000_000, 4_000_000, 5_000_000)
EXPECTED_SEEDS = (42, 43, 44)
N_PREFERENCES = 1_000
PREFERENCE_SEED = 2026
REFERENCE_POINT = (1.1, 1.1, 1.1)
FIXED_COST_UPPER_BOUND = 100.0
HIGH_ACCURACY_THRESHOLD = 0.90

# Qwen3-30B experiment objective weights.  This is an objective value, not a
# monetary bill: cost = 0.11 * mean input tokens + 0.41 * mean output tokens.
QWEN_INPUT_WEIGHT = 0.11
QWEN_OUTPUT_WEIGHT = 0.41
COST_AXIS_LABEL = "Weighted Mean-Token Cost ↓"
COST_SHORT_LABEL = "Cost Objective ↓"
UNFAIRNESS_AXIS_LABEL = "Statistical BBQ Unfairness ↓"

OPTIMIZER_ORDER = ["Tri-Fair", "NSGAII-PO-Fair"]
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


@dataclass(frozen=True)
class CostBounds:
    """Simple normalization bounds for development-side proxy metrics."""

    cost_max: float

    def normalize_minimize(self, raw: np.ndarray) -> np.ndarray:
        """Normalize minimize-all objectives [1-accuracy, cost, unfairness]."""
        out = np.asarray(raw, dtype=float).copy()
        out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
        out[:, 1] = np.clip(out[:, 1] / self.cost_max, 0.0, 1.1)
        out[:, 2] = np.clip(out[:, 2], 0.0, 1.0)
        return out


def parse_csv_ints(raw: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        values = [int(piece.strip().replace("_", "")) for piece in raw.split(",") if piece.strip()]
    else:
        values = [int(value) for value in raw]
    if not values:
        raise ValueError("At least one budget is required")
    return tuple(sorted(set(values)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="results/tri_fair_v2_2",
        help="Root containing the upgraded v2 result namespace.",
    )
    parser.add_argument(
        "--analysis-dir",
        default="analysis/output/bbq_5m_v2",
        help="Dedicated v2 analysis-table directory.",
    )
    parser.add_argument(
        "--figure-dir",
        default=None,
        help="Figure output directory (default: <analysis-dir>/publication_figures).",
    )
    parser.add_argument(
        "--bounds-file",
        default="analysis/normalization_bounds_bbq_v2.json",
        help="Dedicated deterministic v2 bounds file; legacy bounds are rejected.",
    )
    parser.add_argument(
        "--cost-upper-bound",
        type=float,
        default=FIXED_COST_UPPER_BOUND,
        help="Fixed cost-objective upper bound used by the v2 BBQ configuration.",
    )
    parser.add_argument(
        "--overwrite-bounds",
        action="store_true",
        help="Replace an incompatible dedicated v2 bounds file.",
    )
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in DEFAULT_BUDGETS),
        help="Checkpoint ladder used when attaching exact budget labels.",
    )
    parser.add_argument("--n-preferences", type=int, default=N_PREFERENCES)
    parser.add_argument("--preference-seed", type=int, default=PREFERENCE_SEED)
    parser.add_argument(
        "--reference-point",
        default=",".join(str(value) for value in REFERENCE_POINT),
        help="Normalized minimization reference point for hypervolume.",
    )
    parser.add_argument(
        "--high-accuracy-threshold",
        type=float,
        default=HIGH_ACCURACY_THRESHOLD,
        help="Accuracy threshold for the high-accuracy fairness operating point.",
    )
    parser.add_argument(
        "--rebuild-analysis",
        action="store_true",
        help="Force rebuilding the dedicated v2 analysis tables.",
    )
    parser.add_argument(
        "--no-auto-rebuild",
        action="store_true",
        help="Fail instead of rebuilding missing, stale, or legacy analysis tables.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on unreadable artifacts or missing v2 fairness diagnostics.",
    )
    return parser.parse_args(argv)


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


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the intended BBQ/Qwen runs and remove archived copies."""
    out = frame.copy()
    if "run_dir" in out:
        run_dir = out["run_dir"].astype(str)
        excluded = (
            run_dir.str.contains("pilot_reports", case=False, regex=False)
            | run_dir.str.contains("archive", case=False, regex=False)
            | run_dir.str.contains("tri_fair_v2_1", case=False, regex=False)
        )
        out = out[~excluded]
    if "dataset" in out:
        out = out[out["dataset"].astype(str) == DATASET]
    if "model" in out:
        out = out[out["model"].astype(str) == MODEL]
    if "optimizer" in out:
        out = out[out["optimizer"].astype(str).isin(OPTIMIZER_ORDER)]
    if "seed" in out:
        seed = pd.to_numeric(out["seed"], errors="coerce")
        out = out[seed.isin(EXPECTED_SEEDS)]
    return out.reset_index(drop=True)


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


def _diagnostic_column(
    frame: pd.DataFrame,
    source_column: str,
    key: str,
    *,
    default: object,
) -> pd.Series:
    if source_column not in frame:
        return pd.Series([default] * len(frame), index=frame.index)
    return frame[source_column].map(lambda value: _json_object(value).get(key, default))


def attach_bbq_validity(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse v2 diagnostic JSON and attach explicit validity columns."""
    out = frame.copy()
    for split in ("dev", "test"):
        diagnostics = f"{split}_fairness_diagnostics_json"
        out[f"{split}_coverage_valid"] = _diagnostic_column(
            out, diagnostics, "coverage_valid", default=False
        ).fillna(False).astype(bool)
        for key in (
            "disambig_coverage",
            "rms_abs_bias",
            "coverage_valid_unfairness",
            "worst_abs_bias",
            "max_ci_width",
        ):
            out[f"{split}_{key}"] = pd.to_numeric(
                _diagnostic_column(out, diagnostics, key, default=np.nan),
                errors="coerce",
            )

    dev_ready = (
        out["dev_fairness_ready"].fillna(False).astype(bool)
        if "dev_fairness_ready" in out
        else pd.Series(False, index=out.index)
    )
    test_ready = (
        out["test_fairness_ready"].fillna(False).astype(bool)
        if "test_fairness_ready" in out
        else pd.Series(False, index=out.index)
    )
    out["publication_valid"] = test_ready & out["test_coverage_valid"]
    out["strict_publication_valid"] = (
        dev_ready
        & out["dev_coverage_valid"]
        & test_ready
        & out["test_coverage_valid"]
    )
    return out


def publication_valid_evaluations(
    frame: pd.DataFrame,
    *,
    strict: bool,
) -> pd.DataFrame:
    """Keep holdout rows with statistical readiness and valid BBQ coverage."""
    enriched = attach_bbq_validity(frame)
    if strict:
        missing = [
            column
            for column in (
                "test_fairness_ready",
                "test_fairness_diagnostics_json",
            )
            if column not in enriched
        ]
        if missing:
            raise ValueError(f"Evaluation table lacks v2 validity fields: {missing}")
    valid = enriched[enriched["publication_valid"]].copy()
    if valid.empty:
        raise RuntimeError(
            "No publication-valid rows remain after requiring test fairness readiness "
            "and diagnostic coverage_valid=True"
        )
    return valid.reset_index(drop=True)


def analysis_paths(analysis_dir: Path) -> dict[str, Path]:
    return {
        "run_metrics": analysis_dir / "run_metrics.csv",
        "summary": analysis_dir / "summary.csv",
        "trajectory": analysis_dir / "trajectory_metrics.csv",
        "evaluations": analysis_dir / "all_evaluations_valid.parquet",
        "evaluations_raw": analysis_dir / "all_evaluations_raw.parquet",
        "manifest": analysis_dir / "analysis_manifest.json",
    }


def analysis_has_5m(
    analysis_dir: Path,
    *,
    results_root: Path,
    bounds_file: Path,
    cost_upper_bound: float,
) -> bool:
    paths = analysis_paths(analysis_dir)
    required = [
        paths["run_metrics"],
        paths["trajectory"],
        paths["evaluations"],
        paths["evaluations_raw"],
        paths["manifest"],
    ]
    if any(not path.is_file() for path in required):
        return False
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        run_metrics = clean_frame(pd.read_csv(paths["run_metrics"]))
    except Exception:
        return False
    if int(manifest.get("schema_version", -1)) != ANALYSIS_SCHEMA_VERSION:
        return False
    if Path(str(manifest.get("results_root", ""))).resolve() != results_root.resolve():
        return False
    if Path(str(manifest.get("bounds_file", ""))).resolve() != bounds_file.resolve():
        return False
    if not math.isclose(
        float(manifest.get("cost_upper_bound", float("nan"))),
        float(cost_upper_bound),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return False
    if "budget_checkpoint" not in run_metrics or run_metrics.empty:
        return False
    final = run_metrics[
        pd.to_numeric(run_metrics["budget_checkpoint"], errors="coerce") == FINAL_BUDGET
    ].copy()
    observed = {
        (str(row.optimizer), int(row.seed))
        for row in final[["optimizer", "seed"]].dropna().itertuples(index=False)
    }
    expected = {
        (optimizer, seed)
        for optimizer in OPTIMIZER_ORDER
        for seed in EXPECTED_SEEDS
    }
    return observed == expected


def attach_configured_budget(frame: pd.DataFrame, evaluations: pd.DataFrame) -> pd.DataFrame:
    """Attach configured_budget to pipeline outputs when the source eval has it."""
    if frame.empty or "run_dir" not in frame or "configured_budget" not in evaluations:
        return frame
    mapping = (
        evaluations[["run_dir", "configured_budget"]]
        .copy()
        .assign(
            configured_budget=lambda x: pd.to_numeric(
                x["configured_budget"], errors="coerce"
            )
        )
        .dropna(subset=["configured_budget"])
        .groupby("run_dir", as_index=False)["configured_budget"]
        .max()
    )
    if "configured_budget" in frame:
        frame = frame.drop(columns=["configured_budget"])
    return frame.merge(mapping, on="run_dir", how="left")


def ensure_v2_bounds(
    path: Path,
    *,
    cost_upper_bound: float,
    overwrite: bool,
):
    """Create or validate deterministic BBQ v2 normalization bounds."""
    if not np.isfinite(cost_upper_bound) or cost_upper_bound <= 0:
        raise ValueError("--cost-upper-bound must be a positive finite number")
    from analysis.objectives import Bounds, BoundsStore  # type: ignore

    expected_min = np.asarray([0.0, 0.0, 0.0], dtype=float)
    expected_max = np.asarray([1.0, float(cost_upper_bound), 1.0], dtype=float)
    if path.is_file() and not overwrite:
        store = BoundsStore.load(path)
        bounds = store[(DATASET, MODEL)]
        if not (
            np.allclose(bounds.minimum, expected_min, rtol=0.0, atol=1e-12)
            and np.allclose(bounds.maximum, expected_max, rtol=0.0, atol=1e-12)
        ):
            raise RuntimeError(
                f"Incompatible bounds in {path}. Expected minimum={expected_min.tolist()} "
                f"and maximum={expected_max.tolist()}. Use the dedicated v2 file or "
                "pass --overwrite-bounds deliberately."
            )
        return store

    store = BoundsStore(
        {
            (DATASET, MODEL): Bounds(
                minimum=expected_min,
                maximum=expected_max,
                source="bbq_v2_fixed_objective_bounds",
            )
        }
    )
    store.save(path)
    print(f"Wrote deterministic BBQ v2 bounds: {path.resolve()}")
    return store


def write_analysis_manifest(
    analysis_dir: Path,
    *,
    results_root: Path,
    bounds_file: Path,
    cost_upper_bound: float,
    budgets: tuple[int, ...],
    raw_rows: int,
    valid_rows: int,
) -> None:
    payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root.resolve()),
        "bounds_file": str(bounds_file.resolve()),
        "cost_upper_bound": float(cost_upper_bound),
        "budgets": list(map(int, budgets)),
        "final_budget": FINAL_BUDGET,
        "model": MODEL,
        "dataset": DATASET,
        "optimizers": OPTIMIZER_ORDER,
        "seeds": list(EXPECTED_SEEDS),
        "validity_rule": "test_fairness_ready and test diagnostics coverage_valid",
        "cost_definition": (
            f"{QWEN_INPUT_WEIGHT} * mean_input_tokens + "
            f"{QWEN_OUTPUT_WEIGHT} * mean_output_tokens"
        ),
        "raw_evaluation_rows": int(raw_rows),
        "publication_valid_rows": int(valid_rows),
    }
    (analysis_dir / "analysis_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rebuild_analysis_outputs(
    *,
    results_root: Path,
    analysis_dir: Path,
    bounds_file: Path,
    budgets: tuple[int, ...],
    n_preferences: int,
    preference_seed: int,
    reference_point: tuple[float, float, float],
    cost_upper_bound: float,
    overwrite_bounds: bool,
    strict: bool,
) -> None:
    """Rebuild exact v2 tables and recompute metrics on coverage-valid rows."""
    if not results_root.is_dir():
        raise FileNotFoundError(
            f"Results root does not exist: {results_root}\n"
            "Expected the live namespace results/tri_fair_v2_2 or pass --results-root."
        )

    from analysis.analysis_pipeline import (  # type: ignore
        compute_run_metrics,
        compute_trajectory_metrics,
    )
    from analysis.io import load_all_evaluations, load_all_step_results  # type: ignore
    from analysis.metrics import aggregate_run_metrics  # type: ignore

    print(f"\nRebuilding upgraded BBQ 5M tables from: {results_root.resolve()}")
    raw_evaluations = clean_frame(
        load_all_evaluations(
            results_root,
            budget_checkpoints=budgets,
            strict=strict,
        )
    )
    if raw_evaluations.empty:
        raise RuntimeError(
            f"No {DATASET}/{MODEL} evaluations were discovered beneath {results_root}"
        )
    raw_evaluations = attach_bbq_validity(raw_evaluations)
    valid_evaluations = publication_valid_evaluations(
        raw_evaluations,
        strict=strict,
    )

    bounds = ensure_v2_bounds(
        bounds_file,
        cost_upper_bound=cost_upper_bound,
        overwrite=overwrite_bounds,
    )
    run_metrics = compute_run_metrics(
        valid_evaluations,
        bounds,
        n_preferences=n_preferences,
        preference_seed=preference_seed,
        reference_point=reference_point,
        strict=strict,
    )
    run_metrics = attach_configured_budget(run_metrics, raw_evaluations)
    summary = aggregate_run_metrics(run_metrics)

    try:
        steps = clean_frame(load_all_step_results(results_root, strict=strict))
        trajectory = compute_trajectory_metrics(
            steps,
            bounds,
            budgets=budgets,
            reference_point=reference_point,
            strict=strict,
        )
        trajectory = attach_configured_budget(trajectory, raw_evaluations)
    except FileNotFoundError:
        warnings.warn("No step logs found; trajectory_metrics.csv will be empty")
        trajectory = pd.DataFrame()

    analysis_dir.mkdir(parents=True, exist_ok=True)
    raw_evaluations.to_parquet(
        analysis_dir / "all_evaluations_raw.parquet", index=False
    )
    valid_evaluations.to_parquet(
        analysis_dir / "all_evaluations_valid.parquet", index=False
    )
    run_metrics.to_csv(analysis_dir / "run_metrics.csv", index=False)
    summary.to_csv(analysis_dir / "summary.csv", index=False)
    trajectory.to_csv(analysis_dir / "trajectory_metrics.csv", index=False)
    write_analysis_manifest(
        analysis_dir,
        results_root=results_root,
        bounds_file=bounds_file,
        cost_upper_bound=cost_upper_bound,
        budgets=budgets,
        raw_rows=len(raw_evaluations),
        valid_rows=len(valid_evaluations),
    )
    print(
        f"Rebuilt analysis tables in {analysis_dir.resolve()} "
        f"({len(valid_evaluations)}/{len(raw_evaluations)} rows publication-valid)"
    )


def read_inputs(
    analysis_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = analysis_paths(analysis_dir)
    required = [
        paths["run_metrics"],
        paths["trajectory"],
        paths["evaluations"],
        paths["evaluations_raw"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required BBQ v2 analysis outputs:\n"
            + "\n".join(missing)
            + "\nRun with --rebuild-analysis or allow the default auto-rebuild."
        )

    run_metrics = clean_frame(pd.read_csv(paths["run_metrics"]))
    try:
        trajectory = clean_frame(pd.read_csv(paths["trajectory"]))
    except pd.errors.EmptyDataError:
        trajectory = pd.DataFrame()
    evaluations = publication_valid_evaluations(
        clean_frame(pd.read_parquet(paths["evaluations"])),
        strict=True,
    )
    raw_evaluations = attach_bbq_validity(
        clean_frame(pd.read_parquet(paths["evaluations_raw"]))
    )
    return run_metrics, trajectory, evaluations, raw_evaluations


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def selected_final_runs(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Select exactly one 5M run per method/seed and report duplicates clearly."""
    require_columns(
        run_metrics,
        ["optimizer", "seed", "budget_checkpoint", "run_key", "run_dir", "actual_budget_tokens"],
        "run_metrics.csv",
    )
    final = run_metrics[
        pd.to_numeric(run_metrics["budget_checkpoint"], errors="coerce") == FINAL_BUDGET
    ].copy()
    if "configured_budget" in final and final["configured_budget"].notna().any():
        configured = pd.to_numeric(final["configured_budget"], errors="coerce")
        preferred = final[configured >= FINAL_BUDGET].copy()
        if not preferred.empty:
            final = preferred

    if final.empty:
        observed = sorted(
            set(
                pd.to_numeric(run_metrics["budget_checkpoint"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
        )
        raise RuntimeError(
            "No 5M checkpoint rows were found after cleaning. "
            f"Observed checkpoints: {observed}. The tables are stale or the 5M bundle "
            "is not under --results-root."
        )

    final["seed"] = pd.to_numeric(final["seed"], errors="raise").astype(int)
    final["actual_budget_tokens"] = pd.to_numeric(
        final["actual_budget_tokens"], errors="coerce"
    )

    chosen_rows: list[pd.Series] = []
    duplicate_notes: list[str] = []
    for optimizer in OPTIMIZER_ORDER:
        for seed in EXPECTED_SEEDS:
            group = final[(final["optimizer"] == optimizer) & (final["seed"] == seed)].copy()
            if group.empty:
                continue
            group = group.sort_values(
                ["actual_budget_tokens", "run_dir"], ascending=[False, True]
            )
            if len(group) > 1:
                duplicate_notes.append(
                    f"{optimizer} seed {seed}: selected {group.iloc[0]['run_dir']} from {len(group)} candidates"
                )
            chosen_rows.append(group.iloc[0])

    if chosen_rows:
        selected = pd.DataFrame(chosen_rows).reset_index(drop=True)
    else:
        selected = pd.DataFrame(columns=final.columns)

    missing_pairs = [
        (optimizer, seed)
        for optimizer in OPTIMIZER_ORDER
        for seed in EXPECTED_SEEDS
        if selected.empty
        or selected[
            (selected["optimizer"] == optimizer) & (selected["seed"] == seed)
        ].empty
    ]
    if missing_pairs:
        raise RuntimeError(
            "The complete six-run 5M grid was not found. Missing: "
            + ", ".join(f"{optimizer}/seed{seed}" for optimizer, seed in missing_pairs)
        )

    if duplicate_notes:
        warnings.warn("Duplicate 5M candidates were found:\n" + "\n".join(duplicate_notes))

    counts = (
        run_metrics.groupby(["optimizer", "budget_checkpoint"])
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["optimizer", "budget_checkpoint"])
    )
    print("\nRun-metric counts after cleaning")
    print(counts.to_string(index=False))
    print("\nSelected 5M runs")
    print(
        selected[
            ["optimizer", "seed", "actual_budget_tokens", "run_dir"]
        ].sort_values(["optimizer", "seed"]).to_string(index=False)
    )
    return selected


def filter_to_selected_runs(frame: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "run_key" not in frame:
        return frame.copy()
    keys = set(selected["run_key"].astype(str))
    return frame[frame["run_key"].astype(str).isin(keys)].reset_index(drop=True)


def final_checkpoint_evaluations(
    evaluations: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    require_columns(
        evaluations,
        [
            "optimizer",
            "seed",
            "run_key",
            "budget_checkpoint",
            "test_quality",
            "test_cost",
            "test_fairness",
        ],
        "all_evaluations.parquet",
    )
    data = filter_to_selected_runs(evaluations, selected)
    data = data[
        pd.to_numeric(data["budget_checkpoint"], errors="coerce") == FINAL_BUDGET
    ].copy()
    if data.empty:
        raise RuntimeError("No selected 5M evaluation rows were found")
    return data.reset_index(drop=True)


def checkpoint_run_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Choose one run per method/seed/evaluated checkpoint without duplication."""
    frame = run_metrics.copy()
    frame["budget_checkpoint"] = pd.to_numeric(
        frame["budget_checkpoint"], errors="coerce"
    )
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce")
    frame = frame.dropna(subset=["budget_checkpoint", "seed"])
    frame["budget_checkpoint"] = frame["budget_checkpoint"].astype(int)
    frame["seed"] = frame["seed"].astype(int)
    frame = frame[frame["budget_checkpoint"].isin(DEFAULT_BUDGETS)]

    selected_rows: list[pd.Series] = []
    for (_, _, checkpoint_value), group in frame.groupby(
        ["optimizer", "seed", "budget_checkpoint"], sort=True
    ):
        candidate = group.copy()
        if "configured_budget" in candidate and candidate["configured_budget"].notna().any():
            candidate["configured_budget"] = pd.to_numeric(
                candidate["configured_budget"], errors="coerce"
            )
            exact = candidate[candidate["configured_budget"] == checkpoint_value]
            if not exact.empty:
                candidate = exact
            else:
                eligible = candidate[candidate["configured_budget"] >= checkpoint_value]
                if not eligible.empty:
                    minimum = eligible["configured_budget"].min()
                    candidate = eligible[eligible["configured_budget"] == minimum]
        candidate = candidate.sort_values(
            ["actual_budget_tokens", "run_dir"], ascending=[False, True]
        )
        selected_rows.append(candidate.iloc[0])
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def complete_checkpoint_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain checkpoints represented by all six independent method/seed runs."""
    if frame.empty:
        raise RuntimeError("No exact checkpoint metrics are available")
    expected = {
        (optimizer, seed)
        for optimizer in OPTIMIZER_ORDER
        for seed in EXPECTED_SEEDS
    }
    complete: list[int] = []
    for budget, group in frame.groupby("budget_checkpoint", sort=True):
        observed = {
            (str(row.optimizer), int(row.seed))
            for row in group[["optimizer", "seed"]].dropna().itertuples(index=False)
        }
        if observed == expected:
            complete.append(int(budget))
    if FINAL_BUDGET not in complete:
        raise RuntimeError(
            "The exact 5M checkpoint is not represented by all six method/seed runs"
        )
    dropped = sorted(set(frame["budget_checkpoint"].astype(int)) - set(complete))
    if dropped:
        warnings.warn(
            "Omitting incomplete exact holdout checkpoints: "
            + ", ".join(format_budget(value) for value in dropped)
        )
    return frame[frame["budget_checkpoint"].isin(complete)].reset_index(drop=True)


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


def make_preferences(n_preferences: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(3), size=n_preferences)


def chebyshev_r2(normalized_minimize_front: np.ndarray, weights: np.ndarray) -> float:
    """Lower is better; development-side proxy for nR2/R2-style utility."""
    if len(normalized_minimize_front) == 0:
        return float("nan")
    front = np.asarray(normalized_minimize_front, dtype=float)
    utilities = np.max(weights[:, None, :] * front[None, :, :], axis=2)
    return float(np.mean(np.min(utilities, axis=1)))


def finite_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def load_step_results_from_selected_runs(selected: pd.DataFrame) -> pd.DataFrame:
    unique_runs = selected[
        ["run_key", "run_dir", "model", "dataset", "optimizer", "seed"]
    ].drop_duplicates()
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for _, row in unique_runs.iterrows():
        path = Path(str(row["run_dir"])) / "step_results.parquet"
        if not path.is_file():
            missing.append(str(path))
            continue
        frame = pd.read_parquet(path).copy()
        frame["run_key"] = row["run_key"]
        frame["run_dir"] = row["run_dir"]
        frame["model"] = row["model"]
        frame["dataset"] = row["dataset"]
        frame["optimizer"] = row["optimizer"]
        frame["seed"] = int(row["seed"])
        frames.append(frame)
    if missing:
        raise FileNotFoundError(
            "Missing step_results.parquet for selected 5M runs:\n" + "\n".join(missing)
        )
    if not frames:
        raise RuntimeError("No selected 5M step_results.parquet files were loaded")
    return pd.concat(frames, ignore_index=True, sort=False)


def format_budget(value: int | float) -> str:
    number = float(value)
    if number >= 1_000_000:
        return f"{number / 1_000_000:g}M"
    if number >= 1_000:
        return f"{number / 1_000:g}K"
    return f"{number:g}"


def infer_cost_bounds(
    step_results: pd.DataFrame,
    evaluations: pd.DataFrame,
    *,
    cost_upper_bound: float = FIXED_COST_UPPER_BOUND,
) -> CostBounds:
    """Return the fixed v2 cost bound; data-dependent rescaling is forbidden."""
    del step_results, evaluations
    if not np.isfinite(cost_upper_bound) or cost_upper_bound <= 0:
        raise ValueError("cost_upper_bound must be positive and finite")
    return CostBounds(cost_max=float(cost_upper_bound))


def stepwise_development_r2_proxy(
    step_results: pd.DataFrame,
    evaluations: pd.DataFrame,
    *,
    cost_upper_bound: float,
) -> pd.DataFrame:
    bounds = infer_cost_bounds(
        step_results,
        evaluations,
        cost_upper_bound=cost_upper_bound,
    )
    weights = make_preferences(N_PREFERENCES, PREFERENCE_SEED)
    rows: list[dict[str, object]] = []
    group_cols = ["run_key", "run_dir", "model", "dataset", "optimizer", "seed", "step"]

    for keys, group in step_results.groupby(group_cols, sort=True, dropna=False):
        meta = dict(zip(group_cols, keys))
        require_columns(group, ["quality", "cost", "fairness"], "step_results.parquet")
        quality = finite_numeric(group["quality"])
        cost = finite_numeric(group["cost"])
        fairness = finite_numeric(group["fairness"])
        valid_mask = np.isfinite(quality) & np.isfinite(cost) & np.isfinite(fairness)
        if "fairness_ready" in group:
            valid_mask &= group["fairness_ready"].fillna(False).astype(bool).to_numpy()
        if not valid_mask.any():
            continue

        raw = np.column_stack(
            [1.0 - quality[valid_mask], cost[valid_mask], fairness[valid_mask]]
        )
        normalized = bounds.normalize_minimize(raw)
        front_mask = pareto_mask_minimize(normalized)

        if "fairness_diagnostics_json" in group:
            coverage_valid = group["fairness_diagnostics_json"].map(
                lambda value: bool(_json_object(value).get("coverage_valid", False))
            ).to_numpy(dtype=bool)
            coverage_fraction = float(np.mean(coverage_valid[valid_mask]))
        else:
            coverage_fraction = float("nan")

        if "total_tokens_downstream" in group:
            tokens = int(np.nanmax(finite_numeric(group["total_tokens_downstream"])))
        else:
            input_tokens = (
                np.nanmax(finite_numeric(group["input_tokens_downstream"]))
                if "input_tokens_downstream" in group
                else 0.0
            )
            output_tokens = (
                np.nanmax(finite_numeric(group["output_tokens_downstream"]))
                if "output_tokens_downstream" in group
                else 0.0
            )
            tokens = int(input_tokens + output_tokens)

        rows.append(
            {
                **meta,
                "actual_budget_tokens": tokens,
                "dev_noisy_r2_proxy": chebyshev_r2(normalized[front_mask], weights),
                "dev_proxy_front_size": int(front_mask.sum()),
                "dev_coverage_valid_fraction": coverage_fraction,
            }
        )
    if not rows:
        raise RuntimeError("Could not compute stepwise development nR2 proxy")
    return pd.DataFrame(rows)


def prepare_staircase_grid(data: pd.DataFrame, max_budget: int | None = None) -> np.ndarray:
    values = finite_numeric(data["actual_budget_tokens"])
    values = values[np.isfinite(values)]
    if max_budget is not None:
        values = values[values <= max_budget]
    values = np.unique(values.astype(int))
    return np.sort(values[values > 0])


def interpolate_step_values(run_data: pd.DataFrame, grid: np.ndarray, value_col: str) -> np.ndarray:
    data = run_data.sort_values("actual_budget_tokens")
    x = finite_numeric(data["actual_budget_tokens"])
    y = finite_numeric(data[value_col])
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return np.full(len(grid), np.nan)
    collapsed = (
        pd.DataFrame({"x": x[mask].astype(int), "y": y[mask]})
        .groupby("x", as_index=False)["y"]
        .last()
    )
    x_values = collapsed["x"].to_numpy(dtype=int)
    y_values = collapsed["y"].to_numpy(dtype=float)
    result = np.full(len(grid), np.nan)
    for index, grid_value in enumerate(grid):
        position = np.searchsorted(x_values, grid_value, side="right") - 1
        if position >= 0:
            result[index] = y_values[position]
    return result


def step_band_statistics(
    data: pd.DataFrame,
    value_col: str,
    *,
    max_budget: int | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    if data.empty or value_col not in data:
        return result
    for optimizer in OPTIMIZER_ORDER:
        optimizer_data = data[data["optimizer"] == optimizer].copy()
        if optimizer_data.empty:
            continue
        grid = prepare_staircase_grid(optimizer_data, max_budget=max_budget)
        if len(grid) == 0:
            continue
        seed_values = [
            interpolate_step_values(seed_group, grid, value_col)
            for _, seed_group in optimizer_data.groupby("seed", sort=True)
        ]
        matrix = np.vstack(seed_values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(matrix, axis=0)
            std = np.nanstd(matrix, axis=0)
        valid = np.isfinite(mean)
        result[optimizer] = (
            grid[valid],
            mean[valid],
            (mean - std)[valid],
            (mean + std)[valid],
        )
    return result


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_stepwise_nr2_proxy(
    step_proxy: pd.DataFrame,
    outdir: Path,
    *,
    max_budget: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    stats = step_band_statistics(
        step_proxy,
        "dev_noisy_r2_proxy",
        max_budget=max_budget,
    )
    for optimizer in OPTIMIZER_ORDER:
        if optimizer not in stats:
            continue
        grid, mean, lower, upper = stats[optimizer]
        x = grid / 1_000_000.0
        ax.step(
            x,
            mean,
            where="post",
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            markevery=max(1, len(x) // 8),
            linewidth=2,
            markersize=4,
            label=DISPLAY_NAME[optimizer],
        )
        ax.fill_between(
            x, lower, upper, step="post", color=COLORS[optimizer], alpha=0.16
        )
    ax.set_title("BBQ v2 — Qwen-3-30B")
    ax.set_xlabel("Cumulative Downstream Tokens [×10⁶]")
    ax.set_ylabel("Development nR2 Proxy ↓")
    ax.set_xlim(left=0.0, right=max_budget / 1_000_000.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.02,
        "Development-only proxy; exact holdout nR2 is available at 5M only.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        alpha=0.75,
    )
    save_figure(fig, outdir, "bbq_qwen3_5m_v2_stepwise_dev_nr2_proxy")


def plot_stepwise_coverage_validity(
    step_proxy: pd.DataFrame,
    outdir: Path,
    *,
    max_budget: int,
) -> None:
    if "dev_coverage_valid_fraction" not in step_proxy:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    stats = step_band_statistics(
        step_proxy,
        "dev_coverage_valid_fraction",
        max_budget=max_budget,
    )
    for optimizer in OPTIMIZER_ORDER:
        if optimizer not in stats:
            continue
        grid, mean, lower, upper = stats[optimizer]
        x = grid / 1_000_000.0
        ax.step(
            x,
            mean,
            where="post",
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            markevery=max(1, len(x) // 8),
            linewidth=2,
            markersize=4,
            label=DISPLAY_NAME[optimizer],
        )
        ax.fill_between(
            x,
            np.clip(lower, 0.0, 1.0),
            np.clip(upper, 0.0, 1.0),
            step="post",
            color=COLORS[optimizer],
            alpha=0.16,
        )
    ax.axhline(1.0, color="0.5", linewidth=1, linestyle="--")
    ax.set_title("BBQ v2 — Development Coverage Validity")
    ax.set_xlabel("Cumulative Downstream Tokens [×10⁶]")
    ax.set_ylabel("Coverage-Valid Candidate Fraction")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(left=0.0, right=max_budget / 1_000_000.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, outdir, "bbq_qwen3_5m_v2_stepwise_coverage_validity")


def plot_stepwise_hv_and_gap(
    selected_trajectory: pd.DataFrame,
    checkpoint_metrics: pd.DataFrame,
    outdir: Path,
    *,
    max_budget: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    hv_stats = step_band_statistics(
        selected_trajectory,
        "hv_dev_3d",
        max_budget=max_budget,
    )
    for optimizer in OPTIMIZER_ORDER:
        if optimizer not in hv_stats:
            continue
        grid, mean, lower, upper = hv_stats[optimizer]
        x = grid / 1_000_000.0
        axes[0].step(
            x,
            mean,
            where="post",
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            markevery=max(1, len(x) // 8),
            linewidth=2,
            markersize=4,
            label=DISPLAY_NAME[optimizer],
        )
        axes[0].fill_between(
            x, lower, upper, step="post", color=COLORS[optimizer], alpha=0.16
        )
    axes[0].set_title("Development Hypervolume ↑")
    axes[0].set_xlabel("Cumulative Downstream Tokens [×10⁶]")
    axes[0].set_ylabel("Development HV")
    axes[0].set_xlim(left=0.0, right=max_budget / 1_000_000.0)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    plot_checkpoint_metric_on_axis(
        axes[1],
        checkpoint_metrics,
        metric="approximation_gap_3d",
        title="Exact Holdout Approximation Gap ↓",
        ylabel="Approximation Gap",
    )
    fig.suptitle("BBQ v2 — Qwen-3-30B")
    save_figure(fig, outdir, "bbq_qwen3_5m_v2_stepwise_hv_exact_gap")


def plot_checkpoint_metric_on_axis(
    ax: plt.Axes,
    checkpoint_metrics: pd.DataFrame,
    *,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    if metric not in checkpoint_metrics:
        ax.text(0.5, 0.5, f"Missing {metric}", ha="center", va="center")
        ax.set_axis_off()
        return
    budgets = sorted(
        pd.to_numeric(
            checkpoint_metrics["budget_checkpoint"], errors="coerce"
        ).dropna().astype(int).unique()
    )
    if not budgets:
        ax.text(0.5, 0.5, "No complete checkpoint", ha="center", va="center")
        ax.set_axis_off()
        return

    if len(budgets) == 1:
        budget = budgets[0]
        means: list[float] = []
        stds: list[float] = []
        labels: list[str] = []
        colors: list[str] = []
        for optimizer in OPTIMIZER_ORDER:
            values = pd.to_numeric(
                checkpoint_metrics.loc[
                    (checkpoint_metrics["optimizer"] == optimizer)
                    & (checkpoint_metrics["budget_checkpoint"] == budget),
                    metric,
                ],
                errors="coerce",
            ).dropna()
            means.append(float(values.mean()))
            stds.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
            labels.append(DISPLAY_NAME[optimizer])
            colors.append(COLORS[optimizer])
        positions = np.arange(len(labels))
        ax.bar(
            positions,
            means,
            yerr=stds,
            capsize=4,
            color=colors,
            alpha=0.82,
        )
        ax.set_xticks(positions, labels, rotation=8)
        ax.set_xlabel(f"Exact {format_budget(budget)} evaluation")
    else:
        for optimizer in OPTIMIZER_ORDER:
            data = checkpoint_metrics[checkpoint_metrics["optimizer"] == optimizer]
            grouped = (
                data.groupby("budget_checkpoint")[metric]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("budget_checkpoint")
            )
            if grouped.empty:
                continue
            x = grouped["budget_checkpoint"].to_numpy(dtype=float) / 1_000_000.0
            mean = grouped["mean"].to_numpy(dtype=float)
            std = grouped["std"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(
                x,
                mean,
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                linewidth=2,
                markersize=5,
                label=DISPLAY_NAME[optimizer],
            )
            ax.fill_between(
                x, mean - std, mean + std, color=COLORS[optimizer], alpha=0.16
            )
        ax.set_xlabel("Exact Evaluated Checkpoint [×10⁶ tokens]")
        ax.set_xticks(
            [value / 1_000_000.0 for value in budgets],
            [format_budget(value) for value in budgets],
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def plot_exact_checkpoint_mo_metrics(checkpoint_metrics: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    specifications = [
        ("noisy_r2_3d", "Exact Holdout nR2 ↓", "nR2"),
        ("hv_test_pessimistic_3d", "Pessimistic Test HV ↑", "Hypervolume"),
        ("approximation_gap_3d", "Approximation Gap ↓", "Gap"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, specifications):
        plot_checkpoint_metric_on_axis(
            ax,
            checkpoint_metrics,
            metric=metric,
            title=title,
            ylabel=ylabel,
        )
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("BBQ v2 — Qwen-3-30B: Exact Complete Checkpoints")
    save_figure(fig, outdir, "bbq_qwen3_5m_v2_exact_nr2_hv_gap")


def plot_balanced_budget_trajectory(checkpoint_metrics: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    specifications = [
        ("balanced_test_quality", "Balanced Test Accuracy ↑", "Accuracy"),
        ("balanced_test_cost", "Balanced Test Cost ↓", COST_SHORT_LABEL),
        ("balanced_test_fairness", "Balanced Test Unfairness ↓", UNFAIRNESS_AXIS_LABEL),
    ]
    for ax, (metric, title, ylabel) in zip(axes, specifications):
        plot_checkpoint_metric_on_axis(
            ax,
            checkpoint_metrics,
            metric=metric,
            title=title,
            ylabel=ylabel,
        )
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle("BBQ v2 — Qwen-3-30B: Exact Balanced Operating Point")
    save_figure(fig, outdir, "bbq_qwen3_5m_v2_balanced_exact_metrics")


def test_objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    quality = finite_numeric(frame["test_quality"])
    cost = finite_numeric(frame["test_cost"])
    fairness = finite_numeric(frame["test_fairness"])
    return np.column_stack([1.0 - quality, cost, fairness])


def y_attained_at_x(data: pd.DataFrame, x_grid: np.ndarray, x_col: str) -> np.ndarray:
    x = finite_numeric(data[x_col])
    y = finite_numeric(data["test_quality"])
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    result = np.full(len(x_grid), np.nan)
    for index, grid_value in enumerate(x_grid):
        eligible = y[x <= grid_value]
        if len(eligible):
            result[index] = np.max(eligible)
    return result


def plot_empirical_attainment(
    final: pd.DataFrame,
    outdir: Path,
    *,
    x_col: str,
    xlabel: str,
    filename: str,
) -> None:
    xmin = float(np.nanmin(finite_numeric(final[x_col])))
    xmax = float(np.nanmax(finite_numeric(final[x_col])))
    padding = 0.03 * max(xmax - xmin, 1e-6)
    x_grid = np.linspace(xmin - padding, xmax + padding, 400)
    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)

    for optimizer in OPTIMIZER_ORDER:
        optimizer_data = final[final["optimizer"] == optimizer]
        curves: list[np.ndarray] = []
        for _, seed_group in optimizer_data.groupby("seed", sort=True):
            valid = np.all(np.isfinite(test_objective_matrix(seed_group)), axis=1)
            seed_group = seed_group.loc[valid].reset_index(drop=True)
            if seed_group.empty:
                continue
            front = seed_group.loc[pareto_mask_minimize(test_objective_matrix(seed_group))]
            curves.append(y_attained_at_x(front, x_grid, x_col))
        if not curves:
            continue
        matrix = np.vstack(curves)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
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
            label=DISPLAY_NAME[optimizer],
        )
        ax.fill_between(
            x_grid[valid], lower[valid], upper[valid], step="post", color=COLORS[optimizer], alpha=0.16
        )
    ax.set_title("BBQ v2 — Qwen-3-30B at 5M")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, outdir, filename)


def plot_cost_unfairness_projection(final: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    for optimizer in OPTIMIZER_ORDER:
        optimizer_data = final[final["optimizer"] == optimizer]
        pooled_fronts: list[pd.DataFrame] = []
        for seed, seed_group in optimizer_data.groupby("seed", sort=True):
            valid = np.all(np.isfinite(test_objective_matrix(seed_group)), axis=1)
            seed_group = seed_group.loc[valid].reset_index(drop=True)
            if seed_group.empty:
                continue
            front = seed_group.loc[pareto_mask_minimize(test_objective_matrix(seed_group))].copy()
            pooled_fronts.append(front)
            ax.scatter(
                front["test_cost"],
                front["test_fairness"],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                alpha=0.22,
                s=25,
            )
        if not pooled_fronts:
            continue
        pooled = pd.concat(pooled_fronts, ignore_index=True)
        projection = np.column_stack(
            [finite_numeric(pooled["test_cost"]), finite_numeric(pooled["test_fairness"])]
        )
        valid = np.all(np.isfinite(projection), axis=1)
        pooled = pooled.loc[valid].reset_index(drop=True)
        projection = projection[valid]
        front = pooled.loc[pareto_mask_minimize(projection)].sort_values("test_cost")
        ax.plot(
            front["test_cost"],
            front["test_fairness"],
            color=COLORS[optimizer],
            marker=MARKERS[optimizer],
            linewidth=2,
            markersize=5,
            label=DISPLAY_NAME[optimizer],
        )
    ax.set_title("BBQ v2 — 5M Cost vs Test Unfairness")
    ax.set_xlabel(COST_AXIS_LABEL)
    ax.set_ylabel(UNFAIRNESS_AXIS_LABEL)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, outdir, "bbq_qwen3_5m_v2_pareto_cost_unfairness")


def plot_three_objective_pareto(final: pd.DataFrame, outdir: Path) -> None:
    fig = plt.figure(figsize=(7.2, 5.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for optimizer in OPTIMIZER_ORDER:
        optimizer_data = final[final["optimizer"] == optimizer]
        for seed, seed_group in optimizer_data.groupby("seed", sort=True):
            matrix = test_objective_matrix(seed_group)
            valid = np.all(np.isfinite(matrix), axis=1)
            seed_group = seed_group.loc[valid].reset_index(drop=True)
            matrix = matrix[valid]
            if seed_group.empty:
                continue
            front = seed_group.loc[pareto_mask_minimize(matrix)]
            ax.scatter(
                front["test_cost"],
                front["test_fairness"],
                front["test_quality"],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                alpha=0.78,
                s=34,
                label=DISPLAY_NAME[optimizer] if seed == EXPECTED_SEEDS[0] else None,
            )
    ax.set_title("BBQ v2 — Qwen-3-30B 5M Test Pareto Fronts")
    ax.set_xlabel(COST_SHORT_LABEL)
    ax.set_ylabel(UNFAIRNESS_AXIS_LABEL)
    ax.set_zlabel("Accuracy ↑")
    ax.legend(frameon=False)
    save_figure(fig, outdir, "bbq_qwen3_5m_v2_test_pareto_3d")


def plot_method_comparison_bars(final_metrics: pd.DataFrame, outdir: Path) -> None:
    specifications = [
        ("hv_test_optimistic_3d", "Optimistic Test HV ↑"),
        ("hv_test_pessimistic_3d", "Pessimistic Test HV ↑"),
        ("noisy_r2_3d", "Holdout nR2 ↓"),
        ("approximation_gap_3d", "Approximation Gap ↓"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), specifications):
        means: list[float] = []
        stds: list[float] = []
        labels: list[str] = []
        colors: list[str] = []
        for optimizer in OPTIMIZER_ORDER:
            values = pd.to_numeric(
                final_metrics.loc[final_metrics["optimizer"] == optimizer, metric],
                errors="coerce",
            ).dropna()
            means.append(float(values.mean()))
            stds.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
            labels.append(DISPLAY_NAME[optimizer])
            colors.append(COLORS[optimizer])
        positions = np.arange(len(labels))
        bars = ax.bar(positions, means, yerr=stds, capsize=4, color=colors, alpha=0.82)
        ax.set_xticks(positions, labels, rotation=8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("BBQ v2 — Qwen-3-30B at 5M (mean ± SD, 3 seeds)")
    save_figure(fig, outdir, "bbq_qwen3_5m_v2_method_comparison_hv_nr2_gap")


def parse_fewshot_count(value: object) -> int:
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


def output_cost_share(frame: pd.DataFrame) -> np.ndarray:
    require_columns(frame, ["test_input_tokens", "test_output_tokens"], "evaluations")
    input_tokens = finite_numeric(frame["test_input_tokens"])
    output_tokens = finite_numeric(frame["test_output_tokens"])
    weighted_input = QWEN_INPUT_WEIGHT * input_tokens
    weighted_output = QWEN_OUTPUT_WEIGHT * output_tokens
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


def trifair_final_candidates(final: pd.DataFrame) -> pd.DataFrame:
    data = final[final["optimizer"] == "Tri-Fair"].copy()
    if "is_incumbent" in data and data["is_incumbent"].notna().any():
        incumbent = data[data["is_incumbent"].fillna(False).astype(bool)]
        if not incumbent.empty:
            data = incumbent.copy()
    if data.empty:
        raise RuntimeError("No Tri-Fair 5M candidate rows were found")
    if "few_shots_json" in data:
        data["fewshot_count"] = data["few_shots_json"].apply(parse_fewshot_count)
    else:
        data["fewshot_count"] = 0
    data["output_cost_share"] = output_cost_share(data)
    return data.reset_index(drop=True)


def plot_trifair_diagnostic(
    final: pd.DataFrame,
    outdir: Path,
    *,
    color_col: str,
    color_label: str,
    filename: str,
) -> None:
    data = trifair_final_candidates(final)
    values = finite_numeric(data[color_col])
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
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
    ax.set_title("Tri-Fair on BBQ v2 — Qwen-3-30B at 5M")
    ax.set_xlabel(COST_AXIS_LABEL)
    ax.set_ylabel("Test Accuracy")
    ax.grid(True, alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(color_label)
    save_figure(fig, outdir, filename)


def coverage_validity_summary(final_raw: pd.DataFrame) -> pd.DataFrame:
    data = attach_bbq_validity(final_raw)
    rows: list[dict[str, object]] = []
    for optimizer in OPTIMIZER_ORDER:
        for seed in EXPECTED_SEEDS:
            group = data[
                (data["optimizer"] == optimizer)
                & (pd.to_numeric(data["seed"], errors="coerce") == seed)
            ]
            total = int(len(group))
            test_valid = int(group["publication_valid"].sum()) if total else 0
            strict_valid = int(group["strict_publication_valid"].sum()) if total else 0
            rows.append(
                {
                    "optimizer": optimizer,
                    "method": DISPLAY_NAME[optimizer],
                    "seed": seed,
                    "total_candidates": total,
                    "test_coverage_valid_candidates": test_valid,
                    "strict_dev_and_test_valid_candidates": strict_valid,
                    "test_coverage_valid_fraction": (
                        float(test_valid / total) if total else float("nan")
                    ),
                    "strict_valid_fraction": (
                        float(strict_valid / total) if total else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_coverage_validity_diagnostics(final_raw: pd.DataFrame, outdir: Path) -> None:
    data = attach_bbq_validity(final_raw)
    summary = coverage_validity_summary(data)
    summary.to_csv(
        outdir / "bbq_qwen3_5m_v2_coverage_validity_by_seed.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    positions = np.arange(len(EXPECTED_SEEDS), dtype=float)
    width = 0.36
    for index, optimizer in enumerate(OPTIMIZER_ORDER):
        method = summary[summary["optimizer"] == optimizer].sort_values("seed")
        offset = (index - 0.5) * width
        axes[0].bar(
            positions + offset,
            method["test_coverage_valid_fraction"],
            width=width,
            color=COLORS[optimizer],
            alpha=0.82,
            label=DISPLAY_NAME[optimizer],
        )
    axes[0].set_xticks(positions, [str(seed) for seed in EXPECTED_SEEDS])
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Test Coverage-Valid Fraction")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Coverage-Valid Candidate Rate")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    valid = data[data["publication_valid"]]
    invalid = data[~data["publication_valid"]]
    axes[1].scatter(
        invalid["test_fairness"],
        invalid["test_quality"],
        marker="x",
        color="0.55",
        alpha=0.7,
        s=34,
        label="Coverage-invalid",
    )
    for optimizer in OPTIMIZER_ORDER:
        group = valid[valid["optimizer"] == optimizer]
        axes[1].scatter(
            group["test_fairness"],
            group["test_quality"],
            marker=MARKERS[optimizer],
            color=COLORS[optimizer],
            alpha=0.65,
            s=30,
            label=f"{DISPLAY_NAME[optimizer]} valid",
        )
    axes[1].set_xlabel(UNFAIRNESS_AXIS_LABEL)
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("Why Coverage Validity Matters")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("BBQ v2 — Qwen-3-30B at 5M")
    save_figure(fig, outdir, "bbq_qwen3_5m_v2_coverage_validity_diagnostics")


def best_quality_points(final: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for (optimizer, seed), group in final.groupby(["optimizer", "seed"], sort=True):
        candidate = group.copy()
        candidate["_quality"] = pd.to_numeric(candidate["test_quality"], errors="coerce")
        candidate["_fairness"] = pd.to_numeric(candidate["test_fairness"], errors="coerce")
        candidate["_cost"] = pd.to_numeric(candidate["test_cost"], errors="coerce")
        candidate = candidate.dropna(subset=["_quality", "_fairness", "_cost"])
        candidate = candidate.sort_values(
            ["_quality", "_fairness", "_cost"],
            ascending=[False, True, True],
        )
        if not candidate.empty:
            rows.append(candidate.iloc[0].drop(labels=["_quality", "_fairness", "_cost"]))
    return pd.DataFrame(rows).reset_index(drop=True)


def high_accuracy_operating_points(
    final: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--high-accuracy-threshold must be between 0 and 1")
    rows: list[dict[str, object]] = []
    for optimizer in OPTIMIZER_ORDER:
        for seed in EXPECTED_SEEDS:
            group = final[
                (final["optimizer"] == optimizer)
                & (pd.to_numeric(final["seed"], errors="coerce") == seed)
            ].copy()
            group["_quality"] = pd.to_numeric(group["test_quality"], errors="coerce")
            group["_fairness"] = pd.to_numeric(group["test_fairness"], errors="coerce")
            group["_cost"] = pd.to_numeric(group["test_cost"], errors="coerce")
            group = group.dropna(subset=["_quality", "_fairness", "_cost"])
            eligible = group[group["_quality"] >= threshold].copy()
            threshold_met = not eligible.empty
            candidate = eligible if threshold_met else group
            candidate = candidate.sort_values(
                ["_fairness", "_cost", "_quality"],
                ascending=[True, True, False],
            )
            if candidate.empty:
                continue
            row = candidate.iloc[0]
            rows.append(
                {
                    "optimizer": optimizer,
                    "method": DISPLAY_NAME[optimizer],
                    "seed": seed,
                    "threshold": float(threshold),
                    "threshold_met": bool(threshold_met),
                    "test_quality": float(row["_quality"]),
                    "test_cost": float(row["_cost"]),
                    "test_fairness": float(row["_fairness"]),
                    "prompt_id": row.get("prompt_id", ""),
                    "prompt": row.get("prompt", ""),
                }
            )
    return pd.DataFrame(rows)


def plot_high_accuracy_operating_points(
    final: pd.DataFrame,
    outdir: Path,
    *,
    threshold: float,
) -> pd.DataFrame:
    points = high_accuracy_operating_points(final, threshold=threshold)
    points.to_csv(
        outdir / "bbq_qwen3_5m_v2_high_accuracy_operating_points.csv",
        index=False,
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    specifications = [
        ("test_quality", "Accuracy ↑", "Test Accuracy"),
        ("test_cost", "Cost ↓", COST_SHORT_LABEL),
        ("test_fairness", "Unfairness ↓", UNFAIRNESS_AXIS_LABEL),
    ]
    for ax, (column, title, ylabel) in zip(axes, specifications):
        for optimizer in OPTIMIZER_ORDER:
            group = points[points["optimizer"] == optimizer].sort_values("seed")
            ax.plot(
                group["seed"],
                group[column],
                color=COLORS[optimizer],
                marker=MARKERS[optimizer],
                linewidth=2,
                markersize=6,
                label=DISPLAY_NAME[optimizer],
            )
        ax.set_xticks(EXPECTED_SEEDS)
        ax.set_xlabel("Seed")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].axhline(threshold, color="0.5", linestyle="--", linewidth=1)
    axes[0].legend(frameon=False)
    fig.suptitle(
        f"BBQ v2 — Lowest Unfairness with Test Accuracy ≥ {threshold:.2f}"
    )
    save_figure(fig, outdir, "bbq_qwen3_5m_v2_high_accuracy_operating_points")
    return points


def write_summary_tables(
    final_metrics: pd.DataFrame,
    final_evaluations: pd.DataFrame,
    final_raw_evaluations: pd.DataFrame,
    high_accuracy_points: pd.DataFrame,
    outdir: Path,
) -> None:
    coverage = coverage_validity_summary(final_raw_evaluations)
    best_points = best_quality_points(final_evaluations)
    rows: list[dict[str, object]] = []
    for optimizer in OPTIMIZER_ORDER:
        metrics = final_metrics[final_metrics["optimizer"] == optimizer]
        valid_eval = final_evaluations[final_evaluations["optimizer"] == optimizer]
        raw_eval = final_raw_evaluations[
            final_raw_evaluations["optimizer"] == optimizer
        ]
        best = best_points[best_points["optimizer"] == optimizer]
        high = high_accuracy_points[
            high_accuracy_points["optimizer"] == optimizer
        ]
        coverage_method = coverage[coverage["optimizer"] == optimizer]
        rows.append(
            {
                "Method": DISPLAY_NAME[optimizer],
                "Seeds": int(metrics["seed"].nunique()),
                "Raw 5M candidates": int(len(raw_eval)),
                "Publication-valid 5M candidates": int(len(valid_eval)),
                "Coverage-valid fraction mean": coverage_method[
                    "test_coverage_valid_fraction"
                ].mean(),
                "Actual tokens mean": metrics["actual_budget_tokens"].mean(),
                "Holdout nR2 mean ↓": metrics["noisy_r2_3d"].mean(),
                "Holdout nR2 SD": metrics["noisy_r2_3d"].std(ddof=1),
                "HV optimistic mean ↑": metrics["hv_test_optimistic_3d"].mean(),
                "HV pessimistic mean ↑": metrics["hv_test_pessimistic_3d"].mean(),
                "Gap mean ↓": metrics["approximation_gap_3d"].mean(),
                "Balanced test accuracy mean ↑": metrics["balanced_test_quality"].mean(),
                "Balanced cost objective mean ↓": metrics["balanced_test_cost"].mean(),
                "Balanced test unfairness mean ↓": metrics[
                    "balanced_test_fairness"
                ].mean(),
                "Best-quality point mean accuracy ↑": best["test_quality"].mean(),
                "Best-quality point mean cost ↓": best["test_cost"].mean(),
                "Best-quality point mean unfairness ↓": best[
                    "test_fairness"
                ].mean(),
                "High-accuracy point mean accuracy ↑": high["test_quality"].mean(),
                "High-accuracy point mean cost ↓": high["test_cost"].mean(),
                "High-accuracy point mean unfairness ↓": high[
                    "test_fairness"
                ].mean(),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "bbq_qwen3_5m_v2_summary_table.csv", index=False)
    (outdir / "bbq_qwen3_5m_v2_summary_table.md").write_text(
        table.to_markdown(index=False, floatfmt=".4f") + "\n",
        encoding="utf-8",
    )
    coverage.to_csv(
        outdir / "bbq_qwen3_5m_v2_coverage_summary.csv", index=False
    )
    (outdir / "bbq_qwen3_5m_v2_coverage_summary.md").write_text(
        coverage.to_markdown(index=False, floatfmt=".4f") + "\n",
        encoding="utf-8",
    )
    print("\nBBQ v2 5M method summary")
    print(table.to_string(index=False))


def write_readme(
    outdir: Path,
    analysis_dir: Path,
    results_root: Path,
    *,
    cost_upper_bound: float,
    high_accuracy_threshold: float,
) -> None:
    content = f"""# BBQ v2 / Qwen-3-30B / 5M publication figures

Generated by `python -m analysis.make_bbq_5m_figures_v2`.

- Results root: `{results_root}`
- Dedicated v2 analysis tables: `{analysis_dir}`
- Exact final checkpoint: `{FINAL_BUDGET:,}` downstream tokens
- Seeds: {', '.join(str(seed) for seed in EXPECTED_SEEDS)}
- Publication validity: `test_fairness_ready == True` and test diagnostic `coverage_valid == True`
- Fixed objective bounds: quality loss `[0,1]`, cost `[0,{cost_upper_bound:g}]`, unfairness `[0,1]`
- High-accuracy threshold: `{high_accuracy_threshold:.2f}`

## Cost interpretation

`test_cost` is the configured weighted mean-token objective:

`{QWEN_INPUT_WEIGHT} × mean input tokens + {QWEN_OUTPUT_WEIGHT} × mean output tokens`

It is not a dollar amount and is not the Rocket GPU bill.

## Exact versus anytime metrics

- Exact holdout metrics are shown only for checkpoints represented by all six method/seed runs.  With the current direct-to-5M study this is the 5M checkpoint.
- `noisy_r2_3d`: exact checkpoint-level holdout nR2, lower is better.
- `hv_test_optimistic_3d` / `hv_test_pessimistic_3d`: exact test hypervolume, higher is better.
- `approximation_gap_3d`: exact approximation gap, lower is better.
- `bbq_qwen3_5m_v2_stepwise_dev_nr2_proxy.*`: development-only anytime proxy, not exact holdout nR2.
- Coverage-invalid rows are excluded from main publication figures and retained in `all_evaluations_raw.parquet` and the coverage diagnostic figure.

## Main figure files

- `bbq_qwen3_5m_v2_stepwise_dev_nr2_proxy.*`
- `bbq_qwen3_5m_v2_stepwise_coverage_validity.*`
- `bbq_qwen3_5m_v2_stepwise_hv_exact_gap.*`
- `bbq_qwen3_5m_v2_exact_nr2_hv_gap.*`
- `bbq_qwen3_5m_v2_balanced_exact_metrics.*`
- `bbq_qwen3_5m_v2_attainment_accuracy_cost.*`
- `bbq_qwen3_5m_v2_attainment_accuracy_unfairness.*`
- `bbq_qwen3_5m_v2_pareto_cost_unfairness.*`
- `bbq_qwen3_5m_v2_test_pareto_3d.*`
- `bbq_qwen3_5m_v2_method_comparison_hv_nr2_gap.*`
- `bbq_qwen3_5m_v2_coverage_validity_diagnostics.*`
- `bbq_qwen3_5m_v2_high_accuracy_operating_points.*`
- `bbq_qwen3_5m_v2_trifair_fewshot_outputshare.*`
- `bbq_qwen3_5m_v2_trifair_fewshot_unfairness.*`
"""
    (outdir / "README.md").write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    configure_style()

    results_root = Path(args.results_root)
    analysis_dir = Path(args.analysis_dir)
    figure_dir = (
        Path(args.figure_dir)
        if args.figure_dir
        else analysis_dir / "publication_figures"
    )
    bounds_file = Path(args.bounds_file)
    budgets = parse_csv_ints(args.budgets)
    reference_point_values = tuple(
        float(piece.strip()) for piece in str(args.reference_point).split(",")
    )
    if len(reference_point_values) != 3:
        raise ValueError("--reference-point must contain exactly three values")
    reference_point = (
        float(reference_point_values[0]),
        float(reference_point_values[1]),
        float(reference_point_values[2]),
    )
    if FINAL_BUDGET not in budgets:
        budgets = tuple(sorted(set(budgets + (FINAL_BUDGET,))))
    cost_upper_bound = float(args.cost_upper_bound)
    high_accuracy_threshold = float(args.high_accuracy_threshold)

    needs_rebuild = args.rebuild_analysis or not analysis_has_5m(
        analysis_dir,
        results_root=results_root,
        bounds_file=bounds_file,
        cost_upper_bound=cost_upper_bound,
    )
    if needs_rebuild:
        if args.no_auto_rebuild and not args.rebuild_analysis:
            raise RuntimeError(
                f"{analysis_dir} is missing, stale, or pre-v2. "
                "Re-run with --rebuild-analysis."
            )
        rebuild_analysis_outputs(
            results_root=results_root,
            analysis_dir=analysis_dir,
            bounds_file=bounds_file,
            budgets=budgets,
            n_preferences=int(args.n_preferences),
            preference_seed=int(args.preference_seed),
            reference_point=reference_point,
            cost_upper_bound=cost_upper_bound,
            overwrite_bounds=bool(args.overwrite_bounds),
            strict=bool(args.strict),
        )

    run_metrics, trajectory, evaluations, raw_evaluations = read_inputs(analysis_dir)
    selected = selected_final_runs(run_metrics)
    final_evaluations = final_checkpoint_evaluations(evaluations, selected)
    final_raw_evaluations = final_checkpoint_evaluations(raw_evaluations, selected)
    checkpoint_metrics = complete_checkpoint_metrics(
        checkpoint_run_metrics(run_metrics)
    )
    final_metrics = checkpoint_metrics[
        checkpoint_metrics["budget_checkpoint"] == FINAL_BUDGET
    ].copy()

    require_columns(
        final_metrics,
        [
            "noisy_r2_3d",
            "hv_test_optimistic_3d",
            "hv_test_pessimistic_3d",
            "approximation_gap_3d",
            "balanced_test_quality",
            "balanced_test_cost",
            "balanced_test_fairness",
        ],
        "5M run metrics",
    )

    selected_trajectory = filter_to_selected_runs(trajectory, selected)
    step_results = clean_frame(load_step_results_from_selected_runs(selected))
    step_proxy = stepwise_development_r2_proxy(
        step_results,
        final_evaluations,
        cost_upper_bound=cost_upper_bound,
    )
    max_actual_tokens = int(
        max(
            FINAL_BUDGET,
            pd.to_numeric(
                selected["actual_budget_tokens"], errors="coerce"
            ).max(),
        )
    )
    max_plot_budget = int(math.ceil(max_actual_tokens / 50_000.0) * 50_000)

    figure_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(
        figure_dir / "bbq_qwen3_5m_v2_selected_runs.csv", index=False
    )
    checkpoint_metrics.to_csv(
        figure_dir / "bbq_qwen3_5m_v2_checkpoint_run_metrics.csv", index=False
    )
    final_evaluations.to_parquet(
        figure_dir / "bbq_qwen3_5m_v2_publication_valid_evaluations.parquet",
        index=False,
    )
    final_raw_evaluations.to_parquet(
        figure_dir / "bbq_qwen3_5m_v2_raw_final_evaluations.parquet",
        index=False,
    )
    step_proxy.to_csv(
        figure_dir / "bbq_qwen3_5m_v2_stepwise_dev_nr2_proxy.csv", index=False
    )

    plot_stepwise_nr2_proxy(
        step_proxy,
        figure_dir,
        max_budget=max_plot_budget,
    )
    plot_stepwise_coverage_validity(
        step_proxy,
        figure_dir,
        max_budget=max_plot_budget,
    )
    plot_stepwise_hv_and_gap(
        selected_trajectory,
        checkpoint_metrics,
        figure_dir,
        max_budget=max_plot_budget,
    )
    plot_exact_checkpoint_mo_metrics(checkpoint_metrics, figure_dir)
    plot_balanced_budget_trajectory(checkpoint_metrics, figure_dir)
    plot_empirical_attainment(
        final_evaluations,
        figure_dir,
        x_col="test_cost",
        xlabel=COST_AXIS_LABEL,
        filename="bbq_qwen3_5m_v2_attainment_accuracy_cost",
    )
    plot_empirical_attainment(
        final_evaluations,
        figure_dir,
        x_col="test_fairness",
        xlabel=UNFAIRNESS_AXIS_LABEL,
        filename="bbq_qwen3_5m_v2_attainment_accuracy_unfairness",
    )
    plot_cost_unfairness_projection(final_evaluations, figure_dir)
    plot_three_objective_pareto(final_evaluations, figure_dir)
    plot_method_comparison_bars(final_metrics, figure_dir)
    plot_coverage_validity_diagnostics(final_raw_evaluations, figure_dir)
    high_accuracy_points = plot_high_accuracy_operating_points(
        final_evaluations,
        figure_dir,
        threshold=high_accuracy_threshold,
    )
    plot_trifair_diagnostic(
        final_evaluations,
        figure_dir,
        color_col="output_cost_share",
        color_label="Output Token Cost Share",
        filename="bbq_qwen3_5m_v2_trifair_fewshot_outputshare",
    )
    plot_trifair_diagnostic(
        final_evaluations,
        figure_dir,
        color_col="test_fairness",
        color_label=UNFAIRNESS_AXIS_LABEL,
        filename="bbq_qwen3_5m_v2_trifair_fewshot_unfairness",
    )
    write_summary_tables(
        final_metrics,
        final_evaluations,
        final_raw_evaluations,
        high_accuracy_points,
        figure_dir,
    )
    write_readme(
        figure_dir,
        analysis_dir,
        results_root,
        cost_upper_bound=cost_upper_bound,
        high_accuracy_threshold=high_accuracy_threshold,
    )

    generated = sorted(path.name for path in figure_dir.iterdir() if path.is_file())
    print("\nGenerated files")
    for name in generated:
        print(f"  {name}")
    print(f"\nBBQ v2 5M figures written to: {figure_dir.resolve()}")


if __name__ == "__main__":
    main()
