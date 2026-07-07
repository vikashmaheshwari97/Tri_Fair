"""Pairwise optimistic/pessimistic front comparisons for Tri-Fair methods.

This replaces ``pareto_front_criterions.ipynb`` with reproducible criteria for
all datasets, models, seeds, and budget checkpoints.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_BUDGETS, parse_csv_ints
from .io import load_all_evaluations
from .metrics import (
    DEFAULT_REFERENCE_POINT,
    approximation_sets,
    hypervolume_3d,
)
from .objectives import (
    BoundsStore,
    objective_matrix,
    pareto_mask,
    set_weakly_dominates,
    valid_objective_rows,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/tri_fair")
    parser.add_argument("--bounds-file", default="analysis/normalization_bounds.json")
    parser.add_argument("--optimizer-a", default="Tri-Fair")
    parser.add_argument("--optimizer-b", default="NSGAII-PO-Fair")
    parser.add_argument(
        "--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS)
    )
    parser.add_argument("--output", default="analysis/output/front_comparisons.csv")
    parser.add_argument("--atol", type=float, default=1e-12)
    return parser.parse_args()


def development_selected_holdout_sets(
    frame: pd.DataFrame,
    bounds,
) -> dict[str, np.ndarray]:
    mask = valid_objective_rows(frame, require_test=True)
    valid = frame.loc[mask].reset_index(drop=True)
    if valid.empty:
        raise ValueError("No fairness-ready candidates")
    dev = bounds.normalize(objective_matrix(valid, "dev"), clip=False)
    test = bounds.normalize(objective_matrix(valid, "test"), clip=False)
    selected = pareto_mask(dev)
    test_selected = test[selected]
    sets = approximation_sets(test_selected)
    return {
        "all_selected": test_selected,
        "optimistic": test_selected[sets.optimistic_indices],
        "pessimistic": test_selected[sets.pessimistic_indices],
    }


def _choose_group(groups: list[pd.DataFrame]) -> pd.DataFrame:
    if len(groups) == 1:
        return groups[0]
    return max(groups, key=lambda frame: (len(frame), str(frame["run_dir"].iloc[0])))


def compare_pair(
    a_sets: dict[str, np.ndarray],
    b_sets: dict[str, np.ndarray],
    *,
    reference_point: Sequence[float],
    atol: float,
) -> dict[str, Any]:
    a_opt_hv = hypervolume_3d(a_sets["optimistic"], reference_point)
    a_pes_hv = hypervolume_3d(a_sets["pessimistic"], reference_point)
    b_opt_hv = hypervolume_3d(b_sets["optimistic"], reference_point)
    b_pes_hv = hypervolume_3d(b_sets["pessimistic"], reference_point)

    a_hv_separates = a_pes_hv > b_opt_hv + atol
    b_hv_separates = b_pes_hv > a_opt_hv + atol
    a_front_dominates = set_weakly_dominates(
        a_sets["pessimistic"], b_sets["optimistic"], atol=atol
    )
    b_front_dominates = set_weakly_dominates(
        b_sets["pessimistic"], a_sets["optimistic"], atol=atol
    )

    if a_front_dominates and not b_front_dominates:
        criterion_2 = "A_dominates_B"
    elif b_front_dominates and not a_front_dominates:
        criterion_2 = "B_dominates_A"
    elif a_front_dominates and b_front_dominates:
        criterion_2 = "equivalent_or_tied"
    else:
        criterion_2 = "incomparable"

    if a_hv_separates and not b_hv_separates:
        criterion_1 = "A_better"
    elif b_hv_separates and not a_hv_separates:
        criterion_1 = "B_better"
    else:
        criterion_1 = "not_separated"

    return {
        "a_hv_optimistic": a_opt_hv,
        "a_hv_pessimistic": a_pes_hv,
        "a_hv_gap": max(0.0, a_opt_hv - a_pes_hv),
        "b_hv_optimistic": b_opt_hv,
        "b_hv_pessimistic": b_pes_hv,
        "b_hv_gap": max(0.0, b_opt_hv - b_pes_hv),
        "criterion_1_hv_separation": criterion_1,
        "criterion_2_front_dominance": criterion_2,
        "a_pessimistic_dominates_b_optimistic": a_front_dominates,
        "b_pessimistic_dominates_a_optimistic": b_front_dominates,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    budgets = parse_csv_ints(args.budgets)
    evaluations = load_all_evaluations(args.results_root, budget_checkpoints=budgets)
    bounds = BoundsStore.load(args.bounds_file)

    key_columns = ["model", "dataset", "seed", "budget_checkpoint"]
    rows: list[dict[str, Any]] = []
    for keys, matched in evaluations.groupby(key_columns, sort=True):
        metadata = dict(zip(key_columns, keys))
        if int(metadata["budget_checkpoint"]) <= 0:
            continue
        a_groups = [
            group
            for _, group in matched[matched["optimizer"] == args.optimizer_a].groupby(
                "run_key"
            )
        ]
        b_groups = [
            group
            for _, group in matched[matched["optimizer"] == args.optimizer_b].groupby(
                "run_key"
            )
        ]
        if not a_groups or not b_groups:
            continue
        a = _choose_group(a_groups)
        b = _choose_group(b_groups)
        try:
            run_bounds = bounds[(str(metadata["dataset"]), str(metadata["model"]))]
            comparison = compare_pair(
                development_selected_holdout_sets(a, run_bounds),
                development_selected_holdout_sets(b, run_bounds),
                reference_point=DEFAULT_REFERENCE_POINT,
                atol=args.atol,
            )
        except Exception:
            logger.exception("Could not compare %s", metadata)
            continue
        rows.append(
            {
                **metadata,
                "optimizer_a": args.optimizer_a,
                "optimizer_b": args.optimizer_b,
                "run_a": str(a["run_dir"].iloc[0]),
                "run_b": str(b["run_dir"].iloc[0]),
                **comparison,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    if result.empty:
        print("No matched method pairs were found.")
    else:
        print(result["criterion_2_front_dominance"].value_counts().to_string())
        print(f"\nSaved {len(result)} comparisons to {output}")


if __name__ == "__main__":
    main()
