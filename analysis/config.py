"""Shared constants and parsing helpers for Tri-Fair analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

DEFAULT_DATASETS: tuple[str, ...] = ("bbq", "bias_in_bios", "civil_comments")
DEFAULT_OPTIMIZERS: tuple[str, ...] = ("Tri-Fair", "NSGAII-PO-Fair")
DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44)

# The first four are diagnostic checkpoints inside the 1M run.  The final three
# are the publication-scale budget ladder.
DEFAULT_BUDGETS: tuple[int, ...] = (
    250_000,
    500_000,
    750_000,
    1_000_000,
    5_000_000,
    7_500_000,
)
PRIMARY_BUDGETS: tuple[int, ...] = (1_000_000, 5_000_000, 7_500_000)

OBJECTIVE_NAMES: tuple[str, ...] = ("quality_loss", "cost", "unfairness")
OBJECTIVE_DIMENSION = 3

DATASET_LABELS = {
    "bbq": "BBQ",
    "bias_in_bios": "Bias in Bios",
    "civil_comments": "Civil Comments-WILDS",
}

OPTIMIZER_LABELS = {
    "Tri-Fair": "Tri-Fair",
    "NSGAII-PO-Fair": "NSGA-II-PO-Fair",
    "MO-CAPO": "MO-CAPO",
    "init": "Initial Prompts",
}

METRIC_DIRECTIONS = {
    "hv_dev_3d": "max",
    "hv_test_optimistic_3d": "max",
    "hv_test_pessimistic_3d": "max",
    "approximation_gap_3d": "min",
    "noisy_r2_3d": "min",
    "fairness_generalization_gap_abs": "min",
    "balanced_test_quality": "max",
    "balanced_test_cost": "min",
    "balanced_test_fairness": "min",
    "quality_first_test_quality": "max",
    "quality_first_test_cost": "min",
    "quality_first_test_fairness": "min",
    "fairness_first_test_quality": "max",
    "fairness_first_test_cost": "min",
    "fairness_first_test_fairness": "min",
}


@dataclass(frozen=True)
class AnalysisGrid:
    """Expected experiment grid used by the status checker."""

    models: tuple[str, ...]
    datasets: tuple[str, ...] = DEFAULT_DATASETS
    optimizers: tuple[str, ...] = DEFAULT_OPTIMIZERS
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    budgets: tuple[int, ...] = DEFAULT_BUDGETS


def parse_csv_strings(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    result = tuple(item for item in items if item)
    if not result:
        raise ValueError("At least one value is required")
    return result


def parse_csv_ints(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        result = tuple(int(item.replace("_", "")) for item in items)
    else:
        result = tuple(int(item) for item in value)
    if not result:
        raise ValueError("At least one integer is required")
    if any(item < 0 for item in result):
        raise ValueError(f"Negative values are not allowed: {result}")
    return tuple(sorted(set(result)))


def budget_label(value: int) -> str:
    value = int(value)
    if value >= 1_000_000:
        number = value / 1_000_000
        return f"{number:g}M"
    if value >= 1_000:
        number = value / 1_000
        return f"{number:g}k"
    return str(value)
