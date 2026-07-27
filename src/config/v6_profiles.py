"""Tri-Fair v6 study profile.

V6 keeps the shared v5 instruction pool and the frozen v3 data profile for both
methods.  Only the Tri-Fair optimizer is changed.  The NSGA-II-PO-Fair baseline
remains untouched and receives the same prompts, manifests, model and 5M budget.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from src.config.base_config import DatasetConfig, OptimizerConfig
from src.config.v5_profiles import apply_v5_dataset_profile


V6_STUDY_VERSION = (
    "6.0-adaptive-reference-portfolio-quality-floor-budget-aware"
)

TRI_FAIR_V6_CONFIG = OptimizerConfig(
    name="Tri-Fair-v6",
    optimizer="Tri-Fair-v6",
    optimizer_params={
        "crossovers_per_iter": 6,
        "upper_shots": 4,
        "check_fs_accuracy": True,
        "create_fs_reasoning": False,
        "objective_aware_variation": True,
        "min_initial_ready_blocks": 2,
        "min_racing_blocks": 3,
        "dominance_epsilons": (0.002, 0.10, 0.002),
        "archive_confirmation_fraction": 0.999,
        "max_confirmation_streak": 1,
        "quality_guard_scale": 0.08,
        "fairness_guard_scale": 0.06,
        "smart_start_count": 6,
        "archive_cap": 12,
        "verified_parent_blocks": 3,
        "quality_floor_margin": 0.035,
        "stagnation_window": 3,
        "late_phase_crossovers": 2,
        "critical_phase_crossovers": 1,
        "exploration_interval": 4,
    },
    eval_strategy="sequential_block",
    n_subsamples=1,
)


def apply_v6_dataset_profile(config: DatasetConfig) -> DatasetConfig:
    """Apply the same data/prompt profile used for the matched v5 comparison."""
    output = apply_v5_dataset_profile(deepcopy(config))
    if output.task_type == "Fairness" and output.fairness is not None:
        from dataclasses import replace

        kwargs = dict(output.fairness.fairness_kwargs)
        kwargs["study_profile"] = V6_STUDY_VERSION
        output.fairness = replace(output.fairness, fairness_kwargs=kwargs)
    output.validate()
    return output


def build_v6_dataset_registry(
    registry: Mapping[str, DatasetConfig],
) -> dict[str, DatasetConfig]:
    return {
        str(name): (
            apply_v6_dataset_profile(config)
            if config.task_type == "Fairness"
            else deepcopy(config)
        )
        for name, config in registry.items()
    }
