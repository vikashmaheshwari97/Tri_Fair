"""Matched Tri-Fair v7 study profile.

V7 keeps the v6/v5 dataset definitions and raw instruction pool.  The compared
methods use the same model, manifests, initial instructions, few-shot split,
downstream decoding, meta-model sampling configuration, and downstream-token
budget.  Only Tri-Fair's search logic differs from the unchanged
NSGA-II-PO-Fair baseline.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from src.config.base_config import DatasetConfig, OptimizerConfig
from src.config.v6_profiles import apply_v6_dataset_profile


V7_STUDY_VERSION = (
    "7.0-robust-reference-hv-operator-adaptation-deterministic-downstream"
)

TRI_FAIR_V7_CONFIG = OptimizerConfig(
    name="Tri-Fair-v7",
    optimizer="Tri-Fair-v7",
    optimizer_params={
        "crossovers_per_iter": 6,
        "upper_shots": 4,
        "check_fs_accuracy": True,
        "create_fs_reasoning": False,
        "objective_aware_variation": True,
        "min_initial_ready_blocks": 2,
        "min_racing_blocks": 3,
        "dominance_epsilons": (0.002, 0.08, 0.002),
        "archive_confirmation_fraction": 0.999,
        "max_confirmation_streak": 1,
        "quality_guard_scale": 0.06,
        "fairness_guard_scale": 0.08,
        "smart_start_count": 8,
        "archive_cap": 15,
        "verified_parent_blocks": 3,
        "quality_floor_margin": 0.040,
        "stagnation_window": 3,
        "late_phase_crossovers": 2,
        "critical_phase_crossovers": 1,
        "exploration_interval": 4,
        "reference_partitions": 4,
        "robustness_beta_quality": 0.55,
        "robustness_beta_cost": 0.25,
        "robustness_beta_fairness": 0.75,
        "robustness_beta_end_fraction": 0.35,
        "operator_ucb_scale": 0.18,
        "quality_floor_start": 0.055,
        "quality_floor_end": 0.020,
        "parent_pool_cap": 10,
    },
    eval_strategy="sequential_block",
    n_subsamples=1,
)


def apply_v7_dataset_profile(config: DatasetConfig) -> DatasetConfig:
    output = apply_v6_dataset_profile(deepcopy(config))
    if output.task_type == "Fairness" and output.fairness is not None:
        from dataclasses import replace

        kwargs = dict(output.fairness.fairness_kwargs)
        kwargs["study_profile"] = V7_STUDY_VERSION
        output.fairness = replace(output.fairness, fairness_kwargs=kwargs)
    output.validate()
    return output


def build_v7_dataset_registry(
    registry: Mapping[str, DatasetConfig],
) -> dict[str, DatasetConfig]:
    return {
        str(name): (
            apply_v7_dataset_profile(config)
            if config.task_type == "Fairness"
            else deepcopy(config)
        )
        for name, config in registry.items()
    }
