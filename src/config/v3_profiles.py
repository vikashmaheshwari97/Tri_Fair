"""Frozen Tri-Fair v3 study profile.

The v2 repository keeps compact pilot manifests.  This module provides a
separate, explicit v3 profile with more group-complete fidelity levels so that
Tri-Fair can exploit racing without altering the published v2 configuration.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Mapping

from src.config.base_config import DatasetConfig, OptimizerConfig

V3_PROFILE_VERSION = "3.0-ready-init-guarded-racing-confirmation"

# The profile is structural rather than result-tuned: each development set is
# enlarged by complete fairness units and remains disjoint from few-shot/test.
_V3_DATASET_OVERRIDES: dict[str, dict[str, object]] = {
    "bbq": {
        "dev_size": 440,
        "fs_size": 88,
        "test_size": 500,
        "block_size": 44,
        "max_accuracy_ci_width": 0.60,
    },
    "civil_comments": {
        "dev_size": 576,
        "fs_size": 96,
        "test_size": 500,
        "block_size": 96,
        "max_rate_ci_width": 0.55,
    },
    "bias_in_bios": {
        "dev_size": 560,
        "fs_size": 112,
        "test_size": 500,
        "block_size": 56,
        "max_rate_ci_width": 0.65,
    },
}

TRI_FAIR_V3_CONFIG = OptimizerConfig(
    name="Tri-Fair-v3",
    optimizer="Tri-Fair-v3",  # Runtime-valid; the legacy Literal is only a type hint.
    optimizer_params={
        "crossovers_per_iter": 8,
        "upper_shots": 3,
        "check_fs_accuracy": True,
        "create_fs_reasoning": True,
        "objective_aware_variation": True,
        "min_initial_ready_blocks": 2,
        "min_racing_blocks": 2,
        "dominance_epsilons": (0.010, 0.50, 0.010),
        "archive_confirmation_fraction": 0.85,
    },
    eval_strategy="sequential_block",
    n_subsamples=1,
)


def apply_v3_dataset_profile(config: DatasetConfig) -> DatasetConfig:
    """Return an isolated v3 copy of one dataset configuration."""

    output = deepcopy(config)
    if output.task_type != "Fairness" or output.fairness is None:
        return output

    override = _V3_DATASET_OVERRIDES.get(str(output.alias))
    if override is None:
        raise ValueError(f"No Tri-Fair v3 profile is defined for {output.alias!r}")

    fairness_kwargs = dict(output.fairness.fairness_kwargs)
    fairness_kwargs["study_profile"] = V3_PROFILE_VERSION
    for key in ("max_accuracy_ci_width", "max_rate_ci_width"):
        if key in override:
            fairness_kwargs[key] = float(override[key])

    output.fairness = replace(
        output.fairness,
        dev_size=int(override["dev_size"]),
        fs_size=int(override["fs_size"]),
        test_size=int(override["test_size"]),
        block_size=int(override["block_size"]),
        fairness_kwargs=fairness_kwargs,
    )
    output.validate()
    return output


def build_v3_dataset_registry(
    registry: Mapping[str, DatasetConfig],
) -> dict[str, DatasetConfig]:
    """Clone a dataset registry and apply v3 only to fairness datasets."""

    return {
        str(name): (
            apply_v3_dataset_profile(config)
            if config.task_type == "Fairness"
            else deepcopy(config)
        )
        for name, config in registry.items()
    }
