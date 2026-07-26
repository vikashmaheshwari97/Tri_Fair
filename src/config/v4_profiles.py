"""Tri-Fair v4 method configuration.

The data sizes and fairness definitions remain identical to the frozen v3
profile.  Only the optimizer is revised.  Use a new manifest directory and new
results namespace so v3 and v4 artifacts cannot be mixed accidentally.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.config.base_config import DatasetConfig, OptimizerConfig
from src.config.v3_profiles import build_v3_dataset_registry

V4_STUDY_VERSION = (
    "4.0-progressive-shared-fidelity-reference-variation-signed-error-cells"
)

TRI_FAIR_V4_CONFIG = OptimizerConfig(
    name="Tri-Fair-v4",
    optimizer="Tri-Fair-v4",  # Runtime-valid; the legacy Literal is only a type hint.
    optimizer_params={
        # Match the baseline's offspring count.  v3 used eight children and spent
        # too much budget on shallow races; v4 favours four reliable directions.
        "crossovers_per_iter": 4,
        "upper_shots": 3,
        "check_fs_accuracy": True,
        "create_fs_reasoning": True,
        "objective_aware_variation": True,
        "min_initial_ready_blocks": 2,
        "min_racing_blocks": 3,
        "dominance_epsilons": (0.005, 0.25, 0.005),
        "archive_confirmation_fraction": 0.999,
        "fidelity_utilization_thresholds": (0.0, 0.30, 0.60, 0.85),
        "fidelity_block_targets": (2, 3, 4, 5),
        "max_confirmation_streak": 2,
        "quality_guard_scale": 0.20,
        "fairness_guard_scale": 0.10,
    },
    eval_strategy="sequential_block",
    n_subsamples=1,
)


def build_v4_dataset_registry(
    registry: Mapping[str, DatasetConfig],
) -> dict[str, DatasetConfig]:
    """Use the exact frozen v3 fairness profile under a separate v4 manifest root."""

    return build_v3_dataset_registry(registry)
