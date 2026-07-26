"""Frozen Tri-Fair v5 study profile and shared initial instruction pool.

The development/few-shot/test sizes and fairness definitions remain identical to
v3/v4.  V5 changes the optimizer and replaces the *shared* initial instruction
pool for both compared methods.  Use a new manifest directory and results
namespace so no v3/v4 artefact can be mixed into the study.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from src.config.base_config import DatasetConfig, OptimizerConfig
from src.config.v3_profiles import apply_v3_dataset_profile
from src.config.v5_seed_pool import V5_INITIAL_PROMPTS

V5_STUDY_VERSION = (
    "5.0-shared-diverse-seeds-smart-warm-start-contrastive-reference-archive"
)

TRI_FAIR_V5_CONFIG = OptimizerConfig(
    name="Tri-Fair-v5",
    optimizer="Tri-Fair-v5",  # Runtime-valid; the legacy Literal is only a type hint.
    optimizer_params={
        "crossovers_per_iter": 4,
        "upper_shots": 4,
        "check_fs_accuracy": True,
        # Label-only few-shots are shorter and more reproducible than generated
        # rationales, while the downstream model is still free to reason internally.
        "create_fs_reasoning": False,
        "objective_aware_variation": True,
        "min_initial_ready_blocks": 2,
        "min_racing_blocks": 3,
        "dominance_epsilons": (0.003, 0.15, 0.003),
        "archive_confirmation_fraction": 0.999,
        "max_confirmation_streak": 2,
        "quality_guard_scale": 0.12,
        "fairness_guard_scale": 0.08,
        "smart_start_count": 4,
        "archive_cap": 8,
        "verified_parent_blocks": 3,
    },
    eval_strategy="sequential_block",
    n_subsamples=1,
)


def apply_v5_dataset_profile(config: DatasetConfig) -> DatasetConfig:
    """Apply the frozen v3 data profile plus the shared v5 prompt pool."""

    output = apply_v3_dataset_profile(deepcopy(config))
    if output.task_type != "Fairness":
        return output
    try:
        prompts = V5_INITIAL_PROMPTS[str(output.alias)]
    except KeyError as error:
        raise ValueError(f"No v5 prompt pool for {output.alias!r}") from error
    output.initial_prompts = list(prompts)
    if output.fairness is not None:
        kwargs = dict(output.fairness.fairness_kwargs)
        kwargs["study_profile"] = V5_STUDY_VERSION
        from dataclasses import replace

        output.fairness = replace(output.fairness, fairness_kwargs=kwargs)
    output.validate()
    return output


def build_v5_dataset_registry(
    registry: Mapping[str, DatasetConfig],
) -> dict[str, DatasetConfig]:
    return {
        str(name): (
            apply_v5_dataset_profile(config)
            if config.task_type == "Fairness"
            else deepcopy(config)
        )
        for name, config in registry.items()
    }
