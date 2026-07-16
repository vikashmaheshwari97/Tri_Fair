"""Optimizer configurations for MO-CAPO and statistical Tri-Fair v2."""

from __future__ import annotations

from typing import Dict

from src.config.base_config import OptimizerConfig

_CAPO_CONFIG = OptimizerConfig(
    name="CAPO",
    optimizer="CAPO",
    optimizer_params={
        "length_penalty": 0.05,
        "crossovers_per_iter": 4,
        "upper_shots": 5,
        "max_n_blocks_eval": 10,
        "alpha": 0.2,
    },
    eval_strategy="sequential_block",
    n_subsamples=30,
)

_EVOPROMPTGA_CONFIG = OptimizerConfig(
    name="EvoPromptGA",
    optimizer="EvoPromptGA",
    optimizer_params={},
    eval_strategy="full",
)
_GEPA_CONFIG = OptimizerConfig(
    name="GEPA", optimizer="GEPA", optimizer_params={}, eval_strategy="full"
)
_NSGAII_PO_CONFIG = OptimizerConfig(
    name="NSGAII-PO",
    optimizer="NSGAII-PO",
    optimizer_params={"crossovers_per_iter": 4, "upper_shots": 5},
    eval_strategy="full",
)
_MO_CAPO_CONFIG = OptimizerConfig(
    name="MO-CAPO",
    optimizer="MO-CAPO",
    optimizer_params={"crossovers_per_iter": 4, "upper_shots": 5},
    eval_strategy="sequential_block",
    n_subsamples=30,
)
_TRI_FAIR_CONFIG = OptimizerConfig(
    name="Tri-Fair",
    optimizer="Tri-Fair",
    optimizer_params={
        "crossovers_per_iter": 4,
        "upper_shots": 3,
        "check_fs_accuracy": True,
        "create_fs_reasoning": True,
        "objective_aware_variation": True,
        "objective_mutation_weights": {
            "fairness": 0.35,
            "quality": 0.25,
            "cost": 0.20,
            "balanced": 0.15,
            "explore": 0.05,
        },
    },
    eval_strategy="sequential_block",
    n_subsamples=1,
)
_NSGAII_PO_FAIR_CONFIG = OptimizerConfig(
    name="NSGAII-PO-Fair",
    optimizer="NSGAII-PO-Fair",
    optimizer_params={
        "crossovers_per_iter": 4,
        "upper_shots": 3,
        "check_fs_accuracy": True,
        "create_fs_reasoning": True,
        "objective_aware_variation": True,
        "objective_mutation_weights": {
            "fairness": 0.35,
            "quality": 0.25,
            "cost": 0.20,
            "balanced": 0.15,
            "explore": 0.05,
        },
    },
    eval_strategy="full",
    n_subsamples=1,
)

ALL_OPTIMIZERS: Dict[str, OptimizerConfig] = {
    config.name: config
    for config in (
        _CAPO_CONFIG,
        _EVOPROMPTGA_CONFIG,
        _GEPA_CONFIG,
        _NSGAII_PO_CONFIG,
        _MO_CAPO_CONFIG,
        _TRI_FAIR_CONFIG,
        _NSGAII_PO_FAIR_CONFIG,
    )
}
