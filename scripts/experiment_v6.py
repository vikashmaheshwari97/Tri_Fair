"""Tri-Fair v6 experiment entry point.

Tri-Fair-v6 and the unchanged NSGA-II-PO-Fair baseline use the same raw
instruction pool, manifests, model, few-shot split and downstream-token budget.
"""

from __future__ import annotations

from typing import Any

import scripts.experiment as base
from src.config.v6_profiles import (
    TRI_FAIR_V6_CONFIG,
    V6_STUDY_VERSION,
    build_v6_dataset_registry,
)
from src.tri_fair_v6 import TRI_FAIR_V6_METHOD_VERSION, TriFairV6


def _fidelity_schedule(task: Any) -> tuple[tuple[float, ...], tuple[int, ...]]:
    n_blocks = int(task.n_blocks)
    thresholds = (0.0, 0.28, 0.52, 0.74, 0.90)
    if n_blocks <= 6:
        targets = (2, 3, 4, 5, n_blocks)
    else:
        targets = (2, 3, 5, 7, min(8, n_blocks))
    targets = tuple(
        min(n_blocks, max(1, int(value)))
        for value in targets
    )
    repaired: list[int] = []
    for value in targets:
        repaired.append(max(value, repaired[-1] if repaired else 1))
    return thresholds, tuple(repaired)


def _build_v6_optimizer(
    *,
    args: Any,
    optimizer_config: Any,
    model_config: Any,
    predictor: Any,
    task: Any,
    meta_llm: Any,
    initial_prompts: list[str],
    df_fewshots: Any,
    callbacks: list[Any],
):
    params = dict(optimizer_config.optimizer_params)
    if args.crossovers_per_iteration is not None:
        params["crossovers_per_iter"] = int(
            args.crossovers_per_iteration
        )
    if args.max_few_shot_examples is not None:
        params["upper_shots"] = int(
            args.max_few_shot_examples
        )
    params["objective_aware_variation"] = not bool(
        args.disable_objective_aware_variation
    )
    thresholds, targets = _fidelity_schedule(task)
    params["fidelity_utilization_thresholds"] = thresholds
    params["fidelity_block_targets"] = targets

    configured_bound = task.fairness_kwargs.get(
        "normalization_cost_upper_bound"
    )
    cost_upper_bound = (
        args.fixed_cost_upper_bound
        if args.fixed_cost_upper_bound is not None
        else configured_bound
    )
    if cost_upper_bound is not None:
        params["fixed_objective_bounds"] = (
            (0.0, -float(cost_upper_bound), -1.0),
            (1.0, 0.0, 0.0),
        )

    return TriFairV6(
        predictor=predictor,
        task=task,
        meta_llm=meta_llm,
        initial_prompts=initial_prompts,
        callbacks=callbacks,
        df_few_shots=df_fewshots,
        cost_per_input_token=(
            float(model_config.input_costs)
            * float(args.input_cost_multiplier)
        ),
        cost_per_output_token=(
            float(model_config.output_costs)
            * float(args.output_cost_multiplier)
        ),
        random_selection=args.random_selection,
        no_weaker_dominance=args.no_weaker_dominance,
        **params,
    )


def _install_v6_safely() -> None:
    original_datasets = dict(base.ALL_DATASETS)
    base.ALL_DATASETS.clear()
    base.ALL_DATASETS.update(
        build_v6_dataset_registry(original_datasets)
    )
    base.ALL_OPTIMIZERS[
        TRI_FAIR_V6_CONFIG.name
    ] = TRI_FAIR_V6_CONFIG
    base.FAIRNESS_OPTIMIZERS.add(TRI_FAIR_V6_CONFIG.name)

    original_builder = base._build_optimizer

    def builder(**kwargs: Any):
        optimizer_config = kwargs["optimizer_config"]
        if optimizer_config.optimizer == "Tri-Fair-v6":
            return _build_v6_optimizer(**kwargs)
        return original_builder(**kwargs)

    base._build_optimizer = builder

    original_run = base.run

    def run_with_v6_metadata(args: Any, *, logging_dir=None):
        result = original_run(args, logging_dir=logging_dir)
        summary_path = result / "run_summary.json"
        if summary_path.is_file():
            import json

            with summary_path.open(
                "r", encoding="utf-8"
            ) as handle:
                summary = json.load(handle)
            summary["study_profile"] = V6_STUDY_VERSION
            summary["method_version"] = (
                TRI_FAIR_V6_METHOD_VERSION
                if args.optimizer == "Tri-Fair-v6"
                else (
                    "NSGAII-PO-Fair unchanged matched baseline "
                    "under v6 shared-pool profile"
                )
            )
            summary["initial_pool_policy"] = (
                "shared_v5_pool_on_fresh_v6_manifests"
            )
            summary["holdout_feedback_used"] = False
            base.atomic_write_json(summary_path, summary)
        return result

    base.run = run_with_v6_metadata


_install_v6_safely()


if __name__ == "__main__":
    base.main()
