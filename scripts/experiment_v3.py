"""Tri-Fair v3 experiment entry point.

This wrapper leaves the published v2 files untouched.  It installs the frozen v3
fairness profile and Tri-Fair-v3 optimizer into the existing experiment runner,
then delegates all budget control, checkpointing, logging, and resume validation
to ``scripts.experiment``.
"""

from __future__ import annotations

from typing import Any

import scripts.experiment as base
from src.config.v3_profiles import (
    TRI_FAIR_V3_CONFIG,
    V3_PROFILE_VERSION,
    build_v3_dataset_registry,
)
from src.tri_fair_v3 import TRI_FAIR_V3_METHOD_VERSION, TriFairV3


def _build_v3_optimizer(
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
    params = base._optimizer_parameters(args, optimizer_config)
    params["objective_aware_variation"] = not bool(
        args.disable_objective_aware_variation
    )

    configured_bound = task.fairness_kwargs.get("normalization_cost_upper_bound")
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

    return TriFairV3(
        predictor=predictor,
        task=task,
        meta_llm=meta_llm,
        initial_prompts=initial_prompts,
        callbacks=callbacks,
        df_few_shots=df_fewshots,
        cost_per_input_token=(
            float(model_config.input_costs) * float(args.input_cost_multiplier)
        ),
        cost_per_output_token=(
            float(model_config.output_costs) * float(args.output_cost_multiplier)
        ),
        random_selection=args.random_selection,
        no_weaker_dominance=args.no_weaker_dominance,
        **params,
    )


def _install_v3_safely() -> None:
    # Preserve a copy before clearing the imported registry.
    original_datasets = dict(base.ALL_DATASETS)
    base.ALL_DATASETS.clear()
    base.ALL_DATASETS.update(build_v3_dataset_registry(original_datasets))
    base.ALL_OPTIMIZERS[TRI_FAIR_V3_CONFIG.name] = TRI_FAIR_V3_CONFIG
    base.FAIRNESS_OPTIMIZERS.add(TRI_FAIR_V3_CONFIG.name)

    original_builder = base._build_optimizer

    def builder(**kwargs: Any):
        optimizer_config = kwargs["optimizer_config"]
        if optimizer_config.optimizer == "Tri-Fair-v3":
            return _build_v3_optimizer(**kwargs)
        return original_builder(**kwargs)

    base._build_optimizer = builder

    original_run = base.run

    def run_with_v3_metadata(args: Any, *, logging_dir=None):
        result = original_run(args, logging_dir=logging_dir)
        summary_path = result / "run_summary.json"
        if summary_path.is_file():
            import json

            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            summary["study_profile"] = V3_PROFILE_VERSION
            summary["method_version"] = (
                TRI_FAIR_V3_METHOD_VERSION
                if args.optimizer == "Tri-Fair-v3"
                else "NSGAII-PO-Fair matched baseline under v3 profile"
            )
            base.atomic_write_json(summary_path, summary)
        return result

    base.run = run_with_v3_metadata


_install_v3_safely()


if __name__ == "__main__":
    base.main()
