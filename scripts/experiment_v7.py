"""Tri-Fair v7 matched experiment entry point.

Both Tri-Fair-v7 and NSGA-II-PO-Fair use greedy downstream classification and
the same moderately stochastic meta-model configuration.  This prevents the
same prompt from receiving different labels merely because it appeared in a
different inference batch.
"""

from __future__ import annotations

from typing import Any

import scripts.experiment as base
from src.config.v7_profiles import (
    TRI_FAIR_V7_CONFIG,
    V7_STUDY_VERSION,
    build_v7_dataset_registry,
)
from src.helpers.generation_control import (
    configure_downstream_greedy,
    configure_meta_search,
)
from src.tri_fair_v7 import (
    TRI_FAIR_V7_METHOD_VERSION,
    TriFairV7,
)


def _fidelity_schedule(
    task: Any,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    n_blocks = int(task.n_blocks)
    thresholds = (0.0, 0.22, 0.48, 0.70, 0.86, 0.95)
    if n_blocks <= 6:
        targets = (2, 3, 4, 5, n_blocks, n_blocks)
    else:
        targets = (2, 3, 5, 7, min(9, n_blocks), n_blocks)
    repaired: list[int] = []
    for value in targets:
        clipped = min(n_blocks, max(1, int(value)))
        repaired.append(max(clipped, repaired[-1] if repaired else 1))
    return thresholds, tuple(repaired)


def _build_v7_optimizer(
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

    return TriFairV7(
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


def _install_v7_safely() -> None:
    original_datasets = dict(base.ALL_DATASETS)
    base.ALL_DATASETS.clear()
    base.ALL_DATASETS.update(
        build_v7_dataset_registry(original_datasets)
    )
    base.ALL_OPTIMIZERS[
        TRI_FAIR_V7_CONFIG.name
    ] = TRI_FAIR_V7_CONFIG
    base.FAIRNESS_OPTIMIZERS.add(TRI_FAIR_V7_CONFIG.name)

    original_create_llm = base.create_llm

    def create_downstream_llm(*, model_config: Any, seed: int):
        llm = original_create_llm(
            model_config=model_config,
            seed=seed,
        )
        return configure_downstream_greedy(llm, seed=seed)

    base.create_llm = create_downstream_llm

    original_copy_llm = base.copy_llm

    def copy_meta_llm(model_obj: Any, llm_attr_name: str = "llm"):
        meta = original_copy_llm(
            model_obj,
            llm_attr_name=llm_attr_name,
        )
        seed = int(
            getattr(
                getattr(model_obj, "sampling_params", None),
                "seed",
                42,
            )
        )
        return configure_meta_search(
            meta,
            seed=seed + 1_000_003,
            temperature=0.55,
            top_p=0.95,
        )

    base.copy_llm = copy_meta_llm

    original_builder = base._build_optimizer

    def builder(**kwargs: Any):
        optimizer_config = kwargs["optimizer_config"]
        if optimizer_config.optimizer == "Tri-Fair-v7":
            return _build_v7_optimizer(**kwargs)
        return original_builder(**kwargs)

    base._build_optimizer = builder

    original_run = base.run

    def run_with_v7_metadata(args: Any, *, logging_dir=None):
        result = original_run(args, logging_dir=logging_dir)
        summary_path = result / "run_summary.json"
        if summary_path.is_file():
            import json

            with summary_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                summary = json.load(handle)
            summary["study_profile"] = V7_STUDY_VERSION
            summary["method_version"] = (
                TRI_FAIR_V7_METHOD_VERSION
                if args.optimizer == "Tri-Fair-v7"
                else (
                    "NSGAII-PO-Fair unchanged matched baseline "
                    "under v7 deterministic-downstream profile"
                )
            )
            summary["downstream_decoding"] = {
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": int(args.random_seed),
            }
            summary["meta_search_decoding"] = {
                "temperature": 0.55,
                "top_p": 0.95,
                "seed_offset": 1_000_003,
            }
            summary["holdout_feedback_used"] = False
            base.atomic_write_json(summary_path, summary)
        return result

    base.run = run_with_v7_metadata


_install_v7_safely()


if __name__ == "__main__":
    base.main()
