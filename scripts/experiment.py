"""Run Tri-Fair and compatible MO-CAPO-family optimizers.

The canonical main-study entry point supports fresh 1M runs and deterministic
continuation to larger budgets from trusted local checkpoints.

Examples
--------
Fresh 1M Tri-Fair run::

    uv run python scripts/experiment.py \
      --experiment-name tri_fair_bbq \
      --random-seed 42 \
      --budget-per-run 1000000 \
      --output-dir results/tri_fair/qwen-3-30b/bbq/Tri-Fair/seed42 \
      --dataset bbq --model qwen-3-30b --optimizer Tri-Fair \
      --n-init-prompts 6 \
      --budget-checkpoints 250000,500000,750000,1000000

Resume the same run to 5M::

    uv run python scripts/experiment.py ... \
      --budget-per-run 5000000 \
      --resume-from <logging_dir>/checkpoints/latest.pkl \
      --budget-checkpoints 2000000,3000000,4000000,5000000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import traceback
from pathlib import Path
from typing import Any

from promptolution.optimizers import CAPO, EvoPromptGA
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.predictors import MarkerBasedPredictor
from promptolution.utils import LoggerCallback
from promptolution.utils.callbacks import TokenCountCallback
from promptolution.utils.templates import EVOPROMPT_GA_TEMPLATE

from src.mo_capo import MoCAPO
from src.nsgaii_po import NSGAiiPO
from src.callbacks import ExperimentCallback
from src.checkpointing import OptimizerCheckpointCallback
from src.config.dataset_configs import ALL_DATASETS
from src.config.model_configs import ALL_MODELS
from src.config.optimizer_configs import ALL_OPTIMIZERS
from src.config.setup_config import SETUP
from src.gepa import Gepa
from src.helpers.llm_creation import create_llm
from src.helpers.task_creation import create_dev_tasks
from src.nsgaii_po_fair import NSGAiiPOFair
from src.tri_fair import TriFair
from src.utils import copy_llm, generate_random_hash, seed_everything

try:  # Supports both ``python -m scripts...`` and ``python scripts/...``.
    from scripts._common import (
        append_jsonl,
        atomic_write_json,
        compare_immutable_args,
        configure_logging,
        directory_lock,
        manifest_lock_path,
        parse_positive_int_csv,
        set_generation_limit,
        token_counts,
        utc_now_iso,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    from _common import (  # type: ignore[no-redef]
        append_jsonl,
        atomic_write_json,
        compare_immutable_args,
        configure_logging,
        directory_lock,
        manifest_lock_path,
        parse_positive_int_csv,
        set_generation_limit,
        token_counts,
        utc_now_iso,
    )

LOGGER = logging.getLogger(__name__)
FAIRNESS_OPTIMIZERS = {"Tri-Fair", "NSGAII-PO-Fair"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one prompt-optimization configuration with resumable checkpoints."
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--budget-per-run", type=int, required=True)
    parser.add_argument("--output-dir", default="results/main_results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset", choices=sorted(ALL_DATASETS), required=True)
    parser.add_argument("--model", choices=sorted(ALL_MODELS), required=True)
    parser.add_argument("--optimizer", choices=sorted(ALL_OPTIMIZERS), required=True)

    parser.add_argument("--n-init-prompts", type=int, default=6)
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--meta-max-output-tokens", type=int, default=256)
    parser.add_argument(
        "--budget-checkpoints",
        default="250000,500000,750000,1000000",
        help="Comma-separated downstream/evaluation-model token thresholds.",
    )
    parser.add_argument("--manifest-dir", default="data/splits")
    parser.add_argument("--regenerate-manifest", action="store_true")
    parser.add_argument("--manifest-lock-timeout", type=float, default=3600.0)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-steps", type=int, default=SETUP.n_steps)

    # Existing MO-CAPO ablation switches remain available for compatibility.
    parser.add_argument("--random-selection", action="store_true")
    parser.add_argument("--no-weaker-dominance", action="store_true")

    # Controlled overrides for later sensitivity studies.  Main-study jobs leave
    # these unset so src/config/optimizer_configs.py remains the source of truth.
    parser.add_argument("--crossovers-per-iteration", type=int, default=None)
    parser.add_argument("--max-few-shot-examples", type=int, default=None)
    parser.add_argument("--input-cost-multiplier", type=float, default=1.0)
    parser.add_argument("--output-cost-multiplier", type=float, default=1.0)

    parser.add_argument(
        "--allow-under-budget-exit",
        action="store_true",
        help="Permit a run to exit below the requested token budget (debug only).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive_fields = {
        "budget_per_run": args.budget_per_run,
        "n_init_prompts": args.n_init_prompts,
        "max_output_tokens": args.max_output_tokens,
        "meta_max_output_tokens": args.meta_max_output_tokens,
        "max_steps": args.max_steps,
    }
    for name, value in positive_fields.items():
        if int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.crossovers_per_iteration is not None and args.crossovers_per_iteration <= 0:
        raise ValueError("--crossovers-per-iteration must be positive")
    if args.max_few_shot_examples is not None and args.max_few_shot_examples < 0:
        raise ValueError("--max-few-shot-examples cannot be negative")
    if args.input_cost_multiplier < 0 or args.output_cost_multiplier < 0:
        raise ValueError("Cost multipliers cannot be negative")


def _resolve_logging_dir(args: argparse.Namespace) -> Path:
    if args.resume_from:
        checkpoint = Path(args.resume_from).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if checkpoint.parent.name != "checkpoints":
            raise ValueError(
                "--resume-from must point to <logging_dir>/checkpoints/<file>.pkl"
            )
        return checkpoint.parent.parent

    run_id = args.run_id or generate_random_hash()[:8]
    return (Path(args.output_dir).expanduser().resolve() / run_id).resolve()


def _validate_resume_metadata(logging_dir: Path, args: argparse.Namespace) -> None:
    if not args.resume_from:
        return
    prior_path = logging_dir / "args.json"
    if not prior_path.exists():
        raise FileNotFoundError(
            f"Cannot validate resumed run because {prior_path} does not exist"
        )
    with prior_path.open("r", encoding="utf-8") as handle:
        previous = json.load(handle)
    current = vars(args)
    compare_immutable_args(
        previous,
        current,
        keys=(
            "random_seed",
            "dataset",
            "model",
            "optimizer",
            "n_init_prompts",
            "manifest_dir",
            "max_output_tokens",
            "meta_max_output_tokens",
            "input_cost_multiplier",
            "output_cost_multiplier",
            "crossovers_per_iteration",
            "max_few_shot_examples",
        ),
    )


def _optimizer_parameters(
    args: argparse.Namespace, optimizer_config: Any
) -> dict[str, Any]:
    params = dict(optimizer_config.optimizer_params)
    if args.crossovers_per_iteration is not None:
        params["crossovers_per_iter"] = int(args.crossovers_per_iteration)
    if args.max_few_shot_examples is not None:
        params["upper_shots"] = int(args.max_few_shot_examples)
    return params


def _build_optimizer(
    *,
    args: argparse.Namespace,
    optimizer_config: Any,
    model_config: Any,
    predictor: MarkerBasedPredictor,
    task: Any,
    meta_llm: Any,
    initial_prompts: list[str],
    df_fewshots: Any,
    callbacks: list[Any],
) -> BaseOptimizer:
    params = _optimizer_parameters(args, optimizer_config)
    common = {
        "predictor": predictor,
        "task": task,
        "meta_llm": meta_llm,
        "initial_prompts": initial_prompts,
        "callbacks": callbacks,
    }
    input_cost = float(model_config.input_costs) * float(args.input_cost_multiplier)
    output_cost = float(model_config.output_costs) * float(args.output_cost_multiplier)

    name = optimizer_config.optimizer
    if name == "EvoPromptGA":
        return EvoPromptGA(
            **common,
            prompt_template=EVOPROMPT_GA_TEMPLATE,
            **params,
        )
    if name == "CAPO":
        return CAPO(**common, df_few_shots=df_fewshots, **params)
    if name == "GEPA":
        return Gepa(**common, **params)
    if name == "NSGAII-PO":
        return NSGAiiPO(
            **common,
            df_few_shots=df_fewshots,
            cost_per_input_token=input_cost,
            cost_per_output_token=output_cost,
            **params,
        )
    if name == "MO-CAPO":
        return MoCAPO(
            **common,
            df_few_shots=df_fewshots,
            cost_per_input_token=input_cost,
            cost_per_output_token=output_cost,
            random_selection=args.random_selection,
            no_weaker_dominance=args.no_weaker_dominance,
            **params,
        )
    if name == "Tri-Fair":
        return TriFair(
            **common,
            df_few_shots=df_fewshots,
            cost_per_input_token=input_cost,
            cost_per_output_token=output_cost,
            random_selection=args.random_selection,
            no_weaker_dominance=args.no_weaker_dominance,
            **params,
        )
    if name == "NSGAII-PO-Fair":
        return NSGAiiPOFair(
            **common,
            df_few_shots=df_fewshots,
            cost_per_input_token=input_cost,
            cost_per_output_token=output_cost,
            **params,
        )
    raise ValueError(f"Optimizer {name!r} is not implemented")


def _write_stage_metadata(
    logging_dir: Path,
    args: argparse.Namespace,
    *,
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "timestamp": utc_now_iso(),
        "experiment_name": args.experiment_name,
        "model": args.model,
        "dataset": args.dataset,
        "optimizer": args.optimizer,
        "seed": args.random_seed,
        "requested_budget": args.budget_per_run,
        "resume_from": args.resume_from,
        "slurm_job_id": None,
        "slurm_array_task_id": None,
    }
    import os

    payload["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    payload["slurm_array_task_id"] = os.environ.get("SLURM_ARRAY_TASK_ID")
    if extra:
        payload.update(extra)
    append_jsonl(logging_dir / "stage_history.jsonl", payload)


def run(args: argparse.Namespace, *, logging_dir: Path | None = None) -> Path:
    _validate_args(args)
    dataset_config = ALL_DATASETS[args.dataset]
    optimizer_config = ALL_OPTIMIZERS[args.optimizer]
    model_config = ALL_MODELS[args.model]

    is_fairness_dataset = dataset_config.task_type == "Fairness"
    is_fairness_optimizer = args.optimizer in FAIRNESS_OPTIMIZERS
    if is_fairness_optimizer and not is_fairness_dataset:
        raise ValueError(f"{args.optimizer} requires a fairness-aware dataset")
    if is_fairness_dataset and not is_fairness_optimizer:
        raise ValueError(
            "Fairness datasets are restricted to Tri-Fair and NSGAII-PO-Fair in "
            "the main implementation. This prevents accidental optimization of "
            "quality and cost while silently ignoring fairness."
        )
    if args.n_init_prompts > len(dataset_config.initial_prompts):
        raise ValueError(
            f"Requested {args.n_init_prompts} initial prompts, but {args.dataset} "
            f"contains only {len(dataset_config.initial_prompts)}"
        )

    checkpoints = parse_positive_int_csv(
        args.budget_checkpoints,
        include=args.budget_per_run,
    )
    logging_dir = logging_dir or _resolve_logging_dir(args)
    logging_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_resume_metadata(logging_dir, args)

    # One stable pointer per model/dataset/optimizer/seed output directory.
    pointer = output_dir / "logging_dir.txt"
    pointer.write_text(str(logging_dir), encoding="utf-8")

    seed_everything(args.random_seed)
    atomic_write_json(logging_dir / "args.json", vars(args))
    _write_stage_metadata(
        logging_dir, args, status="starting", extra={"checkpoints": checkpoints}
    )

    downstream_llm = create_llm(model_config=model_config, seed=args.random_seed)
    meta_llm = copy_llm(downstream_llm)
    set_generation_limit(downstream_llm, args.max_output_tokens)
    set_generation_limit(meta_llm, args.meta_max_output_tokens)

    lock = manifest_lock_path(args.manifest_dir, args.dataset, args.random_seed)
    with directory_lock(lock, timeout_seconds=args.manifest_lock_timeout):
        dev_task, df_fewshots = create_dev_tasks(
            dataset_config=dataset_config,
            eval_strategy=optimizer_config.eval_strategy,
            n_subsamples=optimizer_config.n_subsamples,
            dev_size=SETUP.dev_size,
            fs_size=SETUP.fs_size,
            seed=args.random_seed,
            manifest_dir=args.manifest_dir,
            regenerate_manifest=args.regenerate_manifest,
        )

    initial_prompts = random.Random(args.random_seed).sample(
        list(dataset_config.initial_prompts),
        args.n_init_prompts,
    )
    predictor = MarkerBasedPredictor(downstream_llm, dev_task.classes)

    callbacks = [
        LoggerCallback(LOGGER),
        ExperimentCallback(dir=str(logging_dir)),
        OptimizerCheckpointCallback(logging_dir, checkpoints),
        # This is intentionally last: logs and checkpoints are written for the
        # budget-crossing step before the optimizer loop is stopped.
        TokenCountCallback(args.budget_per_run, "total_tokens"),
    ]

    optimizer = _build_optimizer(
        args=args,
        optimizer_config=optimizer_config,
        model_config=model_config,
        predictor=predictor,
        task=dev_task,
        meta_llm=meta_llm,
        initial_prompts=initial_prompts,
        df_fewshots=df_fewshots,
        callbacks=callbacks,
    )

    if args.resume_from:
        if not hasattr(optimizer, "load_checkpoint"):
            raise TypeError(
                f"{optimizer.__class__.__name__} does not support checkpoint resume"
            )
        optimizer.load_checkpoint(args.resume_from)  # type: ignore[attr-defined]

    starting_downstream = token_counts(downstream_llm)
    starting_meta = token_counts(meta_llm)
    LOGGER.info(
        "Starting %s/%s/%s/seed%d at %d downstream tokens; target=%d",
        args.model,
        args.dataset,
        args.optimizer,
        args.random_seed,
        starting_downstream["total_tokens"],
        args.budget_per_run,
    )

    if starting_downstream["total_tokens"] < args.budget_per_run:
        optimizer.optimize(n_steps=args.max_steps)
    else:
        LOGGER.warning(
            "Checkpoint already contains %d downstream tokens, which meets target %d; "
            "no additional optimization steps were executed.",
            starting_downstream["total_tokens"],
            args.budget_per_run,
        )

    final_downstream = token_counts(downstream_llm)
    final_meta = token_counts(meta_llm)
    if (
        final_downstream["total_tokens"] < args.budget_per_run
        and not args.allow_under_budget_exit
    ):
        raise RuntimeError(
            "Optimizer exited below the requested evaluation-token budget: "
            f"actual={final_downstream['total_tokens']}, target={args.budget_per_run}. "
            "This often indicates an internal optimizer exception or an insufficient "
            "--max-steps value."
        )

    summary = {
        "status": "complete",
        "completed_at": utc_now_iso(),
        "model": args.model,
        "dataset": args.dataset,
        "optimizer": args.optimizer,
        "seed": args.random_seed,
        "requested_budget": args.budget_per_run,
        "budget_checkpoints": checkpoints,
        "starting_downstream_tokens": starting_downstream,
        "final_downstream_tokens": final_downstream,
        "starting_meta_tokens": starting_meta,
        "final_meta_tokens": final_meta,
        "budget_overshoot_tokens": max(
            0, final_downstream["total_tokens"] - args.budget_per_run
        ),
        "manifest_path": getattr(dev_task, "manifest_path", None),
        "logging_dir": str(logging_dir),
        "checkpoint_latest": str(logging_dir / "checkpoints" / "latest.pkl"),
    }
    atomic_write_json(logging_dir / "run_summary.json", summary)
    _write_stage_metadata(logging_dir, args, status="complete", extra=summary)
    LOGGER.info("Run completed: %s", logging_dir)
    return logging_dir


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    # Resolve once so a failed fresh run cannot write failure metadata into a
    # second randomly generated directory.
    logging_dir = _resolve_logging_dir(args)
    try:
        run(args, logging_dir=logging_dir)
    except Exception as error:
        # Best-effort failure metadata.  Re-raise so SLURM sees a non-zero exit.
        try:
            logging_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "status": "failed",
                "failed_at": utc_now_iso(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "model": args.model,
                "dataset": args.dataset,
                "optimizer": args.optimizer,
                "seed": args.random_seed,
                "requested_budget": args.budget_per_run,
            }
            atomic_write_json(logging_dir / "run_summary.json", failure)
            _write_stage_metadata(logging_dir, args, status="failed", extra=failure)
        except Exception:
            LOGGER.exception("Could not persist failure metadata")
        raise


if __name__ == "__main__":
    main()
