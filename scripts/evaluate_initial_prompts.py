"""Evaluate the complete initial-instruction pool as the budget-zero baseline.

Tri-Fair keeps initial prompts inside each ``DatasetConfig``.  This script does
not import or maintain a second prompt registry.  It evaluates the fixed pool
on development and test manifests and writes one analysis-compatible
``eval.parquet`` under an ``optimizer=init`` run directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from promptolution.predictors import MarkerBasedPredictor
from promptolution.utils import Prompt

from src.config.dataset_configs import ALL_DATASETS
from src.config.model_configs import ALL_MODELS
from src.config.setup_config import SETUP
from src.helpers.llm_creation import create_llm
from src.helpers.task_creation import create_dev_tasks, create_test_task
from src.utils import seed_everything

try:
    from scripts._common import (
        add_result_columns,
        atomic_write_json,
        atomic_write_parquet,
        configure_logging,
        directory_lock,
        manifest_lock_path,
        prompts_to_frame,
        resolve_manifest_path,
        set_generation_limit,
        sha256_file,
        utc_now_iso,
    )
except ModuleNotFoundError:  # pragma: no cover
    from _common import (  # type: ignore[no-redef]
        add_result_columns,
        atomic_write_json,
        atomic_write_parquet,
        configure_logging,
        directory_lock,
        manifest_lock_path,
        prompts_to_frame,
        resolve_manifest_path,
        set_generation_limit,
        sha256_file,
        utc_now_iso,
    )

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all configured initial prompts on fixed dev/test manifests."
    )
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--dataset", choices=sorted(ALL_DATASETS), required=True)
    parser.add_argument("--model", choices=sorted(ALL_MODELS), required=True)
    parser.add_argument(
        "--results-root",
        default="results/tri_fair",
        help="Root used when --output-dir is not supplied.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Exact run directory. Defaults to <results-root>/<model>/<dataset>/init/seedN/initial.",
    )
    parser.add_argument(
        "--split",
        choices=("both", "dev", "test"),
        default="both",
        help="For publication baselines use the default 'both'.",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=0,
        help="0 evaluates the full configured pool; positive values take the first N prompts.",
    )
    parser.add_argument("--expected-pool-size", type=int, default=15)
    parser.add_argument("--manifest-dir", default="data/splits")
    parser.add_argument("--regenerate-manifest", action="store_true")
    parser.add_argument("--manifest-lock-timeout", type=float, default=3600.0)
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return (
        Path(args.results_root).expanduser().resolve()
        / args.model
        / args.dataset
        / "init"
        / f"seed{args.random_seed}"
        / "initial"
    )


def _configured_prompts(args: argparse.Namespace) -> list[Prompt]:
    values = list(ALL_DATASETS[args.dataset].initial_prompts)
    if args.expected_pool_size > 0 and len(values) != args.expected_pool_size:
        raise ValueError(
            f"Dataset {args.dataset!r} contains {len(values)} initial prompts; "
            f"the main-study protocol expects {args.expected_pool_size}. Update the "
            "dataset configuration or pass --expected-pool-size 0 deliberately."
        )
    if args.n_prompts < 0:
        raise ValueError("--n-prompts cannot be negative")
    if args.n_prompts:
        if args.n_prompts > len(values):
            raise ValueError(
                f"Requested {args.n_prompts} prompts but only {len(values)} are configured"
            )
        values = values[: args.n_prompts]
    return [Prompt(instruction=value) for value in values]


def run(args: argparse.Namespace) -> Path:
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "eval.parquet"
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"{output_path} already exists. Use --force only for a deliberate replacement."
        )

    seed_everything(args.random_seed)
    dataset_config = ALL_DATASETS[args.dataset]
    model_config = ALL_MODELS[args.model]
    prompts = _configured_prompts(args)
    base = prompts_to_frame(prompts)
    base["initial_prompt_index"] = np.arange(len(base), dtype=int)
    base["chosen_step"] = 0
    base["optimizer"] = "init"
    base["model"] = args.model
    base["dataset"] = args.dataset
    base["seed"] = int(args.random_seed)
    base["configured_budget"] = 0
    base["budget_per_run"] = 0
    base["selection_policy"] = "complete_initial_pool"

    llm = create_llm(model_config=model_config, seed=args.random_seed)
    set_generation_limit(llm, args.max_output_tokens)
    predictor = None

    lock = manifest_lock_path(args.manifest_dir, args.dataset, args.random_seed)
    with directory_lock(lock, timeout_seconds=args.manifest_lock_timeout):
        dev_task = None
        test_task = None
        if args.split in {"both", "dev"}:
            dev_task, _ = create_dev_tasks(
                dataset_config=dataset_config,
                eval_strategy="full",
                n_subsamples=0,
                dev_size=SETUP.dev_size,
                fs_size=SETUP.fs_size,
                seed=args.random_seed,
                manifest_dir=args.manifest_dir,
                regenerate_manifest=args.regenerate_manifest,
            )
        if args.split in {"both", "test"}:
            test_task = create_test_task(
                dataset_config=dataset_config,
                eval_strategy="full",
                n_subsamples=0,
                test_size=SETUP.test_size,
                seed=args.random_seed,
                manifest_dir=args.manifest_dir,
                regenerate_manifest=args.regenerate_manifest,
            )

    output = base.copy()
    if dev_task is not None:
        predictor = MarkerBasedPredictor(llm, dev_task.classes)
        dev_result = dev_task.evaluate(
            prompts=prompts,
            predictor=predictor,
            eval_strategy="full",
        )
        output = add_result_columns(
            output,
            dev_result,
            prefix="dev",
            model_config=model_config,
        )
    else:
        for column in (
            "dev_quality",
            "dev_cost",
            "dev_input_tokens",
            "dev_output_tokens",
            "dev_fairness",
        ):
            output[column] = np.nan
        output["dev_fairness_ready"] = False
        output["dev_fairness_diagnostics_json"] = "{}"
        output["dev_group_support_json"] = "{}"
        output["dev_objective_vector"] = None

    if test_task is not None:
        predictor = MarkerBasedPredictor(llm, test_task.classes)
        test_result = test_task.evaluate(
            prompts=prompts,
            predictor=predictor,
            eval_strategy="full",
        )
        output = add_result_columns(
            output,
            test_result,
            prefix="test",
            model_config=model_config,
        )
    else:
        for column in (
            "test_quality",
            "test_cost",
            "test_input_tokens",
            "test_output_tokens",
            "test_fairness",
        ):
            output[column] = np.nan
        output["test_fairness_ready"] = False
        output["test_fairness_diagnostics_json"] = "{}"
        output["test_group_support_json"] = "{}"
        output["test_objective_vector"] = None

    manifest_path = None
    if test_task is not None:
        manifest_path = getattr(test_task, "manifest_path", None)
    if manifest_path is None and dev_task is not None:
        manifest_path = getattr(dev_task, "manifest_path", None)
    if manifest_path is None:
        manifest_path = str(
            resolve_manifest_path(args.manifest_dir, args.dataset, args.random_seed)
        )
    output["manifest_path"] = manifest_path
    output["manifest_sha256"] = sha256_file(manifest_path)
    output["evaluation_timestamp"] = utc_now_iso()
    output["run_dir"] = str(output_dir)

    atomic_write_parquet(output, output_path)
    args_payload = vars(args).copy()
    args_payload.update(
        {
            "optimizer": "init",
            "budget_per_run": 0,
            "output_dir": str(output_dir),
            "initial_prompt_pool_size": len(prompts),
            "manifest_path": manifest_path,
        }
    )
    atomic_write_json(output_dir / "args.json", args_payload)
    atomic_write_json(
        output_dir / "run_summary.json",
        {
            "status": "complete",
            "completed_at": utc_now_iso(),
            "optimizer": "init",
            "model": args.model,
            "dataset": args.dataset,
            "seed": args.random_seed,
            "n_prompts": len(prompts),
            "split": args.split,
            "manifest_path": manifest_path,
            "eval_path": str(output_path),
        },
    )
    LOGGER.info("Initial-prompt baseline written to %s", output_path)
    return output_path


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
