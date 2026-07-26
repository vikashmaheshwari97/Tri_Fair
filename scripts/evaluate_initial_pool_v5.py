"""Evaluate the shared v5 raw initial instruction pool on dev and holdout data.

The initial baseline is separate from the 5M optimizer budget.  Both compared
optimizers sample from this same enhanced pool; Tri-Fair-v5's additional smart
warm-start proposals are not part of the Initial Instructions baseline.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from promptolution.predictors import MarkerBasedPredictor
from promptolution.utils.prompt import Prompt

from scripts._common import (
    atomic_write_parquet,
    configure_logging,
    prompt_id,
    set_generation_limit,
    sha256_file,
    utc_now_iso,
)
from src.config.dataset_configs import ALL_DATASETS
from src.config.model_configs import ALL_MODELS
from src.config.setup_config import SETUP
from src.config.v5_profiles import apply_v5_dataset_profile
from src.helpers.llm_creation import create_llm
from src.helpers.task_creation import create_dev_tasks, create_test_task
from src.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(ALL_DATASETS), required=True)
    parser.add_argument("--model", choices=sorted(ALL_MODELS), required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--n-init-prompts", type=int, default=12)
    parser.add_argument("--manifest-dir", default="data/splits_v5")
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _attach_result(
    frame: pd.DataFrame,
    result,
    *,
    prefix: str,
    model_config,
) -> pd.DataFrame:
    output = frame.reset_index(drop=True).copy()
    quality = np.asarray(result.agg_scores, dtype=float)
    input_tokens = np.asarray(result.agg_input_tokens, dtype=float)
    output_tokens = np.asarray(result.agg_output_tokens, dtype=float)
    cost = (
        float(model_config.input_costs) * input_tokens
        + float(model_config.output_costs) * output_tokens
    )
    output[f"{prefix}_quality"] = quality
    output[f"{prefix}_cost"] = cost
    output[f"{prefix}_input_tokens"] = input_tokens
    output[f"{prefix}_output_tokens"] = output_tokens
    output[f"{prefix}_fairness"] = np.asarray(result.fairness_loss, dtype=float)
    output[f"{prefix}_fairness_ready"] = np.asarray(
        result.fairness_ready, dtype=bool
    )
    output[f"{prefix}_fairness_diagnostics_json"] = [
        json.dumps(value, sort_keys=True, default=str)
        for value in result.fairness_diagnostics
    ]
    output[f"{prefix}_group_support_json"] = [
        json.dumps(value, sort_keys=True, default=str)
        for value in result.fairness_support
    ]
    output[f"{prefix}_objective_vector"] = [
        [float(q), -float(c), -float(f)]
        for q, c, f in zip(
            output[f"{prefix}_quality"],
            output[f"{prefix}_cost"],
            output[f"{prefix}_fairness"],
        )
    ]
    return output


def run(args: argparse.Namespace) -> Path:
    if args.n_init_prompts <= 0 or args.max_output_tokens <= 0:
        raise ValueError("n-init-prompts and max-output-tokens must be positive")
    target = Path(args.output_file).expanduser().resolve()
    if target.exists() and not args.force:
        raise FileExistsError(f"{target} already exists; pass --force to replace it")

    dataset_config = apply_v5_dataset_profile(ALL_DATASETS[args.dataset])
    if args.n_init_prompts > len(dataset_config.initial_prompts):
        raise ValueError(
            f"Requested {args.n_init_prompts} prompts but only "
            f"{len(dataset_config.initial_prompts)} are configured"
        )
    model_config = ALL_MODELS[args.model]
    seed_everything(args.random_seed)
    instructions = random.Random(args.random_seed).sample(
        list(dataset_config.initial_prompts), args.n_init_prompts
    )
    prompts = [Prompt(instruction=value, few_shots=[]) for value in instructions]

    llm = create_llm(model_config=model_config, seed=args.random_seed)
    set_generation_limit(llm, args.max_output_tokens)
    dev_task, _ = create_dev_tasks(
        dataset_config=dataset_config,
        eval_strategy="full",
        n_subsamples=0,
        dev_size=SETUP.dev_size,
        fs_size=SETUP.fs_size,
        seed=args.random_seed,
        manifest_dir=args.manifest_dir,
        regenerate_manifest=False,
    )
    test_task = create_test_task(
        dataset_config=dataset_config,
        eval_strategy="full",
        n_subsamples=0,
        test_size=SETUP.test_size,
        seed=args.random_seed,
        manifest_dir=args.manifest_dir,
        regenerate_manifest=False,
    )

    base = pd.DataFrame(
        {
            "prompt": [prompt.construct_prompt() for prompt in prompts],
            "instruction": instructions,
            "few_shots_json": ["[]"] * len(prompts),
            "downstream_template": [prompt.downstream_template for prompt in prompts],
        }
    )
    base["prompt_id"] = base["prompt"].map(prompt_id)
    dev_predictor = MarkerBasedPredictor(llm, dev_task.classes)
    dev_result = dev_task.evaluate(
        prompts=prompts,
        predictor=dev_predictor,
        eval_strategy="full",
    )
    output = _attach_result(base, dev_result, prefix="dev", model_config=model_config)

    test_predictor = MarkerBasedPredictor(llm, test_task.classes)
    test_result = test_task.evaluate(
        prompts=prompts,
        predictor=test_predictor,
        eval_strategy="full",
    )
    output = _attach_result(output, test_result, prefix="test", model_config=model_config)
    output["optimizer"] = "Initial"
    output["model"] = args.model
    output["dataset"] = args.dataset
    output["seed"] = int(args.random_seed)
    output["chosen_step"] = 0
    output["budget_checkpoint"] = 0
    output["actual_budget_tokens"] = 0
    output["selection_policy"] = "shared_v5_initial_pool_dev_selected_during_analysis"
    output["study_profile"] = "v5_shared_diverse_initial_pool"
    output["evaluation_timestamp"] = utc_now_iso()
    output["manifest_path"] = str(getattr(test_task, "manifest_path", ""))
    output["manifest_sha256"] = sha256_file(output["manifest_path"].iloc[0])

    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(output, target)
    return target


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
