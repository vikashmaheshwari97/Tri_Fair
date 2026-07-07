"""Script for running cost ablation experiments with MO-CAPO."""

import argparse
import json
import os
import random
from logging import getLogger

from promptolution.predictors import MarkerBasedPredictor
from promptolution.utils import LoggerCallback
from promptolution.utils.callbacks import TokenCountCallback

from mo_capo import MoCAPO
from src.callbacks import ExperimentCallback
from src.config.dataset_configs import ALL_DATASETS
from src.config.model_configs import ALL_MODELS
from src.config.optimizer_configs import ALL_OPTIMIZERS
from src.config.setup_config import SETUP
from src.helpers.llm_creation import create_llm
from src.helpers.task_creation import create_dev_tasks
from src.utils import copy_llm, generate_random_hash, seed_everything

parser = argparse.ArgumentParser()

parser.add_argument("--experiment-name", required=True)
parser.add_argument("--random-seed", type=int, required=True)
parser.add_argument("--budget-per-run", type=int, required=True)
parser.add_argument("--output-dir", default="results/cost_ablation/")
parser.add_argument("--n-init-prompts", default=10, type=int)
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)

args = parser.parse_args()

logger = getLogger(__name__)


def run_single_ablation(
    ablation_name,
    cost_per_input_token,
    cost_per_output_token,
):
    output_dir = os.path.join(args.output_dir, ablation_name)
    logging_dir = os.path.join(output_dir, generate_random_hash()[:4])
    os.makedirs(logging_dir, exist_ok=True)

    with open(os.path.join(output_dir, "logging_dir.txt"), "w") as f:
        f.write(logging_dir)

    seed_everything(args.random_seed)

    ablation_args = vars(args).copy()
    ablation_args["ablation_name"] = ablation_name
    ablation_args["cost_per_input_token"] = cost_per_input_token
    ablation_args["cost_per_output_token"] = cost_per_output_token
    with open(os.path.join(logging_dir, "args.json"), "w") as f:
        json.dump(ablation_args, f)

    callbacks = [
        LoggerCallback(logger),
        ExperimentCallback(dir=logging_dir),
        TokenCountCallback(args.budget_per_run, "total_tokens"),
    ]

    dataset_config = ALL_DATASETS[args.dataset]
    optimizer_config = ALL_OPTIMIZERS["MO-CAPO"]
    model_config = ALL_MODELS[args.model]

    llm = create_llm(
        model_config=model_config,
        seed=args.random_seed,
    )

    downstream_llm = llm
    meta_llm = copy_llm(llm)

    dev_task, df_fewshots = create_dev_tasks(
        dataset_config=dataset_config,
        eval_strategy=optimizer_config.eval_strategy,
        n_subsamples=optimizer_config.n_subsamples,
        dev_size=SETUP.dev_size,
        fs_size=SETUP.fs_size,
        seed=args.random_seed,
    )

    init_prompt_rng = random.Random(args.random_seed)
    initial_prompts = init_prompt_rng.sample(
        dataset_config.initial_prompts,
        args.n_init_prompts,
    )

    predictor = MarkerBasedPredictor(downstream_llm, dev_task.classes)

    optimizer = MoCAPO(
        predictor=predictor,
        task=dev_task,
        meta_llm=meta_llm,
        initial_prompts=initial_prompts,
        df_few_shots=df_fewshots,
        callbacks=callbacks,
        cost_per_input_token=cost_per_input_token,
        cost_per_output_token=cost_per_output_token,
        **optimizer_config.optimizer_params,
    )

    logger.info(
        f"Input cost: {cost_per_input_token}, Output cost: {cost_per_output_token}"
    )
    best_prompts = optimizer.optimize(n_steps=SETUP.n_steps)  # noqa: F841


if __name__ == "__main__":
    model_config = ALL_MODELS[args.model]
    original_input_cost = model_config.input_costs
    original_output_cost = model_config.output_costs

    run_single_ablation(
        ablation_name="input_cost_zero",
        cost_per_input_token=0.0,
        cost_per_output_token=original_output_cost,
    )

    run_single_ablation(
        ablation_name="output_cost_zero",
        cost_per_input_token=original_input_cost,
        cost_per_output_token=0.0,
    )

    run_single_ablation(
        ablation_name="both_costs_zero",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
    )

    logger.info("All cost ablation experiments completed successfully")
