"""Script for running cost ablation experiments with MO-CAPO."""

import argparse
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
parser.add_argument("--output-dir", default="results/hp_sens/")
parser.add_argument("--n-init-prompts", default=10, type=int)
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--block-size", type=int, default=30)
parser.add_argument("--crossover-rate", type=int, default=7)

args = parser.parse_args()

logger = getLogger(__name__)


if __name__ == "__main__":
    logging_dir = os.path.join(args.output_dir, generate_random_hash()[:4])
    os.makedirs(logging_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "logging_dir.txt"), "w") as f:
        f.write(logging_dir)

    seed_everything(args.random_seed)

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
        n_subsamples=args.block_size,
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
        crossovers_per_iter=args.crossover_rate,
        upper_shots=5,
        cost_per_input_token=model_config.input_costs,
        cost_per_output_token=model_config.output_costs,
        df_few_shots=df_fewshots,
        callbacks=callbacks,
    )

    best_prompts = optimizer.optimize(n_steps=SETUP.n_steps)
