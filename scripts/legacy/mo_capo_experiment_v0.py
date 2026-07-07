"""
Main script for running all experiments.

for minimal example run:
python scripts/experiment.py --experiment-name test --random-seed 42 \
    --budget-per-run 7500000 --output-dir results/ --dataset subj \
    --model qwen-3-30b --optimizer CAPO
"""

import argparse
import json
import os
import random
from logging import getLogger

from promptolution.optimizers import CAPO, EvoPromptGA
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.predictors import MarkerBasedPredictor
from promptolution.utils import LoggerCallback
from promptolution.utils.callbacks import TokenCountCallback
from promptolution.utils.templates import EVOPROMPT_GA_TEMPLATE

from mo_capo import MoCAPO
from nsgaii_po import NSGAiiPO
from src.callbacks import ExperimentCallback
from src.config.dataset_configs import ALL_DATASETS
from src.config.model_configs import ALL_MODELS
from src.config.optimizer_configs import ALL_OPTIMIZERS
from src.config.setup_config import SETUP
from src.gepa import Gepa
from src.helpers.llm_creation import create_llm
from src.helpers.task_creation import create_dev_tasks
from src.utils import copy_llm, generate_random_hash, seed_everything

parser = argparse.ArgumentParser()

parser.add_argument("--experiment-name", required=True)
parser.add_argument("--random-seed", type=int, required=True)
parser.add_argument("--budget-per-run", type=int, required=True)
parser.add_argument("--output-dir", default="results/main_results/")
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--optimizer", required=True)
parser.add_argument("--n-init-prompts", type=int, required=True)
parser.add_argument("--random-selection", action="store_true")
parser.add_argument("--no-weaker-dominance", action="store_true")

args = parser.parse_args()

logger = getLogger(__name__)

if __name__ == "__main__":
    logging_dir = args.output_dir + "/" + generate_random_hash()[:4] + "/"
    os.makedirs(logging_dir, exist_ok=True)

    with open(args.output_dir + "logging_dir.txt", "w") as f:
        f.write(logging_dir)

    seed_everything(args.random_seed)

    with open(logging_dir + "/args.json", "w") as f:
        json.dump(vars(args), f)

    callbacks = [
        LoggerCallback(logger),
        ExperimentCallback(dir=logging_dir),
        TokenCountCallback(args.budget_per_run, "total_tokens"),
    ]

    # get configs
    dataset_config = ALL_DATASETS[args.dataset]
    optimizer_config = ALL_OPTIMIZERS[args.optimizer]
    model_config = ALL_MODELS[args.model]

    # Set up LLMs
    llm = create_llm(
        model_config=model_config,
        seed=args.random_seed,
    )

    downstream_llm = llm
    meta_llm = copy_llm(llm)

    # set-up task and few-shots
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

    # set-up predictor
    predictor = MarkerBasedPredictor(downstream_llm, dev_task.classes)

    # initialize optimizer
    optimizer: BaseOptimizer
    if optimizer_config.optimizer == "EvoPromptGA":
        optimizer = EvoPromptGA(
            predictor=predictor,
            task=dev_task,
            meta_llm=meta_llm,
            initial_prompts=initial_prompts,
            prompt_template=EVOPROMPT_GA_TEMPLATE,
            callbacks=callbacks,
            **optimizer_config.optimizer_params,
        )
    elif optimizer_config.optimizer == "CAPO":
        optimizer = CAPO(
            predictor=predictor,
            task=dev_task,
            meta_llm=meta_llm,
            initial_prompts=initial_prompts,
            df_few_shots=df_fewshots,
            callbacks=callbacks,
            **optimizer_config.optimizer_params,
        )
    elif optimizer_config.optimizer == "GEPA":
        optimizer = Gepa(
            predictor=predictor,
            task=dev_task,
            initial_prompts=initial_prompts,
            callbacks=callbacks,
            meta_llm=meta_llm,
            **optimizer_config.optimizer_params,
        )
    elif optimizer_config.optimizer == "NSGAII-PO":
        optimizer = NSGAiiPO(
            predictor=predictor,
            task=dev_task,
            meta_llm=meta_llm,
            initial_prompts=initial_prompts,
            df_few_shots=df_fewshots,
            callbacks=callbacks,
            cost_per_input_token=model_config.input_costs,
            cost_per_output_token=model_config.output_costs,
            **optimizer_config.optimizer_params,
        )
    elif optimizer_config.optimizer == "MO-CAPO":
        optimizer = MoCAPO(
            predictor=predictor,
            task=dev_task,
            meta_llm=meta_llm,
            initial_prompts=initial_prompts,
            df_few_shots=df_fewshots,
            callbacks=callbacks,
            cost_per_input_token=model_config.input_costs,
            cost_per_output_token=model_config.output_costs,
            random_selection=args.random_selection,
            no_weaker_dominance=args.no_weaker_dominance,
            **optimizer_config.optimizer_params,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_config.optimizer} not recognized.")

    # run optimization
    best_prompts = optimizer.optimize(n_steps=SETUP.n_steps)
