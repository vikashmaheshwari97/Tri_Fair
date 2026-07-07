"""
Script for evaluating all initial prompts on test data.

Example usage:
uv run scripts/evaluate_initial_prompts.py --random-seed 42 --dataset subj --model mistral-3-24b
"""

import argparse
import os
from logging import getLogger

import pandas as pd
from promptolution.predictors import MarkerBasedPredictor
from promptolution.utils import Prompt

from src.config.dataset_configs import ALL_DATASETS
from src.config.initial_prompts import INITIAL_PROMPTS
from src.config.model_configs import ALL_MODELS
from src.config.setup_config import SETUP
from src.helpers.llm_creation import create_llm
from src.helpers.task_creation import create_dev_tasks, create_test_task
from src.utils import seed_everything

parser = argparse.ArgumentParser()

parser.add_argument("--random-seed", type=int, required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--dev-set", action="store_true")

args = parser.parse_args()

logger = getLogger(__name__)

if __name__ == "__main__":
    seed_everything(args.random_seed)

    dataset_config = ALL_DATASETS[args.dataset]
    model_config = ALL_MODELS[args.model]

    set_name = "dev" if args.dev_set else "test"

    output_dir = f"results/init_results/{args.model}/{args.dataset}/{set_name}/seed_{args.random_seed}/"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Results will be saved to {output_dir}eval.csv")

    llm = create_llm(
        model_config=model_config,
        seed=args.random_seed,
    )

    if args.dev_set:
        task, _ = create_dev_tasks(
            dataset_config=dataset_config,
            eval_strategy="full",
            n_subsamples=0,
            dev_size=SETUP.dev_size,
            fs_size=SETUP.fs_size,
            seed=args.random_seed,
        )
    else:
        task = create_test_task(
            dataset_config=dataset_config,
            eval_strategy="full",
            n_subsamples=0,  # No subsampling for test evaluation
            test_size=SETUP.test_size,
            seed=args.random_seed,
        )

    predictor = MarkerBasedPredictor(llm, task.classes)

    initial_prompts = INITIAL_PROMPTS[dataset_config.alias]
    prompt_list = [Prompt(instruction=prompt) for prompt in initial_prompts]

    logger.info(
        f"Evaluating {len(initial_prompts)} initial prompts on {dataset_config.alias} dataset"
    )

    results = task.evaluate(prompts=prompt_list, predictor=predictor)
    results_dict = {
        "prompt": initial_prompts,
        "score": results.agg_scores,
        "input_tokens": results.agg_input_tokens,
        "output_tokens": results.agg_output_tokens,
    }

    df_results = pd.DataFrame(results_dict)
    csv_path = os.path.join(output_dir, "eval.csv")
    df_results.to_csv(csv_path, index=False)

    logger.info(f"Results saved to {csv_path}")
