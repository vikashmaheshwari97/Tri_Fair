"""
Evaluate logged prompts from ExperimentCallback parquet logs.

Usage example:
uv run scripts/evaluate_logged_prompts.py \
    --random-seed 42 --dataset subj --model mistral-3-24b \
    --log-path results/main_results/run123/step_logs/step_results.parquet \
    --incumbents True --step 10
"""

import argparse
import logging
import os

import pandas as pd
from promptolution.predictors import MarkerBasedPredictor
from promptolution.utils import Prompt

from src.config.dataset_configs import ALL_DATASETS
from src.config.model_configs import ALL_MODELS
from src.config.setup_config import SETUP
from src.helpers.llm_creation import create_llm
from src.helpers.task_creation import create_test_task
from src.utils import seed_everything

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--incumbents", type=bool, default=False)
    parser.add_argument(
        "--log-path",
        required=True,
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
    )
    return parser.parse_args()


def load_prompts_from_log(log_path, step, incumbents):
    df = pd.read_parquet(log_path + "/step_results.parquet")

    if step == -1:
        step = df["step"].max()

    if step is not None:
        df_step = df[df["step"] == step]

        if "GEPA" in log_path:
            df_step = df_step.sort_values(by="score", ascending=False).head(1)

        if incumbents:
            df_incumbents = df[df["is_incumbent"]]
            df = pd.concat([df_step, df_incumbents])
        else:
            df = df_step
    else:
        if incumbents:
            df = df[df["is_incumbent"]]

    # remove already evaluated prompts
    eval_path = log_path + "/eval.parquet"
    if os.path.exists(eval_path):
        df_eval = pd.read_parquet(eval_path)
        evaluated_prompts = set(df_eval["prompt"].tolist())
        df = df[~df["prompt"].isin(evaluated_prompts)]

    prompt_texts = [str(p) for p in df["prompt"].tolist()]

    # de-duplicate while preserving order
    seen = set()
    unique_prompts = []
    for p in prompt_texts:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)

    return unique_prompts, step


def main():
    args = parse_args()
    logger.info(f"Log path: {args.log_path}")
    logger.info(f"Dataset: {args.dataset}, Model: {args.model}, Step: {args.step}")
    seed_everything(args.random_seed)

    dataset_config = ALL_DATASETS[args.dataset]
    model_config = ALL_MODELS[args.model]

    prompts_text, chosen_step = load_prompts_from_log(
        log_path=args.log_path,
        step=args.step,
        incumbents=args.incumbents,
    )
    logger.info(f"Loaded {len(prompts_text)} prompts from step {chosen_step}")

    llm = create_llm(
        model_config=model_config,
        seed=args.random_seed,
    )

    test_task = create_test_task(
        dataset_config=dataset_config,
        eval_strategy="full",
        n_subsamples=0,
        test_size=SETUP.test_size,
        seed=args.random_seed,
    )

    predictor = MarkerBasedPredictor(llm, test_task.classes)

    prompt_list = [Prompt(instruction=p) for p in prompts_text]

    logger.info(f"Evaluating {len(prompt_list)} prompts...")
    results = test_task.evaluate(prompts=prompt_list, predictor=predictor)
    logger.info(f"Evaluation complete. Best score: {max(results.agg_scores):.4f}")

    results_dict = {
        "prompt": prompts_text,
        "score": results.agg_scores,
        "input_tokens": results.agg_input_tokens,
        "output_tokens": results.agg_output_tokens,
        "step": [chosen_step] * len(prompts_text),
    }

    df_results = pd.DataFrame(results_dict)

    output_path = f"{args.log_path}/eval.parquet"

    if os.path.exists(output_path):
        df_existing = pd.read_parquet(output_path)
        df_results = pd.concat([df_existing, df_results], ignore_index=True)

    df_results.to_parquet(output_path, index=False)
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
