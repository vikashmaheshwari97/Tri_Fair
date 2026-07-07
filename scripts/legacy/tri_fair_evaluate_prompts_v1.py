"""Evaluate development-selected prompts on the untouched holdout manifest."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from promptolution.predictors import MarkerBasedPredictor
from promptolution.utils import Prompt

from src.config.dataset_configs import ALL_DATASETS
from src.config.model_configs import ALL_MODELS
from src.config.setup_config import SETUP
from src.helpers.llm_creation import create_llm
from src.helpers.task_creation import create_test_task
from src.tasks.fairness_task import FairnessEvalResult
from src.utils import seed_everything

logger = logging.getLogger(__name__)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).casefold()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--dataset", choices=sorted(ALL_DATASETS), required=True)
    parser.add_argument("--model", choices=sorted(ALL_MODELS), required=True)
    parser.add_argument("--incumbents", type=str2bool, default=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--manifest-dir", default="data/splits")
    parser.add_argument("--regenerate-manifest", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=16)
    return parser.parse_args()


def _latest_per_prompt(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("step").groupby("prompt", as_index=False).tail(1)


def load_prompts_from_log(
    log_path: str | Path, step: int, incumbents: bool
) -> pd.DataFrame:
    path = Path(log_path) / "step_results.parquet"
    frame = pd.read_parquet(path)
    chosen_step = int(frame["step"].max()) if step == -1 else int(step)
    if chosen_step not in set(frame["step"].astype(int)):
        raise ValueError(f"Step {chosen_step} is not present in {path}")

    current = frame[frame["step"].astype(int) == chosen_step]
    if incumbents:
        historical = frame[
            (frame["step"] <= chosen_step) & frame["is_incumbent"].fillna(False)
        ]
        selected = pd.concat([current, historical], ignore_index=True)
    else:
        selected = current
    selected = _latest_per_prompt(selected).copy()
    selected["chosen_step"] = chosen_step
    return selected


def reconstruct_prompts(frame: pd.DataFrame) -> list[Prompt]:
    prompts: list[Prompt] = []
    structured = {"instruction", "few_shots_json", "downstream_template"}.issubset(
        frame.columns
    )
    for _, row in frame.iterrows():
        if structured and pd.notna(row.get("instruction")):
            few_shots = json.loads(row.get("few_shots_json") or "[]")
            prompts.append(
                Prompt(
                    instruction=str(row["instruction"]),
                    few_shots=list(few_shots),
                    downstream_template=row.get("downstream_template")
                    if pd.notna(row.get("downstream_template"))
                    else None,
                )
            )
        else:
            prompts.append(Prompt(instruction=str(row["prompt"])))
    return prompts


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    seed_everything(args.random_seed)

    selected = load_prompts_from_log(args.log_path, args.step, args.incumbents)
    prompts = reconstruct_prompts(selected)
    logger.info("Loaded %d unique prompts for holdout evaluation", len(prompts))

    dataset_config = ALL_DATASETS[args.dataset]
    model_config = ALL_MODELS[args.model]
    llm = create_llm(model_config=model_config, seed=args.random_seed)
    if hasattr(llm, "sampling_params"):
        llm.sampling_params.max_tokens = int(args.max_output_tokens)

    test_task = create_test_task(
        dataset_config=dataset_config,
        eval_strategy="full",
        n_subsamples=0,
        test_size=SETUP.test_size,
        seed=args.random_seed,
        manifest_dir=args.manifest_dir,
        regenerate_manifest=args.regenerate_manifest,
    )
    predictor = MarkerBasedPredictor(llm, test_task.classes)
    result = test_task.evaluate(
        prompts=prompts, predictor=predictor, eval_strategy="full"
    )

    weighted_cost = model_config.input_costs * np.asarray(
        result.agg_input_tokens, dtype=float
    ) + model_config.output_costs * np.asarray(result.agg_output_tokens, dtype=float)
    output = selected.reset_index(drop=True).copy()
    output["test_quality"] = np.asarray(result.agg_scores, dtype=float)
    output["test_cost"] = weighted_cost
    output["test_input_tokens"] = np.asarray(result.agg_input_tokens, dtype=float)
    output["test_output_tokens"] = np.asarray(result.agg_output_tokens, dtype=float)

    if isinstance(result, FairnessEvalResult):
        output["test_fairness"] = np.asarray(result.fairness_loss, dtype=float)
        output["test_fairness_ready"] = np.asarray(result.fairness_ready, dtype=bool)
        output["test_fairness_diagnostics_json"] = [
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            for value in result.fairness_diagnostics
        ]
        output["test_group_support_json"] = [
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            for value in result.fairness_support
        ]
        output["test_objective_vector"] = [
            [float(q), -float(c), -float(f)]
            for q, c, f in zip(
                output["test_quality"], output["test_cost"], output["test_fairness"]
            )
        ]
    else:
        output["test_fairness"] = np.nan
        output["test_fairness_ready"] = False
        output["test_fairness_diagnostics_json"] = "{}"
        output["test_group_support_json"] = "{}"
        output["test_objective_vector"] = [
            [float(q), -float(c)]
            for q, c in zip(output["test_quality"], output["test_cost"])
        ]

    # Preserve development selection values under explicit names.
    output = output.rename(
        columns={
            "quality": "dev_quality",
            "cost": "dev_cost",
            "fairness": "dev_fairness",
            "fairness_ready": "dev_fairness_ready",
            "fairness_diagnostics_json": "dev_fairness_diagnostics_json",
            "group_support_json": "dev_group_support_json",
        }
    )
    output["evaluation_seed"] = args.random_seed
    output["dataset"] = args.dataset
    output["model"] = args.model
    output["manifest_path"] = getattr(test_task, "manifest_path", None)

    output_path = Path(args.log_path) / "eval.parquet"
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        output = pd.concat([existing, output], ignore_index=True)
        dedup_columns = [
            column for column in ("prompt", "chosen_step") if column in output
        ]
        output = output.drop_duplicates(subset=dedup_columns, keep="last")
    temporary = output_path.with_suffix(".parquet.tmp")
    output.to_parquet(temporary, index=False)
    temporary.replace(output_path)
    logger.info("Holdout results written to %s", output_path)


if __name__ == "__main__":
    main()
