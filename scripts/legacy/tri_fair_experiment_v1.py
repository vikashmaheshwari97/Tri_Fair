"""Run MO-CAPO or Tri-Fair experiments.

Examples
--------
Fresh 1M Tri-Fair run::

    uv run scripts/experiment.py \
      --experiment-name tri_fair_bbq \
      --random-seed 42 --budget-per-run 1000000 \
      --output-dir results/tri_fair/qwen-3-30b/bbq/Tri-Fair/seed42/ \
      --dataset bbq --model qwen-3-30b --optimizer Tri-Fair \
      --n-init-prompts 6 --budget-checkpoints 250000,500000,750000,1000000

Resume the saved run to 5M::

    uv run scripts/experiment.py ... --budget-per-run 5000000 \
      --resume-from <logging_dir>/checkpoints/latest.pkl \
      --budget-checkpoints 2000000,3000000,4000000,5000000
"""

from __future__ import annotations

import argparse
import json
import random
from logging import getLogger
from pathlib import Path

from promptolution.optimizers import CAPO, EvoPromptGA
from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.predictors import MarkerBasedPredictor
from promptolution.utils import LoggerCallback
from promptolution.utils.callbacks import TokenCountCallback
from promptolution.utils.templates import EVOPROMPT_GA_TEMPLATE

from mo_capo import MoCAPO
from nsgaii_po import NSGAiiPO
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

logger = getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--budget-per-run", type=int, required=True)
    parser.add_argument("--output-dir", default="results/main_results/")
    parser.add_argument("--dataset", choices=sorted(ALL_DATASETS), required=True)
    parser.add_argument("--model", choices=sorted(ALL_MODELS), required=True)
    parser.add_argument("--optimizer", choices=sorted(ALL_OPTIMIZERS), required=True)
    parser.add_argument("--n-init-prompts", type=int, default=6)
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--meta-max-output-tokens", type=int, default=256)
    parser.add_argument("--budget-checkpoints", default="250000,500000,750000,1000000")
    parser.add_argument("--manifest-dir", default="data/splits")
    parser.add_argument("--regenerate-manifest", action="store_true")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-steps", type=int, default=SETUP.n_steps)
    parser.add_argument("--random-selection", action="store_true")
    parser.add_argument("--no-weaker-dominance", action="store_true")
    return parser.parse_args()


def parse_checkpoints(raw: str, budget: int) -> list[int]:
    values = {int(part.strip()) for part in raw.split(",") if part.strip()}
    values.add(int(budget))
    invalid = [value for value in values if value <= 0]
    if invalid:
        raise ValueError(f"Budget checkpoints must be positive: {invalid}")
    return sorted(values)


def resolve_logging_dir(args: argparse.Namespace) -> Path:
    if args.resume_from:
        checkpoint = Path(args.resume_from).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if checkpoint.parent.name != "checkpoints":
            raise ValueError(
                "--resume-from should point to <logging_dir>/checkpoints/*.pkl"
            )
        return checkpoint.parent.parent
    return Path(args.output_dir) / generate_random_hash()[:4]


def set_generation_limit(llm, max_tokens: int) -> None:
    if max_tokens <= 0:
        raise ValueError("generation token limits must be positive")
    if not hasattr(llm, "sampling_params"):
        raise TypeError("The configured LLM does not expose vLLM sampling_params")
    llm.sampling_params.max_tokens = int(max_tokens)


def main() -> None:
    args = parse_args()
    dataset_config = ALL_DATASETS[args.dataset]
    optimizer_config = ALL_OPTIMIZERS[args.optimizer]
    model_config = ALL_MODELS[args.model]

    fairness_optimizer = args.optimizer in {"Tri-Fair", "NSGAII-PO-Fair"}
    if fairness_optimizer and dataset_config.task_type != "Fairness":
        raise ValueError(f"{args.optimizer} requires a fairness dataset")
    if dataset_config.task_type == "Fairness" and not fairness_optimizer:
        raise ValueError(
            "The first Tri-Fair implementation intentionally allows only Tri-Fair and "
            "NSGAII-PO-Fair on fairness datasets, preventing accidental scalarization."
        )
    if args.n_init_prompts > len(dataset_config.initial_prompts):
        raise ValueError(
            f"Requested {args.n_init_prompts} initial prompts but {args.dataset} provides "
            f"only {len(dataset_config.initial_prompts)}"
        )

    logging_dir = resolve_logging_dir(args)
    logging_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logging_dir.txt").write_text(str(logging_dir), encoding="utf-8")

    seed_everything(args.random_seed)
    with (logging_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    downstream_llm = create_llm(model_config=model_config, seed=args.random_seed)
    meta_llm = copy_llm(downstream_llm)
    set_generation_limit(downstream_llm, args.max_output_tokens)
    set_generation_limit(meta_llm, args.meta_max_output_tokens)

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
        dataset_config.initial_prompts,
        args.n_init_prompts,
    )
    predictor = MarkerBasedPredictor(downstream_llm, dev_task.classes)

    checkpoints = parse_checkpoints(args.budget_checkpoints, args.budget_per_run)
    callbacks = [
        LoggerCallback(logger),
        ExperimentCallback(dir=str(logging_dir)),
        OptimizerCheckpointCallback(logging_dir, checkpoints),
        TokenCountCallback(args.budget_per_run, "total_tokens"),
    ]

    common = {
        "predictor": predictor,
        "task": dev_task,
        "meta_llm": meta_llm,
        "initial_prompts": initial_prompts,
        "callbacks": callbacks,
    }

    optimizer: BaseOptimizer
    if optimizer_config.optimizer == "EvoPromptGA":
        optimizer = EvoPromptGA(
            **common,
            prompt_template=EVOPROMPT_GA_TEMPLATE,
            **optimizer_config.optimizer_params,
        )
    elif optimizer_config.optimizer == "CAPO":
        optimizer = CAPO(
            **common,
            df_few_shots=df_fewshots,
            **optimizer_config.optimizer_params,
        )
    elif optimizer_config.optimizer == "GEPA":
        optimizer = Gepa(**common, **optimizer_config.optimizer_params)
    elif optimizer_config.optimizer == "NSGAII-PO":
        optimizer = NSGAiiPO(
            **common,
            df_few_shots=df_fewshots,
            cost_per_input_token=model_config.input_costs,
            cost_per_output_token=model_config.output_costs,
            **optimizer_config.optimizer_params,
        )
    elif optimizer_config.optimizer == "MO-CAPO":
        optimizer = MoCAPO(
            **common,
            df_few_shots=df_fewshots,
            cost_per_input_token=model_config.input_costs,
            cost_per_output_token=model_config.output_costs,
            random_selection=args.random_selection,
            no_weaker_dominance=args.no_weaker_dominance,
            **optimizer_config.optimizer_params,
        )
    elif optimizer_config.optimizer == "Tri-Fair":
        optimizer = TriFair(
            **common,
            df_few_shots=df_fewshots,
            cost_per_input_token=model_config.input_costs,
            cost_per_output_token=model_config.output_costs,
            random_selection=args.random_selection,
            no_weaker_dominance=args.no_weaker_dominance,
            **optimizer_config.optimizer_params,
        )
    elif optimizer_config.optimizer == "NSGAII-PO-Fair":
        optimizer = NSGAiiPOFair(
            **common,
            df_few_shots=df_fewshots,
            cost_per_input_token=model_config.input_costs,
            cost_per_output_token=model_config.output_costs,
            **optimizer_config.optimizer_params,
        )
    else:  # pragma: no cover - choices prevent this branch
        raise ValueError(f"Optimizer {optimizer_config.optimizer!r} not recognized")

    if args.resume_from:
        if not hasattr(optimizer, "load_checkpoint"):
            raise TypeError(
                f"{optimizer.__class__.__name__} does not support checkpoint resume"
            )
        optimizer.load_checkpoint(args.resume_from)  # type: ignore[attr-defined]

    optimizer.optimize(n_steps=args.max_steps)


if __name__ == "__main__":
    main()
