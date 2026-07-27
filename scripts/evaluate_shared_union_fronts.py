"""Evaluate the union of both methods' unique logged prompts in one held-out job.

A prompt is evaluated exactly once per dataset/model/seed and its held-out
result is joined back to every method/checkpoint membership row.  Greedy
downstream decoding and stable prompt-ID ordering make results independent of
method-specific batching.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from promptolution.predictors import MarkerBasedPredictor

from scripts._common import (
    atomic_write_parquet,
    configure_logging,
    prompt_id,
    reconstruct_prompts,
    set_generation_limit,
    sha256_file,
    stable_latest_per_prompt,
    utc_now_iso,
)
from scripts.evaluate_checkpoint_fronts import _step_token_table
from scripts.evaluate_prompts import (
    _rename_development_columns,
    load_prompt_candidates,
)
from src.config.dataset_configs import ALL_DATASETS
from src.config.model_configs import ALL_MODELS
from src.config.setup_config import SETUP
from src.config.v7_profiles import apply_v7_dataset_profile
from src.helpers.generation_control import configure_downstream_greedy
from src.helpers.llm_creation import create_llm
from src.helpers.task_creation import create_test_task
from src.utils import seed_everything


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--dataset", choices=sorted(ALL_DATASETS), required=True)
    parser.add_argument("--model", choices=sorted(ALL_MODELS), required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument(
        "--optimizers",
        default="Tri-Fair-v7,NSGAII-PO-Fair",
    )
    parser.add_argument("--minimum-actual-tokens", type=int, default=0)
    parser.add_argument("--maximum-actual-tokens", type=int, default=5_000_000)
    parser.add_argument("--manifest-dir", default="data/splits_v7")
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _parse_optimizers(raw: str) -> tuple[str, ...]:
    values = tuple(
        dict.fromkeys(
            piece.strip()
            for piece in str(raw).split(",")
            if piece.strip()
        )
    )
    if len(values) < 2:
        raise ValueError("The shared union requires at least two optimizers")
    return values


def _logging_dir(
    study_root: Path,
    dataset: str,
    optimizer: str,
    seed: int,
) -> Path:
    pointer = (
        study_root
        / dataset
        / optimizer
        / f"seed{seed}"
        / "logging_dir.txt"
    )
    if not pointer.is_file():
        raise FileNotFoundError(pointer)
    resolved = Path(
        pointer.read_text(encoding="utf-8").strip()
    ).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def _run_args(log_path: Path) -> dict[str, object]:
    path = log_path / "args.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _membership_for_run(
    log_path: Path,
    *,
    optimizer: str,
    dataset: str,
    model: str,
    seed: int,
    minimum_tokens: int,
    maximum_tokens: int,
) -> pd.DataFrame:
    step_path = log_path / "step_results.parquet"
    if not step_path.is_file():
        raise FileNotFoundError(step_path)

    frame = pd.read_parquet(step_path)
    per_step = _step_token_table(frame)
    per_step = per_step[
        (per_step >= int(minimum_tokens))
        & (per_step <= int(maximum_tokens))
    ]
    if per_step.empty:
        raise RuntimeError(
            f"No real states for {optimizer} in "
            f"{minimum_tokens}..{maximum_tokens} tokens"
        )

    metadata = _run_args(log_path)
    parts: list[pd.DataFrame] = []
    for step, actual_tokens in per_step.items():
        selected, resolved_step = load_prompt_candidates(
            log_path,
            step=int(step),
            selection="current",
        )
        if "is_incumbent" not in selected:
            raise ValueError(
                f"{step_path} lacks is_incumbent; current archive cannot be reconstructed"
            )
        selected = selected[
            selected["is_incumbent"].fillna(False).astype(bool)
        ].reset_index(drop=True)
        if selected.empty:
            continue

        selected = _rename_development_columns(selected)
        selected["optimizer"] = optimizer
        selected["dataset"] = dataset
        selected["model"] = model
        selected["seed"] = int(seed)
        selected["chosen_step"] = int(resolved_step)
        selected["actual_budget_tokens"] = int(actual_tokens)
        selected["budget_checkpoint"] = int(actual_tokens)
        selected["checkpoint_signed_error"] = 0
        selected["checkpoint_relative_error"] = 0.0
        selected["checkpoint_policy"] = "shared_union_exact_real_state"
        selected["selection_policy"] = "current_incumbents"
        selected["configured_budget"] = int(
            metadata.get("budget_per_run", maximum_tokens)
        )
        selected["source_log_path"] = str(log_path.resolve())
        if "prompt_id" not in selected:
            selected["prompt_id"] = selected["prompt"].astype(str).map(
                prompt_id
            )
        parts.append(selected)

    if not parts:
        raise RuntimeError(f"No incumbent memberships found for {optimizer}")
    return pd.concat(parts, ignore_index=True, sort=False)


def _test_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith("test_")
        or column
        in {
            "prompt_id",
            "evaluation_timestamp",
            "manifest_path",
            "manifest_sha256",
            "shared_union_evaluation",
        }
    ]


def run(args: argparse.Namespace) -> Path:
    if args.maximum_actual_tokens <= args.minimum_actual_tokens:
        raise ValueError(
            "maximum-actual-tokens must exceed minimum-actual-tokens"
        )
    optimizers = _parse_optimizers(args.optimizers)
    study_root = Path(args.study_root).expanduser().resolve()

    memberships = []
    for optimizer in optimizers:
        log_path = _logging_dir(
            study_root,
            args.dataset,
            optimizer,
            args.random_seed,
        )
        memberships.append(
            _membership_for_run(
                log_path,
                optimizer=optimizer,
                dataset=args.dataset,
                model=args.model,
                seed=args.random_seed,
                minimum_tokens=args.minimum_actual_tokens,
                maximum_tokens=args.maximum_actual_tokens,
            )
        )

    membership = pd.concat(
        memberships,
        ignore_index=True,
        sort=False,
    )
    unique_candidates = (
        stable_latest_per_prompt(membership)
        .sort_values("prompt_id", kind="stable")
        .reset_index(drop=True)
    )

    output_path = (
        Path(args.output_file).expanduser().resolve()
        if args.output_file
        else (
            study_root
            / "shared_union_holdout"
            / args.dataset
            / f"seed{args.random_seed}"
            / "eval_checkpoints_union.parquet"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cached = pd.DataFrame()
    if output_path.is_file() and not args.force:
        cached = pd.read_parquet(output_path)
    cached_ids = set(
        cached.get("prompt_id", pd.Series(dtype=str)).astype(str)
    )
    pending = unique_candidates[
        ~unique_candidates["prompt_id"].astype(str).isin(cached_ids)
    ].reset_index(drop=True)

    evaluated = pd.DataFrame()
    if not pending.empty:
        seed_everything(args.random_seed)
        dataset_config = apply_v7_dataset_profile(
            ALL_DATASETS[args.dataset]
        )
        model_config = ALL_MODELS[args.model]
        llm = create_llm(
            model_config=model_config,
            seed=args.random_seed,
        )
        configure_downstream_greedy(
            llm,
            seed=args.random_seed,
        )
        set_generation_limit(llm, args.max_output_tokens)

        test_task = create_test_task(
            dataset_config=dataset_config,
            eval_strategy="full",
            n_subsamples=0,
            test_size=SETUP.test_size,
            seed=args.random_seed,
            manifest_dir=args.manifest_dir,
            regenerate_manifest=False,
        )
        prompts = reconstruct_prompts(pending)
        result = test_task.evaluate(
            prompts=prompts,
            predictor=MarkerBasedPredictor(
                llm,
                test_task.classes,
            ),
            eval_strategy="full",
        )

        evaluated = pending[["prompt_id"]].copy()
        quality = np.asarray(result.agg_scores, dtype=float)
        input_tokens = np.asarray(
            result.agg_input_tokens,
            dtype=float,
        )
        output_tokens = np.asarray(
            result.agg_output_tokens,
            dtype=float,
        )
        evaluated["test_quality"] = quality
        evaluated["test_cost"] = (
            float(model_config.input_costs) * input_tokens
            + float(model_config.output_costs) * output_tokens
        )
        evaluated["test_input_tokens"] = input_tokens
        evaluated["test_output_tokens"] = output_tokens
        evaluated["test_fairness"] = np.asarray(
            result.fairness_loss,
            dtype=float,
        )
        evaluated["test_fairness_ready"] = np.asarray(
            result.fairness_ready,
            dtype=bool,
        )
        evaluated["test_fairness_diagnostics_json"] = [
            json.dumps(value, sort_keys=True, default=str)
            for value in result.fairness_diagnostics
        ]
        evaluated["test_group_support_json"] = [
            json.dumps(value, sort_keys=True, default=str)
            for value in result.fairness_support
        ]
        evaluated["test_objective_vector"] = [
            [float(q), -float(c), -float(f)]
            for q, c, f in zip(
                evaluated["test_quality"],
                evaluated["test_cost"],
                evaluated["test_fairness"],
            )
        ]
        evaluated["evaluation_timestamp"] = utc_now_iso()
        evaluated["manifest_path"] = str(
            getattr(test_task, "manifest_path", "")
        )
        evaluated["manifest_sha256"] = sha256_file(
            evaluated["manifest_path"].iloc[0]
        )
        evaluated["shared_union_evaluation"] = True

    lookup_parts: list[pd.DataFrame] = []
    if not cached.empty:
        lookup_parts.append(
            stable_latest_per_prompt(
                cached[_test_columns(cached)]
            )
        )
    if not evaluated.empty:
        lookup_parts.append(evaluated)
    if not lookup_parts:
        raise RuntimeError("No cached or newly evaluated held-out rows")

    lookup = stable_latest_per_prompt(
        pd.concat(lookup_parts, ignore_index=True, sort=False)
    )
    test_only = [
        column for column in lookup.columns if column != "prompt_id"
    ]
    output = membership.merge(
        lookup[["prompt_id", *test_only]],
        on="prompt_id",
        how="left",
        validate="many_to_one",
    )
    if output["test_quality"].isna().any():
        raise RuntimeError("Some membership rows lack shared held-out results")

    dedup = [
        "optimizer",
        "prompt_id",
        "chosen_step",
        "budget_checkpoint",
    ]
    output = (
        output.sort_values(dedup, kind="stable")
        .drop_duplicates(dedup, keep="last")
        .reset_index(drop=True)
    )
    atomic_write_parquet(output, output_path)
    LOGGER.info(
        "Wrote %d memberships for %d unique prompts to %s",
        len(output),
        output["prompt_id"].nunique(),
        output_path,
    )
    return output_path


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
