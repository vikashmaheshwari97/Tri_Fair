"""Evaluate development-selected prompts on the untouched holdout manifest.

This script reads ``step_results.parquet`` produced by ``ExperimentCallback``,
reconstructs complete prompts including few-shot examples, evaluates only
previously unseen prompt/step pairs, and atomically appends canonical
quality-cost-fairness records to ``eval.parquet``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from promptolution.predictors import MarkerBasedPredictor

from src.config.dataset_configs import ALL_DATASETS
from src.config.model_configs import ALL_MODELS
from src.config.setup_config import SETUP
from src.helpers.llm_creation import create_llm
from src.helpers.task_creation import create_test_task
from src.utils import seed_everything

try:  # Supports both ``python -m scripts...`` and ``python scripts/...``.
    from scripts._common import (
        add_result_columns,
        configure_logging,
        directory_lock,
        manifest_lock_path,
        merge_parquet_rows,
        prompt_id,
        reconstruct_prompts,
        resolve_manifest_path,
        set_generation_limit,
        sha256_file,
        stable_latest_per_prompt,
        str2bool,
        utc_now_iso,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    from _common import (  # type: ignore[no-redef]
        add_result_columns,
        configure_logging,
        directory_lock,
        manifest_lock_path,
        merge_parquet_rows,
        prompt_id,
        reconstruct_prompts,
        resolve_manifest_path,
        set_generation_limit,
        sha256_file,
        stable_latest_per_prompt,
        str2bool,
        utc_now_iso,
    )

LOGGER = logging.getLogger(__name__)
SELECTIONS = ("current", "incumbents", "current_and_incumbents")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate logged prompts on the fixed test manifest."
    )
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--dataset", choices=sorted(ALL_DATASETS), required=True)
    parser.add_argument("--model", choices=sorted(ALL_MODELS), required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument(
        "--selection",
        choices=SELECTIONS,
        default="current_and_incumbents",
    )
    # Compatibility with existing jobs/run_eval.sbatch.
    parser.add_argument("--incumbents", type=str2bool, default=None)
    parser.add_argument("--manifest-dir", default="data/splits")
    parser.add_argument("--regenerate-manifest", action="store_true")
    parser.add_argument("--manifest-lock-timeout", type=float, default=3600.0)
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--output-file", default="eval.parquet")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def resolve_selection(args: argparse.Namespace) -> str:
    if args.incumbents is None:
        return str(args.selection)
    return "current_and_incumbents" if args.incumbents else "current"


def load_prompt_candidates(
    log_path: str | Path,
    *,
    step: int,
    selection: str,
) -> tuple[pd.DataFrame, int]:
    path = Path(log_path) / "step_results.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"step", "prompt"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} is empty")

    frame = frame.copy()
    frame["step"] = pd.to_numeric(frame["step"], errors="raise").astype(int)
    chosen_step = int(frame["step"].max()) if int(step) == -1 else int(step)
    if chosen_step not in set(frame["step"]):
        available = sorted(frame["step"].unique().tolist())
        raise ValueError(
            f"Step {chosen_step} is absent from {path}; available range is "
            f"{available[0]}..{available[-1]}"
        )

    current = frame[frame["step"] == chosen_step].copy()
    incumbent_history = pd.DataFrame(columns=frame.columns)
    if "is_incumbent" in frame:
        flags = frame["is_incumbent"].fillna(False).astype(bool)
        incumbent_history = frame[(frame["step"] <= chosen_step) & flags].copy()

    if selection == "current":
        selected = current
    elif selection == "incumbents":
        selected = incumbent_history
    elif selection == "current_and_incumbents":
        selected = pd.concat(
            [current, incumbent_history], ignore_index=True, sort=False
        )
    else:  # pragma: no cover - argparse prevents this branch
        raise ValueError(f"Unknown selection policy: {selection}")

    if selected.empty:
        raise ValueError(
            f"Selection policy {selection!r} produced no prompts at step {chosen_step}"
        )
    if "prompt_id" not in selected:
        selected["prompt_id"] = selected["prompt"].astype(str).map(prompt_id)
    selected = stable_latest_per_prompt(selected)
    selected["chosen_step"] = chosen_step
    selected["selection_policy"] = selection
    return selected.reset_index(drop=True), chosen_step


def _rename_development_columns(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "quality": "dev_quality",
        "cost": "dev_cost",
        "fairness": "dev_fairness",
        "fairness_ready": "dev_fairness_ready",
        "fairness_diagnostics_json": "dev_fairness_diagnostics_json",
        "group_support_json": "dev_group_support_json",
        "objective_vector": "dev_objective_vector",
    }
    safe_mapping = {
        source: target
        for source, target in mapping.items()
        if source in frame.columns and target not in frame.columns
    }
    return frame.rename(columns=safe_mapping)


def _pending_candidates(
    selected: pd.DataFrame,
    output_path: Path,
    *,
    force: bool,
) -> pd.DataFrame:
    if force or not output_path.exists():
        return selected.copy()
    existing = pd.read_parquet(output_path)
    if existing.empty:
        return selected.copy()
    if "prompt_id" not in existing and "prompt" in existing:
        existing["prompt_id"] = existing["prompt"].astype(str).map(prompt_id)
    if not {"prompt_id", "chosen_step"}.issubset(existing.columns):
        raise ValueError(
            f"Existing {output_path} lacks prompt_id/chosen_step. Re-run with --force "
            "after archiving the incompatible file."
        )
    done = set(
        zip(
            existing["prompt_id"].astype(str),
            pd.to_numeric(existing["chosen_step"], errors="coerce")
            .fillna(-1)
            .astype(int),
        )
    )
    mask = [
        (str(row.prompt_id), int(row.chosen_step)) not in done
        for row in selected[["prompt_id", "chosen_step"]].itertuples(index=False)
    ]
    return selected.loc[mask].reset_index(drop=True)


def run(args: argparse.Namespace) -> Path:
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")
    selection = resolve_selection(args)
    log_path = Path(args.log_path).expanduser().resolve()
    output_path = log_path / args.output_file

    selected, chosen_step = load_prompt_candidates(
        log_path,
        step=args.step,
        selection=selection,
    )
    pending = _pending_candidates(selected, output_path, force=args.force)
    if pending.empty:
        LOGGER.info(
            "No new prompts require evaluation for step=%d selection=%s",
            chosen_step,
            selection,
        )
        return output_path

    LOGGER.info(
        "Evaluating %d/%d selected prompts at step %d on %s",
        len(pending),
        len(selected),
        chosen_step,
        args.dataset,
    )
    seed_everything(args.random_seed)
    dataset_config = ALL_DATASETS[args.dataset]
    model_config = ALL_MODELS[args.model]

    llm = create_llm(model_config=model_config, seed=args.random_seed)
    set_generation_limit(llm, args.max_output_tokens)

    lock = manifest_lock_path(args.manifest_dir, args.dataset, args.random_seed)
    with directory_lock(lock, timeout_seconds=args.manifest_lock_timeout):
        test_task = create_test_task(
            dataset_config=dataset_config,
            eval_strategy="full",
            n_subsamples=0,
            test_size=SETUP.test_size,
            seed=args.random_seed,
            manifest_dir=args.manifest_dir,
            regenerate_manifest=args.regenerate_manifest,
        )

    prompts = reconstruct_prompts(pending)
    predictor = MarkerBasedPredictor(llm, test_task.classes)
    result = test_task.evaluate(
        prompts=prompts,
        predictor=predictor,
        eval_strategy="full",
    )

    output = _rename_development_columns(pending.reset_index(drop=True))
    output = add_result_columns(
        output,
        result,
        prefix="test",
        model_config=model_config,
    )
    output["evaluation_seed"] = int(args.random_seed)
    output["dataset"] = args.dataset
    output["model"] = args.model
    output["evaluation_timestamp"] = utc_now_iso()
    output["manifest_path"] = getattr(
        test_task,
        "manifest_path",
        str(resolve_manifest_path(args.manifest_dir, args.dataset, args.random_seed)),
    )
    output["manifest_sha256"] = sha256_file(output["manifest_path"].iloc[0])

    merged = merge_parquet_rows(
        output,
        output_path,
        deduplicate_on=("prompt_id", "chosen_step"),
    )
    not_ready = int((~output["test_fairness_ready"].astype(bool)).sum())
    if not_ready:
        LOGGER.warning(
            "%d evaluated prompts did not meet test fairness support requirements; "
            "their diagnostics remain in eval.parquet",
            not_ready,
        )
    LOGGER.info(
        "Wrote %d new rows (%d total) to %s",
        len(output),
        len(merged),
        output_path,
    )
    return output_path


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
