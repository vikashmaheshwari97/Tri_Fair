"""Evaluate development-selected checkpoint fronts on the fixed v3 holdout set.

The optimizer is never given holdout feedback.  After a run is complete, this
script maps token checkpoints to logged steps, selects candidates using only
development history, evaluates each unique prompt once on holdout data, and
writes a checkpoint-aware parquet table for holdout nR2/HV/gap analysis.\nNominal checkpoints may use the nearest real logged state within a frozen\nsymmetric token tolerance; no interpolation or synthetic state is introduced.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

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
from scripts.evaluate_prompts import _rename_development_columns, load_prompt_candidates
from src.config.dataset_configs import ALL_DATASETS
from src.config.model_configs import ALL_MODELS
from src.config.setup_config import SETUP
from src.config.v3_profiles import apply_v3_dataset_profile
from src.helpers.llm_creation import create_llm
from src.helpers.task_creation import create_test_task
from src.utils import seed_everything

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--dataset", choices=sorted(ALL_DATASETS), default=None)
    parser.add_argument("--model", choices=sorted(ALL_MODELS), default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument(
        "--checkpoints",
        default="2000000,3000000,4000000,5000000",
    )
    parser.add_argument(
        "--selection",
        choices=("current", "current_incumbents", "incumbents", "current_and_incumbents"),
        default="current_incumbents",
    )
    parser.add_argument(
        "--checkpoint-policy",
        choices=("nearest", "prior"),
        default="nearest",
        help=(
            "Map each nominal token checkpoint to either the nearest completed logged "
            "step (symmetric tolerance) or the latest completed step at/below the target."
        ),
    )
    parser.add_argument(
        "--maximum-checkpoint-relative-error",
        type=float,
        default=0.12,
        help=(
            "Maximum |actual-target|/target accepted by the nearest policy. "
            "The main v3 protocol uses 0.12."
        ),
    )
    parser.add_argument(
        "--minimum-checkpoint-utilization",
        type=float,
        default=0.90,
        help="Minimum actual/target ratio used only by the legacy prior policy.",
    )
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help=(
            "Rebuild checkpoint membership instead of appending old checkpoint rows. "
            "Cached prompt-level holdout evaluations are still reused."
        ),
    )
    parser.add_argument(
        "--backup-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Back up an existing output before --replace-output overwrites it.",
    )
    parser.add_argument("--manifest-dir", default="data/splits_v3")
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--output-file", default="eval_checkpoints.parquet")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _positive_csv(raw: str) -> list[int]:
    values = sorted(
        {
            int(piece.strip().replace("_", ""))
            for piece in str(raw).split(",")
            if piece.strip()
        }
    )
    if not values or any(value <= 0 for value in values):
        raise ValueError("--checkpoints must contain positive integers")
    return values


def _run_metadata(log_path: Path) -> dict[str, object]:
    path = log_path / "args.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _resolve_identity(args: argparse.Namespace, log_path: Path) -> tuple[str, str, int, str]:
    metadata = _run_metadata(log_path)
    dataset = str(args.dataset or metadata.get("dataset") or "")
    model = str(args.model or metadata.get("model") or "")
    seed_raw = args.random_seed if args.random_seed is not None else metadata.get("random_seed")
    optimizer = str(metadata.get("optimizer") or "unknown")
    if dataset not in ALL_DATASETS:
        raise ValueError(f"Could not resolve dataset from {log_path}/args.json")
    if model not in ALL_MODELS:
        raise ValueError(f"Could not resolve model from {log_path}/args.json")
    if seed_raw is None:
        raise ValueError(f"Could not resolve random seed from {log_path}/args.json")
    return dataset, model, int(seed_raw), optimizer


def _step_token_table(frame: pd.DataFrame) -> pd.Series:
    if "total_tokens_downstream" in frame:
        tokens = pd.to_numeric(frame["total_tokens_downstream"], errors="coerce")
    else:
        input_source = (
            frame["input_tokens_downstream"]
            if "input_tokens_downstream" in frame
            else pd.Series(0, index=frame.index, dtype=float)
        )
        output_source = (
            frame["output_tokens_downstream"]
            if "output_tokens_downstream" in frame
            else pd.Series(0, index=frame.index, dtype=float)
        )
        input_tokens = pd.to_numeric(input_source, errors="coerce").fillna(0)
        output_tokens = pd.to_numeric(output_source, errors="coerce").fillna(0)
        tokens = input_tokens + output_tokens
    work = frame.assign(_tokens=tokens)
    return work.groupby("step", sort=True)["_tokens"].max().dropna().astype(int)


def _checkpoint_steps(
    frame: pd.DataFrame,
    checkpoints: list[int],
    *,
    policy: str,
    maximum_relative_error: float,
    minimum_utilization: float,
) -> list[tuple[int, int, int, int, float, str]]:
    """Map nominal token checkpoints to real completed optimization steps.

    ``nearest`` is the v3 publication protocol.  It chooses the completed logged
    step with the smallest absolute token difference, prefers an under-target
    step when two candidates are equally distant, and requires a symmetric
    relative error no larger than ``maximum_relative_error``.

    ``prior`` preserves the former one-sided rule for audit/reproduction.
    """

    if not 0.0 <= maximum_relative_error <= 1.0:
        raise ValueError(
            "--maximum-checkpoint-relative-error must lie in [0, 1]"
        )
    if not 0.0 < minimum_utilization <= 1.0:
        raise ValueError("--minimum-checkpoint-utilization must lie in (0, 1]")

    per_step = _step_token_table(frame)
    if per_step.empty:
        raise RuntimeError("step_results.parquet contains no usable token counts")

    candidates = pd.DataFrame(
        {
            "step": per_step.index.to_numpy(dtype=int),
            "actual_tokens": per_step.to_numpy(dtype=int),
        }
    )

    rows: list[tuple[int, int, int, int, float, str]] = []
    for checkpoint in checkpoints:
        if policy == "nearest":
            ranked = candidates.copy()
            ranked["signed_error"] = ranked["actual_tokens"] - int(checkpoint)
            ranked["absolute_error"] = ranked["signed_error"].abs()
            ranked["relative_error"] = (
                ranked["absolute_error"] / float(checkpoint)
            )
            # Prefer the smaller absolute error.  On an exact tie, prefer the
            # under-target state, then the later logged step.
            ranked["overshoot"] = (ranked["signed_error"] > 0).astype(int)
            ranked = ranked.sort_values(
                ["absolute_error", "overshoot", "step"],
                ascending=[True, True, False],
                kind="stable",
            )
            chosen = ranked.iloc[0]
            chosen_step = int(chosen["step"])
            actual = int(chosen["actual_tokens"])
            signed_error = int(chosen["signed_error"])
            relative_error = float(chosen["relative_error"])
            if relative_error > maximum_relative_error:
                LOGGER.warning(
                    "Skipping nominal checkpoint %d: nearest completed step %d "
                    "has %d tokens (relative error %.4f > %.4f)",
                    checkpoint,
                    chosen_step,
                    actual,
                    relative_error,
                    maximum_relative_error,
                )
                continue
            resolved_policy = "nearest_logged_step"
        elif policy == "prior":
            eligible = per_step[per_step <= checkpoint]
            if eligible.empty:
                LOGGER.warning(
                    "Skipping checkpoint %d: no completed step exists at/below target",
                    checkpoint,
                )
                continue
            chosen_step = int(eligible.index[-1])
            actual = int(eligible.iloc[-1])
            if actual < minimum_utilization * checkpoint:
                LOGGER.warning(
                    "Skipping checkpoint %d: latest completed step has only %d tokens",
                    checkpoint,
                    actual,
                )
                continue
            signed_error = int(actual - checkpoint)
            relative_error = float(abs(signed_error) / checkpoint)
            resolved_policy = "latest_prior_step"
        else:  # guarded by argparse; retained for direct function callers
            raise ValueError(f"Unsupported checkpoint policy: {policy}")

        LOGGER.info(
            "Nominal checkpoint %d -> step %d at %d tokens "
            "(signed error %+d, relative error %.4f, policy=%s)",
            checkpoint,
            chosen_step,
            actual,
            signed_error,
            relative_error,
            resolved_policy,
        )
        rows.append(
            (
                int(checkpoint),
                chosen_step,
                actual,
                signed_error,
                relative_error,
                resolved_policy,
            )
        )

    if not rows:
        raise RuntimeError("No logged step satisfies the requested checkpoint rules")
    return rows


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
        }
    ]


def run(args: argparse.Namespace) -> Path:
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")
    log_path = Path(args.log_path).expanduser().resolve()
    step_path = log_path / "step_results.parquet"
    if not step_path.is_file():
        raise FileNotFoundError(step_path)

    dataset, model, seed, optimizer = _resolve_identity(args, log_path)
    step_frame = pd.read_parquet(step_path)
    checkpoint_rows = _checkpoint_steps(
        step_frame,
        _positive_csv(args.checkpoints),
        policy=str(args.checkpoint_policy),
        maximum_relative_error=float(args.maximum_checkpoint_relative_error),
        minimum_utilization=float(args.minimum_checkpoint_utilization),
    )

    membership_frames: list[pd.DataFrame] = []
    for (
        checkpoint,
        chosen_step,
        actual_tokens,
        signed_error,
        relative_error,
        checkpoint_policy,
    ) in checkpoint_rows:
        loader_selection = (
            "current" if args.selection == "current_incumbents" else args.selection
        )
        selected, resolved_step = load_prompt_candidates(
            log_path,
            step=chosen_step,
            selection=loader_selection,
        )
        if args.selection == "current_incumbents":
            if "is_incumbent" not in selected:
                raise ValueError(
                    "step_results.parquet lacks is_incumbent, so the current archive "
                    "cannot be reconstructed"
                )
            selected = selected[
                selected["is_incumbent"].fillna(False).astype(bool)
            ].reset_index(drop=True)
            if selected.empty:
                raise RuntimeError(
                    f"No current incumbents were logged at step {resolved_step}"
                )
            selected["selection_policy"] = "current_incumbents"
        selected = _rename_development_columns(selected)
        selected["budget_checkpoint"] = int(checkpoint)
        selected["actual_budget_tokens"] = int(actual_tokens)
        selected["checkpoint_signed_error"] = int(signed_error)
        selected["checkpoint_relative_error"] = float(relative_error)
        selected["checkpoint_policy"] = str(checkpoint_policy)
        selected["chosen_step"] = int(resolved_step)
        selected["model"] = model
        selected["dataset"] = dataset
        selected["optimizer"] = optimizer
        selected["seed"] = seed
        metadata = _run_metadata(log_path)
        selected["configured_budget"] = int(
            metadata.get("budget_per_run", checkpoint)
        )
        if "prompt_id" not in selected:
            selected["prompt_id"] = selected["prompt"].astype(str).map(prompt_id)
        membership_frames.append(selected)

    membership = pd.concat(membership_frames, ignore_index=True, sort=False)
    unique_candidates = stable_latest_per_prompt(membership)
    output_path = log_path / args.output_file

    cached = pd.DataFrame()
    if output_path.is_file() and not args.force:
        cached = pd.read_parquet(output_path)
    cached_ids = set(cached.get("prompt_id", pd.Series(dtype=str)).astype(str))
    pending = unique_candidates[
        ~unique_candidates["prompt_id"].astype(str).isin(cached_ids)
    ].reset_index(drop=True)

    evaluated = pd.DataFrame()
    if not pending.empty:
        seed_everything(seed)
        dataset_config = apply_v3_dataset_profile(ALL_DATASETS[dataset])
        model_config = ALL_MODELS[model]
        llm = create_llm(model_config=model_config, seed=seed)
        set_generation_limit(llm, args.max_output_tokens)
        test_task = create_test_task(
            dataset_config=dataset_config,
            eval_strategy="full",
            n_subsamples=0,
            test_size=SETUP.test_size,
            seed=seed,
            manifest_dir=args.manifest_dir,
            regenerate_manifest=False,
        )
        prompts = reconstruct_prompts(pending)
        predictor = MarkerBasedPredictor(llm, test_task.classes)
        result = test_task.evaluate(
            prompts=prompts,
            predictor=predictor,
            eval_strategy="full",
        )

        evaluated = pending[["prompt_id"]].reset_index(drop=True).copy()
        quality = np.asarray(result.agg_scores, dtype=float)
        input_tokens = np.asarray(result.agg_input_tokens, dtype=float)
        output_tokens = np.asarray(result.agg_output_tokens, dtype=float)
        evaluated["test_quality"] = quality
        evaluated["test_cost"] = (
            float(model_config.input_costs) * input_tokens
            + float(model_config.output_costs) * output_tokens
        )
        evaluated["test_input_tokens"] = input_tokens
        evaluated["test_output_tokens"] = output_tokens
        evaluated["test_fairness"] = np.asarray(result.fairness_loss, dtype=float)
        evaluated["test_fairness_ready"] = np.asarray(
            result.fairness_ready, dtype=bool
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
        evaluated["manifest_path"] = str(getattr(test_task, "manifest_path", ""))
        evaluated["manifest_sha256"] = sha256_file(
            evaluated["manifest_path"].iloc[0]
        )

    lookup_parts: list[pd.DataFrame] = []
    if not cached.empty:
        columns = _test_columns(cached)
        lookup_parts.append(stable_latest_per_prompt(cached[columns]))
    if not evaluated.empty:
        lookup_parts.append(evaluated)
    if not lookup_parts:
        raise RuntimeError("No cached or newly evaluated holdout records are available")
    lookup = stable_latest_per_prompt(
        pd.concat(lookup_parts, ignore_index=True, sort=False)
    )

    test_only = [column for column in lookup.columns if column != "prompt_id"]
    output = membership.merge(
        lookup[["prompt_id", *test_only]],
        on="prompt_id",
        how="left",
        validate="many_to_one",
    )
    missing_test = output["test_quality"].isna().sum()
    if missing_test:
        raise RuntimeError(f"{missing_test} checkpoint rows lack holdout evaluation")

    if not args.replace_output and not args.force and not cached.empty:
        output = pd.concat([cached, output], ignore_index=True, sort=False)
    dedup = ["prompt_id", "chosen_step", "budget_checkpoint"]
    output = (
        output.sort_values(dedup, kind="stable")
        .drop_duplicates(dedup, keep="last")
        .reset_index(drop=True)
    )
    if (
        args.replace_output
        and args.backup_existing
        and output_path.is_file()
    ):
        backup_path = output_path.with_name(
            f"{output_path.stem}_prior_policy_backup{output_path.suffix}"
        )
        if not backup_path.exists():
            shutil.copy2(output_path, backup_path)
            LOGGER.info("Backed up previous checkpoint table to %s", backup_path)

    atomic_write_parquet(output, output_path)
    LOGGER.info(
        "Wrote %d checkpoint rows covering %d unique prompts to %s",
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
