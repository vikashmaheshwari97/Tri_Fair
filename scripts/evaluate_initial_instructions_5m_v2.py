"""Evaluate the exact 12-prompt initial populations used by the 5M v2 study.

For each dataset/seed pair this script recovers the initial population from the
completed Tri-Fair and NSGAII-PO-Fair 5M runs, verifies that both optimizers used
the same prompt set, and evaluates those prompts on the full immutable v2
development and test manifests.

The baseline is written to:

    results/tri_fair_v2_2/qwen-3-30b/<dataset>/init/seed<seed>/initial/

It is a budget-zero baseline linked to the nominal 5M comparison through the
``comparison_budget`` and ``source_budget`` columns. Existing evaluations are
validated and reused idempotently; this script never overwrites them.
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Iterable

import pandas as pd
from promptolution.predictors import MarkerBasedPredictor

from src.config.dataset_configs import ALL_DATASETS
from src.config.model_configs import ALL_MODELS
from src.config.setup_config import SETUP
from src.helpers.llm_creation import create_llm
from src.helpers.task_creation import create_dev_tasks, create_test_task
from src.utils import seed_everything

try:
    from scripts._common import (
        add_result_columns,
        atomic_write_json,
        atomic_write_parquet,
        configure_logging,
        directory_lock,
        manifest_lock_path,
        prompt_id,
        reconstruct_prompts,
        resolve_manifest_path,
        set_generation_limit,
        sha256_file,
        stable_latest_per_prompt,
        utc_now_iso,
    )
except ModuleNotFoundError:  # pragma: no cover
    from _common import (  # type: ignore[no-redef]
        add_result_columns,
        atomic_write_json,
        atomic_write_parquet,
        configure_logging,
        directory_lock,
        manifest_lock_path,
        prompt_id,
        reconstruct_prompts,
        resolve_manifest_path,
        set_generation_limit,
        sha256_file,
        stable_latest_per_prompt,
        utc_now_iso,
    )


LOGGER = logging.getLogger(__name__)

DEFAULT_RESULTS_ROOT = "results/tri_fair_v2_2"
DEFAULT_MANIFEST_DIR = "data/splits_v2_2"
DEFAULT_MODEL = "qwen-3-30b"
DEFAULT_SOURCE_BUDGET = 5_000_000
DEFAULT_SOURCE_OPTIMIZERS = ("Tri-Fair", "NSGAII-PO-Fair")
DEFAULT_EXPECTED_INITIAL_PROMPTS = 12

PROMPT_COLUMNS = (
    "prompt",
    "prompt_id",
    "instruction",
    "few_shots_json",
    "downstream_template",
    "prompt_order",
)


def parse_csv(raw: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = [str(part).strip() for part in raw if str(part).strip()]
    if not values:
        raise ValueError("At least one value is required")
    return tuple(values)


def require_file(path: Path, *, nonempty: bool = True) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if nonempty and path.stat().st_size == 0:
        raise RuntimeError(f"Required file is empty: {path}")


def read_json(path: Path) -> dict[str, object]:
    require_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def resolve_logging_dir(output_dir: Path) -> Path:
    pointer = output_dir / "logging_dir.txt"
    require_file(pointer)
    raw_text = pointer.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise RuntimeError(f"Empty logging-directory pointer: {pointer}")

    raw = Path(raw_text).expanduser()
    candidates = (
        [raw.resolve()]
        if raw.is_absolute()
        else [
            (Path.cwd() / raw).resolve(),
            (output_dir / raw).resolve(),
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve logging directory from {pointer}; tried {candidates}"
    )


def source_output_dir(
    results_root: Path,
    *,
    model: str,
    dataset: str,
    optimizer: str,
    seed: int,
) -> Path:
    return results_root / model / dataset / optimizer / f"seed{seed}"


def validate_source_stage(
    output_dir: Path,
    *,
    model: str,
    dataset: str,
    optimizer: str,
    seed: int,
    source_budget: int,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    status_dir = output_dir / "status"
    stage_done = status_dir / f"stage_{source_budget}.done.json"
    eval_done = status_dir / f"eval_{source_budget}.done.json"

    stage = read_json(stage_done)
    evaluation = read_json(eval_done)

    expected_stage = {
        "model": model,
        "dataset": dataset,
        "optimizer": optimizer,
        "seed": seed,
        "budget": source_budget,
    }
    expected_eval = {
        "model": model,
        "dataset": dataset,
        "optimizer": optimizer,
        "seed": seed,
        "requested_budget": source_budget,
    }

    for key, expected in expected_stage.items():
        observed = stage.get(key)
        if observed is not None and str(observed) != str(expected):
            raise RuntimeError(
                f"{stage_done}: {key}={observed!r}, expected {expected!r}"
            )

    for key, expected in expected_eval.items():
        observed = evaluation.get(key)
        if observed is not None and str(observed) != str(expected):
            raise RuntimeError(
                f"{eval_done}: {key}={observed!r}, expected {expected!r}"
            )

    for marker in (
        status_dir / f"stage_{source_budget}.failed.json",
        status_dir / f"stage_{source_budget}.running.json",
        status_dir / f"eval_{source_budget}.failed.json",
        status_dir / f"eval_{source_budget}.running.json",
    ):
        if marker.exists():
            raise RuntimeError(f"Incomplete/failure marker exists: {marker}")

    logging_dir = resolve_logging_dir(output_dir)
    require_file(logging_dir / "step_results.parquet")
    return logging_dir, stage, evaluation


def initial_population(run_dir: Path) -> tuple[pd.DataFrame, int]:
    path = run_dir / "step_results.parquet"
    require_file(path)
    frame = pd.read_parquet(path)

    required = {"step", "prompt"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} lacks required columns: {sorted(missing)}")
    if frame.empty:
        raise RuntimeError(f"{path} is empty")

    work = frame.copy()
    work["step"] = pd.to_numeric(work["step"], errors="raise").astype(int)
    initial_step = int(work["step"].min())
    selected = work[work["step"] == initial_step].copy()

    if "prompt_id" not in selected:
        selected["prompt_id"] = selected["prompt"].astype(str).map(prompt_id)

    selected = stable_latest_per_prompt(selected)
    sort_columns = [
        column
        for column in ("prompt_order", "prompt_id")
        if column in selected
    ]
    if sort_columns:
        selected = selected.sort_values(sort_columns, kind="stable")

    keep = [column for column in PROMPT_COLUMNS if column in selected]
    selected = selected[keep].copy()

    if "prompt_order" not in selected:
        selected["prompt_order"] = range(len(selected))

    return selected.reset_index(drop=True), initial_step


def verify_shared_initial_population(
    populations: dict[str, pd.DataFrame],
    *,
    expected_count: int,
) -> None:
    if not populations:
        raise RuntimeError("No source optimizer populations were loaded")

    reference_optimizer = next(iter(populations))
    reference = populations[reference_optimizer]
    reference_ids = set(reference["prompt_id"].astype(str))

    if len(reference) != expected_count or len(reference_ids) != expected_count:
        raise RuntimeError(
            f"{reference_optimizer} initial population has {len(reference)} rows and "
            f"{len(reference_ids)} unique prompts; expected exactly {expected_count}"
        )

    for optimizer, frame in populations.items():
        ids = set(frame["prompt_id"].astype(str))
        if len(frame) != expected_count or len(ids) != expected_count:
            raise RuntimeError(
                f"{optimizer} initial population has {len(frame)} rows and "
                f"{len(ids)} unique prompts; expected exactly {expected_count}"
            )
        if ids != reference_ids:
            only_reference = sorted(reference_ids - ids)[:3]
            only_current = sorted(ids - reference_ids)[:3]
            raise RuntimeError(
                "The two optimizers did not use the same initial population: "
                f"reference={reference_optimizer}, optimizer={optimizer}, "
                f"reference_only={only_reference}, optimizer_only={only_current}"
            )


def output_dir(
    results_root: Path,
    *,
    model: str,
    dataset: str,
    seed: int,
) -> Path:
    return results_root / model / dataset / "init" / f"seed{seed}" / "initial"


def validate_existing_output(
    eval_path: Path,
    *,
    model: str,
    dataset: str,
    seed: int,
    expected_count: int,
    comparison_budget: int,
    manifest_sha256: str | None,
    expected_prompt_ids: set[str],
) -> pd.DataFrame:
    frame = pd.read_parquet(eval_path)
    required = {
        "model",
        "dataset",
        "seed",
        "optimizer",
        "prompt_id",
        "comparison_budget",
        "test_quality",
        "test_cost",
        "test_fairness",
        "test_fairness_ready",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Existing baseline {eval_path} is incompatible; missing {sorted(missing)}"
        )

    observed_prompt_ids = set(frame["prompt_id"].astype(str))
    if len(frame) != expected_count or len(observed_prompt_ids) != expected_count:
        raise RuntimeError(
            f"Existing baseline {eval_path} has {len(frame)} rows and "
            f"{len(observed_prompt_ids)} unique prompts; expected {expected_count}"
        )
    if observed_prompt_ids != expected_prompt_ids:
        raise RuntimeError(
            f"Existing baseline {eval_path} does not contain the exact initial "
            "population recovered from the completed 5M runs"
        )

    checks = {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "optimizer": "Initial Instructions",
        "comparison_budget": comparison_budget,
    }
    for column, expected in checks.items():
        observed = set(frame[column].astype(str))
        if observed != {str(expected)}:
            raise RuntimeError(
                f"Existing baseline {eval_path}: {column}={sorted(observed)}, "
                f"expected only {expected!r}"
            )

    if manifest_sha256 and "manifest_sha256" in frame:
        observed_hashes = set(frame["manifest_sha256"].dropna().astype(str))
        if observed_hashes and observed_hashes != {manifest_sha256}:
            raise RuntimeError(
                f"Existing baseline {eval_path} uses manifest hashes "
                f"{sorted(observed_hashes)}, expected {manifest_sha256}"
            )

    return frame.reset_index(drop=True)


def evaluate(
    args: argparse.Namespace,
) -> Path:
    if args.dataset not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset: {args.dataset!r}")
    if args.model not in ALL_MODELS:
        raise ValueError(f"Unknown model: {args.model!r}")
    if args.source_budget <= 0:
        raise ValueError("--source-budget must be positive")
    if args.expected_initial_prompts <= 0:
        raise ValueError("--expected-initial-prompts must be positive")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")

    results_root = Path(args.results_root).expanduser().resolve()
    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    manifest_path = resolve_manifest_path(
        manifest_dir,
        args.dataset,
        args.random_seed,
    )
    require_file(manifest_path)
    expected_manifest_hash = sha256_file(manifest_path)

    source_optimizers = parse_csv(args.source_optimizers)
    populations: dict[str, pd.DataFrame] = {}
    source_dirs: dict[str, Path] = {}
    source_steps: dict[str, int] = {}
    source_status: dict[str, dict[str, object]] = {}

    for optimizer in source_optimizers:
        source_output = source_output_dir(
            results_root,
            model=args.model,
            dataset=args.dataset,
            optimizer=optimizer,
            seed=args.random_seed,
        )
        logging_dir, stage_status, eval_status = validate_source_stage(
            source_output,
            model=args.model,
            dataset=args.dataset,
            optimizer=optimizer,
            seed=args.random_seed,
            source_budget=args.source_budget,
        )
        population, initial_step = initial_population(logging_dir)
        populations[optimizer] = population
        source_dirs[optimizer] = logging_dir
        source_steps[optimizer] = initial_step
        source_status[optimizer] = {
            "stage": stage_status,
            "evaluation": eval_status,
        }

    verify_shared_initial_population(
        populations,
        expected_count=args.expected_initial_prompts,
    )

    preferred = args.preferred_optimizer
    if preferred not in populations:
        raise ValueError(
            f"--preferred-optimizer={preferred!r} is not present in "
            f"--source-optimizers={source_optimizers}"
        )

    selected = populations[preferred].copy()
    baseline_dir = output_dir(
        results_root,
        model=args.model,
        dataset=args.dataset,
        seed=args.random_seed,
    )
    baseline_dir.mkdir(parents=True, exist_ok=True)
    eval_path = baseline_dir / "eval.parquet"
    status_dir = baseline_dir / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    running_marker = status_dir / f"initial_{args.source_budget}.running.json"
    done_marker = status_dir / f"initial_{args.source_budget}.done.json"
    failed_marker = status_dir / f"initial_{args.source_budget}.failed.json"

    if eval_path.is_file():
        validate_existing_output(
            eval_path,
            model=args.model,
            dataset=args.dataset,
            seed=args.random_seed,
            expected_count=args.expected_initial_prompts,
            comparison_budget=args.source_budget,
            manifest_sha256=expected_manifest_hash,
            expected_prompt_ids=set(selected["prompt_id"].astype(str)),
        )
        if not done_marker.is_file():
            atomic_write_json(
                done_marker,
                {
                    "state": "complete",
                    "reused_existing": True,
                    "completed_at": utc_now_iso(),
                    "model": args.model,
                    "dataset": args.dataset,
                    "seed": args.random_seed,
                    "source_budget": args.source_budget,
                    "n_prompts": args.expected_initial_prompts,
                    "eval_path": str(eval_path),
                },
            )
        LOGGER.info("Validated existing initial baseline; skipping: %s", eval_path)
        return eval_path

    started_at = utc_now_iso()
    running_payload = {
        "state": "running",
        "started_at": started_at,
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.random_seed,
        "source_budget": args.source_budget,
        "source_optimizers": list(source_optimizers),
        "source_run_dirs": {key: str(value) for key, value in source_dirs.items()},
        "source_initial_steps": source_steps,
        "expected_initial_prompts": args.expected_initial_prompts,
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_manifest_hash,
    }
    atomic_write_json(running_marker, running_payload)
    done_marker.unlink(missing_ok=True)
    failed_marker.unlink(missing_ok=True)

    try:
        dataset_config = ALL_DATASETS[args.dataset]
        model_config = ALL_MODELS[args.model]

        lock = manifest_lock_path(
            manifest_dir,
            args.dataset,
            args.random_seed,
        )
        with directory_lock(lock, timeout_seconds=args.manifest_lock_timeout):
            dev_task, _ = create_dev_tasks(
                dataset_config=dataset_config,
                eval_strategy="full",
                n_subsamples=0,
                dev_size=SETUP.dev_size,
                fs_size=SETUP.fs_size,
                seed=args.random_seed,
                manifest_dir=manifest_dir,
                regenerate_manifest=False,
            )
            test_task = create_test_task(
                dataset_config=dataset_config,
                eval_strategy="full",
                n_subsamples=0,
                test_size=SETUP.test_size,
                seed=args.random_seed,
                manifest_dir=manifest_dir,
                regenerate_manifest=False,
            )

        prompts = reconstruct_prompts(selected)
        seed_everything(args.random_seed)
        llm = create_llm(model_config=model_config, seed=args.random_seed)
        set_generation_limit(llm, args.max_output_tokens)

        LOGGER.info(
            "Evaluating exact initial population: model=%s dataset=%s seed=%s "
            "n_prompts=%s source_budget=%s",
            args.model,
            args.dataset,
            args.random_seed,
            len(prompts),
            args.source_budget,
        )

        dev_predictor = MarkerBasedPredictor(llm, dev_task.classes)
        dev_result = dev_task.evaluate(
            prompts=prompts,
            predictor=dev_predictor,
            eval_strategy="full",
        )
        output = add_result_columns(
            selected.reset_index(drop=True),
            dev_result,
            prefix="dev",
            model_config=model_config,
        )

        test_predictor = MarkerBasedPredictor(llm, test_task.classes)
        test_result = test_task.evaluate(
            prompts=prompts,
            predictor=test_predictor,
            eval_strategy="full",
        )
        output = add_result_columns(
            output,
            test_result,
            prefix="test",
            model_config=model_config,
        )

        output["optimizer"] = "Initial Instructions"
        output["source_optimizer"] = preferred
        output["source_optimizers"] = ",".join(source_optimizers)
        output["dataset"] = args.dataset
        output["model"] = args.model
        output["seed"] = int(args.random_seed)
        output["evaluation_seed"] = int(args.random_seed)
        output["chosen_step"] = 0
        output["budget_checkpoint"] = 0
        output["configured_budget"] = 0
        output["budget_per_run"] = 0
        output["comparison_budget"] = int(args.source_budget)
        output["source_budget"] = int(args.source_budget)
        output["actual_budget_tokens"] = 0
        output["selection_policy"] = "exact_5m_optimizer_initial_population"
        output["run_key"] = (
            f"initial/{args.model}/{args.dataset}/seed{args.random_seed}/"
            f"source_budget{args.source_budget}"
        )
        output["run_dir"] = str(baseline_dir)
        output["source_run_dir"] = str(source_dirs[preferred])
        output["source_initial_step"] = int(source_steps[preferred])
        output["evaluation_timestamp"] = utc_now_iso()
        output["manifest_path"] = str(manifest_path)
        output["manifest_sha256"] = expected_manifest_hash

        atomic_write_parquet(output, eval_path)
        output.to_csv(baseline_dir / "eval.csv", index=False)

        atomic_write_json(
            baseline_dir / "args.json",
            {
                **vars(args),
                "optimizer": "Initial Instructions",
                "output_dir": str(baseline_dir),
                "manifest_path": str(manifest_path),
                "manifest_sha256": expected_manifest_hash,
                "source_run_dirs": {
                    key: str(value)
                    for key, value in source_dirs.items()
                },
                "source_initial_steps": source_steps,
            },
        )
        atomic_write_json(
            baseline_dir / "run_summary.json",
            {
                "status": "complete",
                "completed_at": utc_now_iso(),
                "optimizer": "Initial Instructions",
                "model": args.model,
                "dataset": args.dataset,
                "seed": args.random_seed,
                "comparison_budget": args.source_budget,
                "n_prompts": len(output),
                "dev_ready": int(
                    output["dev_fairness_ready"].fillna(False).astype(bool).sum()
                ),
                "test_ready": int(
                    output["test_fairness_ready"].fillna(False).astype(bool).sum()
                ),
                "manifest_path": str(manifest_path),
                "manifest_sha256": expected_manifest_hash,
                "eval_path": str(eval_path),
            },
        )

        running_marker.unlink(missing_ok=True)
        atomic_write_json(
            done_marker,
            {
                "state": "complete",
                "started_at": started_at,
                "completed_at": utc_now_iso(),
                "model": args.model,
                "dataset": args.dataset,
                "seed": args.random_seed,
                "source_budget": args.source_budget,
                "n_prompts": len(output),
                "eval_path": str(eval_path),
                "manifest_sha256": expected_manifest_hash,
            },
        )
        return eval_path

    except Exception as error:
        running_marker.unlink(missing_ok=True)
        atomic_write_json(
            failed_marker,
            {
                "state": "failed",
                "failed_at": utc_now_iso(),
                "model": args.model,
                "dataset": args.dataset,
                "seed": args.random_seed,
                "source_budget": args.source_budget,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random-seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(ALL_DATASETS),
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=sorted(ALL_MODELS),
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
    )
    parser.add_argument(
        "--manifest-dir",
        default=DEFAULT_MANIFEST_DIR,
    )
    parser.add_argument(
        "--source-budget",
        type=int,
        default=DEFAULT_SOURCE_BUDGET,
    )
    parser.add_argument(
        "--source-optimizers",
        default=",".join(DEFAULT_SOURCE_OPTIMIZERS),
        help="Optimizers whose step-one initial populations must match.",
    )
    parser.add_argument(
        "--preferred-optimizer",
        default="Tri-Fair",
        help="Source run used to preserve deterministic prompt ordering.",
    )
    parser.add_argument(
        "--expected-initial-prompts",
        type=int,
        default=DEFAULT_EXPECTED_INITIAL_PROMPTS,
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--manifest-lock-timeout",
        type=float,
        default=3600.0,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    path = evaluate(args)
    print(f"\nInitial baseline ready:\n{path}")


if __name__ == "__main__":
    main()
