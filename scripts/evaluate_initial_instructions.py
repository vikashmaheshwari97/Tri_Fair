from __future__ import annotations

import argparse
import gc
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

try:
    from scripts._common import (
        add_result_columns,
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
except ModuleNotFoundError:
    from _common import (  # type: ignore[no-redef]
        add_result_columns,
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


DEV_COLUMN_RENAME = {
    "quality": "dev_quality",
    "cost": "dev_cost",
    "fairness": "dev_fairness",
    "fairness_ready": "dev_fairness_ready",
    "fairness_diagnostics_json": "dev_fairness_diagnostics_json",
    "group_support_json": "dev_group_support_json",
    "objective_vector": "dev_objective_vector",
}


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def parse_int_csv(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def rename_dev_columns(frame: pd.DataFrame) -> pd.DataFrame:
    safe = {
        source: target
        for source, target in DEV_COLUMN_RENAME.items()
        if source in frame.columns and target not in frame.columns
    }
    return frame.rename(columns=safe)


def load_run_metrics(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing {source}. Run analysis.analysis_pipeline first."
        )
    return pd.read_csv(source)


def source_run_dir(
    run_metrics: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    seed: int,
    budget: int,
    preferred_optimizer: str,
) -> Path:
    data = run_metrics[
        (run_metrics["dataset"] == dataset)
        & (run_metrics["model"] == model)
        & (pd.to_numeric(run_metrics["seed"], errors="coerce").astype(int) == int(seed))
        & (
            pd.to_numeric(run_metrics["budget_checkpoint"], errors="coerce").astype(int)
            == int(budget)
        )
    ].copy()

    if data.empty:
        raise RuntimeError(
            f"No run_metrics row for dataset={dataset}, model={model}, seed={seed}, budget={budget}"
        )

    preferred = data[data["optimizer"] == preferred_optimizer].copy()
    if not preferred.empty:
        data = preferred

    if "run_dir" not in data.columns:
        raise RuntimeError("run_metrics.csv does not contain run_dir")

    run_dir = Path(str(data.sort_values("run_dir").iloc[0]["run_dir"]))
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir from run_metrics does not exist: {run_dir}")
    return run_dir


def select_initial_candidates(
    run_dir: Path,
    *,
    max_initial_prompts: int,
) -> tuple[pd.DataFrame, int]:
    step_path = run_dir / "step_results.parquet"
    if not step_path.is_file():
        raise FileNotFoundError(step_path)

    frame = pd.read_parquet(step_path)
    if frame.empty:
        raise RuntimeError(f"{step_path} is empty")
    if "step" not in frame or "prompt" not in frame:
        raise RuntimeError(f"{step_path} must contain step and prompt columns")

    frame = frame.copy()
    frame["step"] = pd.to_numeric(frame["step"], errors="raise").astype(int)
    initial_step = int(frame["step"].min())

    selected = frame[frame["step"] == initial_step].copy()
    if "prompt_id" not in selected:
        selected["prompt_id"] = selected["prompt"].astype(str).map(prompt_id)

    selected = stable_latest_per_prompt(selected)

    sort_columns = [col for col in ["prompt_order", "prompt_id"] if col in selected.columns]
    if sort_columns:
        selected = selected.sort_values(sort_columns, kind="stable")

    if max_initial_prompts > 0 and len(selected) > max_initial_prompts:
        selected = selected.head(max_initial_prompts).copy()

    selected["chosen_step"] = initial_step
    selected["selection_policy"] = "initial_instructions"
    return selected.reset_index(drop=True), initial_step


def evaluate_one_dataset_seed(
    *,
    llm,
    run_metrics: pd.DataFrame,
    dataset: str,
    model: str,
    seed: int,
    budget: int,
    preferred_optimizer: str,
    max_initial_prompts: int,
    manifest_dir: str,
    regenerate_manifest: bool,
    manifest_lock_timeout: float,
) -> pd.DataFrame:
    dataset_config = ALL_DATASETS[dataset]
    model_config = ALL_MODELS[model]

    run_dir = source_run_dir(
        run_metrics,
        dataset=dataset,
        model=model,
        seed=seed,
        budget=budget,
        preferred_optimizer=preferred_optimizer,
    )

    selected, initial_step = select_initial_candidates(
        run_dir,
        max_initial_prompts=max_initial_prompts,
    )

    lock = manifest_lock_path(manifest_dir, dataset, seed)
    with directory_lock(lock, timeout_seconds=manifest_lock_timeout):
        test_task = create_test_task(
            dataset_config=dataset_config,
            eval_strategy="full",
            n_subsamples=0,
            test_size=SETUP.test_size,
            seed=seed,
            manifest_dir=manifest_dir,
            regenerate_manifest=regenerate_manifest,
        )

    prompts = reconstruct_prompts(selected)
    predictor = MarkerBasedPredictor(llm, test_task.classes)

    LOGGER.info(
        "Evaluating %d initial prompts: dataset=%s seed=%s source_run=%s",
        len(prompts),
        dataset,
        seed,
        run_dir,
    )

    result = test_task.evaluate(
        prompts=prompts,
        predictor=predictor,
        eval_strategy="full",
    )

    output = rename_dev_columns(selected.reset_index(drop=True))
    output = add_result_columns(
        output,
        result,
        prefix="test",
        model_config=model_config,
    )

    output["optimizer"] = "Initial Instructions"
    output["source_optimizer"] = preferred_optimizer
    output["dataset"] = dataset
    output["model"] = model
    output["seed"] = int(seed)
    output["evaluation_seed"] = int(seed)
    output["budget_checkpoint"] = int(budget)
    output["actual_budget_tokens"] = 0
    output["run_key"] = f"initial/{model}/{dataset}/seed{seed}"
    output["run_dir"] = str(run_dir)
    output["source_run_dir"] = str(run_dir)
    output["initial_step"] = int(initial_step)
    output["evaluation_timestamp"] = utc_now_iso()
    output["manifest_path"] = getattr(
        test_task,
        "manifest_path",
        str(resolve_manifest_path(manifest_dir, dataset, seed)),
    )
    output["manifest_sha256"] = sha256_file(output["manifest_path"].iloc[0])

    not_ready = int((~output["test_fairness_ready"].astype(bool)).sum())
    if not_ready:
        LOGGER.warning(
            "%d initial prompts did not meet fairness support for %s seed %s",
            not_ready,
            dataset,
            seed,
        )

    return output.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="bbq,bias_in_bios,civil_comments")
    parser.add_argument("--model", default="qwen-3-30b")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--budget", type=int, default=1_000_000)
    parser.add_argument("--run-metrics", default="analysis/output/run_metrics.csv")
    parser.add_argument(
        "--preferred-optimizer",
        default="Tri-Fair",
        help="Use this optimizer run to recover the exact initial prompts used in that run.",
    )
    parser.add_argument("--max-initial-prompts", type=int, default=6)
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--manifest-dir", default="data/splits")
    parser.add_argument("--regenerate-manifest", action="store_true")
    parser.add_argument("--manifest-lock-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--output-file",
        default="analysis/output/initial_instructions_evaluations.parquet",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    datasets = parse_csv(args.datasets)
    seeds = parse_int_csv(args.seeds)
    run_metrics = load_run_metrics(args.run_metrics)

    model_config = ALL_MODELS[args.model]
    rows: list[pd.DataFrame] = []

    for seed in seeds:
        LOGGER.info("Loading model %s for seed %s", args.model, seed)
        seed_everything(seed)
        llm = create_llm(model_config=model_config, seed=seed)
        set_generation_limit(llm, args.max_output_tokens)

        try:
            for dataset in datasets:
                rows.append(
                    evaluate_one_dataset_seed(
                        llm=llm,
                        run_metrics=run_metrics,
                        dataset=dataset,
                        model=args.model,
                        seed=seed,
                        budget=args.budget,
                        preferred_optimizer=args.preferred_optimizer,
                        max_initial_prompts=args.max_initial_prompts,
                        manifest_dir=args.manifest_dir,
                        regenerate_manifest=args.regenerate_manifest,
                        manifest_lock_timeout=args.manifest_lock_timeout,
                    )
                )
        finally:
            del llm
            gc.collect()

    if not rows:
        raise RuntimeError("No initial-instruction evaluations were produced")

    output = pd.concat(rows, ignore_index=True)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    csv_path = output_path.with_suffix(".csv")
    output.to_csv(csv_path, index=False)

    print(f"\nWrote initial-instruction evaluations to:\n{output_path}")
    print(f"CSV copy:\n{csv_path}")
    print("\nCounts:")
    print(
        output.groupby(["dataset", "model", "seed", "optimizer"])
        .size()
        .rename("n_prompts")
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()