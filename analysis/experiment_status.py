"""Audit the Tri-Fair experiment grid and generate recovery commands.

This replaces the ad-hoc MO-CAPO ``experiment_status.ipynb``.  It understands
budget checkpoints, resumable runs, fairness readiness, and checkpoint-specific
holdout evaluations.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .config import (
    DEFAULT_BUDGETS,
    DEFAULT_DATASETS,
    DEFAULT_OPTIMIZERS,
    DEFAULT_SEEDS,
    parse_csv_ints,
    parse_csv_strings,
)
from .io import (
    RunArtifacts,
    checkpoint_files,
    discover_runs,
    latest_checkpoint,
    read_eval_results,
    read_json,
    read_step_results,
    step_token_table,
    steps_for_checkpoints,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/tri_fair")
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model aliases, for example qwen-3-30b",
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--optimizers", default=",".join(DEFAULT_OPTIMIZERS))
    parser.add_argument(
        "--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS)
    )
    parser.add_argument(
        "--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS)
    )
    parser.add_argument("--target-budget", type=int, default=1_000_000)
    parser.add_argument("--output-dir", default="analysis/status")
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _duration_minutes(frame: pd.DataFrame) -> float | None:
    if frame.empty or "time" not in frame:
        return None
    values = pd.to_numeric(frame["time"], errors="coerce").dropna()
    if len(values) < 2:
        return None
    return float((values.max() - values.min()) / 60.0)


def _eval_steps(run: RunArtifacts) -> set[int]:
    if run.eval_path is None:
        return set()
    try:
        frame = read_eval_results(run)
    except Exception:
        logger.exception("Could not read %s", run.eval_path)
        return set()
    return {int(value) for value in frame["chosen_step"].dropna().astype(int).unique()}


def _fairness_ready_ratios(run: RunArtifacts) -> tuple[float | None, float | None]:
    dev_ratio: float | None = None
    test_ratio: float | None = None
    if run.step_results_path is not None:
        try:
            frame = read_step_results(run)
            final_step = int(frame["step"].max())
            final = frame[frame["step"] == final_step]
            if "fairness_ready" in final:
                dev_ratio = float(
                    final["fairness_ready"].fillna(False).astype(bool).mean()
                )
        except Exception:
            logger.exception(
                "Could not calculate development fairness readiness for %s", run.run_dir
            )
    if run.eval_path is not None:
        try:
            frame = read_eval_results(run)
            if "test_fairness_ready" in frame:
                test_ratio = float(
                    frame["test_fairness_ready"].fillna(False).astype(bool).mean()
                )
        except Exception:
            logger.exception(
                "Could not calculate test fairness readiness for %s", run.run_dir
            )
    return dev_ratio, test_ratio


def inspect_run(
    run: RunArtifacts, budgets: Sequence[int], target_budget: int
) -> dict[str, Any]:
    step_frame = pd.DataFrame()
    step_table = pd.DataFrame()
    if run.step_results_path is not None:
        step_frame = read_step_results(run)
        step_table = step_token_table(step_frame)
    actual_tokens = (
        int(step_table["total_tokens_downstream"].max()) if not step_table.empty else 0
    )
    checkpoint_steps = (
        steps_for_checkpoints(step_frame, budgets) if not step_frame.empty else {}
    )
    eval_steps = _eval_steps(run)
    evaluated_checkpoints = {
        checkpoint
        for checkpoint, step in checkpoint_steps.items()
        if int(step) in eval_steps
    }
    reached_checkpoints = {
        checkpoint for checkpoint in budgets if checkpoint <= actual_tokens
    }
    missing_eval_checkpoints = sorted(reached_checkpoints - evaluated_checkpoints)
    checkpoint_map = checkpoint_files(run)
    latest = latest_checkpoint(run)
    dev_ready, test_ready = _fairness_ready_ratios(run)

    if run.step_results_path is None:
        status = "missing_optimization"
    elif actual_tokens < target_budget:
        status = "needs_resume"
    elif missing_eval_checkpoints:
        status = "missing_checkpoint_evaluation"
    elif run.eval_path is None:
        status = "missing_evaluation"
    elif test_ready is not None and test_ready < 1.0:
        status = "evaluation_not_fairness_ready"
    else:
        status = "complete"

    return {
        **run.to_dict(),
        "actual_tokens": actual_tokens,
        "target_budget": int(target_budget),
        "budget_deficit": max(0, int(target_budget) - actual_tokens),
        "max_step": int(step_table["step"].max()) if not step_table.empty else None,
        "duration_minutes": _duration_minutes(step_frame),
        "checkpoint_files": len(checkpoint_map),
        "checkpoint_tokens": json.dumps(sorted(checkpoint_map)),
        "latest_checkpoint": str(latest) if latest else None,
        "reached_checkpoints": json.dumps(sorted(reached_checkpoints)),
        "evaluated_checkpoints": json.dumps(sorted(evaluated_checkpoints)),
        "missing_eval_checkpoints": json.dumps(missing_eval_checkpoints),
        "eval_steps": json.dumps(sorted(eval_steps)),
        "dev_fairness_ready_ratio": dev_ready,
        "test_fairness_ready_ratio": test_ready,
        "status": status,
        "checkpoint_step_map": checkpoint_steps,
    }


def _quote(value: Any) -> str:
    return shlex.quote(str(value))


def _experiment_command(
    *,
    model: str,
    dataset: str,
    optimizer: str,
    seed: int,
    budget: int,
    output_dir: str | Path,
    checkpoints: Sequence[int],
    args: dict[str, Any] | None = None,
    resume_from: str | Path | None = None,
) -> str:
    args = args or {}
    command = [
        "uv",
        "run",
        "scripts/experiment.py",
        "--experiment-name",
        str(
            args.get(
                "experiment_name", f"tri_fair_{optimizer}_{model}_{dataset}_{seed}"
            )
        ),
        "--random-seed",
        str(seed),
        "--budget-per-run",
        str(budget),
        "--output-dir",
        str(output_dir),
        "--dataset",
        dataset,
        "--model",
        model,
        "--optimizer",
        optimizer,
        "--n-init-prompts",
        str(args.get("n_init_prompts", 6)),
        "--max-output-tokens",
        str(args.get("max_output_tokens", 16)),
        "--meta-max-output-tokens",
        str(args.get("meta_max_output_tokens", 256)),
        "--budget-checkpoints",
        ",".join(str(value) for value in checkpoints if value <= budget),
        "--manifest-dir",
        str(args.get("manifest_dir", "data/splits")),
        "--max-steps",
        str(args.get("max_steps", 2000)),
    ]
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from)])
    return " ".join(_quote(part) for part in command)


def _evaluation_command(
    run: RunArtifacts,
    *,
    step: int,
    max_output_tokens: int,
) -> str:
    command = [
        "uv",
        "run",
        "scripts/evaluate_prompts.py",
        "--random-seed",
        str(run.seed),
        "--dataset",
        run.dataset,
        "--model",
        run.model,
        "--log-path",
        str(run.run_dir),
        "--incumbents",
        "True",
        "--step",
        str(step),
        "--max-output-tokens",
        str(max_output_tokens),
    ]
    return " ".join(_quote(part) for part in command)


def _choose_primary_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        records,
        key=lambda row: (
            int(row.get("actual_tokens", 0)),
            int(row.get("eval_path") is not None),
            str(row.get("run_dir", "")),
        ),
    )


def build_expected_status(
    discovered: list[RunArtifacts],
    *,
    models: Sequence[str],
    datasets: Sequence[str],
    optimizers: Sequence[str],
    seeds: Sequence[int],
    budgets: Sequence[int],
    target_budget: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str, int], RunArtifacts]]:
    by_grid: dict[tuple[str, str, str, int], list[RunArtifacts]] = {}
    for run in discovered:
        key = (run.model, run.dataset, run.optimizer, run.seed)
        by_grid.setdefault(key, []).append(run)

    rows: list[dict[str, Any]] = []
    chosen_runs: dict[tuple[str, str, str, int], RunArtifacts] = {}
    for key in product(models, datasets, optimizers, seeds):
        candidates = by_grid.get(key, [])
        if not candidates:
            model, dataset, optimizer, seed = key
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "optimizer": optimizer,
                    "seed": seed,
                    "run_dir": None,
                    "actual_tokens": 0,
                    "target_budget": target_budget,
                    "budget_deficit": target_budget,
                    "duplicate_run_count": 0,
                    "status": "missing_optimization",
                }
            )
            continue
        inspected = [inspect_run(run, budgets, target_budget) for run in candidates]
        primary = _choose_primary_run(inspected)
        selected_run = next(
            run for run in candidates if str(run.run_dir) == primary["run_dir"]
        )
        chosen_runs[key] = selected_run
        primary["duplicate_run_count"] = len(candidates)
        rows.append(primary)
    return pd.DataFrame(rows), chosen_runs


def generate_commands(
    status: pd.DataFrame,
    chosen_runs: dict[tuple[str, str, str, int], RunArtifacts],
    *,
    results_root: Path,
    budgets: Sequence[int],
    target_budget: int,
    max_output_tokens: int,
) -> tuple[list[str], list[str]]:
    optimization_commands: list[str] = []
    evaluation_commands: list[str] = []

    for _, row in status.iterrows():
        key = (
            str(row["model"]),
            str(row["dataset"]),
            str(row["optimizer"]),
            int(row["seed"]),
        )
        run = chosen_runs.get(key)
        state = str(row["status"])
        if state == "missing_optimization":
            output_dir = results_root / key[0] / key[1] / key[2] / f"seed{key[3]}"
            optimization_commands.append(
                _experiment_command(
                    model=key[0],
                    dataset=key[1],
                    optimizer=key[2],
                    seed=key[3],
                    budget=target_budget,
                    output_dir=output_dir,
                    checkpoints=budgets,
                )
            )
            continue
        if run is None:
            continue

        if state == "needs_resume":
            checkpoint = latest_checkpoint(run)
            if checkpoint is None:
                logger.warning(
                    "Cannot generate resume command for %s; no checkpoint", run.run_dir
                )
            else:
                args = read_json(run.args_path) if run.args_path else {}
                output_dir = args.get("output_dir", str(run.run_dir.parent))
                optimization_commands.append(
                    _experiment_command(
                        model=run.model,
                        dataset=run.dataset,
                        optimizer=run.optimizer,
                        seed=run.seed,
                        budget=target_budget,
                        output_dir=output_dir,
                        checkpoints=budgets,
                        args=args,
                        resume_from=checkpoint,
                    )
                )

        if run.step_results_path is not None:
            mapping = steps_for_checkpoints(read_step_results(run), budgets)
            evaluated = _eval_steps(run)
            for checkpoint, step in mapping.items():
                if checkpoint > target_budget or step in evaluated:
                    continue
                evaluation_commands.append(
                    f"# checkpoint={checkpoint:,}\n"
                    + _evaluation_command(
                        run,
                        step=step,
                        max_output_tokens=max_output_tokens,
                    )
                )
    return optimization_commands, evaluation_commands


def write_command_file(path: Path, commands: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for command in commands:
            handle.write(command)
            handle.write("\n")
    path.chmod(0o755)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    models = parse_csv_strings(args.models)
    datasets = parse_csv_strings(args.datasets)
    optimizers = parse_csv_strings(args.optimizers)
    seeds = parse_csv_ints(args.seeds)
    budgets = parse_csv_ints(args.budgets)
    target_budget = int(args.target_budget)
    if target_budget not in budgets:
        budgets = tuple(sorted(set(budgets) | {target_budget}))

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir)
    discovered = discover_runs(results_root)
    status, chosen_runs = build_expected_status(
        discovered,
        models=models,
        datasets=datasets,
        optimizers=optimizers,
        seeds=seeds,
        budgets=budgets,
        target_budget=target_budget,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    status.to_csv(output_dir / "experiment_status.csv", index=False)

    optimization_commands, evaluation_commands = generate_commands(
        status,
        chosen_runs,
        results_root=results_root,
        budgets=budgets,
        target_budget=target_budget,
        max_output_tokens=args.max_output_tokens,
    )
    write_command_file(output_dir / "run_or_resume_missing.sh", optimization_commands)
    write_command_file(
        output_dir / "evaluate_missing_checkpoints.sh", evaluation_commands
    )

    print("\nTri-Fair experiment status")
    print("=" * 72)
    print(status["status"].value_counts(dropna=False).to_string())
    print(f"\nExpected configurations: {len(status)}")
    print(f"Discovered logging directories: {len(discovered)}")
    print(f"Optimization/recovery commands: {len(optimization_commands)}")
    print(f"Checkpoint-evaluation commands: {len(evaluation_commands)}")
    print(f"Reports written to: {output_dir}")


if __name__ == "__main__":
    main()
