"""Discovery and loading utilities for Tri-Fair experiment artifacts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_BUDGETS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunArtifacts:
    """Files belonging to one concrete logging directory."""

    run_dir: Path
    args_path: Path | None
    step_results_path: Path | None
    eval_path: Path | None
    runhistory_path: Path | None
    checkpoint_dir: Path | None
    model: str
    dataset: str
    optimizer: str
    seed: int
    configured_budget: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _infer_metadata_from_path(run_dir: Path) -> tuple[str, str, str, int]:
    """Infer ``model/dataset/optimizer/seed`` from a conventional results path.

    The function searches backwards for a ``seedNN`` component and uses the
    preceding three components.  Args from ``args.json`` override this fallback.
    """

    parts = run_dir.parts
    for index in range(len(parts) - 1, -1, -1):
        part = parts[index]
        if part.startswith("seed") and part[4:].isdigit() and index >= 3:
            return parts[index - 3], parts[index - 2], parts[index - 1], int(part[4:])
    return "unknown", "unknown", "unknown", 0


def _candidate_run_dirs(results_root: Path) -> set[Path]:
    candidates: set[Path] = set()
    for filename in (
        "args.json",
        "step_results.parquet",
        "eval.parquet",
        "runhistory.json",
    ):
        for path in results_root.rglob(filename):
            candidates.add(path.parent.resolve())
    return candidates


def discover_runs(results_root: str | Path) -> list[RunArtifacts]:
    """Discover logging directories recursively beneath ``results_root``."""

    root = Path(results_root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    runs: list[RunArtifacts] = []
    for run_dir in sorted(_candidate_run_dirs(root)):
        args_path = run_dir / "args.json"
        args = read_json(args_path) if args_path.exists() else {}
        path_model, path_dataset, path_optimizer, path_seed = _infer_metadata_from_path(
            run_dir
        )

        model = str(args.get("model", path_model))
        dataset = str(args.get("dataset", path_dataset))
        optimizer = str(args.get("optimizer", path_optimizer))
        seed = _safe_int(args.get("random_seed", path_seed), path_seed)
        configured_budget = _safe_int(args.get("budget_per_run", 0), 0)

        step_path = run_dir / "step_results.parquet"
        eval_path = run_dir / "eval.parquet"
        runhistory_path = run_dir / "runhistory.json"
        checkpoint_dir = run_dir / "checkpoints"

        runs.append(
            RunArtifacts(
                run_dir=run_dir,
                args_path=args_path if args_path.exists() else None,
                step_results_path=step_path if step_path.exists() else None,
                eval_path=eval_path if eval_path.exists() else None,
                runhistory_path=runhistory_path if runhistory_path.exists() else None,
                checkpoint_dir=checkpoint_dir if checkpoint_dir.exists() else None,
                model=model,
                dataset=dataset,
                optimizer=optimizer,
                seed=seed,
                configured_budget=configured_budget,
            )
        )
    return runs


def read_step_results(run: RunArtifacts | str | Path) -> pd.DataFrame:
    path = run.step_results_path if isinstance(run, RunArtifacts) else Path(run)
    if path is None or not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path).copy()
    if frame.empty:
        return frame
    if "step" not in frame:
        raise ValueError(f"{path} has no 'step' column")
    frame["step"] = pd.to_numeric(frame["step"], errors="raise").astype(int)
    return frame


def read_eval_results(run: RunArtifacts | str | Path) -> pd.DataFrame:
    path = run.eval_path if isinstance(run, RunArtifacts) else Path(run)
    if path is None or not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path).copy()
    if "chosen_step" not in frame:
        frame["chosen_step"] = -1
    frame["chosen_step"] = (
        pd.to_numeric(frame["chosen_step"], errors="coerce").fillna(-1).astype(int)
    )
    return frame


def step_token_table(step_results: pd.DataFrame) -> pd.DataFrame:
    """Return one row per optimizer step with cumulative token counts."""

    if step_results.empty:
        return pd.DataFrame(
            columns=[
                "step",
                "input_tokens_downstream",
                "output_tokens_downstream",
                "total_tokens_downstream",
                "total_tokens_meta",
                "time",
            ]
        )

    frame = step_results.copy()
    if "total_tokens_downstream" not in frame:
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
        frame["total_tokens_downstream"] = input_tokens + output_tokens

    aggregation: dict[str, str] = {"total_tokens_downstream": "max"}
    for column in (
        "input_tokens_downstream",
        "output_tokens_downstream",
        "input_tokens_meta",
        "output_tokens_meta",
        "total_tokens_meta",
        "time",
    ):
        if column in frame:
            aggregation[column] = "max"

    result = frame.groupby("step", as_index=False).agg(aggregation).sort_values("step")
    for column in result.columns:
        if column != "step":
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.reset_index(drop=True)


def actual_tokens_for_step(step_table: pd.DataFrame, chosen_step: int) -> int:
    if step_table.empty:
        return 0
    exact = step_table[step_table["step"] == int(chosen_step)]
    if exact.empty:
        prior = step_table[step_table["step"] <= int(chosen_step)]
        selected = prior.tail(1) if not prior.empty else step_table.head(1)
    else:
        selected = exact
    return _safe_int(selected.iloc[0]["total_tokens_downstream"], 0)


def checkpoint_for_tokens(
    actual_tokens: int, checkpoints: Sequence[int] = DEFAULT_BUDGETS
) -> int:
    """Map an overshooting actual token count to the crossed target checkpoint."""

    actual_tokens = int(actual_tokens)
    eligible = [int(value) for value in checkpoints if int(value) <= actual_tokens]
    return max(eligible) if eligible else 0


def steps_for_checkpoints(
    step_results: pd.DataFrame,
    checkpoints: Sequence[int] = DEFAULT_BUDGETS,
    *,
    prefer_at_or_after: bool = True,
) -> dict[int, int]:
    """Map budget checkpoints to the closest recorded optimizer step.

    Checkpoint callbacks fire after a step, so the scientifically faithful
    mapping is the first step whose cumulative tokens are at or above the target.
    If a target was never reached, it is omitted.
    """

    table = step_token_table(step_results)
    mapping: dict[int, int] = {}
    if table.empty:
        return mapping

    tokens = table["total_tokens_downstream"].to_numpy(dtype=float)
    steps = table["step"].to_numpy(dtype=int)
    for checkpoint in sorted({int(value) for value in checkpoints if int(value) > 0}):
        if prefer_at_or_after:
            indices = np.flatnonzero(tokens >= checkpoint)
            if len(indices):
                mapping[checkpoint] = int(steps[indices[0]])
        else:
            indices = np.flatnonzero(tokens <= checkpoint)
            if len(indices):
                mapping[checkpoint] = int(steps[indices[-1]])
    return mapping


def _metadata_columns(run: RunArtifacts, row_count: int) -> dict[str, list[Any]]:
    return {
        "run_dir": [str(run.run_dir)] * row_count,
        "model": [run.model] * row_count,
        "dataset": [run.dataset] * row_count,
        "optimizer": [run.optimizer] * row_count,
        "seed": [run.seed] * row_count,
        "configured_budget": [run.configured_budget] * row_count,
    }


def load_all_evaluations(
    results_root: str | Path,
    *,
    budget_checkpoints: Sequence[int] = DEFAULT_BUDGETS,
    strict: bool = False,
) -> pd.DataFrame:
    """Load every compatible ``eval.parquet`` and attach run/checkpoint metadata."""

    frames: list[pd.DataFrame] = []
    for run in discover_runs(results_root):
        if run.eval_path is None:
            continue
        try:
            evaluation = read_eval_results(run)
            step_table = (
                step_token_table(read_step_results(run))
                if run.step_results_path is not None
                else pd.DataFrame()
            )
        except Exception:
            if strict:
                raise
            logger.exception("Skipping unreadable run %s", run.run_dir)
            continue

        for key, values in _metadata_columns(run, len(evaluation)).items():
            # Explicit columns written by the evaluator are retained when valid;
            # run metadata is the fallback.
            if key not in evaluation or evaluation[key].isna().all():
                evaluation[key] = values

        evaluation["actual_budget_tokens"] = [
            actual_tokens_for_step(step_table, step) if not step_table.empty else 0
            for step in evaluation["chosen_step"]
        ]
        evaluation["budget_checkpoint"] = [
            checkpoint_for_tokens(value, budget_checkpoints)
            for value in evaluation["actual_budget_tokens"]
        ]
        evaluation["run_key"] = (
            evaluation["model"].astype(str)
            + "/"
            + evaluation["dataset"].astype(str)
            + "/"
            + evaluation["optimizer"].astype(str)
            + "/seed"
            + evaluation["seed"].astype(str)
            + "/"
            + evaluation["run_dir"].astype(str).map(lambda value: Path(value).name)
        )
        frames.append(evaluation)

    if not frames:
        raise FileNotFoundError(f"No eval.parquet files found beneath {results_root}")
    combined = pd.concat(frames, ignore_index=True, sort=False)

    dedup = [
        column for column in ("run_dir", "chosen_step", "prompt") if column in combined
    ]
    if dedup:
        combined = combined.drop_duplicates(subset=dedup, keep="last")
    return combined.reset_index(drop=True)


def load_all_step_results(
    results_root: str | Path, *, strict: bool = False
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in discover_runs(results_root):
        if run.step_results_path is None:
            continue
        try:
            frame = read_step_results(run)
        except Exception:
            if strict:
                raise
            logger.exception("Skipping unreadable step log %s", run.run_dir)
            continue
        for key, values in _metadata_columns(run, len(frame)).items():
            frame[key] = values
        frame["run_key"] = (
            frame["model"].astype(str)
            + "/"
            + frame["dataset"].astype(str)
            + "/"
            + frame["optimizer"].astype(str)
            + "/seed"
            + frame["seed"].astype(str)
            + "/"
            + frame["run_dir"].astype(str).map(lambda value: Path(value).name)
        )
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No step_results.parquet files found beneath {results_root}"
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def checkpoint_files(run: RunArtifacts) -> dict[int, Path]:
    result: dict[int, Path] = {}
    if run.checkpoint_dir is None:
        return result
    for path in run.checkpoint_dir.glob("tokens_*.pkl"):
        token_text = path.stem.removeprefix("tokens_")
        if token_text.isdigit():
            result[int(token_text)] = path
    return dict(sorted(result.items()))


def latest_checkpoint(run: RunArtifacts) -> Path | None:
    if run.checkpoint_dir is None:
        return None
    path = run.checkpoint_dir / "latest.pkl"
    if path.exists():
        return path
    files = checkpoint_files(run)
    return files[max(files)] if files else None
