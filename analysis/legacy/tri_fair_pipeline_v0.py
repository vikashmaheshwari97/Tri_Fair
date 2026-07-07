"""Three-objective analysis pipeline for Tri-Fair.

The pipeline consumes ``eval.parquet`` files produced by
``scripts/evaluate_prompts.py`` and reports:

* normalized 3-D hypervolume,
* noisy R2 with Dirichlet preference vectors,
* optimistic/pessimistic holdout hypervolume and approximation gap,
* development-to-test fairness generalization,
* 3-D Pareto and pairwise projection plots.

All prompt selection is performed on development objectives.  Test objectives
are consulted only after a development-selected prompt has been chosen.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

logger = logging.getLogger(__name__)

OBJECTIVE_COLUMNS = {
    "dev": ("dev_quality", "dev_cost", "dev_fairness"),
    "test": ("test_quality", "test_cost", "test_fairness"),
}
REFERENCE_POINT = np.asarray([1.1, 1.1, 1.1], dtype=float)


@dataclass(frozen=True)
class Bounds:
    minimum: np.ndarray
    maximum: np.ndarray

    def normalise(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        span = self.maximum - self.minimum
        span = np.where(span <= 0, 1.0, span)
        return np.clip((values - self.minimum) / span, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/tri_fair")
    parser.add_argument("--output-dir", default="analysis/tri_fair_output")
    parser.add_argument("--n-preferences", type=int, default=500)
    parser.add_argument("--preference-seed", type=int, default=2026)
    return parser.parse_args()


def _read_args(eval_path: Path) -> Dict:
    args_path = eval_path.parent / "args.json"
    if args_path.exists():
        with args_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def discover_evaluations(root: Path) -> Iterator[tuple[Path, pd.DataFrame, Dict]]:
    for path in sorted(root.rglob("eval.parquet")):
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        args = _read_args(path)
        yield path, frame, args


def load_all(root: Path) -> pd.DataFrame:
    frames = []
    for path, frame, args in discover_evaluations(root):
        required = set(sum((list(value) for value in OBJECTIVE_COLUMNS.values()), []))
        missing = required - set(frame.columns)
        if missing:
            logger.warning("Skipping %s; missing columns %s", path, sorted(missing))
            continue
        frame = frame.copy()
        frame["run_path"] = str(path.parent)
        frame["optimizer"] = args.get("optimizer", frame.get("optimizer", "unknown"))
        frame["dataset"] = args.get("dataset", frame.get("dataset", "unknown"))
        frame["model"] = args.get("model", frame.get("model", "unknown"))
        if "random_seed" in args:
            seed_value = args["random_seed"]
        elif "evaluation_seed" in frame.columns:
            seed_value = frame["evaluation_seed"].iloc[0]
        else:
            seed_value = 0
        frame["seed"] = int(seed_value)
        frame["budget_limit"] = int(args.get("budget_per_run", 0))
        if "chosen_step" not in frame:
            frame["chosen_step"] = -1
        frame["run_id"] = (
            frame["model"].astype(str)
            + "/"
            + frame["dataset"].astype(str)
            + "/"
            + frame["optimizer"].astype(str)
            + "/seed"
            + frame["seed"].astype(str)
            + "/step"
            + frame["chosen_step"].astype(str)
        )
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No compatible eval.parquet files found under {root}")
    return pd.concat(frames, ignore_index=True)


def objective_matrix(frame: pd.DataFrame, split: str) -> np.ndarray:
    quality, cost, fairness = OBJECTIVE_COLUMNS[split]
    return np.column_stack(
        [
            1.0 - pd.to_numeric(frame[quality], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(frame[cost], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(frame[fairness], errors="coerce").to_numpy(dtype=float),
        ]
    )


def valid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    dev = objective_matrix(frame, "dev")
    test = objective_matrix(frame, "test")
    mask = np.all(np.isfinite(dev), axis=1) & np.all(np.isfinite(test), axis=1)
    if "dev_fairness_ready" in frame:
        mask &= frame["dev_fairness_ready"].fillna(False).to_numpy(dtype=bool)
    if "test_fairness_ready" in frame:
        mask &= frame["test_fairness_ready"].fillna(False).to_numpy(dtype=bool)
    return frame.loc[mask].reset_index(drop=True)


def pareto_mask(values: np.ndarray) -> np.ndarray:
    """Return non-dominated rows for a minimize-all objective matrix."""

    values = np.asarray(values, dtype=float)
    n = len(values)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates_i = np.all(values <= values[i], axis=1) & np.any(
            values < values[i], axis=1
        )
        if np.any(dominates_i):
            keep[i] = False
    return keep


def dominance_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.all(values[:, None, :] <= values[None, :, :], axis=2) & np.any(
        values[:, None, :] < values[None, :, :], axis=2
    )


def compute_bounds(frame: pd.DataFrame) -> Dict[tuple[str, str], Bounds]:
    result: Dict[tuple[str, str], Bounds] = {}
    for key, group in frame.groupby(["dataset", "model"], sort=True):
        union = np.vstack(
            [objective_matrix(group, "dev"), objective_matrix(group, "test")]
        )
        union = union[np.all(np.isfinite(union), axis=1)]
        if not len(union):
            continue
        minimum = np.min(union, axis=0)
        maximum = np.max(union, axis=0)
        # Add a tiny data-dependent margin so boundary solutions do not map
        # exactly onto the hypervolume reference point after clipping.
        margin = np.maximum((maximum - minimum) * 1e-9, 1e-12)
        result[(str(key[0]), str(key[1]))] = Bounds(minimum - margin, maximum + margin)
    return result


def hypervolume(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    front = values[pareto_mask(values)]
    return float(HV(ref_point=REFERENCE_POINT)(front))


def noisy_r2(
    dev_values: np.ndarray,
    test_values: np.ndarray,
    *,
    n_preferences: int,
    seed: int,
) -> float:
    dev_front = pareto_mask(dev_values)
    dev = dev_values[dev_front]
    test = test_values[dev_front]
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(dev.shape[1]), size=n_preferences)
    utilities = []
    for weight in weights:
        dev_utility = np.max(weight[None, :] * dev, axis=1)
        selected = int(np.argmin(dev_utility))
        utilities.append(float(np.max(weight * test[selected])))
    return float(np.mean(utilities))


def optimistic_pessimistic_hv(test_values: np.ndarray) -> tuple[float, float, float]:
    if len(test_values) == 0:
        return 0.0, 0.0, 0.0
    dominates = dominance_matrix(test_values)
    dominated_by_any = np.any(dominates, axis=0)
    dominates_any = np.any(dominates, axis=1)

    optimistic = test_values[~dominated_by_any]
    # Following the approximation-gap construction described in MO-CAPO:
    # pessimistic boundary points are those that fail to dominate any other
    # development-selected solution on holdout data.
    pessimistic = test_values[~dominates_any]
    optimistic_hv = hypervolume(optimistic)
    pessimistic_hv = hypervolume(pessimistic)
    return optimistic_hv, pessimistic_hv, max(0.0, optimistic_hv - pessimistic_hv)


def analyse_runs(
    frame: pd.DataFrame,
    bounds: Dict[tuple[str, str], Bounds],
    *,
    n_preferences: int,
    preference_seed: int,
) -> pd.DataFrame:
    rows = []
    for run_id, group in frame.groupby("run_id", sort=True):
        group = valid_rows(group)
        if group.empty:
            logger.warning("No fairness-ready rows for %s", run_id)
            continue
        dataset = str(group["dataset"].iloc[0])
        model = str(group["model"].iloc[0])
        run_bounds = bounds[(dataset, model)]
        dev = run_bounds.normalise(objective_matrix(group, "dev"))
        test = run_bounds.normalise(objective_matrix(group, "test"))

        selected_mask = pareto_mask(dev)
        dev_front = dev[selected_mask]
        test_selected = test[selected_mask]
        hv_dev = hypervolume(dev_front)
        hv_test_opt, hv_test_pes, gap = optimistic_pessimistic_hv(test_selected)
        nr2 = noisy_r2(
            dev,
            test,
            n_preferences=n_preferences,
            seed=preference_seed + int(group["seed"].iloc[0]),
        )

        rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "model": model,
                "optimizer": group["optimizer"].iloc[0],
                "seed": int(group["seed"].iloc[0]),
                "chosen_step": int(group["chosen_step"].iloc[0]),
                "budget_limit": int(group["budget_limit"].iloc[0]),
                "n_candidates": len(group),
                "n_dev_pareto": int(selected_mask.sum()),
                "hv_dev_3d": hv_dev,
                "hv_test_optimistic_3d": hv_test_opt,
                "hv_test_pessimistic_3d": hv_test_pes,
                "approximation_gap_3d": gap,
                "noisy_r2_3d": nr2,
                "mean_dev_fairness": float(
                    group.loc[selected_mask, "dev_fairness"].mean()
                ),
                "mean_test_fairness": float(
                    group.loc[selected_mask, "test_fairness"].mean()
                ),
                "fairness_generalization_gap": float(
                    (
                        group.loc[selected_mask, "test_fairness"].to_numpy(dtype=float)
                        - group.loc[selected_mask, "dev_fairness"].to_numpy(dtype=float)
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def plot_run(group: pd.DataFrame, bounds: Bounds, output_dir: Path) -> None:
    group = valid_rows(group)
    if group.empty:
        return
    dev = bounds.normalise(objective_matrix(group, "dev"))
    mask = pareto_mask(dev)
    pareto = group.loc[mask].reset_index(drop=True)
    title = str(group["run_id"].iloc[0])
    stem = _safe_name(title)

    # Plot raw, interpretable objective values rather than normalized losses.
    quality = pareto["test_quality"].to_numpy(dtype=float)
    cost = pareto["test_cost"].to_numpy(dtype=float)
    fairness = pareto["test_fairness"].to_numpy(dtype=float)

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(cost, fairness, quality)
    axis.set_xlabel("Inference cost")
    axis.set_ylabel("Unfairness")
    axis.set_zlabel("Quality")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}_pareto_3d.png", dpi=200)
    plt.close(figure)

    for x, y, xlabel, ylabel, suffix in (
        (cost, quality, "Inference cost", "Quality", "cost_quality"),
        (fairness, quality, "Unfairness", "Quality", "fairness_quality"),
        (cost, fairness, "Inference cost", "Unfairness", "cost_fairness"),
    ):
        figure = plt.figure(figsize=(7, 5))
        axis = figure.add_axes((0.13, 0.13, 0.82, 0.78))
        axis.scatter(x, y)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        figure.savefig(output_dir / f"{stem}_{suffix}.png", dpi=200)
        plt.close(figure)


def aggregate_summary(run_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "hv_dev_3d",
        "hv_test_optimistic_3d",
        "hv_test_pessimistic_3d",
        "approximation_gap_3d",
        "noisy_r2_3d",
        "fairness_generalization_gap",
    ]
    grouped = run_metrics.groupby(
        ["dataset", "model", "optimizer", "budget_limit"], sort=True
    )
    rows = []
    for keys, group in grouped:
        row = dict(zip(("dataset", "model", "optimizer", "budget_limit"), keys))
        row["n_runs"] = len(group)
        for column in numeric:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = (
                float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    root = Path(args.results_root)
    output = Path(args.output_dir)
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    frame = load_all(root)
    bounds = compute_bounds(frame)
    run_metrics = analyse_runs(
        frame,
        bounds,
        n_preferences=args.n_preferences,
        preference_seed=args.preference_seed,
    )
    summary = aggregate_summary(run_metrics)

    output.mkdir(parents=True, exist_ok=True)
    run_metrics.to_csv(output / "run_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    with (output / "normalization_bounds.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                f"{dataset}/{model}": {
                    "minimum": value.minimum.tolist(),
                    "maximum": value.maximum.tolist(),
                }
                for (dataset, model), value in bounds.items()
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    for _, group in frame.groupby("run_id", sort=True):
        dataset = str(group["dataset"].iloc[0])
        model = str(group["model"].iloc[0])
        plot_run(group, bounds[(dataset, model)], plot_dir)

    logger.info("Wrote Tri-Fair analysis to %s", output)


if __name__ == "__main__":
    main()
