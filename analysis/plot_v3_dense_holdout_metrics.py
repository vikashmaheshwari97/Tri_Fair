"""Plot MO-CAPO-style post-hoc holdout trajectories on actual token usage.

Input is ``v3_checkpoint_run_metrics.csv`` produced by
``analysis/summarize_v3_checkpoints.py``.

Each seed is first reduced to its unique real evaluated optimizer states using
``actual_budget_tokens``/``chosen_step``.  The three seed trajectories are then
placed on the union of their real token positions with a right-continuous
last-observation-carried-forward rule.  Mean and sample standard deviation are
computed at each event position.

This is an anytime-trajectory visualization of real holdout evaluations:
no objective interpolation, smoothing, invented checkpoint, or synthetic metric
value is introduced.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ("Tri-Fair-v3", "NSGAII-PO-Fair")
DISPLAY = {
    "Tri-Fair-v3": "Tri-Fair v3",
    "NSGAII-PO-Fair": "NSGA-II-PO-Fair",
    "Initial": "Initial Instructions",
}
COLORS = {
    "Tri-Fair-v3": "black",
    "NSGAII-PO-Fair": "#E69F00",
    "Initial": "#888888",
}
MARKERS = {"Tri-Fair-v3": "o", "NSGAII-PO-Fair": "s"}

DATASET_LABELS = {
    "bbq": "BBQ",
    "civil_comments": "Civil Comments",
    "bias_in_bios": "Bias in Bios",
}
MODEL_LABELS = {
    "qwen-3-30b": "Qwen-3-30B",
    "gpt-oss-120b": "GPT-OSS-120B",
}

SPECS = (
    ("noisy_r2_3d", "nR2 Indicator ↓", "actual_token_holdout_nr2"),
    (
        "hv_test_optimistic_3d",
        "Optimistic Holdout Hypervolume ↑",
        "actual_token_holdout_hv_optimistic",
    ),
    (
        "hv_test_pessimistic_3d",
        "Pessimistic Holdout Hypervolume ↑",
        "actual_token_holdout_hv_pessimistic",
    ),
    (
        "approximation_gap_3d",
        "Holdout Approximation Gap ↓",
        "actual_token_holdout_gap",
    ),
)


class CurveStats(NamedTuple):
    tokens: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    parser.add_argument(
        "--strict-three-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require exactly three independent seed trajectories per method.",
    )
    parser.add_argument(
        "--minimum-budget",
        type=int,
        default=None,
        help="Optional lower limit in actual cumulative downstream tokens.",
    )
    parser.add_argument(
        "--maximum-budget",
        type=int,
        default=None,
        help="Optional upper limit in actual cumulative downstream tokens.",
    )
    parser.add_argument(
        "--x-label",
        default="Token Budget [×10⁶]",
        help=(
            "Axis label. 'Token Budget [×10⁶]' matches the MO-CAPO presentation; "
            "'Cumulative Downstream Tokens [×10⁶]' is the more explicit equivalent."
        ),
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _unique_real_states(
    frame: pd.DataFrame,
    *,
    method: str,
    metric: str,
    minimum_budget: int | None,
    maximum_budget: int | None,
) -> dict[int, pd.DataFrame]:
    required = {"seed", "actual_budget_tokens", metric}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Run metrics are missing {sorted(missing)}")

    data = frame[frame["optimizer"] == method].copy()
    data = _numeric(
        data,
        (
            "seed",
            "budget_checkpoint",
            "chosen_step",
            "actual_budget_tokens",
            metric,
        ),
    )
    data = data.dropna(subset=["seed", "actual_budget_tokens", metric])
    data["seed"] = data["seed"].astype(int)

    if minimum_budget is not None:
        data = data[data["actual_budget_tokens"] >= minimum_budget]
    if maximum_budget is not None:
        data = data[data["actual_budget_tokens"] <= maximum_budget]

    result: dict[int, pd.DataFrame] = {}
    for seed, group in data.groupby("seed", sort=True):
        order = [
            column
            for column in ("actual_budget_tokens", "chosen_step", "budget_checkpoint")
            if column in group
        ]
        group = group.sort_values(order, kind="stable")

        # Several nominal checkpoints may map to the same real logged state.
        # Keep that state only once.
        dedup = (
            ["chosen_step"]
            if "chosen_step" in group and group["chosen_step"].notna().all()
            else ["actual_budget_tokens"]
        )
        group = group.drop_duplicates(dedup, keep="last")
        group = group.drop_duplicates(["actual_budget_tokens"], keep="last")
        group = group.sort_values("actual_budget_tokens", kind="stable")

        if not group.empty:
            result[int(seed)] = group.reset_index(drop=True)

    return result


def _locf_stats(seed_states: dict[int, pd.DataFrame], metric: str) -> CurveStats:
    if not seed_states:
        raise RuntimeError(f"No seed states are available for {metric}")

    # Start only when all seeds have at least one real evaluated state.
    common_start = max(
        int(group["actual_budget_tokens"].iloc[0])
        for group in seed_states.values()
    )

    grid = np.unique(
        np.concatenate(
            [
                group["actual_budget_tokens"].to_numpy(dtype=int)
                for group in seed_states.values()
            ]
        )
    )
    grid = np.sort(grid[grid >= common_start])
    if grid.size == 0:
        raise RuntimeError(f"No common actual-token grid is available for {metric}")

    trajectories: list[np.ndarray] = []
    for group in seed_states.values():
        x = group["actual_budget_tokens"].to_numpy(dtype=int)
        y = group[metric].to_numpy(dtype=float)

        values = np.full(grid.size, np.nan, dtype=float)
        positions = np.searchsorted(x, grid, side="right") - 1
        valid = positions >= 0
        values[valid] = y[positions[valid]]
        trajectories.append(values)

    matrix = np.vstack(trajectories)
    n = np.sum(np.isfinite(matrix), axis=0)
    mean = np.nanmean(matrix, axis=0)

    std = np.zeros(grid.size, dtype=float)
    for index in range(grid.size):
        values = matrix[:, index]
        values = values[np.isfinite(values)]
        std[index] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    return CurveStats(tokens=grid, mean=mean, std=std, n=n)


def _initial_stats(frame: pd.DataFrame, metric: str) -> tuple[float, float] | None:
    data = frame[frame["optimizer"] == "Initial"].copy()
    if metric not in data:
        return None
    values = pd.to_numeric(data[metric], errors="coerce").dropna()
    if values.empty:
        return None
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def _build_curves(
    frame: pd.DataFrame,
    *,
    metric: str,
    minimum_budget: int | None,
    maximum_budget: int | None,
    strict_three_seeds: bool,
) -> dict[str, CurveStats]:
    curves: dict[str, CurveStats] = {}
    for method in METHOD_ORDER:
        seed_states = _unique_real_states(
            frame,
            method=method,
            metric=metric,
            minimum_budget=minimum_budget,
            maximum_budget=maximum_budget,
        )
        if strict_three_seeds and set(seed_states) != {42, 43, 44}:
            raise RuntimeError(
                f"{method}/{metric} has seeds {sorted(seed_states)}; "
                "expected 42, 43, and 44"
            )
        curves[method] = _locf_stats(seed_states, metric)
    return curves


def _draw_metric(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    x_label: str,
    minimum_budget: int | None,
    maximum_budget: int | None,
    strict_three_seeds: bool,
) -> pd.DataFrame:
    curves = _build_curves(
        frame,
        metric=metric,
        minimum_budget=minimum_budget,
        maximum_budget=maximum_budget,
        strict_three_seeds=strict_three_seeds,
    )

    xmin = min(float(curve.tokens[0]) for curve in curves.values()) / 1_000_000.0
    xmax = max(float(curve.tokens[-1]) for curve in curves.values()) / 1_000_000.0

    audit_rows: list[pd.DataFrame] = []
    method_handles = []

    for method in METHOD_ORDER:
        curve = curves[method]
        x = curve.tokens.astype(float) / 1_000_000.0

        (line,) = ax.step(
            x,
            curve.mean,
            where="post",
            color=COLORS[method],
            linewidth=2.0,
            label=DISPLAY[method],
            zorder=3,
        )
        method_handles.append(line)

        # Match the reference style: markers only at the first and final point.
        ax.scatter(
            [x[0], x[-1]],
            [curve.mean[0], curve.mean[-1]],
            color=COLORS[method],
            marker=MARKERS[method],
            s=38,
            zorder=4,
        )

        ax.fill_between(
            x,
            curve.mean - curve.std,
            curve.mean + curve.std,
            step="post",
            color=COLORS[method],
            alpha=0.16,
            zorder=2,
        )

        audit_rows.append(
            pd.DataFrame(
                {
                    "optimizer": method,
                    "metric": metric,
                    "actual_budget_tokens": curve.tokens,
                    "mean": curve.mean,
                    "std": curve.std,
                    "n_seeds": curve.n,
                }
            )
        )

    initial_handle = None
    initial = _initial_stats(frame, metric)
    if initial is not None:
        initial_mean, initial_std = initial
        initial_handle = ax.axhline(
            initial_mean,
            color=COLORS["Initial"],
            linestyle="--",
            linewidth=1.6,
            label=DISPLAY["Initial"],
            zorder=1,
        )
        if initial_std > 0:
            ax.fill_between(
                [xmin, xmax],
                [initial_mean - initial_std] * 2,
                [initial_mean + initial_std] * 2,
                color=COLORS["Initial"],
                alpha=0.08,
                zorder=0,
            )

    padding = max(0.03, 0.02 * (xmax - xmin))
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.7)

    handles = method_handles + ([initial_handle] if initial_handle is not None else [])
    labels = [DISPLAY[method] for method in METHOD_ORDER]
    if initial_handle is not None:
        labels.append(DISPLAY["Initial"])
    ax.legend(handles, labels, frameon=False)

    return pd.concat(audit_rows, ignore_index=True)


def save_figure(fig: plt.Figure, outdir: Path, stem: str, output_format: str) -> None:
    if output_format in {"png", "both"}:
        fig.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches="tight")
    if output_format in {"pdf", "both"}:
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()

    source = Path(args.run_metrics).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    frame = pd.read_csv(source)
    required = {
        "model",
        "dataset",
        "optimizer",
        "seed",
        "budget_checkpoint",
        "actual_budget_tokens",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Run-metric table is missing {sorted(missing)}")

    output_root = Path(args.output_dir).expanduser().resolve()

    for (model, dataset), group in frame.groupby(["model", "dataset"], sort=True):
        outdir = output_root / str(model) / str(dataset)
        outdir.mkdir(parents=True, exist_ok=True)

        heading = (
            f"{DATASET_LABELS.get(str(dataset), dataset)} — "
            f"{MODEL_LABELS.get(str(model), model)}"
        )

        audit_tables: list[pd.DataFrame] = []

        for metric, ylabel, stem in SPECS:
            if metric not in group:
                continue
            fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
            audit = _draw_metric(
                ax,
                group,
                metric=metric,
                ylabel=ylabel,
                x_label=args.x_label,
                minimum_budget=args.minimum_budget,
                maximum_budget=args.maximum_budget,
                strict_three_seeds=args.strict_three_seeds,
            )
            ax.set_title(heading)
            save_figure(fig, outdir, stem, args.format)
            audit_tables.append(audit)

        fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
        for ax, (metric, ylabel, _) in zip(axes.ravel(), SPECS):
            if metric not in group:
                ax.set_axis_off()
                continue
            _draw_metric(
                ax,
                group,
                metric=metric,
                ylabel=ylabel,
                x_label=args.x_label,
                minimum_budget=args.minimum_budget,
                maximum_budget=args.maximum_budget,
                strict_three_seeds=args.strict_three_seeds,
            )
            ax.set_title(ylabel)

        # One shared legend, matching Tri-Fair / NSGA-II / Initial order.
        for ax in axes.ravel():
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            order = [
                labels.index(DISPLAY[name])
                for name in (*METHOD_ORDER, "Initial")
                if DISPLAY[name] in labels
            ]
            fig.legend(
                [handles[index] for index in order],
                [labels[index] for index in order],
                loc="upper center",
                ncol=3,
                frameon=False,
                bbox_to_anchor=(0.5, 1.02),
            )

        fig.suptitle(
            f"{heading}: Post-hoc Holdout Anytime Trajectories",
            y=1.055,
            fontsize=14,
        )
        fig.text(
            0.5,
            -0.01,
            "Right-continuous latest real holdout state; "
            "lines and shaded regions show mean ± standard deviation "
            "across three independent seeds.",
            ha="center",
            fontsize=8,
        )
        save_figure(
            fig,
            outdir,
            "actual_token_holdout_nr2_hv_gap_2x2",
            args.format,
        )

        if audit_tables:
            pd.concat(audit_tables, ignore_index=True).to_csv(
                outdir / "actual_token_holdout_mean_std.csv",
                index=False,
            )

        print(f"Wrote actual-token holdout trajectories to {outdir}")


if __name__ == "__main__":
    main()
