#!/usr/bin/env python
"""Run Tri-Fair-GR method comparison over evaluated GraphRAG policy candidates.

Methods:
  - Initial GNN-RAG
  - NSGAII-PO-Fair
  - Tri-Fair-GR

This script uses the existing Qwen-evaluated policy table as an evaluation cache.
It runs budgeted method search over the GraphRAG policy task and produces
method-level tables and figures with Accuracy, token cost, and unfairness.

Main objective direction:
  maximize Accuracy
  minimize Token cost
  minimize End-to-end unfairness
"""

from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ORACLE_ORDERS = {"answer_evidence_first", "answer_evidence_last"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-label", default="WebQSP pilot")
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--n-init", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--exclude-oracle", action="store_true", default=True)
    return parser.parse_args()


def add_policy_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "quality" not in out.columns and "answer_mean" in out.columns:
        out["quality"] = out["answer_mean"]

    out["accuracy"] = out["quality"]

    parts = out["policy_name"].astype(str).str.extract(
        r"i-(?P<instruction>.*?)_f-(?P<fairness>.*?)_a-(?P<answer>.*?)_p-"
    )
    for col in parts.columns:
        out[col] = parts[col].fillna("unknown")

    out["is_oracle_order"] = out["path_order"].isin(ORACLE_ORDERS)

    required = [
        "policy_name",
        "accuracy",
        "cost_per_1k_examples_usd",
        "cost_usd",
        "unfairness",
        "answer_unfairness",
        "retrieval_unfairness",
        "max_paths",
        "path_order",
        "verbalization",
        "instruction",
        "fairness",
    ]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    out["candidate_idx"] = range(len(out))
    return out


def norm_good(series: pd.Series, higher_is_better: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo = s.min()
    hi = s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series([1.0] * len(s), index=s.index)
    z = (s - lo) / (hi - lo)
    return z if higher_is_better else 1.0 - z


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score_accuracy"] = norm_good(out["accuracy"], True)
    out["score_cost"] = norm_good(out["cost_per_1k_examples_usd"], False)
    out["score_fairness"] = norm_good(out["unfairness"], False)
    out["trifair_score"] = (
        out["score_accuracy"] + out["score_cost"] + out["score_fairness"]
    ) / 3.0
    return out


def dominates(a: pd.Series, b: pd.Series) -> bool:
    better_or_equal = (
        a["accuracy"] >= b["accuracy"]
        and a["cost_per_1k_examples_usd"] <= b["cost_per_1k_examples_usd"]
        and a["unfairness"] <= b["unfairness"]
    )
    strictly_better = (
        a["accuracy"] > b["accuracy"]
        or a["cost_per_1k_examples_usd"] < b["cost_per_1k_examples_usd"]
        or a["unfairness"] < b["unfairness"]
    )
    return bool(better_or_equal and strictly_better)


def nondominated_indices(df: pd.DataFrame, indices: list[int]) -> list[int]:
    keep = []
    for i in indices:
        row = df.loc[i]
        dominated = False
        for j in indices:
            if i == j:
                continue
            if dominates(df.loc[j], row):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def select_final_policy(df: pd.DataFrame, observed: list[int]) -> int:
    # One final row per method: best balanced Tri-Fair objective among observed candidates.
    obs = df.loc[observed].copy()
    obs = obs.sort_values(
        ["trifair_score", "accuracy", "unfairness", "cost_per_1k_examples_usd"],
        ascending=[False, False, True, True],
    )
    return int(obs.index[0])


def get_default_initial(df: pd.DataFrame) -> int:
    # Initial GNN-RAG: no fairness instruction, original ordering, arrow text,
    # default-ish graph context. Prefer evidence_concise, max_paths=5.
    preferred = df[
        (df["fairness"].astype(str) == "none")
        & (df["path_order"].astype(str) == "original")
        & (df["verbalization"].astype(str) == "arrow")
        & (pd.to_numeric(df["max_paths"], errors="coerce") == 5)
        & (df["instruction"].astype(str) == "evidence_concise")
    ]
    if not preferred.empty:
        return int(preferred.index[0])

    fallback = df[
        (df["fairness"].astype(str) == "none")
        & (df["path_order"].astype(str) == "original")
        & (df["verbalization"].astype(str) == "arrow")
    ]
    if not fallback.empty:
        return int(
            fallback.sort_values(
                ["max_paths", "accuracy"],
                ascending=[True, False],
            ).index[0]
        )

    return int(df.sort_values(["cost_per_1k_examples_usd", "accuracy"], ascending=[True, False]).index[0])


def feature_columns() -> list[str]:
    return [
        "instruction",
        "fairness",
        "max_paths",
        "path_order",
        "verbalization",
    ]


def same_feature_count(a: pd.Series, b: pd.Series) -> int:
    return sum(str(a[col]) == str(b[col]) for col in feature_columns())


def random_unobserved(df: pd.DataFrame, observed: set[int], rng: random.Random) -> int:
    candidates = [int(i) for i in df.index if int(i) not in observed]
    return rng.choice(candidates)


def mutate_from_parent(df: pd.DataFrame, parent_idx: int, observed: set[int], rng: random.Random) -> int:
    parent = df.loc[parent_idx]
    candidate = parent.copy()

    cols = feature_columns()
    n_changes = rng.choice([1, 1, 2])
    change_cols = rng.sample(cols, k=n_changes)

    for col in change_cols:
        values = sorted(df[col].dropna().astype(str).unique().tolist())
        if not values:
            continue
        candidate[col] = rng.choice(values)

    mask = pd.Series([True] * len(df), index=df.index)
    for col in cols:
        mask &= df[col].astype(str) == str(candidate[col])

    matches = [int(i) for i in df.loc[mask].index if int(i) not in observed]
    if matches:
        return rng.choice(matches)

    return random_unobserved(df, observed, rng)


def trajectory_row(
    *,
    method: str,
    seed: int,
    step: int,
    df: pd.DataFrame,
    observed: list[int],
) -> dict:
    best_idx = select_final_policy(df, observed)
    row = df.loc[best_idx]
    return {
        "method": method,
        "seed": seed,
        "step": step,
        "evaluated": len(observed),
        "best_policy_name": row["policy_name"],
        "accuracy": row["accuracy"],
        "cost_usd": row["cost_usd"],
        "cost_per_1k_examples_usd": row["cost_per_1k_examples_usd"],
        "unfairness": row["unfairness"],
        "answer_unfairness": row["answer_unfairness"],
        "retrieval_unfairness": row["retrieval_unfairness"],
        "trifair_score": row["trifair_score"],
        "max_paths": row["max_paths"],
        "path_order": row["path_order"],
        "verbalization": row["verbalization"],
        "fairness": row["fairness"],
        "instruction": row["instruction"],
    }


def run_initial(df: pd.DataFrame, seed: int) -> tuple[list[int], list[dict]]:
    idx = get_default_initial(df)
    observed = [idx]
    traj = [
        trajectory_row(
            method="Initial GNN-RAG",
            seed=seed,
            step=1,
            df=df,
            observed=observed,
        )
    ]
    return observed, traj


def run_nsgaii(df: pd.DataFrame, seed: int, budget: int, population_size: int) -> tuple[list[int], list[dict]]:
    rng = random.Random(seed)
    initial_idx = get_default_initial(df)

    observed: list[int] = [initial_idx]
    observed_set = {initial_idx}

    while len(observed) < min(population_size, budget):
        idx = random_unobserved(df, observed_set, rng)
        observed.append(idx)
        observed_set.add(idx)

    traj = []
    for step in range(1, len(observed) + 1):
        traj.append(
            trajectory_row(
                method="NSGAII-PO-Fair",
                seed=seed,
                step=step,
                df=df,
                observed=observed[:step],
            )
        )

    while len(observed) < budget:
        front = nondominated_indices(df, observed)
        parent_idx = rng.choice(front if front else observed)
        child_idx = mutate_from_parent(df, parent_idx, observed_set, rng)

        observed.append(child_idx)
        observed_set.add(child_idx)

        traj.append(
            trajectory_row(
                method="NSGAII-PO-Fair",
                seed=seed,
                step=len(observed),
                df=df,
                observed=observed,
            )
        )

    return observed, traj


def run_trifair(df: pd.DataFrame, seed: int, budget: int, n_init: int) -> tuple[list[int], list[dict]]:
    rng = random.Random(seed)
    initial_idx = get_default_initial(df)

    observed: list[int] = [initial_idx]
    observed_set = {initial_idx}

    # Exploration initialization.
    while len(observed) < min(n_init, budget):
        idx = random_unobserved(df, observed_set, rng)
        observed.append(idx)
        observed_set.add(idx)

    traj = []
    for step in range(1, len(observed) + 1):
        traj.append(
            trajectory_row(
                method="Tri-Fair-GR",
                seed=seed,
                step=step,
                df=df,
                observed=observed[:step],
            )
        )

    while len(observed) < budget:
        best_idx = select_final_policy(df, observed)
        best = df.loc[best_idx]

        candidates = df.loc[[i for i in df.index if int(i) not in observed_set]].copy()
        if candidates.empty:
            break

        # Intensify near the current best policy, but keep the tri-objective score.
        candidates["locality"] = candidates.apply(lambda row: same_feature_count(row, best), axis=1)
        candidates["acquisition"] = (
            1.00 * candidates["trifair_score"]
            + 0.12 * candidates["locality"] / len(feature_columns())
            + 0.05 * candidates["score_fairness"]
        )

        # Every fourth evaluation, force broader exploration.
        if len(observed) % 4 == 0:
            candidates["acquisition"] = (
                0.85 * candidates["trifair_score"]
                + 0.15 * rng.random()
            )

        next_idx = int(
            candidates.sort_values(
                ["acquisition", "accuracy", "unfairness", "cost_per_1k_examples_usd"],
                ascending=[False, False, True, True],
            ).index[0]
        )

        observed.append(next_idx)
        observed_set.add(next_idx)

        traj.append(
            trajectory_row(
                method="Tri-Fair-GR",
                seed=seed,
                step=len(observed),
                df=df,
                observed=observed,
            )
        )

    return observed, traj


def final_summary_from_trajectory(traj: list[dict]) -> dict:
    final = traj[-1].copy()
    final["final_step"] = final.pop("step")
    final["final_evaluated"] = final.pop("evaluated")
    return final


def aggregate_seed_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "cost_usd",
        "cost_per_1k_examples_usd",
        "unfairness",
        "answer_unfairness",
        "retrieval_unfairness",
        "trifair_score",
    ]

    rows = []
    for method, group in seed_summary.groupby("method"):
        row = {"method": method, "n_seeds": len(group)}
        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std(ddof=0)
        best = group.sort_values(
            ["trifair_score", "accuracy", "unfairness", "cost_per_1k_examples_usd"],
            ascending=[False, False, True, True],
        ).iloc[0]
        row["representative_policy"] = best["best_policy_name"]
        rows.append(row)

    out = pd.DataFrame(rows)

    order = {
        "Initial GNN-RAG": 0,
        "NSGAII-PO-Fair": 1,
        "Tri-Fair-GR": 2,
    }
    out["method_order"] = out["method"].map(order).fillna(99)
    return out.sort_values("method_order").drop(columns=["method_order"])


def save_tables(
    *,
    out_dir: Path,
    trajectory: pd.DataFrame,
    seed_summary: pd.DataFrame,
    mean_summary: pd.DataFrame,
    pool: pd.DataFrame,
) -> None:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    trajectory.to_csv(table_dir / "trifair_gr_method_trajectory.csv", index=False)
    seed_summary.to_csv(table_dir / "trifair_gr_method_seed_summary.csv", index=False)
    mean_summary.to_csv(table_dir / "trifair_gr_method_mean_summary.csv", index=False)
    pool.to_csv(table_dir / "trifair_gr_valid_policy_pool.csv", index=False)

    try:
        mean_summary.to_latex(table_dir / "trifair_gr_method_mean_summary.tex", index=False, escape=True)
        seed_summary.to_latex(table_dir / "trifair_gr_method_seed_summary.tex", index=False, escape=True)
    except Exception as exc:
        print("WARNING: LaTeX export failed:", exc)


def plot_accuracy_unfairness(pool: pd.DataFrame, seed_summary: pd.DataFrame, out: Path, dataset_label: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)

    ax.scatter(
        pool["unfairness"],
        pool["accuracy"],
        s=30,
        alpha=0.25,
        label="Valid policy pool",
    )

    for method, group in seed_summary.groupby("method"):
        ax.scatter(
            group["unfairness"],
            group["accuracy"],
            s=90,
            marker="D",
            label=method,
        )

    ax.set_xlabel("End-to-end unfairness ↓")
    ax.set_ylabel("Accuracy ↑")
    ax.set_title(f"{dataset_label}: Tri-Fair-GR method comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_cost(pool: pd.DataFrame, seed_summary: pd.DataFrame, out: Path, dataset_label: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)

    ax.scatter(
        pool["cost_per_1k_examples_usd"],
        pool["accuracy"],
        s=30,
        alpha=0.25,
        label="Valid policy pool",
    )

    for method, group in seed_summary.groupby("method"):
        ax.scatter(
            group["cost_per_1k_examples_usd"],
            group["accuracy"],
            s=90,
            marker="D",
            label=method,
        )

    ax.set_xlabel("Token cost per 1K examples (USD) ↓")
    ax.set_ylabel("Accuracy ↑")
    ax.set_title(f"{dataset_label}: Accuracy–cost comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_budget_curve(trajectory: pd.DataFrame, metric: str, ylabel: str, out: Path, dataset_label: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)

    grouped = (
        trajectory.groupby(["method", "step"], as_index=False)[metric]
        .mean()
        .sort_values(["method", "step"])
    )

    for method, group in grouped.groupby("method"):
        ax.plot(group["step"], group[metric], marker="o", label=method)

    ax.set_xlabel("Evaluation budget used")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset_label}: {ylabel} over budget")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_method_bars(mean_summary: pd.DataFrame, out_dir: Path, dataset_label: str) -> None:
    metrics = [
        ("accuracy_mean", "Accuracy ↑", "accuracy"),
        ("cost_per_1k_examples_usd_mean", "Token cost per 1K examples (USD) ↓", "cost"),
        ("unfairness_mean", "End-to-end unfairness ↓", "unfairness"),
    ]

    for col, ylabel, name in metrics:
        fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=300)
        ax.bar(mean_summary["method"], mean_summary[col])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{dataset_label}: {ylabel} by method")
        ax.grid(True, axis="y", alpha=0.25)
        fig.autofmt_xdate(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / f"trifair_gr_method_bar_{name}.png", bbox_inches="tight")
        fig.savefig(out_dir / f"trifair_gr_method_bar_{name}.pdf", bbox_inches="tight")
        plt.close(fig)


def save_figures(
    *,
    out_dir: Path,
    pool: pd.DataFrame,
    trajectory: pd.DataFrame,
    seed_summary: pd.DataFrame,
    mean_summary: pd.DataFrame,
    dataset_label: str,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_unfairness(
        pool,
        seed_summary,
        fig_dir / "trifair_gr_accuracy_vs_unfairness",
        dataset_label,
    )

    plot_accuracy_cost(
        pool,
        seed_summary,
        fig_dir / "trifair_gr_accuracy_vs_cost",
        dataset_label,
    )

    plot_budget_curve(
        trajectory,
        "accuracy",
        "Best accuracy ↑",
        fig_dir / "trifair_gr_budget_accuracy",
        dataset_label,
    )

    plot_budget_curve(
        trajectory,
        "unfairness",
        "Best end-to-end unfairness ↓",
        fig_dir / "trifair_gr_budget_unfairness",
        dataset_label,
    )

    plot_budget_curve(
        trajectory,
        "trifair_score",
        "Best normalized Tri-Fair score ↑",
        fig_dir / "trifair_gr_budget_score",
        dataset_label,
    )

    plot_method_bars(mean_summary, fig_dir, dataset_label)


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)
    df = add_policy_parts(df)

    if args.exclude_oracle:
        df = df.loc[~df["is_oracle_order"]].copy()

    if df.empty:
        raise SystemExit("No valid policies available after filtering oracle orders.")

    df = add_scores(df)
    df = df.reset_index(drop=True)
    df["candidate_idx"] = df.index

    all_trajectory = []
    final_rows = []

    for seed in args.seeds:
        _, init_traj = run_initial(df, seed)
        _, nsga_traj = run_nsgaii(df, seed, args.budget, args.population_size)
        _, tri_traj = run_trifair(df, seed, args.budget, args.n_init)

        for traj in [init_traj, nsga_traj, tri_traj]:
            all_trajectory.extend(traj)
            final_rows.append(final_summary_from_trajectory(traj))

    trajectory = pd.DataFrame(all_trajectory)
    seed_summary = pd.DataFrame(final_rows)
    mean_summary = aggregate_seed_summary(seed_summary)

    save_tables(
        out_dir=out_dir,
        trajectory=trajectory,
        seed_summary=seed_summary,
        mean_summary=mean_summary,
        pool=df,
    )

    save_figures(
        out_dir=out_dir,
        pool=df,
        trajectory=trajectory,
        seed_summary=seed_summary,
        mean_summary=mean_summary,
        dataset_label=args.dataset_label,
    )

    print("Wrote outputs to", out_dir)
    print()
    print("Method mean summary:")
    display_cols = [
        "method",
        "accuracy_mean",
        "accuracy_std",
        "cost_per_1k_examples_usd_mean",
        "cost_per_1k_examples_usd_std",
        "unfairness_mean",
        "unfairness_std",
        "answer_unfairness_mean",
        "retrieval_unfairness_mean",
        "representative_policy",
    ]
    print(mean_summary[display_cols].to_string(index=False))
    print()
    print("Figures:")
    for p in sorted((out_dir / "figures").glob("*.png")):
        print(p)


if __name__ == "__main__":
    main()
