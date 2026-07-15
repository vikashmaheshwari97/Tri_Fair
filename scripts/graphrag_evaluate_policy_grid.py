#!/usr/bin/env python
"""Evaluate all Version-B policy prediction files and compute Pareto inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.fairness.graphrag import compute_graphrag_fairness
from src.graphrag.adapter import build_graphrag_frame, write_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--group-col", default="protected_group")
    parser.add_argument("--min-group-count", type=int, default=5)
    parser.add_argument("--lambda-retrieval", type=float, default=0.4)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.predictions_root)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    group_rows = []
    for pred in sorted(root.glob("*/predictions.jsonl")):
        policy_name = pred.parent.name
        try:
            frame = build_graphrag_frame(
                pred,
                metadata_path=args.metadata,
                group_col=args.group_col,
            )
        except Exception as exc:
            print("SKIP", pred, exc)
            continue

        result = compute_graphrag_fairness(
            y_true=frame["target"].astype(str).tolist(),
            y_pred=frame["prediction"].astype(str).tolist(),
            metadata=frame,
            min_group_count=args.min_group_count,
            group_column=args.group_col,
            retrieval_metric_col="retrieval_hit",
            lambda_retrieval=args.lambda_retrieval,
        )
        policy_payload = {}
        if "policy" in frame.columns and len(frame):
            first = frame["policy"].iloc[0]
            if isinstance(first, dict):
                policy_payload = first
            else:
                try:
                    policy_payload = json.loads(first)
                except Exception:
                    policy_payload = {}

        mean_answer = float(pd.to_numeric(frame["answer_hit"], errors="coerce").mean())
        mean_retrieval = float(pd.to_numeric(frame["retrieval_hit"], errors="coerce").mean())
        mean_paths = float(pd.to_numeric(frame["n_reasoning_paths"], errors="coerce").mean())
        mean_context_chars = float(pd.to_numeric(frame["graph_context_chars"], errors="coerce").mean())
        mean_prompt_chars = float(frame["input"].astype(str).str.len().mean())
        cost_proxy = mean_context_chars

        row = {
            "policy_name": policy_name,
            "n_rows": int(len(frame)),
            "answer_mean": mean_answer,
            "retrieval_mean": mean_retrieval,
            "loss": result.loss,
            "ready": result.ready,
            "answer_unfairness": result.diagnostics.get("answer_unfairness"),
            "retrieval_unfairness": result.diagnostics.get("retrieval_unfairness"),
            "cost_proxy_context_chars": cost_proxy,
            "mean_prompt_chars": mean_prompt_chars,
            "mean_paths": mean_paths,
            "max_paths": policy_payload.get("max_paths"),
            "path_order": policy_payload.get("path_order"),
            "verbalization": policy_payload.get("verbalization"),
            "include_fairness_instruction": policy_payload.get("include_fairness_instruction"),
        }
        rows.append(row)

        policy_out = outdir / "policy_frames" / policy_name
        policy_out.mkdir(parents=True, exist_ok=True)
        write_frame(frame, policy_out / "frame")
        (policy_out / "fairness_summary.json").write_text(
            json.dumps(
                {"summary": row, "support": result.support, "diagnostics": result.diagnostics},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        for component, key in [("answer", "answer_group_performance"), ("retrieval", "retrieval_group_performance")]:
            for group, perf in result.diagnostics.get(key, {}).items():
                group_rows.append(
                    {
                        "policy_name": policy_name,
                        "component": component,
                        "group": group,
                        "performance": perf,
                        "support": result.support.get(group, 0),
                    }
                )

    summary = pd.DataFrame(rows).sort_values(["loss", "cost_proxy_context_chars"], ascending=[True, True])
    groups = pd.DataFrame(group_rows)
    summary.to_csv(outdir / "policy_eval_summary.csv", index=False)
    groups.to_csv(outdir / "policy_group_metrics.csv", index=False)
    print(outdir / "policy_eval_summary.csv")
    if not summary.empty:
        print(summary.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
