#!/usr/bin/env python
"""Evaluate Version-B policy prediction files and compute Tri-Fair-GR objectives.

Objectives:
  quality    = answer_mean, maximize
  cost       = token-dollar cost, minimize
  unfairness = lambda * retrieval_unfairness + (1-lambda) * answer_unfairness, minimize
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

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

    parser.add_argument(
        "--tokenizer-path",
        default="",
        help="Local HF tokenizer/model path. If omitted, token columns are estimated from text length.",
    )
    parser.add_argument("--input-cost-per-mtok", type=float, default=0.11)
    parser.add_argument("--output-cost-per-mtok", type=float, default=0.41)
    return parser.parse_args()


def load_tokenizer(tokenizer_path: str):
    if not tokenizer_path:
        return None

    transformers = importlib.import_module("transformers")
    return transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )


def count_tokens(texts: list[str], tokenizer: Any | None) -> list[int]:
    if tokenizer is None:
        # Conservative fallback for diagnostics only. The paper table should use tokenizer counts.
        return [max(1, int(round(len(str(text)) / 4.0))) for text in texts]

    counts: list[int] = []
    for text in texts:
        encoded = tokenizer(
            str(text),
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        counts.append(len(encoded["input_ids"]))
    return counts


def policy_payload_from_frame(frame: pd.DataFrame) -> dict:
    if "policy" not in frame.columns or len(frame) == 0:
        return {}

    first = frame["policy"].iloc[0]
    if isinstance(first, dict):
        return first

    try:
        return json.loads(first)
    except Exception:
        return {}


def main() -> None:
    args = parse_args()
    root = Path(args.predictions_root)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer_path)
    tokenizer_mode = "hf_tokenizer" if tokenizer is not None else "chars_div_4_estimate"
    print("tokenizer_mode=", tokenizer_mode)
    if args.tokenizer_path:
        print("tokenizer_path=", args.tokenizer_path)

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

        input_texts = frame["input"].astype(str).tolist()
        output_texts = frame["prediction"].astype(str).tolist()

        input_token_counts = count_tokens(input_texts, tokenizer)
        output_token_counts = count_tokens(output_texts, tokenizer)

        frame["input_tokens"] = input_token_counts
        frame["output_tokens"] = output_token_counts
        frame["token_cost_usd"] = (
            frame["input_tokens"] / 1_000_000.0 * args.input_cost_per_mtok
            + frame["output_tokens"] / 1_000_000.0 * args.output_cost_per_mtok
        )

        result = compute_graphrag_fairness(
            y_true=frame["target"].astype(str).tolist(),
            y_pred=frame["prediction"].astype(str).tolist(),
            metadata=frame,
            min_group_count=args.min_group_count,
            group_column=args.group_col,
            retrieval_metric_col="retrieval_hit",
            lambda_retrieval=args.lambda_retrieval,
        )

        policy_payload = policy_payload_from_frame(frame)

        mean_answer = float(pd.to_numeric(frame["answer_hit"], errors="coerce").mean())
        mean_retrieval = float(pd.to_numeric(frame["retrieval_hit"], errors="coerce").mean())
        mean_paths = float(pd.to_numeric(frame["n_reasoning_paths"], errors="coerce").mean())
        mean_context_chars = float(pd.to_numeric(frame["graph_context_chars"], errors="coerce").mean())
        mean_prompt_chars = float(frame["input"].astype(str).str.len().mean())

        total_input_tokens = int(frame["input_tokens"].sum())
        total_output_tokens = int(frame["output_tokens"].sum())
        mean_input_tokens = float(frame["input_tokens"].mean())
        mean_output_tokens = float(frame["output_tokens"].mean())

        input_cost_usd = total_input_tokens / 1_000_000.0 * args.input_cost_per_mtok
        output_cost_usd = total_output_tokens / 1_000_000.0 * args.output_cost_per_mtok
        cost_usd = input_cost_usd + output_cost_usd

        n_rows = int(len(frame))
        cost_per_1k_examples_usd = cost_usd / max(1, n_rows) * 1000.0

        row = {
            "policy_name": policy_name,
            "n_rows": n_rows,

            # Objective 1: quality, maximize
            "quality": mean_answer,
            "answer_mean": mean_answer,
            "retrieval_mean": mean_retrieval,

            # Objective 2: cost, minimize
            "cost_usd": cost_usd,
            "cost_per_1k_examples_usd": cost_per_1k_examples_usd,
            "input_cost_usd": input_cost_usd,
            "output_cost_usd": output_cost_usd,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "mean_input_tokens": mean_input_tokens,
            "mean_output_tokens": mean_output_tokens,
            "input_cost_per_mtok": args.input_cost_per_mtok,
            "output_cost_per_mtok": args.output_cost_per_mtok,
            "tokenizer_mode": tokenizer_mode,

            # Keep old diagnostics for comparison
            "cost_proxy_context_chars": mean_context_chars,
            "mean_prompt_chars": mean_prompt_chars,
            "mean_paths": mean_paths,

            # Objective 3: unfairness, minimize
            "unfairness": result.loss,
            "loss": result.loss,
            "ready": result.ready,
            "answer_unfairness": result.diagnostics.get("answer_unfairness"),
            "retrieval_unfairness": result.diagnostics.get("retrieval_unfairness"),

            # Policy knobs
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
                {
                    "summary": row,
                    "support": result.support,
                    "diagnostics": result.diagnostics,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        for component, key in [
            ("answer", "answer_group_performance"),
            ("retrieval", "retrieval_group_performance"),
        ]:
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

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["unfairness", "cost_usd"], ascending=[True, True])

    groups = pd.DataFrame(group_rows)

    summary.to_csv(outdir / "policy_eval_summary.csv", index=False)
    groups.to_csv(outdir / "policy_group_metrics.csv", index=False)

    print(outdir / "policy_eval_summary.csv")
    if not summary.empty:
        display_cols = [
            "policy_name",
            "quality",
            "cost_usd",
            "cost_per_1k_examples_usd",
            "unfairness",
            "answer_unfairness",
            "retrieval_unfairness",
            "mean_input_tokens",
            "mean_output_tokens",
            "max_paths",
            "path_order",
            "verbalization",
        ]
        print(summary[display_cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
