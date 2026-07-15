#!/usr/bin/env python
"""Materialize Version-B graph-context/prompt-policy datasets.

Each policy gets a GNN-RAG-compatible JSONL file containing modified `input`
prompts.  These files are then passed to `scripts.graphrag_run_policy_vllm`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.graphrag.policy import build_policy_prompt, load_jsonl, read_policies, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--policies", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--policy-name", default=None, help="Materialize only this policy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_rows = load_jsonl(args.base_predictions)
    if args.limit and args.limit > 0:
        base_rows = base_rows[: args.limit]

    policies = read_policies(args.policies)
    if args.policy_name:
        policies = [policy for policy in policies if policy.name == args.policy_name]
        if not policies:
            raise SystemExit(f"No policy named {args.policy_name!r}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for policy in policies:
        rows = []
        for row in base_rows:
            prompt, diagnostics = build_policy_prompt(row, policy)
            rows.append(
                {
                    "id": str(row.get("id", "")),
                    "question": row.get("question", ""),
                    "ground_truth": row.get("ground_truth", []),
                    "input": prompt,
                    "policy_name": policy.name,
                    "policy": policy.to_dict(),
                    "policy_diagnostics": diagnostics,
                }
            )

        policy_dir = out_dir / policy.name
        policy_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(policy_dir / "prompts.jsonl", rows)

    print(out_dir, "policies=", len(policies), "rows_per_policy=", len(base_rows))


if __name__ == "__main__":
    main()
