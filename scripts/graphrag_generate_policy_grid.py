#!/usr/bin/env python
"""Generate candidate graph-context/prompt policies for Tri-Fair-GR Version B."""

from __future__ import annotations

import argparse

from src.graphrag.policy import generate_policy_grid, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["compact", "full"], default="compact")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = generate_policy_grid(size=args.size, seed=args.seed)
    write_jsonl(args.out, [policy.to_dict() for policy in policies])
    print(args.out, "policies=", len(policies))


if __name__ == "__main__":
    main()
