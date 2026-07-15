#!/usr/bin/env python
"""Summarize GraphRAG diagnostic folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="Folder prefix, e.g. webqsp_ or cwq_")
    parser.add_argument("--root", default="analysis/output/graphrag_diagnostics")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    rows = []
    for d in sorted(root.glob(f"{args.prefix}*")):
        p = d / "graphrag_fairness_summary.json"
        if not p.is_file():
            continue

        data = json.loads(p.read_text(encoding="utf-8"))
        diag = data["diagnostics"]
        answer = diag["answer_group_performance"]
        retrieval = diag["retrieval_group_performance"]

        rows.append(
            {
                "run": d.name,
                "loss": data["loss"],
                "answer_unfairness": diag["answer_unfairness"],
                "retrieval_unfairness": diag["retrieval_unfairness"],
                "best_answer_group": max(answer, key=answer.get),
                "best_answer": max(answer.values()),
                "worst_answer_group": min(answer, key=answer.get),
                "worst_answer": min(answer.values()),
                "best_retrieval_group": max(retrieval, key=retrieval.get),
                "best_retrieval": max(retrieval.values()),
                "worst_retrieval_group": min(retrieval, key=retrieval.get),
                "worst_retrieval": min(retrieval.values()),
            }
        )

    if not rows:
        raise SystemExit(f"No diagnostic summaries found under {root} with prefix {args.prefix!r}")

    df = pd.DataFrame(rows).sort_values("loss")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(out)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
