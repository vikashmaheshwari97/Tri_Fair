#!/usr/bin/env python
"""Offline fairness diagnostics for GNN-RAG prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.fairness.graphrag import compute_graphrag_fairness
from src.graphrag.adapter import build_graphrag_frame, write_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--group-col", default="protected_group")
    parser.add_argument("--retrieval-metric-col", default="retrieval_hit")
    parser.add_argument("--min-group-count", type=int, default=5)
    parser.add_argument("--lambda-retrieval", type=float, default=0.4)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    frame = build_graphrag_frame(
        args.predictions,
        metadata_path=args.metadata,
        id_col=args.id_col,
        group_col=args.group_col,
    )
    write_frame(frame, outdir / "graphrag_predictions_frame")

    result = compute_graphrag_fairness(
        y_true=frame["target"].astype(str).tolist(),
        y_pred=frame["prediction"].astype(str).tolist(),
        metadata=frame,
        min_group_count=args.min_group_count,
        group_column=args.group_col,
        retrieval_metric_col=args.retrieval_metric_col,
        lambda_retrieval=args.lambda_retrieval,
    )

    summary = {
        "loss": result.loss,
        "ready": result.ready,
        "support": result.support,
        "diagnostics": result.diagnostics,
    }
    (outdir / "graphrag_fairness_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rows = []
    for group, value in result.diagnostics.get("answer_group_performance", {}).items():
        rows.append({"component": "answer", "group": group, "performance": value})
    for group, value in result.diagnostics.get("retrieval_group_performance", {}).items():
        rows.append({"component": "retrieval", "group": group, "performance": value})
    pd.DataFrame(rows).to_csv(outdir / "graphrag_group_metrics.csv", index=False)

    print("GraphRAG diagnostics written to:", outdir.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
