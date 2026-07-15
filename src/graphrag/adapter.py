"""Adapters for reading GNN-RAG outputs into Tri-Fair-style dataframes."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.fairness.graphrag import answer_hit


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def _coerce_answer(value: object) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return "; ".join(str(v) for v in parsed)
            except Exception:
                pass
        return text
    if isinstance(value, (list, tuple, set)):
        return "; ".join(str(v) for v in value)
    return str(value)


def _extract_reasoning_path_block(prompt_input: str) -> list[str]:
    text = str(prompt_input)
    marker = "Reasoning Paths:"
    if marker not in text:
        return []
    after = text.split(marker, 1)[1]
    for stop in ["\n\nQuestion:", "\nQuestion:"]:
        if stop in after:
            after = after.split(stop, 1)[0]
            break
    lines = [line.strip() for line in after.splitlines()]
    return [line for line in lines if line]


def _average_path_length(paths: Iterable[str]) -> float:
    lengths = []
    for path in paths:
        if "->" in path:
            parts = [p.strip() for p in path.split("->") if p.strip()]
        else:
            parts = [p.strip() for p in re.split(r"\s+-\s+", path) if p.strip()]
        if parts:
            lengths.append(len(parts))
    if not lengths:
        return 0.0
    return float(sum(lengths) / len(lengths))


def build_graphrag_frame(
    predictions_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    id_col: str = "id",
    group_col: str = "protected_group",
) -> pd.DataFrame:
    rows = []
    for row in load_jsonl(predictions_path):
        paths = _extract_reasoning_path_block(str(row.get("input", "")))
        y_true = _coerce_answer(row.get("ground_truth", ""))
        y_pred = _coerce_answer(row.get("prediction", ""))
        hit = answer_hit(y_true, y_pred)
        rows.append(
            {
                id_col: str(row.get("id", "")),
                "question": row.get("question", ""),
                "target": y_true,
                "prediction": y_pred,
                "input": row.get("input", ""),
                "answer_hit": hit,
                "n_reasoning_paths": int(len(paths)),
                "avg_reasoning_path_length": _average_path_length(paths),
                "graph_context_chars": int(sum(len(path) for path in paths)),
                "retrieval_hit": row.get("retrieval_hit", hit),
            }
        )

    frame = pd.DataFrame(rows)
    if metadata_path is not None:
        meta_path = Path(metadata_path)
        meta = pd.read_parquet(meta_path) if meta_path.suffix.lower() == ".parquet" else pd.read_csv(meta_path)
        meta[id_col] = meta[id_col].astype(str)
        frame[id_col] = frame[id_col].astype(str)
        frame = frame.merge(meta, on=id_col, how="left", validate="many_to_one")

    if group_col not in frame:
        frame[group_col] = "unknown"

    return frame


def write_frame(frame: pd.DataFrame, out_prefix: str | Path) -> None:
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out.with_suffix(".csv"), index=False)
    frame.to_parquet(out.with_suffix(".parquet"), index=False)
