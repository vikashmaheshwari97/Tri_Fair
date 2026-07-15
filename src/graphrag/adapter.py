"""Adapters for reading GNN-RAG outputs into Tri-Fair-style dataframes."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.fairness.graphrag import answer_hit, normalize_answer


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


def _answer_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v) for v in parsed if str(v).strip()]
        except Exception:
            pass

    return [part.strip() for part in re.split(r"\n|;|,", text) if part.strip()]


def _extract_reasoning_path_block(prompt_input: str) -> list[str]:
    """Extract graph path lines from a GNN-RAG prompt input when present."""
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


def evidence_answer_hit(ground_truth: object, paths: Iterable[str]) -> float:
    """Approximate retrieval-evidence hit from verbalized reasoning paths.

    Returns 1 when any gold answer string appears in the retrieved reasoning
    paths. This separates graph evidence retrieval from final LLM answer
    correctness.
    """
    context = normalize_answer(" ".join(str(path) for path in paths))
    if not context:
        return 0.0

    gold = [normalize_answer(v) for v in _answer_list(ground_truth)]
    gold = [v for v in gold if v]
    if not gold:
        return 0.0

    return float(any(g == context or g in context for g in gold))


def build_graphrag_frame(
    predictions_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    id_col: str = "id",
    group_col: str = "protected_group",
) -> pd.DataFrame:
    """Convert GNN-RAG predictions.jsonl into a dataframe.

    Output columns include:
    - answer_hit: final LLM answer correctness;
    - retrieval_hit: whether the retrieved reasoning paths contain a gold answer.
    """
    rows = []

    for row in load_jsonl(predictions_path):
        paths = _extract_reasoning_path_block(str(row.get("input", "")))
        y_true = _coerce_answer(row.get("ground_truth", ""))
        y_pred = _coerce_answer(row.get("prediction", ""))

        final_answer_hit = answer_hit(y_true, y_pred)
        retrieval_hit = float(
            row.get(
                "retrieval_hit",
                evidence_answer_hit(row.get("ground_truth", ""), paths),
            )
        )

        rows.append(
            {
                id_col: str(row.get("id", "")),
                "question": row.get("question", ""),
                "target": y_true,
                "prediction": y_pred,
                "input": row.get("input", ""),
                "answer_hit": final_answer_hit,
                "retrieval_hit": retrieval_hit,
                "n_reasoning_paths": int(len(paths)),
                "avg_reasoning_path_length": _average_path_length(paths),
                "graph_context_chars": int(sum(len(path) for path in paths)),
            }
        )

    frame = pd.DataFrame(rows)

    if metadata_path is not None:
        meta_path = Path(metadata_path)
        if meta_path.suffix.lower() == ".parquet":
            meta = pd.read_parquet(meta_path)
        else:
            meta = pd.read_csv(meta_path)

        meta[id_col] = meta[id_col].astype(str)
        frame[id_col] = frame[id_col].astype(str)
        frame = frame.merge(meta, on=id_col, how="left", validate="many_to_one")

    if group_col not in frame:
        frame[group_col] = "unknown"

    return frame


def write_frame(frame: pd.DataFrame, out_prefix: str | Path) -> None:
    """Write CSV and Parquet versions of a dataframe."""
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out.with_suffix(".csv"), index=False)
    frame.to_parquet(out.with_suffix(".parquet"), index=False)
