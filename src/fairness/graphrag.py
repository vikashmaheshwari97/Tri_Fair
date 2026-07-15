"""Fairness objective for a future Tri-Fair × GraphRAG/KGQA setup."""

from __future__ import annotations

import re
import string
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.fairness.base import FairnessMetricResult, safe_rms


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(value: object) -> str:
    """Normalize an answer string for lightweight KGQA exact/contains matching."""
    text = str(value).lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _as_answer_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [normalize_answer(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"\n|;|,", text)
    out = [normalize_answer(part) for part in parts if normalize_answer(part)]
    return out or [normalize_answer(text)]


def answer_hit(y_true: object, y_pred: object) -> float:
    """Return 1.0 if any gold answer is matched by the prediction."""
    gold = _as_answer_list(y_true)
    pred_text = normalize_answer(y_pred)
    if not gold or not pred_text:
        return 0.0
    return float(any(g == pred_text or g in pred_text for g in gold))


def _max_pairwise_gap(values: dict[str, float]) -> float:
    vals = [float(v) for v in values.values() if np.isfinite(v)]
    if len(vals) < 2:
        return float("nan")
    return float(max(vals) - min(vals))


def _group_means(
    values: np.ndarray,
    groups: Sequence[object],
    *,
    min_group_count: int,
) -> tuple[dict[str, float], dict[str, int]]:
    frame = pd.DataFrame({"group": [str(g) for g in groups], "value": values})
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[np.isfinite(frame["value"].to_numpy(dtype=float))]
    support = frame.groupby("group").size().astype(int).to_dict()
    valid_groups = {g for g, n in support.items() if n >= min_group_count}
    means = (
        frame[frame["group"].isin(valid_groups)]
        .groupby("group")["value"]
        .mean()
        .astype(float)
        .to_dict()
    )
    return means, support


def compute_graphrag_fairness(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metadata: pd.DataFrame,
    *,
    min_group_count: int,
    group_column: str = "protected_group",
    retrieval_metric_col: str = "retrieval_hit",
    lambda_retrieval: float = 0.4,
    answer_gap: str = "max_gap",
    retrieval_gap: str = "max_gap",
    **_: Any,
) -> FairnessMetricResult:
    """Compute end-to-end GraphRAG unfairness."""
    if group_column not in metadata.columns:
        raise ValueError(
            f"GraphRAG fairness requires metadata column {group_column!r}; "
            f"available columns: {list(metadata.columns)}"
        )

    groups = metadata[group_column].astype(str).tolist()
    answer_values = np.asarray(
        [answer_hit(gold, pred) for gold, pred in zip(y_true, y_pred)],
        dtype=float,
    )

    answer_means, support = _group_means(
        answer_values,
        groups,
        min_group_count=min_group_count,
    )

    def collapse(means: dict[str, float], mode: str) -> float:
        if len(means) < 2:
            return float("nan")
        if mode == "max_gap":
            return _max_pairwise_gap(means)
        if mode == "rms_gap":
            overall = float(np.mean(list(means.values())))
            return safe_rms([value - overall for value in means.values()])
        raise ValueError(f"Unsupported gap mode: {mode!r}")

    answer_component = collapse(answer_means, answer_gap)

    retrieval_component = float("nan")
    retrieval_means: dict[str, float] = {}
    if retrieval_metric_col in metadata.columns:
        retrieval_values = pd.to_numeric(
            metadata[retrieval_metric_col], errors="coerce"
        ).to_numpy(dtype=float)
        retrieval_means, _ = _group_means(
            retrieval_values,
            groups,
            min_group_count=min_group_count,
        )
        retrieval_component = collapse(retrieval_means, retrieval_gap)

    has_retrieval = np.isfinite(retrieval_component)
    has_answer = np.isfinite(answer_component)

    if has_retrieval and has_answer:
        lam = float(np.clip(lambda_retrieval, 0.0, 1.0))
        loss = lam * retrieval_component + (1.0 - lam) * answer_component
    elif has_answer:
        loss = answer_component
    elif has_retrieval:
        loss = retrieval_component
    else:
        loss = float("nan")

    ready = (
        np.isfinite(loss)
        and sum(1 for n in support.values() if n >= min_group_count) >= 2
    )

    diagnostics = {
        "group_column": group_column,
        "retrieval_metric_col": retrieval_metric_col
        if retrieval_metric_col in metadata.columns
        else None,
        "lambda_retrieval": float(lambda_retrieval),
        "answer_group_performance": answer_means,
        "retrieval_group_performance": retrieval_means,
        "answer_unfairness": answer_component,
        "retrieval_unfairness": retrieval_component,
        "loss_formula": (
            "lambda*retrieval_unfairness + (1-lambda)*answer_unfairness"
            if has_retrieval and has_answer
            else "answer_unfairness_or_retrieval_unfairness"
        ),
    }

    return FairnessMetricResult(
        loss=float(loss) if np.isfinite(loss) else float("nan"),
        ready=bool(ready),
        diagnostics=diagnostics,
        support={str(k): int(v) for k, v in support.items()},
    )
