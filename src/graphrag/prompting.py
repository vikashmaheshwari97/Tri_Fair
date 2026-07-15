"""Graph-to-text prompt construction utilities for GNN-RAG experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class GraphRAGPromptConfig:
    instruction: str = (
        "Based on the reasoning paths, answer the question. "
        "Use only relevant graph evidence and answer concisely."
    )
    fairness_instruction: str = (
        "Do not infer protected attributes or stereotypes beyond the evidence."
    )
    context_header: str = "Reasoning Paths:"
    question_header: str = "Question:"
    answer_format: str = "Return the final answer only."
    max_paths: int = 8
    include_fairness_instruction: bool = True


def build_graph_context(paths: Sequence[str], *, max_paths: int = 8) -> str:
    selected = [str(path).strip() for path in paths if str(path).strip()]
    selected = selected[: max(0, int(max_paths))]
    return "\n".join(f"- {path}" for path in selected)


def build_graphrag_prompt(
    *,
    question: str,
    reasoning_paths: Sequence[str],
    config: GraphRAGPromptConfig | None = None,
) -> str:
    cfg = config or GraphRAGPromptConfig()
    pieces = [cfg.instruction.strip()]
    if cfg.include_fairness_instruction and cfg.fairness_instruction.strip():
        pieces.append(cfg.fairness_instruction.strip())
    pieces.extend(["", cfg.context_header, build_graph_context(reasoning_paths, max_paths=cfg.max_paths)])
    pieces.extend(["", cfg.question_header, str(question).strip(), "", cfg.answer_format.strip()])
    return "\n".join(pieces).strip()
