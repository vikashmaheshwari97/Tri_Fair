"""Graph-context policy utilities for Tri-Fair-GR Version B.

Version B optimizes graph-context and prompt construction while using a fixed
pretrained GNN-RAG retriever.  The policy controls which retrieved reasoning
paths are shown, how they are ordered, how they are verbalized, and how the LLM
is instructed to use them.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.fairness.graphrag import normalize_answer


@dataclass(frozen=True)
class GraphRAGPolicy:
    name: str
    instruction: str
    fairness_instruction: str = ""
    answer_format: str = "Return all possible answers as a short list, one per line."
    max_paths: int = 5
    path_order: str = "original"
    verbalization: str = "arrow"
    max_context_chars: int = 4000
    include_fairness_instruction: bool = True
    shuffle_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_INSTRUCTIONS: dict[str, str] = {
    "evidence_concise": (
        "Based on the reasoning paths, answer the question. Use only relevant "
        "graph evidence and keep the answer concise."
    ),
    "evidence_complete": (
        "Use the supplied reasoning paths as knowledge evidence. Return every "
        "answer supported by the graph context and avoid unsupported guesses."
    ),
    "conservative": (
        "Answer using only the graph context. If several answers are supported, "
        "return all of them. If the context is insufficient, return the best "
        "supported answer only."
    ),
}

DEFAULT_FAIRNESS_INSTRUCTIONS: dict[str, str] = {
    "none": "",
    "fair_evidence": (
        "Do not infer stereotypes or protected attributes beyond the graph "
        "evidence. Treat entities from all regions and groups consistently."
    ),
    "neutral_context": (
        "Use neutral wording. Ignore irrelevant social, regional, or identity "
        "details unless they are necessary to answer the question."
    ),
}

ANSWER_FORMATS: dict[str, str] = {
    "list": "Return all possible answers as a short list, one answer per line.",
    "final_only": "Return only the final answer text. Do not explain.",
}

SUPPORTED_PATH_ORDERS = {
    "original",
    "shortest_first",
    "longest_first",
    "answer_evidence_first",
    "answer_evidence_last",
    "deterministic_shuffle",
}

SUPPORTED_VERBALIZATIONS = {
    "arrow",
    "compact",
    "sentence",
    "neutral_sentence",
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_reasoning_paths(prompt_input: str) -> list[str]:
    text = str(prompt_input)
    marker = "Reasoning Paths:"
    if marker not in text:
        return []
    after = text.split(marker, 1)[1]
    for stop in ["\n\nQuestion:", "\nQuestion:"]:
        if stop in after:
            after = after.split(stop, 1)[0]
            break
    return [line.strip(" -\t") for line in after.splitlines() if line.strip()]


def split_path(path: str) -> list[str]:
    if "->" in path:
        return [part.strip() for part in path.split("->") if part.strip()]
    if "<SEP>" in path:
        cleaned = path.replace("<PATH>", "").replace("</PATH>", "")
        return [part.strip() for part in cleaned.split("<SEP>") if part.strip()]
    return [part.strip() for part in re.split(r"\s+-\s+", path) if part.strip()]


def path_hop_count(path: str) -> int:
    parts = split_path(path)
    return max(1, len(parts) - 1)


def answer_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"\n|;|,", text) if part.strip()]


def path_contains_gold(path: str, ground_truth: object) -> bool:
    context = normalize_answer(path)
    gold = [normalize_answer(v) for v in answer_list(ground_truth)]
    return any(g and (g == context or g in context) for g in gold)


def deterministic_key(seed: int, text: str) -> str:
    payload = f"{seed}\t{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def order_paths(paths: Sequence[str], *, row: dict[str, Any], policy: GraphRAGPolicy) -> list[str]:
    order = policy.path_order
    if order not in SUPPORTED_PATH_ORDERS:
        raise ValueError(f"Unsupported path_order {order!r}; available: {sorted(SUPPORTED_PATH_ORDERS)}")

    out = list(paths)
    if order == "original":
        return out
    if order == "shortest_first":
        return sorted(out, key=lambda p: (path_hop_count(p), p))
    if order == "longest_first":
        return sorted(out, key=lambda p: (-path_hop_count(p), p))
    if order == "answer_evidence_first":
        return sorted(
            out,
            key=lambda p: (not path_contains_gold(p, row.get("ground_truth", "")), path_hop_count(p), p),
        )
    if order == "answer_evidence_last":
        return sorted(
            out,
            key=lambda p: (path_contains_gold(p, row.get("ground_truth", "")), path_hop_count(p), p),
        )
    if order == "deterministic_shuffle":
        return sorted(out, key=lambda p: deterministic_key(policy.shuffle_seed, p))
    raise AssertionError(order)


def verbalize_path(path: str, *, verbalization: str) -> str:
    if verbalization not in SUPPORTED_VERBALIZATIONS:
        raise ValueError(
            f"Unsupported verbalization {verbalization!r}; available: {sorted(SUPPORTED_VERBALIZATIONS)}"
        )
    parts = split_path(path)
    if verbalization == "arrow":
        return " -> ".join(parts)
    if verbalization == "compact":
        return " | ".join(parts)
    if verbalization == "sentence":
        if len(parts) >= 3:
            subject = parts[0]
            obj = parts[-1]
            relations = " then ".join(parts[1:-1])
            return f"{subject} is connected to {obj} through {relations}."
        return " is connected to ".join(parts) + "."
    if verbalization == "neutral_sentence":
        if len(parts) >= 3:
            subject = parts[0]
            obj = parts[-1]
            relations = ", ".join(parts[1:-1])
            return f"Graph evidence links {subject} to {obj} via relation(s): {relations}."
        return "Graph evidence: " + " ; ".join(parts) + "."
    raise AssertionError(verbalization)


def build_context(paths: Sequence[str], *, policy: GraphRAGPolicy) -> str:
    selected = list(paths)[: max(0, int(policy.max_paths))]
    lines: list[str] = []
    total_chars = 0
    for path in selected:
        line = verbalize_path(path, verbalization=policy.verbalization)
        projected = total_chars + len(line) + 3
        if policy.max_context_chars > 0 and projected > policy.max_context_chars:
            break
        lines.append(f"- {line}")
        total_chars = projected
    return "\n".join(lines)


def build_policy_prompt(row: dict[str, Any], policy: GraphRAGPolicy) -> tuple[str, dict[str, Any]]:
    question = str(row.get("question", "")).strip()
    paths = extract_reasoning_paths(str(row.get("input", "")))
    ordered = order_paths(paths, row=row, policy=policy)
    selected = ordered[: max(0, int(policy.max_paths))]
    context = build_context(ordered, policy=policy)

    pieces = ["[INST] <<SYS>>", "<</SYS>>", policy.instruction.strip()]
    if policy.include_fairness_instruction and policy.fairness_instruction.strip():
        pieces.append(policy.fairness_instruction.strip())
    pieces.extend(
        [
            "",
            "Reasoning Paths:",
            context,
            "",
            "Question:",
            question + ("?" if question and not question.endswith("?") else ""),
            "",
            policy.answer_format.strip(),
            "[/INST]",
        ]
    )
    prompt = "\n".join(pieces).strip()
    diagnostics = {
        "n_original_paths": len(paths),
        "n_selected_paths": len(selected),
        "policy_name": policy.name,
        "path_order": policy.path_order,
        "verbalization": policy.verbalization,
        "max_paths": policy.max_paths,
        "max_context_chars": policy.max_context_chars,
    }
    return prompt, diagnostics


def policy_from_dict(payload: dict[str, Any]) -> GraphRAGPolicy:
    return GraphRAGPolicy(**payload)


def read_policies(path: str | Path) -> list[GraphRAGPolicy]:
    policies = []
    for row in load_jsonl(path):
        policies.append(policy_from_dict(row))
    return policies


def make_policy_name(parts: dict[str, Any]) -> str:
    raw = "_".join(f"{key}-{value}" for key, value in parts.items())
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return raw[:160]


def generate_policy_grid(*, size: str = "compact", seed: int = 42) -> list[GraphRAGPolicy]:
    if size not in {"compact", "full"}:
        raise ValueError("size must be 'compact' or 'full'")

    if size == "compact":
        instruction_keys = ["evidence_concise", "conservative"]
        fairness_keys = ["none", "fair_evidence"]
        answer_keys = ["list"]
        max_paths_values = [2, 5, 8]
        path_orders = ["original", "shortest_first", "answer_evidence_first"]
        verbalizations = ["arrow", "neutral_sentence"]
    else:
        instruction_keys = list(DEFAULT_INSTRUCTIONS)
        fairness_keys = list(DEFAULT_FAIRNESS_INSTRUCTIONS)
        answer_keys = list(ANSWER_FORMATS)
        max_paths_values = [1, 2, 3, 5, 8]
        path_orders = [
            "original",
            "shortest_first",
            "longest_first",
            "answer_evidence_first",
            "answer_evidence_last",
            "deterministic_shuffle",
        ]
        verbalizations = ["arrow", "compact", "sentence", "neutral_sentence"]

    policies: list[GraphRAGPolicy] = []
    for instr_key in instruction_keys:
        for fair_key in fairness_keys:
            for answer_key in answer_keys:
                for max_paths in max_paths_values:
                    for path_order in path_orders:
                        for verbalization in verbalizations:
                            name = make_policy_name(
                                {
                                    "i": instr_key,
                                    "f": fair_key,
                                    "a": answer_key,
                                    "p": max_paths,
                                    "o": path_order,
                                    "v": verbalization,
                                }
                            )
                            policies.append(
                                GraphRAGPolicy(
                                    name=name,
                                    instruction=DEFAULT_INSTRUCTIONS[instr_key],
                                    fairness_instruction=DEFAULT_FAIRNESS_INSTRUCTIONS[fair_key],
                                    answer_format=ANSWER_FORMATS[answer_key],
                                    max_paths=max_paths,
                                    path_order=path_order,
                                    verbalization=verbalization,
                                    include_fairness_instruction=fair_key != "none",
                                    shuffle_seed=seed,
                                )
                            )
    return policies
