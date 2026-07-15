#!/usr/bin/env python
"""Run LLM inference for one materialized Tri-Fair-GR policy prompt file.

This script is intended for Rocket/Linux GPU nodes.  It uses vLLM directly and
writes GNN-RAG-compatible `predictions.jsonl` rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-path", default="../models/Qwen3-30B")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_processed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    processed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                processed.add(str(json.loads(line).get("id", "")))
    return processed


def main() -> None:
    args = parse_args()

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:  # pragma: no cover - cluster-only path
        raise RuntimeError(
            "vLLM is required for this script. Run it on a Linux GPU node, not Windows."
        ) from exc

    prompts = load_jsonl(args.prompts)
    out_path = Path(args.out)
    if args.force and out_path.exists():
        out_path.unlink()

    processed = load_processed(out_path)
    todo = [row for row in prompts if str(row.get("id", "")) not in processed]
    print("prompts=", len(prompts), "processed=", len(processed), "todo=", len(todo))
    if not todo:
        return

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_output_tokens,
    )

    for start in range(0, len(todo), args.batch_size):
        batch = todo[start : start + args.batch_size]
        generations = llm.generate([str(row["input"]) for row in batch], sampling)
        out_rows = []
        for row, gen in zip(batch, generations):
            text = gen.outputs[0].text.strip() if gen.outputs else ""
            out_rows.append(
                {
                    "id": row.get("id", ""),
                    "question": row.get("question", ""),
                    "prediction": text,
                    "ground_truth": row.get("ground_truth", []),
                    "input": row.get("input", ""),
                    "policy_name": row.get("policy_name", ""),
                    "policy": row.get("policy", {}),
                    "policy_diagnostics": row.get("policy_diagnostics", {}),
                }
            )
        append_jsonl(out_path, out_rows)
        print("wrote", min(start + len(batch), len(todo)), "/", len(todo))


if __name__ == "__main__":
    main()
