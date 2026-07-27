"""Adaptive, dataset-aware variation for Tri-Fair v6.

The operator uses development data only.  It never reads holdout metrics.
"""

from __future__ import annotations

from typing import Any

from promptolution.utils.capo_utils import build_few_shot_examples
from promptolution.utils.formatting import extract_from_tag
from promptolution.utils.prompt import Prompt

from src.fairness.v4_variation import generate_v4_challengers
from src.fairness.v5_variation import (
    MODE_WORD_LIMIT,
    _MUTATION_TEMPLATE,
    _anchor_for_mode,
    _deduplicate_examples,
    _diverse_partner,
    _ensure_output_contract,
    _mutate_examples_v5,
    _normalise_instruction,
    _repair_prompt,
    _responses,
    _word_count,
    goal_for_mode,
)


def _quality_enrichment(
    optimizer: Any,
    instruction: str,
    examples: list[str],
    mode: str,
) -> list[str]:
    """Add one extra verified example for quality-sensitive multiclass search."""
    if mode != "quality":
        return examples
    upper = int(getattr(optimizer, "upper_shots", 0))
    if len(examples) >= upper:
        return examples
    metric = str(getattr(optimizer.task, "fairness_metric_name", ""))
    if not metric.startswith("bios_tpr_gap"):
        return examples
    additions = build_few_shot_examples(
        instruction=instruction,
        num_examples=1,
        optimizer=optimizer,
    )
    return _deduplicate_examples([*examples, *list(additions)], upper)


def generate_v6_challengers(optimizer: Any) -> list[Prompt]:
    """Generate a budget-aware portfolio of objective-directed challengers."""
    count = int(optimizer.v6_challenger_count())
    if count <= 0:
        return []

    modes = list(optimizer.get_v6_modes(count))
    if len(modes) != count:
        raise RuntimeError("Tri-Fair-v6 mode scheduler returned the wrong count")

    common_blocks = list(
        optimizer._get_common_blocks(optimizer.incumbents) or []
    )

    parent_pairs: list[tuple[Prompt, Prompt]] = []
    guide_records: list[dict[str, Any]] = []
    crossover_prompts: list[str] = []
    inherited: list[list[str]] = []

    for mode in modes:
        anchor, record = _anchor_for_mode(
            optimizer,
            "explore" if mode == "explore" else mode,
            common_blocks,
        )
        partner = _diverse_partner(optimizer, anchor)
        parent_pairs.append((anchor, partner))
        guide_records.append(record)
        crossover_prompts.append(
            optimizer.crossover_template
            .replace("<mother>", anchor.instruction)
            .replace("<father>", partner.instruction)
            .strip()
        )

        combined = _deduplicate_examples(
            [*list(anchor.few_shots), *list(partner.few_shots)],
            int(optimizer.upper_shots),
        )
        if mode == "cost":
            inherited.append([])
        else:
            target = min(
                len(combined),
                int(optimizer.upper_shots),
                max(1, (len(anchor.few_shots) + len(partner.few_shots)) // 2),
            )
            inherited.append(combined[:target])

    crossover_responses = _responses(
        optimizer.meta_llm.get_response(crossover_prompts),
        count,
    )

    children: list[Prompt] = []
    for response, (anchor, _), examples in zip(
        crossover_responses,
        parent_pairs,
        inherited,
    ):
        instruction = _normalise_instruction(
            extract_from_tag(response, "<prompt>", "</prompt>")
        )
        children.append(
            Prompt(
                _ensure_output_contract(instruction or anchor.instruction),
                examples,
            )
        )

    mutation_prompts = [
        _MUTATION_TEMPLATE
        .replace("<task>", str(optimizer.task.task_description))
        .replace(
            "<labels>",
            ", ".join(str(value) for value in optimizer.task.classes),
        )
        .replace("<instruction>", child.instruction)
        .replace("<goal>", goal_for_mode(optimizer, mode))
        .replace(
            "<diagnostics>",
            str(record.get("diagnostics") or {}),
        )
        for child, mode, record in zip(children, modes, guide_records)
    ]
    mutation_responses = _responses(
        optimizer.meta_llm.get_response(mutation_prompts),
        count,
    )

    proposed: list[str] = []
    for child, response in zip(children, mutation_responses):
        instruction = _normalise_instruction(
            extract_from_tag(response, "<prompt>", "</prompt>")
        )
        proposed.append(
            _ensure_output_contract(instruction or child.instruction)
        )

    existing_instructions = {
        _normalise_instruction(prompt.instruction).casefold()
        for prompt in list(getattr(optimizer, "prompts", []))
    }
    repair_positions = [
        index
        for index, (instruction, mode) in enumerate(zip(proposed, modes))
        if _word_count(instruction) > MODE_WORD_LIMIT[mode]
        or instruction.casefold() in existing_instructions
    ]
    if repair_positions:
        repaired = _responses(
            optimizer.meta_llm.get_response(
                [
                    _repair_prompt(
                        optimizer,
                        proposed[index],
                        modes[index],
                    )
                    for index in repair_positions
                ]
            ),
            len(repair_positions),
        )
        for index, response in zip(repair_positions, repaired):
            instruction = _normalise_instruction(
                extract_from_tag(response, "<prompt>", "</prompt>")
            )
            if instruction:
                proposed[index] = _ensure_output_contract(instruction)

    emitted = {
        prompt.construct_prompt().strip().casefold()
        for prompt in list(getattr(optimizer, "prompts", []))
    }
    challengers: list[Prompt] = []
    mode_by_prompt: dict[str, str] = {}

    for child, instruction, mode, record in zip(
        children,
        proposed,
        modes,
        guide_records,
    ):
        examples = _mutate_examples_v5(
            optimizer,
            child,
            instruction,
            mode,
            dict(record.get("diagnostics") or {}),
        )
        examples = _quality_enrichment(
            optimizer,
            instruction,
            examples,
            mode,
        )
        candidate = Prompt(instruction, examples)
        key = candidate.construct_prompt().strip().casefold()
        if not key or key in emitted:
            continue
        emitted.add(key)
        challengers.append(candidate)
        mode_by_prompt[key] = mode

    if not challengers:
        return generate_v4_challengers(optimizer)

    optimizer.last_objective_mutation_modes = [
        mode_by_prompt[prompt.construct_prompt().strip().casefold()]
        for prompt in challengers
    ]
    optimizer.v6_candidate_modes = dict(mode_by_prompt)
    return challengers
