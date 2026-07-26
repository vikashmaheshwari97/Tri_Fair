"""Tri-Fair v5: smart warm start, verified archive, and robust reference racing.

V5 builds on v4's progressive shared-fidelity racing.  It keeps the comparison
scientifically matched by giving both methods the same enhanced raw instruction
pool, then adds a method-internal four-direction warm start that is evaluated on
development data and consumes the ordinary 5M downstream-token budget.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np
from promptolution.utils.capo_utils import build_few_shot_examples
from promptolution.utils.formatting import extract_from_tag
from promptolution.utils.prompt import Prompt

from src.fairness.v5_variation import (
    PRIMARY_MODES,
    _ensure_output_contract,
    _normalise_instruction,
    generate_v5_challengers,
    warm_start_prompt,
)
from src.tri_fair_v4 import TriFairV4

TRI_FAIR_V5_METHOD_VERSION = (
    "5.0-shared-diverse-seeds-smart-warm-start-contrastive-reference-archive"
)


class TriFairV5(TriFairV4):
    """Three-objective optimizer with a quality-preserving verified archive.

    Additions relative to v4:

    * method-internal quality/fairness/cost/balanced warm-start proposals;
    * dataset-specific mutation goals instead of Civil-only toxicity goals;
    * contrastive protected-group few-shot pairs from the frozen few-shot split;
    * a capped archive that always preserves quality, cost, fairness, and balanced
      champions before filling remaining slots by crowding distance;
    * parent selection restricted to sufficiently evaluated, fairness-ready elites.
    """

    method_version = TRI_FAIR_V5_METHOD_VERSION

    def __init__(
        self,
        *args: Any,
        smart_start_count: int = 4,
        archive_cap: int = 8,
        verified_parent_blocks: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.smart_start_count = max(0, int(smart_start_count))
        self.archive_cap = max(4, int(archive_cap))
        self.verified_parent_blocks = max(1, int(verified_parent_blocks))
        self.smart_start_prompt_keys: set[str] = set()
        self.v5_candidate_modes: dict[str, str] = {}
        self.last_step_mode = "not_started"

    # ------------------------------------------------------------------
    # Smart warm start
    # ------------------------------------------------------------------
    @staticmethod
    def _responses(value: Any, expected: int) -> list[str]:
        if isinstance(value, str):
            output = [value]
        elif isinstance(value, Sequence):
            output = [str(item) for item in value]
        else:
            output = []
        if len(output) < expected:
            output.extend([""] * (expected - len(output)))
        return output[:expected]

    def _build_initial_population(self) -> list[Prompt]:
        population = super()._build_initial_population()
        if self.smart_start_count <= 0 or len(population) < 2:
            return population

        modes = [
            PRIMARY_MODES[index % len(PRIMARY_MODES)]
            for index in range(self.smart_start_count)
        ]
        ordered = sorted(
            population,
            key=lambda prompt: (
                len(prompt.instruction.split()),
                prompt.instruction.casefold(),
            ),
        )
        parent_pairs: list[tuple[Prompt, Prompt]] = []
        prompts: list[str] = []
        for index, mode in enumerate(modes):
            mother = ordered[index % len(ordered)]
            father = ordered[(index * 5 + len(ordered) // 2 + 1) % len(ordered)]
            if father == mother:
                father = ordered[(index + 1) % len(ordered)]
            parent_pairs.append((mother, father))
            prompts.append(
                warm_start_prompt(
                    self,
                    mode=mode,
                    mother=mother,
                    father=father,
                )
            )

        responses = self._responses(self.meta_llm.get_response(prompts), len(prompts))
        existing = {
            prompt.construct_prompt().strip().casefold() for prompt in population
        }
        for mode, response, (mother, father) in zip(modes, responses, parent_pairs):
            instruction = _normalise_instruction(
                extract_from_tag(response, "<prompt>", "</prompt>")
            )
            if not instruction:
                instruction = (
                    mother.instruction.rstrip(" .")
                    + ". Apply the same rule across protected groups and keep the decision concise."
                )
            instruction = _ensure_output_contract(instruction)

            if mode == "cost":
                examples: list[str] = []
            else:
                requested = 2 if mode == "fairness" else 1
                examples = list(
                    build_few_shot_examples(
                        instruction=instruction,
                        num_examples=min(requested, int(self.upper_shots)),
                        optimizer=self,
                    )
                )
            candidate = Prompt(instruction, examples[: int(self.upper_shots)])
            key = candidate.construct_prompt().strip().casefold()
            if not key or key in existing:
                # A deterministic, still method-specific fallback avoids silently
                # losing an entire search direction when the meta model duplicates a seed.
                suffix = {
                    "quality": " Use an explicit evidence hierarchy before selecting the label.",
                    "fairness": " Verify that an identity substitution would not change the decision.",
                    "cost": " Use the shortest complete decision rule and output no explanation.",
                    "balanced": " Use a short ordered rule balancing correctness, fairness, and cost.",
                }[mode]
                candidate = Prompt(
                    _ensure_output_contract(mother.instruction.rstrip(" .") + suffix),
                    examples[: int(self.upper_shots)],
                )
                key = candidate.construct_prompt().strip().casefold()
            if key and key not in existing:
                existing.add(key)
                self.smart_start_prompt_keys.add(key)
                population.append(candidate)
        return population

    def _pre_optimization_loop(self) -> None:
        super()._pre_optimization_loop()
        common = self._common_blocks()
        if common:
            self._update_incumbent_front(blocks=common)

        # The augmented warm-start population is environmental-selected back to
        # the configured population size before the first ordinary search step.
        guard = 0
        while len(self.incumbents) + len(self.non_incumbents) > self.population_size:
            before = len(self.incumbents) + len(self.non_incumbents)
            self._select_survivors()
            after = len(self.incumbents) + len(self.non_incumbents)
            guard += 1
            if after >= before or guard > 4 * max(1, before):
                raise RuntimeError("Tri-Fair-v5 initial environmental selection did not converge")

        self.prompts = self.incumbents + self.non_incumbents
        self.scores = self._get_evaluated_vectors(self.prompts).tolist()
        self.last_step_mode = "smart_verified_initialization"

    # ------------------------------------------------------------------
    # Verified champion archive
    # ------------------------------------------------------------------
    def _normalise_vectors(self, vectors: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(vectors, dtype=float))
        if self.fixed_objective_bounds is not None:
            lower, upper = self.fixed_objective_bounds
        else:
            lower = np.nanmin(values, axis=0)
            upper = np.nanmax(values, axis=0)
        span = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
        span[~np.isfinite(span) | (span <= 0)] = 1.0
        output = (values - np.asarray(lower, dtype=float)) / span
        return np.clip(output, 0.0, 1.0)

    def _champion_local_indices(self, vectors: np.ndarray) -> list[int]:
        normalised = self._normalise_vectors(vectors)
        champions = [
            int(np.argmax(normalised[:, 0])),
            int(np.argmax(normalised[:, 1])),
            int(np.argmax(normalised[:, 2])),
            int(np.argmax(np.min(normalised, axis=1))),
        ]
        output: list[int] = []
        for index in champions:
            if index not in output:
                output.append(index)
        return output

    def _cap_front(self, vectors: np.ndarray, capacity: int) -> list[int]:
        if len(vectors) <= capacity:
            return list(range(len(vectors)))
        selected = self._champion_local_indices(vectors)
        crowding = np.asarray(self._calculate_crowding_distance(vectors), dtype=float)
        crowding = np.nan_to_num(crowding, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        ranked = sorted(
            range(len(vectors)),
            key=lambda index: (crowding[index], -index),
            reverse=True,
        )
        for index in ranked:
            if index not in selected:
                selected.append(index)
            if len(selected) >= capacity:
                break
        return selected[:capacity]

    def _update_incumbent_front(self, blocks: list[int]) -> None:
        if not self.incumbents:
            return
        blocks = sorted(set(int(value) for value in blocks))
        vectors = self._get_block_vectors(self.incumbents, blocks)
        ready_indices = [
            index
            for index, prompt in enumerate(self.incumbents)
            if bool(self._record_for(prompt, blocks).get("fairness_ready", False))
        ]
        if not ready_indices:
            self.non_incumbents.extend(self.incumbents)
            self.incumbents = []
            return

        ready_vectors = vectors[ready_indices]
        fronts = self._non_dominated_sort(ready_vectors)
        first_ready_local = list(fronts[0])
        first_original = [ready_indices[index] for index in first_ready_local]
        first_vectors = vectors[first_original]
        kept_local = self._cap_front(
            first_vectors,
            min(self.archive_cap, self.population_size),
        )
        selected_original = {first_original[index] for index in kept_local}

        previous = list(self.incumbents)
        self.incumbents = [
            prompt for index, prompt in enumerate(previous) if index in selected_original
        ]
        demoted = [
            prompt for index, prompt in enumerate(previous) if index not in selected_original
        ]
        existing = {
            prompt.construct_prompt().strip().casefold()
            for prompt in self.non_incumbents
        }
        for prompt in demoted:
            key = prompt.construct_prompt().strip().casefold()
            if key not in existing:
                existing.add(key)
                self.non_incumbents.append(prompt)

    def get_v5_parent_pool(self) -> list[Prompt]:
        if not self.incumbents:
            return []
        blocks_map = self._get_evaluated_blocks(self.incumbents)
        required = min(
            int(self.task.n_blocks),
            max(self.verified_parent_blocks, self._required_common_blocks()),
        )
        verified: list[Prompt] = []
        for prompt in self.incumbents:
            blocks = sorted(set(int(value) for value in blocks_map[prompt]))
            if len(blocks) < required:
                continue
            try:
                ready = bool(self._record_for(prompt, blocks).get("fairness_ready", False))
            except Exception:
                ready = False
            if ready:
                verified.append(prompt)
        return verified or list(self.incumbents)

    # ------------------------------------------------------------------
    # Dataset-specific reference variation
    # ------------------------------------------------------------------
    def _generate_challengers(self) -> list[Prompt]:
        if not self.objective_aware_variation:
            return super()._generate_challengers()
        return generate_v5_challengers(self)
