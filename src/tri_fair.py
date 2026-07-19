"""Tri-Fair: quality × cost × fairness prompt optimization.

Tri-Fair extends MO-CAPO with an exact, non-decomposable fairness objective and
fairness-readiness gates.  Candidate predictions are still cached per block,
but fairness is recomputed over the union of evaluated blocks rather than
incorrectly averaging per-block fairness scores.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from promptolution.utils.prompt import Prompt

from src.mo_capo import MoCAPO
from src.checkpointing import ResumableOptimizerMixin
from src.tasks.fairness_task import FairnessEvalResult, FairnessTask
from src.fairness.objective_mutation import (
    DEFAULT_MUTATION_WEIGHTS,
    generate_objective_aware_challengers,
)

TRI_FAIR_METHOD_VERSION = "2.1-strict-budget-statistical-objective-aware"


class TriFair(ResumableOptimizerMixin, MoCAPO):
    """Three-objective, fairness-aware MO-CAPO."""

    supports_multi_objective = True

    def __init__(
        self,
        *args,
        fixed_objective_bounds: Optional[Sequence[Sequence[float]]] = None,
        objective_aware_variation: bool = True,
        objective_mutation_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        self.fixed_objective_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.objective_aware_variation = bool(objective_aware_variation)
        self.objective_mutation_weights = dict(
            objective_mutation_weights or DEFAULT_MUTATION_WEIGHTS
        )
        super().__init__(*args, **kwargs)
        if not isinstance(self.task, FairnessTask):
            raise TypeError("TriFair requires src.tasks.fairness_task.FairnessTask")
        self.n_objectives = 3
        self.aggregate_records: Dict[Tuple[str, Tuple[int, ...]], Dict] = {}

        if fixed_objective_bounds is not None:
            bounds = np.asarray(fixed_objective_bounds, dtype=float)
            if bounds.shape != (2, self.n_objectives):
                raise ValueError(
                    "fixed_objective_bounds must have shape (2, 3): lower then upper"
                )
            lower, upper = bounds
            if not np.all(np.isfinite(bounds)) or np.any(upper <= lower):
                raise ValueError("fixed objective bounds must be finite and strictly ordered")
            self.fixed_objective_bounds = (lower.copy(), upper.copy())
            self.global_min_bounds = lower.copy()
            self.global_max_bounds = upper.copy()

    def _update_global_bounds(self, vectors: np.ndarray) -> None:
        """Track only fully finite objective vectors for stable normalization."""

        if self.fixed_objective_bounds is not None:
            lower, upper = self.fixed_objective_bounds
            self.global_min_bounds = lower.copy()
            self.global_max_bounds = upper.copy()
            return

        values = np.atleast_2d(np.asarray(vectors, dtype=float))
        values = values[np.all(np.isfinite(values), axis=1)]
        if values.size == 0:
            return
        current_min = np.min(values, axis=0)
        current_max = np.max(values, axis=0)
        if self.global_min_bounds is None:
            self.global_min_bounds = current_min
            self.global_max_bounds = current_max
        else:
            self.global_min_bounds = np.minimum(self.global_min_bounds, current_min)
            self.global_max_bounds = np.maximum(self.global_max_bounds, current_max)

    def _get_objective_vectors(self, result: FairnessEvalResult) -> np.ndarray:
        quality = np.asarray(result.agg_scores, dtype=float).reshape(-1, 1)
        cost = (
            self.cost_per_input_token * np.asarray(result.agg_input_tokens, dtype=float)
            + self.cost_per_output_token
            * np.asarray(result.agg_output_tokens, dtype=float)
        ).reshape(-1, 1)
        unfairness = np.asarray(result.fairness_loss, dtype=float).reshape(-1, 1)
        return np.hstack([quality, -cost, -unfairness])

    def _store_records(
        self,
        prompts: Sequence[Prompt],
        blocks: Sequence[int],
        result: FairnessEvalResult,
        vectors: np.ndarray,
    ) -> None:
        key_blocks = tuple(sorted(set(int(value) for value in blocks)))
        for index, prompt in enumerate(prompts):
            key = (prompt.construct_prompt(), key_blocks)
            self.aggregate_records[key] = {
                "quality": float(result.agg_scores[index]),
                "cost": float(-vectors[index, 1]),
                "fairness": float(result.fairness_loss[index]),
                "fairness_ready": bool(result.fairness_ready[index]),
                "diagnostics": result.fairness_diagnostics[index],
                "support": result.fairness_support[index],
                "blocks": list(key_blocks),
                "mean_input_tokens": float(result.agg_input_tokens[index]),
                "mean_output_tokens": float(result.agg_output_tokens[index]),
            }

    def _get_block_vectors(
        self,
        prompts: List[Prompt],
        block_idx: Union[int, List[int]],
    ) -> np.ndarray:
        """Evaluate requested blocks and aggregate fairness exactly over their union."""

        block_indices = [block_idx] if isinstance(block_idx, int) else list(block_idx)
        if not block_indices:
            raise ValueError("block_idx cannot be empty")
        block_indices = sorted(set(int(value) for value in block_indices))
        prompt_strings = [prompt.construct_prompt() for prompt in prompts]

        # Ensure every prompt has a per-block runhistory entry.  FairnessTask's
        # prediction cache prevents duplicate LLM calls.
        for block in block_indices:
            missing_prompts: list[Prompt] = []
            missing_positions: list[int] = []
            for index, prompt_string in enumerate(prompt_strings):
                if (prompt_string, block) not in self.runhistory:
                    missing_prompts.append(prompts[index])
                    missing_positions.append(index)
            if missing_prompts:
                result = self.task.evaluate(
                    missing_prompts,
                    self.predictor,
                    block_idx=block,
                )
                vectors = self._get_objective_vectors(result)
                self._update_global_bounds(vectors)
                self._store_records(missing_prompts, [block], result, vectors)
                for local_index, original_index in enumerate(missing_positions):
                    key = (prompt_strings[original_index], block)
                    self.runhistory[key] = (vectors[local_index], self.current_step)

        # Recompute set-level fairness over the complete union.  All predictions
        # are cached, so this is CPU-only after the per-block calls above.
        aggregate = self.task.evaluate(
            prompts,
            self.predictor,
            block_idx=block_indices,
        )
        vectors = self._get_objective_vectors(aggregate)
        self._update_global_bounds(vectors)
        self._store_records(prompts, block_indices, aggregate, vectors)
        return vectors

    def _record_for(self, prompt: Prompt, blocks: Sequence[int]) -> Dict:
        key = (
            prompt.construct_prompt(),
            tuple(sorted(set(int(value) for value in blocks))),
        )
        if key not in self.aggregate_records:
            self._get_block_vectors([prompt], list(blocks))
        return dict(self.aggregate_records[key])

    def get_fairness_record(
        self,
        prompt: Prompt,
        blocks: Optional[Sequence[int]] = None,
    ) -> Dict:
        if blocks is None:
            blocks = self._get_evaluated_blocks([prompt])[prompt]
        if not blocks:
            return {}
        return self._record_for(prompt, blocks)

    def _generate_challengers(self) -> List[Prompt]:
        if not self.objective_aware_variation:
            return super()._generate_challengers()
        return generate_objective_aware_challengers(self)

    def _ready_incumbent_vectors(
        self,
        incumbents: Sequence[Prompt],
        blocks: Sequence[int],
    ) -> tuple[list[Prompt], np.ndarray]:
        vectors = self._get_block_vectors(list(incumbents), list(blocks))
        ready_prompts: list[Prompt] = []
        ready_vectors: list[np.ndarray] = []
        for prompt, vector in zip(incumbents, vectors):
            if self._record_for(prompt, blocks)["fairness_ready"]:
                ready_prompts.append(prompt)
                ready_vectors.append(vector)
        if not ready_vectors:
            return [], np.empty((0, self.n_objectives), dtype=float)
        return ready_prompts, np.vstack(ready_vectors)

    def _do_intensification(self, challenger: Prompt) -> None:
        if challenger in self.incumbents:
            return
        if challenger in self.non_incumbents:
            self.non_incumbents.remove(challenger)

        common_blocks = self._get_common_blocks(self.incumbents)
        if not common_blocks:
            # This should only occur in a malformed/restored state.  Evaluate one
            # common block rather than comparing incomparable candidates.
            common_blocks = [random.randrange(self.task.n_blocks)]

        remaining = set(common_blocks)
        challenger_blocks: list[int] = []
        old_vector: Optional[np.ndarray] = None

        while remaining:
            diagnostics = (
                self._record_for(challenger, challenger_blocks).get("diagnostics", {})
                if challenger_blocks
                else {}
            )
            block = self.task.select_fairness_block(
                sorted(remaining),
                current_blocks=challenger_blocks,
                diagnostics=diagnostics,
            )
            remaining.remove(block)
            challenger_blocks.append(block)
            challenger_blocks.sort()

            challenger_vector = self._get_block_vectors(
                [challenger], challenger_blocks
            )[0]
            record = self._record_for(challenger, challenger_blocks)

            # A challenger cannot be rejected until its set contains enough
            # protected-group support for the fairness metric.
            if not record["fairness_ready"]:
                continue

            # Match MO-CAPO's improvement-triggered comparisons: compare after
            # the first ready estimate and whenever the new estimate dominates
            # the previous estimate.
            should_compare = old_vector is None or self._is_dominated(
                old_vector, challenger_vector
            )
            if not should_compare and remaining:
                continue
            old_vector = challenger_vector.copy()

            ready_incumbents, incumbent_vectors = self._ready_incumbent_vectors(
                self.incumbents,
                challenger_blocks,
            )
            if not ready_incumbents:
                continue
            closest_index = self._get_closest_incumbent(
                challenger_vector, incumbent_vectors
            )
            closest_vector = incumbent_vectors[closest_index]
            if self._is_dominated(challenger_vector, closest_vector):
                self.non_incumbents.append(challenger)
                return

        final_record = self._record_for(challenger, common_blocks)
        if not final_record["fairness_ready"]:
            # Insufficient evidence is not interpreted as fairness.  Keep the
            # candidate as genetic material, but do not admit it to the archive.
            self.non_incumbents.append(challenger)
            return

        self.incumbents.append(challenger)
        self._update_incumbent_front(blocks=list(common_blocks))

    def _advance_one_incumbent(self) -> None:
        if not self.incumbents:
            return
        blocks_map = self._get_evaluated_blocks(self.incumbents)
        counts = [len(blocks_map[prompt]) for prompt in self.incumbents]
        minimum = min(counts)
        least = [
            prompt for prompt, count in zip(self.incumbents, counts) if count == minimum
        ]
        chosen = random.choice(least)

        union_blocks: set[int] = set()
        for prompt in self.incumbents:
            union_blocks.update(blocks_map[prompt])
        chosen_blocks = list(blocks_map[chosen])
        gap_blocks = union_blocks - set(chosen_blocks)
        if gap_blocks:
            candidates = sorted(gap_blocks)
        else:
            candidates = sorted(set(range(self.task.n_blocks)) - union_blocks)
        if not candidates:
            return
        diagnostics = (
            self._record_for(chosen, chosen_blocks).get("diagnostics", {})
            if chosen_blocks
            else {}
        )
        block = self.task.select_fairness_block(
            candidates,
            current_blocks=chosen_blocks,
            diagnostics=diagnostics,
        )
        self._get_block_vectors([chosen], block)

    def _update_incumbent_front(self, blocks: List[int]) -> None:
        vectors = self._get_block_vectors(self.incumbents, blocks)
        ready_indices = [
            index
            for index, prompt in enumerate(self.incumbents)
            if self._record_for(prompt, blocks)["fairness_ready"]
        ]
        not_ready = [
            prompt
            for index, prompt in enumerate(self.incumbents)
            if index not in set(ready_indices)
        ]
        if not ready_indices:
            self.non_incumbents.extend(not_ready)
            self.incumbents = []
            return

        ready_vectors = vectors[ready_indices]
        fronts = self._non_dominated_sort(ready_vectors)
        first_front_original = [ready_indices[index] for index in fronts[0]]
        selected = {index for index in first_front_original}
        new_incumbents = [self.incumbents[index] for index in first_front_original]
        demoted = [
            prompt
            for index, prompt in enumerate(self.incumbents)
            if index not in selected
        ]
        self.incumbents = new_incumbents
        self.non_incumbents.extend(demoted)
