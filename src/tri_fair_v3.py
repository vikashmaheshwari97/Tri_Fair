"""Tri-Fair v3: readiness-safe, uncertainty-guarded three-objective racing."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np
from promptolution.utils.capo_utils import build_few_shot_examples
from promptolution.utils.prompt import Prompt

from src.fairness.v3_variation import generate_v3_challengers
from src.tri_fair import TriFair

TRI_FAIR_V3_METHOD_VERSION = "3.0-ready-init-guarded-racing-confirmation"


class TriFairV3(TriFair):
    """Tri-Fair with safer initialization and conservative multi-fidelity racing.

    The optimizer retains the same three objectives and downstream-token budget as
    Tri-Fair v2.  Its additional behaviour is algorithmic:

    * build the initial Pareto archive only after statistically ready common blocks;
    * cover quality/fairness/cost/balanced mutation directions every iteration;
    * reject challengers early only under margin-guarded dominance;
    * use the final budget phase to increase common-block evidence for incumbents.
    """

    method_version = TRI_FAIR_V3_METHOD_VERSION

    def __init__(
        self,
        *args: Any,
        min_initial_ready_blocks: int = 2,
        min_racing_blocks: int = 2,
        dominance_epsilons: Sequence[float] = (0.010, 0.50, 0.010),
        archive_confirmation_fraction: float = 0.85,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.min_initial_ready_blocks = max(1, int(min_initial_ready_blocks))
        self.min_racing_blocks = max(1, int(min_racing_blocks))
        epsilon = np.asarray(tuple(dominance_epsilons), dtype=float).reshape(-1)
        if epsilon.shape != (3,) or np.any(~np.isfinite(epsilon)) or np.any(epsilon < 0):
            raise ValueError("dominance_epsilons must contain three finite non-negative values")
        self.dominance_epsilons = epsilon
        self.archive_confirmation_fraction = float(archive_confirmation_fraction)
        if not 0.0 < self.archive_confirmation_fraction < 1.0:
            raise ValueError("archive_confirmation_fraction must lie in (0, 1)")
        self.initial_common_blocks: list[int] = []
        self.last_step_mode = "not_started"

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _build_initial_population(self) -> list[Prompt]:
        population: list[Prompt] = []
        for prompt in self.prompts:
            num_examples = random.randint(0, self.upper_shots)
            few_shots = build_few_shot_examples(
                instruction=prompt.instruction,
                num_examples=num_examples,
                optimizer=self,
            )
            population.append(Prompt(prompt.instruction, few_shots))
        return population

    def _population_diagnostics(
        self, population: Sequence[Prompt], blocks: Sequence[int]
    ) -> dict[str, Any]:
        widest: dict[str, float] = {}
        max_ci_width = 0.0
        for prompt in population:
            record = self._record_for(prompt, blocks)
            diagnostics = dict(record.get("diagnostics") or {})
            raw = diagnostics.get("uncertainty_by_group") or {}
            if isinstance(raw, Mapping):
                for key, value in raw.items():
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(number):
                        widest[str(key)] = max(widest.get(str(key), 0.0), number)
            try:
                width = float(diagnostics.get("max_ci_width", 0.0))
            except (TypeError, ValueError):
                width = 0.0
            if np.isfinite(width):
                max_ci_width = max(max_ci_width, width)
        return {
            "uncertainty_by_group": widest,
            "max_ci_width": max_ci_width,
        }

    def _pre_optimization_loop(self) -> None:
        self.current_step = 0
        population = self._build_initial_population()
        remaining = set(range(self.task.n_blocks))
        common_blocks: list[int] = []
        vectors: Optional[np.ndarray] = None
        ready: list[bool] = [False] * len(population)

        while remaining:
            diagnostics = (
                self._population_diagnostics(population, common_blocks)
                if common_blocks
                else {}
            )
            block = self.task.select_fairness_block(
                sorted(remaining),
                current_blocks=common_blocks,
                diagnostics=diagnostics,
            )
            remaining.remove(block)
            common_blocks.append(int(block))
            common_blocks.sort()
            vectors = self._get_block_vectors(population, common_blocks)
            ready = [
                bool(self._record_for(prompt, common_blocks).get("fairness_ready", False))
                for prompt in population
            ]
            if (
                len(common_blocks) >= self.min_initial_ready_blocks
                and all(ready)
            ):
                break

        if vectors is None:
            raise RuntimeError("Tri-Fair-v3 could not evaluate the initial population")
        ready_indices = [index for index, value in enumerate(ready) if value]
        if not ready_indices:
            raise RuntimeError(
                "No initial prompt reached statistical fairness readiness even after all "
                "development blocks. Check the v3 manifest and readiness thresholds."
            )

        ready_vectors = vectors[ready_indices]
        fronts = self._non_dominated_sort(ready_vectors)
        first_front = [ready_indices[index] for index in fronts[0]]
        incumbent_set = set(first_front)
        self.incumbents = [population[index] for index in first_front]
        self.non_incumbents = [
            prompt for index, prompt in enumerate(population) if index not in incumbent_set
        ]
        self.prompts = self.incumbents + self.non_incumbents

        vector_by_prompt = {
            prompt.construct_prompt(): vectors[index]
            for index, prompt in enumerate(population)
        }
        self.scores = [
            vector_by_prompt[prompt.construct_prompt()].tolist() for prompt in self.prompts
        ]
        self.initial_common_blocks = list(common_blocks)
        self.last_step_mode = "ready_initialization"

    # ------------------------------------------------------------------
    # Variation
    # ------------------------------------------------------------------
    def _generate_challengers(self) -> list[Prompt]:
        if not self.objective_aware_variation:
            return super()._generate_challengers()
        return generate_v3_challengers(self)

    # ------------------------------------------------------------------
    # Uncertainty-guarded racing
    # ------------------------------------------------------------------
    def _n_examples(self, blocks: Sequence[int]) -> int:
        if not blocks:
            return 0
        task_blocks = getattr(self.task, "blocks", None)
        if task_blocks is None:
            return 0
        return int(sum(len(task_blocks[int(block)]) for block in set(blocks)))

    @staticmethod
    def _diagnostic_width(record: Mapping[str, Any]) -> float:
        diagnostics = dict(record.get("diagnostics") or {})
        try:
            value = float(diagnostics.get("max_ci_width", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return value if np.isfinite(value) else 0.0

    def _comparison_epsilon(
        self,
        challenger_record: Mapping[str, Any],
        incumbent_record: Mapping[str, Any],
        blocks: Sequence[int],
    ) -> np.ndarray:
        epsilon = self.dominance_epsilons.copy()
        n_examples = max(1, self._n_examples(blocks))
        # A small sample-dependent quality guard prevents one early block from
        # deciding the race.  It shrinks automatically as evidence accumulates.
        epsilon[0] = max(epsilon[0], 0.50 / math.sqrt(n_examples))
        width = max(
            self._diagnostic_width(challenger_record),
            self._diagnostic_width(incumbent_record),
        )
        epsilon[2] = max(
            epsilon[2],
            0.20 * width / math.sqrt(max(1, len(set(blocks)))),
        )
        return epsilon

    def _robustly_dominated(
        self,
        challenger_vector: np.ndarray,
        incumbent_vector: np.ndarray,
        challenger_record: Mapping[str, Any],
        incumbent_record: Mapping[str, Any],
        blocks: Sequence[int],
    ) -> bool:
        # Objective representation is maximize-all: quality, -cost, -unfairness.
        delta = np.asarray(incumbent_vector, dtype=float) - np.asarray(
            challenger_vector, dtype=float
        )
        epsilon = self._comparison_epsilon(
            challenger_record, incumbent_record, blocks
        )
        no_meaningful_regression = bool(np.all(delta >= -epsilon))
        meaningful_gain = bool(np.any(delta > epsilon))
        return no_meaningful_regression and meaningful_gain

    def _do_intensification(self, challenger: Prompt) -> None:
        if challenger in self.incumbents:
            return
        if challenger in self.non_incumbents:
            self.non_incumbents.remove(challenger)

        common_blocks = self._get_common_blocks(self.incumbents) or []
        if not common_blocks:
            common_blocks = [
                self.task.select_fairness_block(
                    list(range(self.task.n_blocks)),
                    current_blocks=(),
                    diagnostics={},
                )
            ]

        remaining = set(int(value) for value in common_blocks)
        challenger_blocks: list[int] = []

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
            challenger_blocks.append(int(block))
            challenger_blocks.sort()

            challenger_vector = self._get_block_vectors(
                [challenger], challenger_blocks
            )[0]
            challenger_record = self._record_for(challenger, challenger_blocks)
            if not challenger_record.get("fairness_ready", False):
                continue
            if len(challenger_blocks) < self.min_racing_blocks:
                continue

            ready_incumbents, incumbent_vectors = self._ready_incumbent_vectors(
                self.incumbents, challenger_blocks
            )
            if not ready_incumbents:
                continue
            closest_index = self._get_closest_incumbent(
                challenger_vector, incumbent_vectors
            )
            closest_prompt = ready_incumbents[closest_index]
            closest_vector = incumbent_vectors[closest_index]
            if not self._is_dominated(challenger_vector, closest_vector):
                continue
            closest_record = self._record_for(closest_prompt, challenger_blocks)
            if self._robustly_dominated(
                challenger_vector,
                closest_vector,
                challenger_record,
                closest_record,
                challenger_blocks,
            ):
                self.non_incumbents.append(challenger)
                return

        final_record = self._record_for(challenger, common_blocks)
        if not final_record.get("fairness_ready", False):
            self.non_incumbents.append(challenger)
            return

        self.incumbents.append(challenger)
        self._update_incumbent_front(blocks=list(common_blocks))

    # ------------------------------------------------------------------
    # Near-budget archive confirmation
    # ------------------------------------------------------------------
    def _confirmation_candidate(self) -> tuple[int, Prompt] | None:
        if not self.incumbents:
            return None
        blocks_map = self._get_evaluated_blocks(self.incumbents)
        all_blocks = set(range(self.task.n_blocks))
        common = set(all_blocks)
        for prompt in self.incumbents:
            common &= set(blocks_map[prompt])
        remaining = sorted(all_blocks - common)
        if not remaining:
            return None

        # Finish a partially shared block before opening a new one.
        coverage = {
            block: sum(block in set(blocks_map[prompt]) for prompt in self.incumbents)
            for block in remaining
        }
        best_coverage = max(coverage.values())
        partially_shared = [
            block for block, count in coverage.items() if count == best_coverage and count > 0
        ]
        if partially_shared:
            target_block = min(partially_shared)
        else:
            target_block = self.task.select_fairness_block(
                remaining,
                current_blocks=sorted(common),
                diagnostics={},
            )

        missing = [
            prompt for prompt in self.incumbents if target_block not in set(blocks_map[prompt])
        ]
        if not missing:
            return None

        def uncertainty(prompt: Prompt) -> tuple[float, int, str]:
            prompt_blocks = blocks_map[prompt]
            if prompt_blocks:
                record = self._record_for(prompt, prompt_blocks)
                width = self._diagnostic_width(record)
            else:
                width = float("inf")
            return (-width, len(prompt_blocks), prompt.construct_prompt())

        chosen = sorted(missing, key=uncertainty)[0]
        return int(target_block), chosen

    def _confirmation_step(self) -> list[Prompt]:
        candidate = self._confirmation_candidate()
        if candidate is None:
            return super()._step()
        target_block, prompt = candidate
        self._get_block_vectors([prompt], target_block)

        blocks_map = self._get_evaluated_blocks(self.incumbents)
        if all(target_block in set(blocks_map[item]) for item in self.incumbents):
            common = self._get_common_blocks(self.incumbents) or []
            self._update_incumbent_front(blocks=list(common))

        self.prompts = self.incumbents + self.non_incumbents
        self.scores = self._get_evaluated_vectors(self.prompts).tolist()
        self.last_step_mode = "archive_confirmation"
        return self.prompts

    def _step(self) -> list[Prompt]:
        controller = getattr(self, "budget_controller", None)
        if (
            controller is not None
            and float(controller.utilization) >= self.archive_confirmation_fraction
            and self._confirmation_candidate() is not None
        ):
            self.current_step += 1
            return self._confirmation_step()

        self.last_step_mode = "search"
        return super()._step()
