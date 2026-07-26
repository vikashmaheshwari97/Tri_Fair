"""Tri-Fair v4: progressive shared-fidelity racing and reference variation.

This revision addresses the failure mode observed in Tri-Fair v3 where the
blockwise archive could remain non-dominated on a small common development
subset yet become unstable on holdout.  It keeps holdout completely outside the
optimizer and changes only development-time budget allocation and variation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from promptolution.utils.prompt import Prompt

from src.fairness.v4_variation import generate_v4_challengers
from src.tri_fair import TriFair
from src.tri_fair_v3 import TriFairV3

TRI_FAIR_V4_METHOD_VERSION = (
    "4.0-progressive-shared-fidelity-reference-variation-signed-error-cells"
)


class TriFairV4(TriFairV3):
    """Tri-Fair with progressive archive fidelity and less biased early racing.

    Key changes relative to v3:

    * the whole incumbent archive is progressively brought to 2/3/4/5 shared
      blocks as the downstream budget is consumed;
    * confirmation is interleaved with search instead of being postponed until
      the final 15 percent of the budget;
    * early dominance guards are narrower, so modest quality gains are not
      discarded merely because another candidate is slightly cheaper/fairer;
    * a challenger is checked against every comparable incumbent rather than
      only the geometrically closest incumbent;
    * reference-direction variation deliberately generates quality, fairness,
      cost, and balanced candidates each iteration.
    """

    method_version = TRI_FAIR_V4_METHOD_VERSION

    def __init__(
        self,
        *args: Any,
        fidelity_utilization_thresholds: Sequence[float] = (0.0, 0.30, 0.60, 0.85),
        fidelity_block_targets: Sequence[int] = (2, 3, 4, 5),
        max_confirmation_streak: int = 2,
        quality_guard_scale: float = 0.20,
        fairness_guard_scale: float = 0.10,
        **kwargs: Any,
    ) -> None:
        # TriFairV4 owns confirmation scheduling; retain a valid but inactive v3
        # threshold for checkpoint compatibility.
        kwargs.setdefault("archive_confirmation_fraction", 0.999)
        super().__init__(*args, **kwargs)

        thresholds = np.asarray(tuple(fidelity_utilization_thresholds), dtype=float)
        targets = np.asarray(tuple(fidelity_block_targets), dtype=int)
        if thresholds.ndim != 1 or targets.ndim != 1 or len(thresholds) != len(targets):
            raise ValueError(
                "fidelity thresholds and block targets must be one-dimensional and equal length"
            )
        if len(thresholds) == 0 or thresholds[0] != 0.0:
            raise ValueError("fidelity_utilization_thresholds must begin at 0.0")
        if np.any(~np.isfinite(thresholds)) or np.any(np.diff(thresholds) <= 0):
            raise ValueError("fidelity utilization thresholds must be finite and increasing")
        if thresholds[-1] >= 1.0:
            raise ValueError("the final fidelity utilization threshold must be below 1.0")
        if np.any(targets <= 0) or np.any(np.diff(targets) < 0):
            raise ValueError("fidelity block targets must be positive and non-decreasing")

        self.fidelity_utilization_thresholds = thresholds
        self.fidelity_block_targets = targets
        self.max_confirmation_streak = max(1, int(max_confirmation_streak))
        self.quality_guard_scale = float(quality_guard_scale)
        self.fairness_guard_scale = float(fairness_guard_scale)
        if self.quality_guard_scale < 0 or self.fairness_guard_scale < 0:
            raise ValueError("guard scales must be non-negative")
        self._confirmation_streak = 0

    # ------------------------------------------------------------------
    # Progressive shared fidelity
    # ------------------------------------------------------------------
    def _budget_utilization(self) -> float:
        controller = getattr(self, "budget_controller", None)
        if controller is None:
            return 0.0
        try:
            value = float(controller.utilization)
        except (TypeError, ValueError, AttributeError):
            return 0.0
        return min(1.0, max(0.0, value)) if np.isfinite(value) else 0.0

    def _required_common_blocks(self) -> int:
        utilization = self._budget_utilization()
        index = int(
            np.searchsorted(
                self.fidelity_utilization_thresholds,
                utilization,
                side="right",
            )
            - 1
        )
        index = max(0, min(index, len(self.fidelity_block_targets) - 1))
        return min(int(self.task.n_blocks), int(self.fidelity_block_targets[index]))

    def _common_blocks(self) -> list[int]:
        return sorted(int(value) for value in (self._get_common_blocks(self.incumbents) or []))

    def _confirmation_candidate_for_target(
        self,
        target_count: int,
    ) -> tuple[int, Prompt] | None:
        if not self.incumbents:
            return None

        blocks_map = self._get_evaluated_blocks(self.incumbents)
        all_blocks = set(range(self.task.n_blocks))
        common = set(all_blocks)
        for prompt in self.incumbents:
            common &= set(blocks_map[prompt])
        if len(common) >= min(target_count, self.task.n_blocks):
            return None

        remaining = sorted(all_blocks - common)
        if not remaining:
            return None

        # Complete a partially shared block before opening a new one. This spends
        # the smallest number of downstream calls needed to raise common fidelity.
        coverage = {
            block: sum(block in set(blocks_map[prompt]) for prompt in self.incumbents)
            for block in remaining
        }
        best_coverage = max(coverage.values())
        partially_shared = [
            block
            for block, count in coverage.items()
            if count == best_coverage and count > 0
        ]
        if partially_shared:
            target_block = min(partially_shared)
        else:
            target_block = int(
                self.task.select_fairness_block(
                    remaining,
                    current_blocks=sorted(common),
                    diagnostics={},
                )
            )

        missing = [
            prompt
            for prompt in self.incumbents
            if target_block not in set(blocks_map[prompt])
        ]
        if not missing:
            return None

        def priority(prompt: Prompt) -> tuple[float, int, str]:
            prompt_blocks = blocks_map[prompt]
            if prompt_blocks:
                record = self._record_for(prompt, prompt_blocks)
                width = self._diagnostic_width(record)
            else:
                width = float("inf")
            # Highest uncertainty first, then least evidence, then stable text key.
            return (-width, len(prompt_blocks), prompt.construct_prompt())

        chosen = sorted(missing, key=priority)[0]
        return target_block, chosen

    def _confirmation_step_for_target(self, target_count: int) -> list[Prompt]:
        candidate = self._confirmation_candidate_for_target(target_count)
        if candidate is None:
            self._confirmation_streak = 0
            return TriFair._step(self)

        target_block, prompt = candidate
        self._get_block_vectors([prompt], target_block)

        common = self._common_blocks()
        if common:
            self._update_incumbent_front(blocks=common)

        self.prompts = self.incumbents + self.non_incumbents
        self.scores = self._get_evaluated_vectors(self.prompts).tolist()
        self.last_step_mode = "progressive_archive_confirmation"
        return self.prompts

    # ------------------------------------------------------------------
    # Reference-direction variation
    # ------------------------------------------------------------------
    def _generate_challengers(self) -> list[Prompt]:
        if not self.objective_aware_variation:
            return super()._generate_challengers()
        return generate_v4_challengers(self)

    # ------------------------------------------------------------------
    # Narrower uncertainty guards
    # ------------------------------------------------------------------
    def _comparison_epsilon(
        self,
        challenger_record: Mapping[str, Any],
        incumbent_record: Mapping[str, Any],
        blocks: Sequence[int],
    ) -> np.ndarray:
        epsilon = self.dominance_epsilons.copy()
        n_examples = max(1, self._n_examples(blocks))
        epsilon[0] = max(
            epsilon[0],
            self.quality_guard_scale / math.sqrt(n_examples),
        )
        width = max(
            self._diagnostic_width(challenger_record),
            self._diagnostic_width(incumbent_record),
        )
        epsilon[2] = max(
            epsilon[2],
            self.fairness_guard_scale
            * width
            / math.sqrt(max(1, len(set(blocks)))),
        )
        return epsilon

    # ------------------------------------------------------------------
    # All-incumbent robust racing
    # ------------------------------------------------------------------
    def _do_intensification(self, challenger: Prompt) -> None:
        if challenger in self.incumbents:
            return
        if challenger in self.non_incumbents:
            self.non_incumbents.remove(challenger)

        common_blocks = self._common_blocks()
        if not common_blocks:
            common_blocks = [
                int(
                    self.task.select_fairness_block(
                        list(range(self.task.n_blocks)),
                        current_blocks=(),
                        diagnostics={},
                    )
                )
            ]

        remaining = set(common_blocks)
        challenger_blocks: list[int] = []
        comparison_floor = min(
            len(common_blocks),
            max(self.min_racing_blocks, self._required_common_blocks()),
        )

        while remaining:
            diagnostics = (
                self._record_for(challenger, challenger_blocks).get("diagnostics", {})
                if challenger_blocks
                else {}
            )
            block = int(
                self.task.select_fairness_block(
                    sorted(remaining),
                    current_blocks=challenger_blocks,
                    diagnostics=diagnostics,
                )
            )
            remaining.remove(block)
            challenger_blocks.append(block)
            challenger_blocks.sort()

            challenger_vector = self._get_block_vectors(
                [challenger], challenger_blocks
            )[0]
            challenger_record = self._record_for(challenger, challenger_blocks)
            if not challenger_record.get("fairness_ready", False):
                continue
            if len(challenger_blocks) < comparison_floor:
                continue

            ready_incumbents, incumbent_vectors = self._ready_incumbent_vectors(
                self.incumbents,
                challenger_blocks,
            )
            if not ready_incumbents:
                continue

            # A challenger can be stopped only when a comparable incumbent
            # robustly epsilon-dominates it. Checking every incumbent avoids the
            # instability of selecting one geometrically closest reference point.
            for incumbent, incumbent_vector in zip(
                ready_incumbents,
                incumbent_vectors,
            ):
                if not self._is_dominated(challenger_vector, incumbent_vector):
                    continue
                incumbent_record = self._record_for(incumbent, challenger_blocks)
                if self._robustly_dominated(
                    challenger_vector,
                    incumbent_vector,
                    challenger_record,
                    incumbent_record,
                    challenger_blocks,
                ):
                    self.non_incumbents.append(challenger)
                    return

        final_record = self._record_for(challenger, common_blocks)
        if not final_record.get("fairness_ready", False):
            self.non_incumbents.append(challenger)
            return

        self.incumbents.append(challenger)
        self._update_incumbent_front(blocks=common_blocks)

    def _step(self) -> list[Prompt]:
        target = self._required_common_blocks()
        candidate = self._confirmation_candidate_for_target(target)
        utilization = self._budget_utilization()
        final_phase = utilization >= float(self.fidelity_utilization_thresholds[-1])

        if candidate is not None and (
            self._confirmation_streak < self.max_confirmation_streak or final_phase
        ):
            self.current_step += 1
            self._confirmation_streak += 1
            return self._confirmation_step_for_target(target)

        self._confirmation_streak = 0
        self.last_step_mode = "search"
        # Bypass TriFairV3's late all-block confirmation; v4 owns the progressive
        # schedule above while retaining TriFair's standard challenger loop.
        return TriFair._step(self)
