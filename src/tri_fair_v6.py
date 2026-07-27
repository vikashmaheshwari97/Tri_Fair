"""Tri-Fair v6: adaptive portfolio search with a quality-safe archive.

V6 changes only Tri-Fair.  It does not modify NSGA-II-PO-Fair and it does not
use holdout feedback during optimization.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from promptolution.utils.prompt import Prompt

from src.fairness.v6_variation import generate_v6_challengers
from src.tri_fair_v5 import TriFairV5


TRI_FAIR_V6_METHOD_VERSION = (
    "6.0-adaptive-reference-portfolio-quality-floor-budget-aware"
)


class TriFairV6(TriFairV5):
    """Development-only adaptive three-objective prompt optimizer.

    Additions over v5:

    * adaptive direction allocation without reading holdout results;
    * pure and quality-constrained cost/fairness champions;
    * budget-aware challenger batch shrinking near the hard cap;
    * stagnation-triggered exploration;
    * multiclass quality enrichment for Bias-in-Bios;
    * larger, diversity-preserving verified archive.
    """

    method_version = TRI_FAIR_V6_METHOD_VERSION

    def __init__(
        self,
        *args: Any,
        quality_floor_margin: float = 0.035,
        stagnation_window: int = 3,
        late_phase_crossovers: int = 2,
        critical_phase_crossovers: int = 1,
        exploration_interval: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.quality_floor_margin = float(quality_floor_margin)
        if not 0.0 <= self.quality_floor_margin <= 0.25:
            raise ValueError("quality_floor_margin must lie in [0, 0.25]")
        self.stagnation_window = max(1, int(stagnation_window))
        self.late_phase_crossovers = max(1, int(late_phase_crossovers))
        self.critical_phase_crossovers = max(
            1, int(critical_phase_crossovers)
        )
        self.exploration_interval = max(1, int(exploration_interval))

        self._front_signature: np.ndarray | None = None
        self._stagnation_steps = 0
        self._mode_usage: Counter[str] = Counter()
        self.v6_candidate_modes: dict[str, str] = {}

    def _metric_family(self) -> str:
        metric = str(
            getattr(self.task, "fairness_metric_name", "")
        )
        if metric.startswith("bbq_bias"):
            return "bbq"
        if metric == "civil_equalized_odds":
            return "civil"
        if metric.startswith("bios_tpr_gap"):
            return "bios"
        return "generic"

    def _mode_weights(self) -> dict[str, float]:
        """Structural dataset priorities; no holdout values are consulted."""
        family = self._metric_family()
        weights = {
            "quality": 0.30,
            "fairness": 0.30,
            "cost": 0.18,
            "balanced": 0.17,
            "explore": 0.05,
        }
        if family == "bbq":
            weights.update(
                quality=0.29,
                fairness=0.35,
                cost=0.16,
                balanced=0.15,
                explore=0.05,
            )
        elif family == "civil":
            weights.update(
                quality=0.35,
                fairness=0.29,
                cost=0.15,
                balanced=0.16,
                explore=0.05,
            )
        elif family == "bios":
            weights.update(
                quality=0.35,
                fairness=0.34,
                cost=0.12,
                balanced=0.14,
                explore=0.05,
            )

        if self._stagnation_steps >= self.stagnation_window:
            weights["explore"] += 0.15
            weights["quality"] += 0.05

        total = sum(weights.values())
        return {key: value / total for key, value in weights.items()}

    def get_v6_modes(self, count: int) -> list[str]:
        """Deterministic deficit-round-robin direction scheduler."""
        count = max(0, int(count))
        if count == 0:
            return []

        modes = ["quality", "fairness", "cost", "balanced", "explore"]
        output: list[str] = []

        # Preserve all primary objectives whenever the batch is large enough.
        for mode in ("quality", "fairness", "cost", "balanced"):
            if len(output) >= count:
                break
            output.append(mode)

        weights = self._mode_weights()
        while len(output) < count:
            total_after = sum(self._mode_usage.values()) + len(output) + 1
            best = max(
                modes,
                key=lambda mode: (
                    weights[mode] * total_after
                    - (self._mode_usage[mode] + output.count(mode)),
                    weights[mode],
                    -modes.index(mode),
                ),
            )
            output.append(best)

        if (
            self.current_step > 0
            and self.current_step % self.exploration_interval == 0
            and "explore" not in output
        ):
            output[-1] = "explore"

        self._mode_usage.update(output)
        return output

    def v6_challenger_count(self) -> int:
        """Shrink the final batches so strict atomic budgeting wastes less."""
        base = max(1, int(self.crossovers_per_iter))
        utilization = self._budget_utilization()
        if utilization >= 0.96:
            return min(base, self.critical_phase_crossovers)
        if utilization >= 0.88:
            return min(base, self.late_phase_crossovers)
        if utilization >= 0.72:
            return min(base, 4)
        return base

    def _champion_local_indices(
        self,
        vectors: np.ndarray,
    ) -> list[int]:
        """Preserve pure extremes plus quality-constrained extremes."""
        values = np.atleast_2d(np.asarray(vectors, dtype=float))
        normalised = self._normalise_vectors(values)

        quality = normalised[:, 0]
        cost_score = normalised[:, 1]
        fairness_score = normalised[:, 2]
        quality_floor = float(np.max(quality) - self.quality_floor_margin)
        eligible = np.flatnonzero(quality >= quality_floor)
        if not len(eligible):
            eligible = np.arange(len(values))

        champions = [
            int(np.argmax(quality)),
            int(np.argmax(cost_score)),
            int(np.argmax(fairness_score)),
            int(np.argmax(np.min(normalised, axis=1))),
            int(eligible[np.argmax(cost_score[eligible])]),
            int(eligible[np.argmax(fairness_score[eligible])]),
            int(np.argmax(0.55 * quality + 0.25 * cost_score + 0.20 * fairness_score)),
            int(np.argmax(0.55 * quality + 0.20 * cost_score + 0.25 * fairness_score)),
        ]

        output: list[int] = []
        for index in champions:
            if index not in output:
                output.append(index)
        return output

    def _update_stagnation_signature(
        self,
        blocks: Sequence[int],
    ) -> None:
        if not self.incumbents:
            return
        vectors = self._get_block_vectors(
            list(self.incumbents),
            list(blocks),
        )
        if not len(vectors):
            return
        signature = np.nanmax(vectors, axis=0)
        if self._front_signature is None:
            self._front_signature = signature
            self._stagnation_steps = 0
            return

        improvement = np.asarray((0.0015, 0.05, 0.0015))
        if np.any(signature > self._front_signature + improvement):
            self._stagnation_steps = 0
            self._front_signature = np.maximum(
                self._front_signature,
                signature,
            )
        else:
            self._stagnation_steps += 1

    def _update_incumbent_front(self, blocks: list[int]) -> None:
        super()._update_incumbent_front(blocks)
        if self.incumbents:
            self._update_stagnation_signature(blocks)

    def _generate_challengers(self) -> list[Prompt]:
        if not self.objective_aware_variation:
            return super()._generate_challengers()
        return generate_v6_challengers(self)
