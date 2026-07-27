"""Tri-Fair v7: robust reference-direction and hypervolume-guided prompt search.

V7 changes only Tri-Fair's optimizer.  It uses development data only and never
reads held-out metrics during optimization.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from promptolution.utils.prompt import Prompt

from src.tri_fair_v6 import TriFairV6


TRI_FAIR_V7_METHOD_VERSION = (
    "7.0-robust-reference-hv-operator-adaptation-deterministic-downstream"
)

MODES = ("quality", "fairness", "cost", "balanced", "explore")


def _pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(np.asarray(values, dtype=float))
    keep = np.all(np.isfinite(values), axis=1)
    for index in np.flatnonzero(keep):
        candidate = values[index]
        others = values[keep]
        dominated = np.all(others <= candidate, axis=1) & np.any(
            others < candidate, axis=1
        )
        if np.any(dominated):
            keep[index] = False
    return keep


def _exact_hypervolume_3d(
    values: np.ndarray,
    reference: np.ndarray | None = None,
) -> float:
    """Exact cell-decomposition hypervolume for a small minimization front."""
    points = np.atleast_2d(np.asarray(values, dtype=float))
    reference = (
        np.asarray((1.1, 1.1, 1.1), dtype=float)
        if reference is None
        else np.asarray(reference, dtype=float)
    )
    valid = (
        np.all(np.isfinite(points), axis=1)
        & np.all(points <= reference, axis=1)
    )
    points = points[valid]
    if not len(points):
        return 0.0
    points = points[_pareto_mask_minimize(points)]

    coordinates = [
        np.unique(np.append(points[:, dimension], reference[dimension]))
        for dimension in range(3)
    ]
    volume = 0.0
    for x0, x1 in zip(coordinates[0][:-1], coordinates[0][1:]):
        for y0, y1 in zip(coordinates[1][:-1], coordinates[1][1:]):
            for z0, z1 in zip(coordinates[2][:-1], coordinates[2][1:]):
                if x1 <= x0 or y1 <= y0 or z1 <= z0:
                    continue
                lower = np.asarray((x0, y0, z0), dtype=float)
                if np.any(np.all(points <= lower, axis=1)):
                    volume += float((x1 - x0) * (y1 - y0) * (z1 - z0))
    return volume


def _simplex_reference_directions(partitions: int) -> np.ndarray:
    if partitions <= 0:
        raise ValueError("reference_partitions must be positive")
    directions: list[tuple[float, float, float]] = []
    for first in range(partitions + 1):
        for second in range(partitions + 1 - first):
            third = partitions - first - second
            directions.append(
                (
                    first / partitions,
                    second / partitions,
                    third / partitions,
                )
            )
    return np.asarray(directions, dtype=float)


class TriFairV7(TriFairV6):
    """Robust multi-fidelity search with adaptive operator allocation.

    Main changes over v6:

    * archive decisions use lower-confidence objective vectors computed from the
      already evaluated common development blocks;
    * reference-direction niching preserves broad Pareto-front coverage;
    * exact three-dimensional hypervolume contribution fills remaining archive
      slots;
    * quality-constrained cost and fairness champions receive dedicated slots;
    * mutation directions are allocated by a success-aware UCB portfolio;
    * the quality floor tightens as the 5M budget is consumed.
    """

    method_version = TRI_FAIR_V7_METHOD_VERSION

    def __init__(
        self,
        *args: Any,
        reference_partitions: int = 4,
        robustness_beta_quality: float = 0.55,
        robustness_beta_cost: float = 0.25,
        robustness_beta_fairness: float = 0.75,
        robustness_beta_end_fraction: float = 0.35,
        operator_ucb_scale: float = 0.18,
        quality_floor_start: float = 0.055,
        quality_floor_end: float = 0.020,
        parent_pool_cap: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reference_directions = _simplex_reference_directions(
            int(reference_partitions)
        )
        beta = np.asarray(
            (
                robustness_beta_quality,
                robustness_beta_cost,
                robustness_beta_fairness,
            ),
            dtype=float,
        )
        if np.any(~np.isfinite(beta)) or np.any(beta < 0):
            raise ValueError("robustness betas must be finite and non-negative")
        self.robustness_beta_start = beta
        self.robustness_beta_end_fraction = float(
            robustness_beta_end_fraction
        )
        if not 0.0 <= self.robustness_beta_end_fraction <= 1.0:
            raise ValueError("robustness_beta_end_fraction must lie in [0, 1]")

        self.operator_ucb_scale = float(operator_ucb_scale)
        self.quality_floor_start = float(quality_floor_start)
        self.quality_floor_end = float(quality_floor_end)
        if not (
            0.0 <= self.quality_floor_end <= self.quality_floor_start <= 0.25
        ):
            raise ValueError(
                "quality floor margins must satisfy "
                "0 <= end <= start <= 0.25"
            )
        self.parent_pool_cap = max(4, int(parent_pool_cap))

        self.v7_mode_attempts: Counter[str] = Counter()
        self.v7_mode_successes: Counter[str] = Counter()
        self.v7_candidate_modes: dict[str, str] = {}
        self._v7_front_signature: np.ndarray | None = None
        self._v7_stagnation_steps = 0

    # ------------------------------------------------------------------
    # Robust development objective vectors
    # ------------------------------------------------------------------
    def _robustness_beta(self) -> np.ndarray:
        utilization = self._budget_utilization()
        factor = 1.0 - (
            (1.0 - self.robustness_beta_end_fraction) * utilization
        )
        return self.robustness_beta_start * max(
            self.robustness_beta_end_fraction,
            factor,
        )

    def _robust_vectors(
        self,
        prompts: Sequence[Prompt],
        blocks: Sequence[int],
    ) -> np.ndarray:
        prompts = list(prompts)
        unique_blocks = sorted(set(int(value) for value in blocks))
        if not prompts:
            return np.empty((0, 3), dtype=float)
        if not unique_blocks:
            return self._get_evaluated_vectors(prompts)

        per_block = np.stack(
            [
                self._get_block_vectors(prompts, int(block))
                for block in unique_blocks
            ],
            axis=0,
        )
        mean = np.nanmean(per_block, axis=0)
        if len(unique_blocks) <= 1:
            return mean

        standard_error = np.nanstd(
            per_block,
            axis=0,
            ddof=1,
        ) / math.sqrt(len(unique_blocks))
        return mean - self._robustness_beta()[None, :] * standard_error

    def _adaptive_quality_floor_margin(self) -> float:
        utilization = self._budget_utilization()
        return (
            self.quality_floor_start * (1.0 - utilization)
            + self.quality_floor_end * utilization
        )

    # ------------------------------------------------------------------
    # Reference-direction and hypervolume archive selection
    # ------------------------------------------------------------------
    def _champion_local_indices(
        self,
        vectors: np.ndarray,
    ) -> list[int]:
        values = np.atleast_2d(np.asarray(vectors, dtype=float))
        normalised = self._normalise_vectors(values)
        quality = normalised[:, 0]
        cost_score = normalised[:, 1]
        fairness_score = normalised[:, 2]

        floor = float(
            np.nanmax(quality) - self._adaptive_quality_floor_margin()
        )
        eligible = np.flatnonzero(quality >= floor)
        if not len(eligible):
            eligible = np.arange(len(values))

        deficits = 1.0 - normalised
        utopia_distance = np.linalg.norm(deficits, axis=1)
        geometric = np.prod(
            np.clip(normalised, 1e-9, 1.0),
            axis=1,
        ) ** (1.0 / 3.0)

        champions = [
            int(np.nanargmax(quality)),
            int(np.nanargmax(cost_score)),
            int(np.nanargmax(fairness_score)),
            int(np.nanargmin(utopia_distance)),
            int(np.nanargmax(geometric)),
            int(eligible[np.nanargmax(cost_score[eligible])]),
            int(eligible[np.nanargmax(fairness_score[eligible])]),
            int(
                np.nanargmax(
                    0.55 * quality
                    + 0.25 * cost_score
                    + 0.20 * fairness_score
                )
            ),
            int(
                np.nanargmax(
                    0.55 * quality
                    + 0.20 * cost_score
                    + 0.25 * fairness_score
                )
            ),
        ]
        output: list[int] = []
        for index in champions:
            if index not in output:
                output.append(index)
        return output

    def _reference_representatives(
        self,
        normalised: np.ndarray,
    ) -> list[int]:
        deficits = np.clip(1.0 - normalised, 0.0, 1.1)
        output: list[int] = []
        directions = np.roll(
            self.reference_directions,
            shift=int(self.current_step) % len(self.reference_directions),
            axis=0,
        )
        for direction in directions:
            safe = np.maximum(direction, 1e-4)
            achievement = np.max(deficits / safe[None, :], axis=1)
            order = np.argsort(achievement, kind="stable")
            for index in order:
                candidate = int(index)
                if candidate not in output:
                    output.append(candidate)
                    break
        return output

    def _hypervolume_contributions(
        self,
        normalised: np.ndarray,
    ) -> np.ndarray:
        losses = np.clip(1.0 - normalised, 0.0, 1.1)
        total = _exact_hypervolume_3d(losses)
        contributions = np.zeros(len(losses), dtype=float)
        for index in range(len(losses)):
            without = np.delete(losses, index, axis=0)
            contributions[index] = max(
                0.0,
                total - _exact_hypervolume_3d(without),
            )
        return contributions

    def _cap_front(
        self,
        vectors: np.ndarray,
        capacity: int,
    ) -> list[int]:
        values = np.atleast_2d(np.asarray(vectors, dtype=float))
        if len(values) <= capacity:
            return list(range(len(values)))

        normalised = self._normalise_vectors(values)
        selected = self._champion_local_indices(values)[:capacity]

        for index in self._reference_representatives(normalised):
            if index not in selected:
                selected.append(index)
            if len(selected) >= capacity:
                return selected[:capacity]

        contributions = self._hypervolume_contributions(normalised)
        crowding = np.asarray(
            self._calculate_crowding_distance(values),
            dtype=float,
        )
        crowding = np.nan_to_num(
            crowding,
            nan=0.0,
            posinf=1e6,
            neginf=0.0,
        )

        while len(selected) < capacity:
            remaining = [
                index
                for index in range(len(values))
                if index not in selected
            ]
            if not remaining:
                break

            def score(index: int) -> tuple[float, float, float, int]:
                novelty = min(
                    float(
                        np.linalg.norm(
                            normalised[index] - normalised[chosen]
                        )
                    )
                    for chosen in selected
                )
                return (
                    float(contributions[index]),
                    novelty,
                    float(crowding[index]),
                    -index,
                )

            selected.append(max(remaining, key=score))

        return selected[:capacity]

    # ------------------------------------------------------------------
    # Success-aware mutation-direction portfolio
    # ------------------------------------------------------------------
    def _front_deficit_weights(self) -> dict[str, float]:
        priors = dict(super()._mode_weights())
        common = self._common_blocks()
        if self.incumbents and common:
            robust = self._robust_vectors(self.incumbents, common)
            normalised = self._normalise_vectors(robust)
            best = np.nanmax(normalised, axis=0)
            deficits = np.clip(1.0 - best, 0.0, 1.0)
            priors["quality"] += 0.35 * float(deficits[0])
            priors["cost"] += 0.25 * float(deficits[1])
            priors["fairness"] += 0.35 * float(deficits[2])
            priors["balanced"] += 0.15 * float(np.mean(deficits))

        if self._v7_stagnation_steps >= self.stagnation_window:
            priors["explore"] += 0.20

        total = sum(priors.values())
        return {mode: priors[mode] / total for mode in MODES}

    def get_v6_modes(self, count: int) -> list[str]:
        """Override the v6 scheduler with a success-aware UCB portfolio."""
        count = max(0, int(count))
        if count == 0:
            return []

        selected: list[str] = []
        for mode in ("quality", "fairness", "cost", "balanced"):
            if len(selected) >= count:
                break
            selected.append(mode)

        priors = self._front_deficit_weights()
        total_attempts = sum(self.v7_mode_attempts.values()) + 1
        while len(selected) < count:
            def score(mode: str) -> tuple[float, float, int]:
                attempts = self.v7_mode_attempts[mode] + selected.count(mode)
                successes = self.v7_mode_successes[mode]
                posterior = (successes + 1.0) / (attempts + 2.0)
                exploration = self.operator_ucb_scale * math.sqrt(
                    math.log(total_attempts + len(selected) + 1.0)
                    / (attempts + 1.0)
                )
                return (
                    priors[mode] + posterior + exploration,
                    priors[mode],
                    -MODES.index(mode),
                )

            selected.append(max(MODES, key=score))

        if (
            self.current_step > 0
            and self.current_step % self.exploration_interval == 0
            and "explore" not in selected
        ):
            selected[-1] = "explore"
        return selected

    def _generate_challengers(self) -> list[Prompt]:
        challengers = super()._generate_challengers()
        self.v7_candidate_modes = dict(
            getattr(self, "v6_candidate_modes", {})
        )
        for prompt in challengers:
            key = prompt.construct_prompt().strip().casefold()
            mode = self.v7_candidate_modes.get(key)
            if mode:
                self.v7_mode_attempts[mode] += 1
        return challengers

    # ------------------------------------------------------------------
    # Verified robust archive and parent pool
    # ------------------------------------------------------------------
    def _update_front_signature(self, robust_vectors: np.ndarray) -> None:
        if not len(robust_vectors):
            return
        normalised = self._normalise_vectors(robust_vectors)
        signature = np.concatenate(
            (
                np.nanmax(normalised, axis=0),
                np.asarray(
                    [float(np.nanmax(np.nanmin(normalised, axis=1)))],
                    dtype=float,
                ),
            )
        )
        if self._v7_front_signature is None:
            self._v7_front_signature = signature
            self._v7_stagnation_steps = 0
            return

        threshold = np.asarray(
            (0.0015, 0.0020, 0.0015, 0.0015),
            dtype=float,
        )
        if np.any(signature > self._v7_front_signature + threshold):
            self._v7_front_signature = np.maximum(
                self._v7_front_signature,
                signature,
            )
            self._v7_stagnation_steps = 0
        else:
            self._v7_stagnation_steps += 1

    def _update_incumbent_front(self, blocks: list[int]) -> None:
        if not self.incumbents:
            return

        blocks = sorted(set(int(value) for value in blocks))
        previous = list(self.incumbents)
        robust_vectors = self._robust_vectors(previous, blocks)

        ready_indices = [
            index
            for index, prompt in enumerate(previous)
            if bool(
                self._record_for(prompt, blocks).get(
                    "fairness_ready",
                    False,
                )
            )
        ]
        if not ready_indices:
            self.non_incumbents.extend(previous)
            self.incumbents = []
            return

        ready_vectors = robust_vectors[ready_indices]
        fronts = self._non_dominated_sort(ready_vectors)
        first_ready_local = list(fronts[0])
        first_original = [
            ready_indices[index] for index in first_ready_local
        ]
        first_vectors = robust_vectors[first_original]
        kept_local = self._cap_front(
            first_vectors,
            min(self.archive_cap, self.population_size),
        )
        selected_original = {
            first_original[index] for index in kept_local
        }

        self.incumbents = [
            prompt
            for index, prompt in enumerate(previous)
            if index in selected_original
        ]
        selected_keys = {
            prompt.construct_prompt().strip().casefold()
            for prompt in self.incumbents
        }
        for key in selected_keys:
            mode = self.v7_candidate_modes.get(key)
            if mode:
                self.v7_mode_successes[mode] += 1
                self.v7_candidate_modes.pop(key, None)

        demoted = [
            prompt
            for index, prompt in enumerate(previous)
            if index not in selected_original
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

        if self.incumbents:
            selected_positions = [
                index
                for index, prompt in enumerate(previous)
                if prompt.construct_prompt().strip().casefold()
                in selected_keys
            ]
            self._update_front_signature(
                robust_vectors[selected_positions]
            )

    def get_v5_parent_pool(self) -> list[Prompt]:
        verified = super().get_v5_parent_pool()
        if len(verified) <= self.parent_pool_cap:
            return verified

        common = self._common_blocks()
        if not common:
            return verified[: self.parent_pool_cap]

        robust = self._robust_vectors(verified, common)
        selected = self._cap_front(
            robust,
            min(self.parent_pool_cap, len(verified)),
        )
        return [verified[index] for index in selected]
