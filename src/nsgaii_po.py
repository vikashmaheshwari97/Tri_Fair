"""NSGA-II style Prompt Optimizer that reuses MoCAPO internals but skips intensification.

Implementation of NSGA2-PO: A simplified multi-objective prompt optimization algorithm
based on NSGA-II without intensification mechanisms. Used for ablation studies to compare
against the full MO-CAPO algorithm.

This variant evaluates prompts on the full dataset rather than using block-based evaluation
and racing, making it suitable for studying the impact of intensification strategies.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from promptolution.utils.capo_utils import (
    build_few_shot_examples,
    perform_crossover,
    perform_mutation,
)
from promptolution.utils.prompt import Prompt

from src.mo_capo import MoCAPO


class NSGAiiPO(MoCAPO):
    """NSGA-II variant for Prompt-Optimization (full-pop evaluation, no blockwise intensification).

    Key differences from MO-CAPO:
    - Full dataset evaluation (no block-based racing)
    - Standard NSGA-II environmental selection
    - Tournament selection with crowding distance tie-breaking
    """

    def __init__(
        self,
        predictor,
        task,
        meta_llm,
        initial_prompts: Optional[List[str]] = None,
        crossover_template: Optional[str] = None,
        mutation_template: Optional[str] = None,
        crossovers_per_iter: int = 4,
        upper_shots: int = 5,
        cost_per_input_token: float = 1.0,
        cost_per_output_token: float = 1.0,
        check_fs_accuracy: bool = True,
        create_fs_reasoning: bool = True,
        df_few_shots=None,
        callbacks=None,
        config=None,
    ) -> None:
        super().__init__(
            predictor=predictor,
            task=task,
            meta_llm=meta_llm,
            initial_prompts=initial_prompts,
            crossover_template=crossover_template,
            mutation_template=mutation_template,
            crossovers_per_iter=crossovers_per_iter,
            upper_shots=upper_shots,
            cost_per_input_token=cost_per_input_token,
            cost_per_output_token=cost_per_output_token,
            check_fs_accuracy=check_fs_accuracy,
            create_fs_reasoning=create_fs_reasoning,
            df_few_shots=df_few_shots,
            callbacks=callbacks,
            config=config,
        )

        # Ensure eval_strategy is set correctly for full dataset evaluation
        if self.task.eval_strategy != "full":
            self.task.eval_strategy = "full"

        # Cache for tournament selection metrics (rank, crowding distance)
        self._population_metrics: Dict[int, Dict[str, Any]] = {}

    def _pre_optimization_loop(self) -> None:
        """Initialize population with few-shot examples and perform initial environmental selection."""
        population: List[Prompt] = []
        for prompt in self.prompts:
            num_examples = random.randint(0, self.upper_shots)
            few_shots = build_few_shot_examples(
                instruction=prompt.instruction,
                num_examples=num_examples,
                optimizer=self,
            )
            population.append(Prompt(prompt.instruction, few_shots))

        # Evaluate on full dataset
        result = self.task.evaluate(population, self.predictor)
        vecs = self._get_objective_vectors(result)

        # Set self.prompts to the evaluated population
        self.prompts = population

        # Set up incumbents and non_incumbents for logging (no selection needed, init pop is already pop_size)
        fronts = self._non_dominated_sort(vecs)
        self.incumbents = [population[i] for i in fronts[0]]
        self.non_incumbents = [population[i] for front in fronts[1:] for i in front]

        self.scores = vecs.tolist()

        # Update metrics for tournament selection
        self._precompute_selection_metrics()

    def _step(self) -> List[Prompt]:
        """Execute one generation of NSGA-II: crossover, mutation, evaluation, and environmental selection."""
        # Generate offspring through the overridable variation operator.
        new_challengers = self._generate_challengers()

        # Combine parent and offspring populations
        candidates = self.prompts + new_challengers

        # Evaluate all candidates on full dataset - should used cached results!?
        res = self.task.evaluate(candidates, self.predictor)
        vecs = self._get_objective_vectors(res)

        # Environmental selection (NSGA-II)
        self.prompts = self._environmental_selection(candidates, vecs)

        # Get objective vectors for selected population to update scores and sets for logging
        result = self.task.evaluate(
            self.prompts, self.predictor, eval_strategy="evaluated"
        )
        vecs_new = self._get_objective_vectors(result)

        fronts = self._non_dominated_sort(vecs_new)
        self.incumbents = [self.prompts[i] for i in fronts[0]]
        self.non_incumbents = [self.prompts[i] for front in fronts[1:] for i in front]

        self.scores = vecs_new.tolist()

        # Update metrics for next generation's tournament selection
        self._precompute_selection_metrics()

        return self.prompts

    def _precompute_selection_metrics(self) -> None:
        """
        Calculates NDS rank and Crowding Distance for the *entire* current population
        and caches it in self._population_metrics for efficient tournament selection.
        This should be called *once* per generation, after environmental selection.
        """
        self._population_metrics = {}
        if not self.prompts:
            return

        # Get all objective vectors for current population
        result = self.task.evaluate(
            self.prompts, self.predictor, eval_strategy="evaluated"
        )
        obj_vectors = self._get_objective_vectors(result)

        # Run Non-Dominated Sort
        fronts = self._non_dominated_sort(obj_vectors)

        # Calculate metrics for each front
        for rank, front_indices in enumerate(fronts):
            if not front_indices:
                continue

            # Get vectors for *this* front only
            front_vectors = obj_vectors[front_indices]

            # Calculate crowding distance for *this* front
            distances = self._calculate_crowding_distance(front_vectors)

            # Store metrics in the cache (keyed by population index)
            for i, original_pop_idx in enumerate(front_indices):
                self._population_metrics[original_pop_idx] = {
                    "rank": rank,
                    "crowding_distance": distances[i],
                }

    def _tournament_selection(self, parents=None) -> Tuple[Prompt, Prompt]:
        """
        NSGA-II tournament selection based on dominance rank and crowding distance.

        Selects two distinct parents using binary tournament selection:
        1. Compare rank (lower is better)
        2. Compare crowding distance if ranks are equal (higher is better)
        3. Random tiebreaker if both metrics are equal

        Returns:
            Tuple of two distinct parent Prompts
        """
        # Select first parent

        parent1 = self._tournament_select_one(parents)

        # Select second parent (ensure it's different)
        parent2 = self._tournament_select_one(parents)
        while parent2 == parent1:
            parent2 = self._tournament_select_one(parents)

        return parent1, parent2

    def _tournament_select_one(self, parents=None) -> Prompt:
        """
        Binary tournament selection for a single parent.

        Returns:
                Selected Prompt from tournament
        """
        # Select two random individuals
        idx1, idx2 = random.sample(range(len(self.prompts)), 2)
        candidate1 = self.prompts[idx1]
        candidate2 = self.prompts[idx2]

        # Get their cached metrics
        metrics1 = self._population_metrics[idx1]
        metrics2 = self._population_metrics[idx2]

        # 1. Check Rank (lower is better)
        if metrics1["rank"] < metrics2["rank"]:
            return candidate1
        elif metrics2["rank"] < metrics1["rank"]:
            return candidate2

        # 2. Ranks are equal - check Crowding Distance (higher is better)
        if metrics1["crowding_distance"] > metrics2["crowding_distance"]:
            return candidate1
        elif metrics2["crowding_distance"] > metrics1["crowding_distance"]:
            return candidate2

        # 3. Tie in both: pick randomly
        return random.choice([candidate1, candidate2])

    def _environmental_selection(
        self, candidates: List[Prompt], obj_vectors: np.ndarray
    ) -> List[Prompt]:
        """
        NSGA-II environmental selection: select population_size individuals from candidates.

        This follows the original NSGA-II algorithm:
        - Fill population front by front
        - When a front doesn't fully fit, iteratively select individuals with highest crowding distance
        - Recalculate crowding distance after each removal (for scientific accuracy)

        Args:
            candidates: List of candidate Prompts
            obj_vectors: Objective vectors for all candidates (shape: n_candidates x n_objectives)

        Returns:
            List of selected Prompts (size = population_size)
        """
        if len(candidates) <= self.population_size:
            return candidates

        # Perform non-dominated sorting
        fronts = self._non_dominated_sort(obj_vectors)

        # Select individuals front by front
        selected: List[Prompt] = []

        for front in fronts:
            if len(selected) + len(front) <= self.population_size:
                # Include entire front
                selected.extend([candidates[i] for i in front])
            else:
                # Partially include this front using iterative crowding distance
                remaining_slots = self.population_size - len(selected)
                if remaining_slots > 0:
                    # Use iterative selection with CD recalculation (matches original implementation)
                    remaining_front_indices = list(front)  # Copy of indices

                    for _ in range(remaining_slots):
                        # Recalculate CD for remaining individuals in this front
                        front_vectors = obj_vectors[remaining_front_indices]
                        distances = self._calculate_crowding_distance(front_vectors)

                        # Select individual with largest crowding distance
                        max_distance = np.max(distances)
                        tied_indices = np.where(distances == max_distance)[0]

                        if len(tied_indices) > 1:
                            best_local_idx = random.choice(tied_indices)
                        else:
                            best_local_idx = tied_indices[0]

                        best_original_idx = remaining_front_indices[best_local_idx]
                        selected.append(candidates[best_original_idx])

                        # Remove selected individual and recalculate for next iteration
                        remaining_front_indices.pop(best_local_idx)
                break

        return selected
