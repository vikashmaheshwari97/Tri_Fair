"""Full-evaluation three-objective NSGA-II prompt-optimization baseline."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from promptolution.utils.prompt import Prompt

from src.nsgaii_po import NSGAiiPO
from src.checkpointing import ResumableOptimizerMixin
from src.tasks.fairness_task import FairnessEvalResult, FairnessTask


class NSGAiiPOFair(ResumableOptimizerMixin, NSGAiiPO):
    """NSGA-II-PO with quality, inference cost, and unfairness objectives.

    Unlike Tri-Fair, every offspring is evaluated on the complete development
    set before environmental selection.  It therefore isolates the benefit of
    fairness-aware intensification and budget allocation.
    """

    supports_multi_objective = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.task, FairnessTask):
            raise TypeError("NSGAiiPOFair requires FairnessTask")
        self.n_objectives = 3
        self.current_step = 0

    def _get_objective_vectors(self, result: FairnessEvalResult) -> np.ndarray:
        quality = np.asarray(result.agg_scores, dtype=float).reshape(-1, 1)
        cost = (
            self.cost_per_input_token * np.asarray(result.agg_input_tokens, dtype=float)
            + self.cost_per_output_token
            * np.asarray(result.agg_output_tokens, dtype=float)
        ).reshape(-1, 1)
        unfairness = np.asarray(result.fairness_loss, dtype=float).reshape(-1, 1)
        return np.hstack([quality, -cost, -unfairness])

    def _pre_optimization_loop(self) -> None:
        self.current_step = 0
        super()._pre_optimization_loop()
        self._assert_fairness_ready(self.prompts)

    def _step(self):
        self.current_step += 1
        prompts = super()._step()
        self._assert_fairness_ready(prompts)
        return prompts

    def _assert_fairness_ready(self, prompts: Sequence[Prompt]) -> None:
        result = self.task.evaluate(list(prompts), self.predictor, eval_strategy="full")
        not_ready = [
            prompt.construct_prompt()[:80]
            for prompt, ready in zip(prompts, result.fairness_ready)
            if not bool(ready)
        ]
        if not_ready:
            raise RuntimeError(
                "Full development evaluation did not meet fairness support requirements for "
                f"{len(not_ready)} prompts. Check the manifest/block sampler."
            )

    def get_fairness_record(
        self,
        prompt: Prompt,
        blocks: Optional[Sequence[int]] = None,
    ) -> Dict:
        result = self.task.evaluate([prompt], self.predictor, eval_strategy="full")
        cost = float(
            self.cost_per_input_token * result.agg_input_tokens[0]
            + self.cost_per_output_token * result.agg_output_tokens[0]
        )
        record = self.task.get_last_record(prompt)
        record["cost"] = cost
        return record
