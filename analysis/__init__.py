"""Tri-Fair experiment analysis package.

The package expects the logging schema produced by ``src.callbacks.ExperimentCallback``
and ``scripts/evaluate_prompts.py`` from the Tri-Fair implementation.
"""

from .config import DEFAULT_BUDGETS, DEFAULT_DATASETS, DEFAULT_OPTIMIZERS, DEFAULT_SEEDS

__all__ = [
    "DEFAULT_BUDGETS",
    "DEFAULT_DATASETS",
    "DEFAULT_OPTIMIZERS",
    "DEFAULT_SEEDS",
]
