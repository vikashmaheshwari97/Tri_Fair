"""Fairness objectives for Tri-Fair."""

from src.fairness.base import FairnessMetricResult
from src.fairness.bbq import compute_bbq_fairness
from src.fairness.bios import compute_bios_fairness
from src.fairness.civilcomments import compute_civilcomments_fairness

__all__ = [
    "FairnessMetricResult",
    "compute_bbq_fairness",
    "compute_bios_fairness",
    "compute_civilcomments_fairness",
]
