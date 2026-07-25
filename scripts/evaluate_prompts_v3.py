"""Holdout evaluator for the frozen Tri-Fair v3 manifest profile."""

from __future__ import annotations

import scripts.evaluate_prompts as base
from src.config.v3_profiles import build_v3_dataset_registry

_original = dict(base.ALL_DATASETS)
base.ALL_DATASETS.clear()
base.ALL_DATASETS.update(build_v3_dataset_registry(_original))


if __name__ == "__main__":
    base.main()
