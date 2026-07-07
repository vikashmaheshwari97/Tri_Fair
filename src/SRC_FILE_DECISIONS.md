# Source-file decisions

| Existing or MO-CAPO file | Action |
|---|---|
| `src/callbacks.py` | Replace with three-objective, crash-safe logging version. |
| `src/config/base_config.py` | Replace; adds fairness types and validation. |
| `src/config/dataset_configs.py` | Replace; adds all three fairness adapters and pilot sizes. |
| `src/config/initial_prompts.py` | Keep and upgrade; it is required by `dataset_configs.py`. |
| `src/config/model_configs.py` | Keep with validation and immutable config handling. |
| `src/config/optimizer_configs.py` | Replace; adds Tri-Fair and NSGA-II-PO-Fair defaults. |
| `src/config/setup_config.py` | Keep; fairness datasets use per-dataset overrides. |
| `src/config/task_descriptions.py` | Replace; adds fairness task descriptions. |
| `src/helpers/llm_creation.py` | Add/replace; avoids mutating shared model kwargs. |
| `src/helpers/task_creation.py` | Replace; fixed manifests, row IDs, metadata, and blocks. |
| `src/gepa.py` | Keep for compatibility; GEPA remains an optional baseline. |
| `src/mo_capo.py` | Keep under `src/`; do not duplicate at repository root. |
| `src/nsgaii_po.py` | Keep under `src/` with package-qualified import. |
| `src/utils.py` | Replace; safer shared-engine LLM wrapper copying. |
| `src/checkpointing.py` | Add; required for 1M → 5M → 7.5M continuation. |
| `src/tri_fair.py` | Add; main three-objective optimizer. |
| `src/nsgaii_po_fair.py` | Add; full-evaluation fairness baseline. |
| `src/fairness/*` | Add/replace; dataset-specific fairness metrics and samplers. |
| `src/tasks/fairness_task.py` | Add/replace; set-level quality/fairness evaluation and cache. |
| `src/tasks/__init__.py` | Keep; exports the fairness task types. |
