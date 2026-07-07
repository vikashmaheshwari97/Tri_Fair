# Tri-Fair `src/` migration

This package is a complete source overlay for the Tri-Fair project. It retains
MO-CAPO's original optimizers and baseline datasets while adding the three
fairness datasets, immutable split manifests, group-complete blocks, resumable
checkpoints, Tri-Fair, and NSGA-II-PO-Fair.

## Final structure

```text
src/
├── __init__.py
├── callbacks.py
├── checkpointing.py
├── gepa.py
├── mo_capo.py
├── nsgaii_po.py
├── nsgaii_po_fair.py
├── tri_fair.py
├── utils.py
├── config/
│   ├── __init__.py
│   ├── base_config.py
│   ├── dataset_configs.py
│   ├── initial_prompts.py
│   ├── model_configs.py
│   ├── optimizer_configs.py
│   ├── setup_config.py
│   └── task_descriptions.py
├── helpers/
│   ├── __init__.py
│   ├── llm_creation.py
│   └── task_creation.py
├── fairness/
│   ├── __init__.py
│   ├── base.py
│   ├── bbq.py
│   ├── bios.py
│   ├── block_sampler.py
│   └── civilcomments.py
└── tasks/
    ├── __init__.py
    └── fairness_task.py
```

## Why `initial_prompts.py` is required

The active `dataset_configs.py` imports `src.config.initial_prompts`. The file is
therefore part of the production configuration, not a legacy copy. It contains
all original MO-CAPO pools plus 15 fairness-aware instructions for BBQ,
Bias-in-Bios, and Civil Comments. `evaluate_initial_prompts.py` reads the same
lists through each `DatasetConfig`, so there is one source of truth.

## Import correction

All optimizer modules live inside the `src` package. Use:

```python
from src.mo_capo import MoCAPO
from src.nsgaii_po import NSGAiiPO
```

Do not create duplicate `mo_capo.py` or `nsgaii_po.py` files at repository root.
The included replacement `scripts/experiment.py` contains the corrected imports.

## Migration

1. Commit or back up the current project.
2. Replace the active `src/` directory with the provided `src/` directory.
3. Replace `scripts/experiment.py` with the provided patched version.
4. Optionally replace the two ablation wrappers; they now launch
   `python -m scripts.experiment`, which is safer for package imports.
5. Do not copy source files into `scripts/` or the repository root.

PowerShell example from the project root:

```powershell
Rename-Item src src_before_tri_fair_upgrade
Copy-Item <extracted>\src .\src -Recurse
Copy-Item <extracted>\scripts\experiment.py .\scripts\experiment.py -Force
Copy-Item <extracted>\scripts\run_cost_ablation.py .\scripts\run_cost_ablation.py -Force
Copy-Item <extracted>\scripts\run_hp_sens.py .\scripts\run_hp_sens.py -Force
```

## Validation

```powershell
python -m compileall src scripts
python -m scripts.experiment --help
python -m scripts.evaluate_prompts --help
python -m scripts.evaluate_initial_prompts --help
python -m scripts.prepare_manifests --help
pytest tests -v
```

The manifest preflight should be run on the cluster or another machine with
access to the pinned datasets:

```bash
python -m scripts.prepare_manifests \
  --datasets bbq,bias_in_bios,civil_comments \
  --seeds 42,43,44
```

For publication runs, set `require_official_metadata=True` in the BBQ fairness
configuration after staging or downloading the official commit-pinned metadata
CSV.

## First execution order

1. Prepare manifests.
2. Evaluate the initial-prompt baseline.
3. Run one BBQ / Tri-Fair / seed 42 smoke test at 250k.
4. Resume the same run to 1M.
5. Launch the complete 18-run one-model matrix only after the smoke test passes.
