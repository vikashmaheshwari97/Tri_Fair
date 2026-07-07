# MO-CAPO scripts → Tri-Fair scripts

## Keep as active production files

| Tri-Fair file | Role |
|---|---|
| `scripts/experiment.py` | Canonical fresh/resume optimizer entry point |
| `scripts/evaluate_prompts.py` | Untouched-test evaluation of logged candidates |
| `scripts/evaluate_initial_prompts.py` | Budget-zero initial-pool baseline |
| `scripts/prepare_manifests.py` | Pre-create and validate immutable manifests |
| `scripts/_common.py` | Shared schema, atomic I/O, locks, serialization |

## Keep, but do not run during the first 1M study

| File | Later purpose |
|---|---|
| `scripts/run_cost_ablation.py` | Input/output-cost definition sensitivity |
| `scripts/run_hp_sens.py` | Crossover/few-shot sensitivity |

## Archive

The original MO-CAPO scripts and the first Tri-Fair draft are copied into
`scripts/legacy/`. Current jobs must call only the active files above.

## Initial-prompt decision

- Keep `scripts/evaluate_initial_prompts.py`.
- Do not create a second `src/config/initial_prompts.py` file.
- Use `DatasetConfig.initial_prompts` as the only prompt registry.
- Evaluate all 15 prompts for each dataset/model/seed as an `optimizer=init`,
  budget-zero baseline outside the optimizer's 1M token budget.

## Migration from the current Tri-Fair files

1. Move the current `scripts/experiment.py` to
   `scripts/legacy/tri_fair_experiment_v1.py`.
2. Move the current `scripts/evaluate_prompts.py` to
   `scripts/legacy/tri_fair_evaluate_prompts_v1.py`.
3. Copy the new active files from this bundle into `scripts/`.
4. Keep the new ablation wrappers but do not schedule them for the first 1M run.
5. Run `python -m compileall scripts` before cluster submission.
