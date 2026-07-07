# Tri-Fair scripts

This directory contains the canonical Python entry points used by the SLURM
jobs and the analysis pipeline.

## Active main-study files

- `experiment.py` — fresh and resumable optimization runs.
- `evaluate_prompts.py` — test evaluation of development-selected prompts.
- `evaluate_initial_prompts.py` — budget-zero baseline for the full configured
  initial-instruction pool.
- `prepare_manifests.py` — optional but recommended manifest preflight.
- `_common.py` — shared atomic I/O, prompt serialization, result schema, and
  manifest locking.

## Later-study wrappers

- `run_cost_ablation.py` — one cost-definition variant per invocation.
- `run_hp_sens.py` — one crossover/few-shot sensitivity configuration per
  invocation.

Do not run the two sensitivity wrappers for the first 1M main experiment.

## Initial prompts

Tri-Fair does **not** use a separate `src/config/initial_prompts.py` registry.
Each dataset owns its prompt pool through
`src/config/dataset_configs.py -> DatasetConfig.initial_prompts`. This prevents
two prompt lists from drifting apart.

The evaluation script is still needed because it creates the `optimizer=init`,
budget-zero baseline used by the analysis tables and Pareto comparisons. It
evaluates all 15 configured prompts by default.

## Recommended order

From the repository root:

```bash
python scripts/prepare_manifests.py \
  --datasets bbq,bias_in_bios,civil_comments \
  --seeds 42,43,44
```

Then run one smoke test through the job layer:

```bash
AUTO_EVAL=1 sbatch jobs/tri_fair_single.sbatch \
  qwen-3-30b bbq Tri-Fair 42 250000 fresh
```

Evaluate the initial-prompt baseline separately:

```bash
python scripts/evaluate_initial_prompts.py \
  --random-seed 42 --dataset bbq --model qwen-3-30b
```

The main 1M array continues to use:

```bash
BUDGET=1000000 RUN_MODE=auto bash jobs/submit_tri_fair.sh
```

## Output schema

`step_results.parquet` contains development-side population records and
cumulative downstream/meta token counters. `eval.parquet` contains explicit
`dev_*` and `test_*` columns, including:

- quality
- weighted inference cost
- unfairness
- fairness readiness
- per-group support and diagnostics
- objective vectors in maximize-all form

All Parquet and JSON outputs are written atomically.

## Existing files

Move old MO-CAPO scripts and the previous Tri-Fair script versions into
`scripts/legacy/`. They are retained for provenance only and must not be called
by current jobs.
