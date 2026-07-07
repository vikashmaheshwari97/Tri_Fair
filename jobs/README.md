# Tri_Fair SLURM jobs

This directory replaces the MO-CAPO job layer with a resumable, stage-aware
Tri-Fair workflow. Each array task still executes exactly one
`model × dataset × optimizer × seed` configuration.

## Active files

- `submit_tri_fair.sh` — dynamically computes and submits the array.
- `tri_fair_main.sbatch` — array worker for optimization.
- `tri_fair_single.sbatch` — one-configuration pilot/debug worker.
- `run_eval.sbatch` — idempotent test-manifest evaluation for one completed budget.
- `submit_evaluations.sh` — recovery submission for missing/failed evaluations.
- `lib/common.sh` — shared validation, path, checkpoint, and environment helpers.

## What to do with the existing `tri_fair_main.sbatch`

Preserve the current file once for provenance:

```bash
mkdir -p jobs/legacy
mv jobs/tri_fair_main.sbatch jobs/legacy/tri_fair_main_v0.sbatch
```

Then copy the new `tri_fair_main.sbatch` into `jobs/`. Do not keep two active
workers with the same purpose.

The replacement fixes four important issues in the initial file:

1. It removes the hard-coded `0-53` array, because the first stage uses one model
   and therefore has 18 tasks, not 54.
2. It separates optimization from holdout evaluation.
3. It prevents duplicate jobs from writing the same run concurrently.
4. It supports safe fresh/resume/auto operation for 1M → 5M → 7.5M.

## Installation

From the repository root:

```bash
chmod +x jobs/*.sh jobs/*.sbatch jobs/lib/common.sh
mkdir -p logs
```

No Rocket-specific partition, QoS, or account is hard-coded. Supply those at
submission time through environment variables such as `PARTITION`, `QOS`, and
`ACCOUNT`. Command-line `sbatch` overrides take precedence over the defaults in
the worker.

## First smoke test

Run one BBQ/Tri-Fair/seed-42 configuration at 250k:

```bash
mkdir -p logs
AUTO_EVAL=1 sbatch jobs/tri_fair_single.sbatch \
  qwen-3-30b bbq Tri-Fair 42 250000 fresh
```

After checking the logs and generated checkpoint, continue that same configuration:

```bash
AUTO_EVAL=1 sbatch jobs/tri_fair_single.sbatch \
  qwen-3-30b bbq Tri-Fair 42 1000000 resume
```

## Full first-stage matrix

The default first stage is:

```text
1 model × 3 datasets × 2 optimizers × 3 seeds = 18 tasks
```

Preview without submitting:

```bash
DRY_RUN=1 BUDGET=1000000 bash jobs/submit_tri_fair.sh
```

Submit the 1M stage, with at most three simultaneous GPU jobs:

```bash
BUDGET=1000000 RUN_MODE=auto MAX_CONCURRENT=3 \
  bash jobs/submit_tri_fair.sh
```

The 1M checkpoint schedule is:

```text
250k, 500k, 750k, 1M
```

`RUN_MODE=auto` resumes the pilot configuration when a valid checkpoint exists
and starts the remaining configurations fresh.

## Continue to 5M

Every selected configuration must have a valid 1M checkpoint. The submitter
performs this preflight before creating the array:

```bash
BUDGET=5000000 RUN_MODE=resume MAX_CONCURRENT=3 \
  bash jobs/submit_tri_fair.sh
```

The 5M continuation saves checkpoints at:

```text
2M, 3M, 4M, 5M
```

## Continue to 7.5M

```bash
BUDGET=7500000 RUN_MODE=resume MAX_CONCURRENT=3 \
  bash jobs/submit_tri_fair.sh
```

The final continuation saves:

```text
6M, 7M, 7.5M
```

An optional 2M stage is supported by the helper but is not required by the
current 1M → 5M → 7.5M plan.

## Expanding to all three models later

```bash
TF_MODELS='gpt-oss-120b,qwen-3-30b,mistral-3-24b' \
BUDGET=1000000 RUN_MODE=auto MAX_CONCURRENT=3 \
  bash jobs/submit_tri_fair.sh
```

That matrix has 54 tasks. The submitter computes this automatically.

## Holdout evaluation policy

By default, each successful optimization stage submits one separate evaluation
job for that stage's final checkpoint: 1M, 5M, or 7.5M. Intermediate 250k,
500k, and 750k snapshots remain available for development-side anytime analysis
without repeatedly consulting the untouched test set.

To disable automatic test evaluation:

```bash
AUTO_EVAL=0 BUDGET=1000000 bash jobs/submit_tri_fair.sh
```

Submit missing evaluations later:

```bash
BUDGET=1000000 bash jobs/submit_evaluations.sh
```

`run_eval.sbatch` maps the requested token budget to the first optimizer step at
or above that checkpoint and writes the corresponding prompts into
`eval.parquet`. Re-running it is idempotent unless `FORCE_EVAL=1` is set.

## Initial-prompt baseline

Do not use MO-CAPO's original `evaluate_initial_prompts.sbatch` yet. Its Python
counterpart records only score and token counts and does not preserve Tri-Fair's
fairness diagnostics or fixed split manifests. Upgrade the Python evaluator in
the scripts-folder phase, then add its matching job worker.

## Main-study constants

The jobs enforce or pass these starting settings:

- initial population: 6
- downstream prediction output limit: 16 tokens
- seeds: 42, 43, 44
- datasets: BBQ, Bias in Bios, Civil Comments
- optimizers: Tri-Fair and NSGA-II-PO-Fair

Crossovers per iteration (2), maximum few-shot examples (3), the 15-item
instruction pool, and dataset-specific development/few-shot/block sizes belong
in the Python configuration modules rather than in SLURM scripts.
