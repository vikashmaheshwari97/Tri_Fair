# Tri-Fair v7 resource-aware protocol

## What v7 fixes

1. Downstream classification and held-out evaluation use greedy decoding.
2. The meta-model remains stochastic for search diversity.
3. Held-out evaluation is performed on the union of unique prompts from both
   methods in one dataset/seed job.
4. Tri-Fair archive selection uses robust cross-block objectives, reference
   directions, hypervolume contribution, quality-constrained champions, and
   success-adaptive mutation allocation.

## Do not change the datasets

Keep BBQ, Civil Comments, and Bias-in-Bios.  Changing datasets after seeing
results would make comparisons difficult to interpret.  Use smaller budgets and
fewer seeds for screening, not different final datasets.

## Resource-saving promotion ladder

### Gate 0: static

```bash
python -m py_compile \
  src/helpers/generation_control.py \
  src/config/v7_profiles.py \
  src/tri_fair_v7.py \
  scripts/experiment_v7.py \
  scripts/evaluate_shared_union_fronts.py \
  scripts/evaluate_initial_pool_v7.py

PYTHONPATH="$PWD" python -m scripts.validate_v7_overlay
```

### Gate 1: 500K Tri-Fair-only code smoke

Use BBQ seed 72, one task.

### Gate 2: 2M matched BBQ screen

Run Tri-Fair-v7 and NSGA-II-PO-Fair with seed 72, then one shared-union held-out
job.  Promote only if Tri-Fair has:

- no material held-out accuracy loss;
- better held-out hypervolume or nR2;
- a better high-accuracy unfairness/cost operating point.

### Gate 3: 2M one-seed three-dataset screen

Six optimization runs plus three shared-union held-out jobs.

### Gate 4: full study

Only after Gate 3 passes, run 18 optimizations, nine shared-union holdout jobs,
and nine Initial Instructions jobs at seeds 72, 73, and 74.

## Final study commands

Optimization:

```bash
NODELIST=pegasus2 \
TF_DATASETS=bbq,civil_comments,bias_in_bios \
TF_OPTIMIZERS=Tri-Fair-v7,NSGAII-PO-Fair \
TF_SEEDS=72,73,74 \
TF_RESULTS_NAMESPACE=tri_fair_v7_qwen_5m \
MANIFEST_DIR=data/splits_v7 \
BUDGET=5000000 \
MAX_CONCURRENT=2 \
PARTITION=gpu \
MEMORY=160G \
CPUS_PER_TASK=32 \
TIME_LIMIT=24:00:00 \
bash jobs/submit_tri_fair_v7.sh
```

Shared-union held-out evaluation after all optimization runs finish:

```bash
NODELIST=pegasus2 \
TF_DATASETS=bbq,civil_comments,bias_in_bios \
TF_SEEDS=72,73,74 \
TF_OPTIMIZERS=Tri-Fair-v7,NSGAII-PO-Fair \
TF_RESULTS_NAMESPACE=tri_fair_v7_qwen_5m \
MANIFEST_DIR=data/splits_v7 \
MAX_ACTUAL_TOKENS=5000000 \
MAX_CONCURRENT=2 \
PARTITION=gpu \
MEMORY=160G \
CPUS_PER_TASK=32 \
TIME_LIMIT=24:00:00 \
bash jobs/submit_shared_union_holdout_v7.sh
```

Initial Instructions:

```bash
NODELIST=pegasus2 \
TF_SEEDS=72,73,74 \
TF_RESULTS_NAMESPACE=tri_fair_v7_qwen_5m \
MANIFEST_DIR=data/splits_v7 \
MAX_CONCURRENT=2 \
bash jobs/submit_initial_eval_v7.sh
```
