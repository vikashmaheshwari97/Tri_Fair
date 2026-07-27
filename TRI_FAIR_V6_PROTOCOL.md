# Tri-Fair v6 protocol

## Purpose

Tri-Fair v6 is an algorithm-development study.  It improves only Tri-Fair and
keeps `NSGAII-PO-Fair` unchanged as the matched baseline.

## Non-negotiable comparison rules

1. Both methods receive the same model, raw initial prompt pool, fresh manifests,
   few-shot split, seeds, generation limits and 5M downstream-token cap.
2. No holdout result is used by either optimizer.
3. Use fresh seeds `62,63,64` and `data/splits_v6` because v5 holdout results have
   already been inspected.
4. Run all 18 optimizations, all 18 holdout evaluations and all 9 Initial
   evaluations.
5. Report all seeds and all metrics, including failures and trade-offs.
6. Do not alter baseline scores, discard strong baseline candidates, or use
   different evaluation logic for the two methods.

## V6 changes

- six-direction-capable adaptive portfolio with guaranteed primary coverage;
- dataset-structural direction weighting;
- pure objective champions plus quality-constrained cost/fairness champions;
- archive cap 12 with crowding diversity;
- stagnation-triggered exploration;
- extra multiclass quality example for Bias-in-Bios quality mutations;
- challenger batch shrinking near the hard budget boundary;
- fresh study namespace and manifests.

## Smoke-test ladder

1. `250000` tokens, one dataset/seed/method, syntax and artifact gate.
2. `2000000` tokens, both methods, one dataset/seed.
3. `3500000` tokens, NSGA generation and Tri-Fair adaptive-search gate.
4. Only after those pass, submit the complete 18-task 5M study.

## Full optimization

```bash
NODELIST=pegasus2 \
TF_MODELS=qwen-3-30b \
TF_DATASETS=bbq,civil_comments,bias_in_bios \
TF_OPTIMIZERS=Tri-Fair-v6,NSGAII-PO-Fair \
TF_SEEDS=62,63,64 \
TF_RESULTS_NAMESPACE=tri_fair_v6_qwen_5m \
MANIFEST_DIR=data/splits_v6 \
BUDGET=5000000 \
MAX_OUTPUT_TOKENS=16 \
MAX_CONCURRENT=2 \
PARTITION=gpu \
MEMORY=160G \
CPUS_PER_TASK=32 \
TIME_LIMIT=24:00:00 \
bash jobs/submit_tri_fair_v6.sh
```

## Initial Instructions

```bash
NODELIST=pegasus2 \
TF_SEEDS=62,63,64 \
TF_RESULTS_NAMESPACE=tri_fair_v6_qwen_5m \
MANIFEST_DIR=data/splits_v6 \
MAX_CONCURRENT=2 \
bash jobs/submit_initial_eval_v6.sh
```

## Holdout

Run only after every optimization is complete and the source commit is frozen.

```bash
NODELIST=pegasus2 \
TF_OPTIMIZERS=Tri-Fair-v6,NSGAII-PO-Fair \
TF_SEEDS=62,63,64 \
TF_RESULTS_NAMESPACE=tri_fair_v6_qwen_5m \
MANIFEST_DIR=data/splits_v6 \
MIN_ACTUAL_TOKENS=0 \
MAX_ACTUAL_TOKENS=5000000 \
ALL_STEP_REPLACE_OUTPUT=1 \
MAX_CONCURRENT=2 \
PARTITION=gpu \
MEMORY=160G \
CPUS_PER_TASK=32 \
TIME_LIMIT=24:00:00 \
bash jobs/submit_all_step_holdout_eval_v6.sh
```
