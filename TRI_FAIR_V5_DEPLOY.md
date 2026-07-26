# Deploy and run Tri-Fair v5

## Local installation

From Git Bash in the repository root:

```bash
cd ~/PycharmProjects/Tri_Fair
unzip -o ~/Downloads/tri_fair_v5_robust_archive_patch_20260726.zip -d .

python -m py_compile \
  src/config/v5_seed_pool.py \
  src/config/v5_profiles.py \
  src/fairness/v5_variation.py \
  src/tri_fair_v5.py \
  scripts/experiment_v5.py \
  scripts/evaluate_initial_pool_v5.py \
  analysis/summarize_v5_checkpoints.py

for file in jobs/*v5*.sh jobs/*v5*.sbatch; do bash -n "$file"; done

chmod +x jobs/*v5*.sh jobs/*v5*.sbatch

git add \
  src/config/v5_seed_pool.py \
  src/config/v5_profiles.py \
  src/fairness/v5_variation.py \
  src/tri_fair_v5.py \
  scripts/experiment_v5.py \
  scripts/evaluate_initial_pool_v5.py \
  analysis/summarize_v5_checkpoints.py \
  jobs/submit_tri_fair_v5.sh \
  jobs/tri_fair_v5_main.sbatch \
  jobs/submit_all_step_holdout_eval_v5.sh \
  jobs/all_step_holdout_eval_v5.sbatch \
  jobs/submit_initial_eval_v5.sh \
  jobs/initial_eval_v5.sbatch \
  TRI_FAIR_V5_PROTOCOL.md \
  TRI_FAIR_V5_DEPLOY.md

git commit -m "Add robust shared-pool Tri-Fair v5"
git push origin main
```

## Pull on Rocket

```bash
cd "$HOME/projects/Tri_Fair"
git pull --ff-only origin main
source "$HOME/venvs/tri-fair/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m py_compile \
  src/config/v5_seed_pool.py \
  src/config/v5_profiles.py \
  src/fairness/v5_variation.py \
  src/tri_fair_v5.py \
  scripts/experiment_v5.py \
  scripts/evaluate_initial_pool_v5.py \
  analysis/summarize_v5_checkpoints.py
for file in jobs/*v5*.sh jobs/*v5*.sbatch; do bash -n "$file"; done

git rev-parse HEAD | tee TRI_FAIR_V5_FROZEN_COMMIT.txt
```

## Separate technical smoke test

Do not evaluate holdout for the smoke test.

```bash
NODELIST=pegasus2 \
QWEN_LOCAL_SNAPSHOT="$QWEN_LOCAL_SNAPSHOT" \
TF_HF_HOME="$TF_HF_HOME" \
TF_HF_HUB_CACHE="$TF_HF_HUB_CACHE" \
TF_HF_DATASETS_CACHE="$TF_HF_DATASETS_CACHE" \
TF_MODELS=qwen-3-30b \
TF_DATASETS=bbq \
TF_OPTIMIZERS=Tri-Fair-v5,NSGAII-PO-Fair \
TF_SEEDS=52 \
TF_RESULTS_NAMESPACE=tri_fair_v5_smoke \
MANIFEST_DIR=data/splits_v5_smoke \
BUDGET=250000 \
RUN_MODE=fresh \
AUTO_EVAL=0 \
MAX_OUTPUT_TOKENS=16 \
META_MAX_OUTPUT_TOKENS=256 \
N_INIT_PROMPTS=12 \
MAX_STEPS=4000 \
MAX_CONCURRENT=1 \
PARTITION=gpu \
MEMORY=160G \
CPUS_PER_TASK=32 \
TIME_LIMIT=08:00:00 \
DRY_RUN=0 \
bash jobs/submit_tri_fair_v5.sh
```

Only verify completion, files, and absence of exceptions.  Do not use smoke-test
metrics to tune v5.

## Main Qwen 5M optimization: all three datasets

Run all 18 tasks from the frozen commit.  `MAX_CONCURRENT=2` keeps two GPU tasks
active at a time.

```bash
NODELIST=pegasus2 \
QWEN_LOCAL_SNAPSHOT="$QWEN_LOCAL_SNAPSHOT" \
TF_HF_HOME="$TF_HF_HOME" \
TF_HF_HUB_CACHE="$TF_HF_HUB_CACHE" \
TF_HF_DATASETS_CACHE="$TF_HF_DATASETS_CACHE" \
TF_MODELS=qwen-3-30b \
TF_DATASETS=bbq,civil_comments,bias_in_bios \
TF_OPTIMIZERS=Tri-Fair-v5,NSGAII-PO-Fair \
TF_SEEDS=52,53,54 \
TF_RESULTS_NAMESPACE=tri_fair_v5_qwen_5m \
MANIFEST_DIR=data/splits_v5 \
BUDGET=5000000 \
RUN_MODE=fresh \
AUTO_EVAL=0 \
MAX_OUTPUT_TOKENS=16 \
META_MAX_OUTPUT_TOKENS=256 \
N_INIT_PROMPTS=12 \
MAX_STEPS=4000 \
MAX_CONCURRENT=2 \
PARTITION=gpu \
MEMORY=160G \
CPUS_PER_TASK=32 \
TIME_LIMIT=30:00:00 \
DRY_RUN=0 \
bash jobs/submit_tri_fair_v5.sh
```

Expected matrix: 3 datasets × 2 methods × 3 seeds = 18 tasks.

## All-real-step holdout evaluation

Submit only after all 18 optimization tasks are complete and no failed markers
exist.

```bash
NODELIST=pegasus2 \
QWEN_LOCAL_SNAPSHOT="$QWEN_LOCAL_SNAPSHOT" \
TF_HF_HOME="$TF_HF_HOME" \
TF_HF_HUB_CACHE="$TF_HF_HUB_CACHE" \
TF_HF_DATASETS_CACHE="$TF_HF_DATASETS_CACHE" \
TF_MODELS=qwen-3-30b \
TF_DATASETS=bbq,civil_comments,bias_in_bios \
TF_OPTIMIZERS=Tri-Fair-v5,NSGAII-PO-Fair \
TF_SEEDS=52,53,54 \
TF_RESULTS_NAMESPACE=tri_fair_v5_qwen_5m \
MANIFEST_DIR=data/splits_v5 \
MIN_ACTUAL_TOKENS=0 \
MAX_ACTUAL_TOKENS=5000000 \
MAX_OUTPUT_TOKENS=16 \
ALL_STEP_REPLACE_OUTPUT=1 \
ALL_STEP_FORCE=0 \
MAX_CONCURRENT=2 \
PARTITION=gpu \
MEMORY=160G \
CPUS_PER_TASK=32 \
TIME_LIMIT=30:00:00 \
bash jobs/submit_all_step_holdout_eval_v5.sh
```

## Shared Initial Instructions evaluation

```bash
NODELIST=pegasus2 \
QWEN_LOCAL_SNAPSHOT="$QWEN_LOCAL_SNAPSHOT" \
TF_HF_HOME="$TF_HF_HOME" \
TF_HF_HUB_CACHE="$TF_HF_HUB_CACHE" \
TF_HF_DATASETS_CACHE="$TF_HF_DATASETS_CACHE" \
TF_MODELS=qwen-3-30b \
TF_DATASETS=bbq,civil_comments,bias_in_bios \
TF_SEEDS=52,53,54 \
TF_RESULTS_NAMESPACE=tri_fair_v5_qwen_5m \
MANIFEST_DIR=data/splits_v5 \
N_INIT_PROMPTS=12 \
MAX_OUTPUT_TOKENS=16 \
MAX_CONCURRENT=2 \
PARTITION=gpu \
MEMORY=160G \
CPUS_PER_TASK=32 \
TIME_LIMIT=16:00:00 \
bash jobs/submit_initial_eval_v5.sh
```

## Summaries

```bash
for dataset in bbq civil_comments bias_in_bios; do
  rm -rf "analysis/output/tri_fair_v5_qwen_${dataset}_allstep"
  python analysis/summarize_v5_checkpoints.py \
    --results-root \
      "results/tri_fair_v5_qwen_5m/qwen-3-30b/${dataset}" \
    --initial-root \
      "results/tri_fair_v5_qwen_5m/initial/qwen-3-30b/${dataset}" \
    --output-dir \
      "analysis/output/tri_fair_v5_qwen_${dataset}_allstep"
done
```
