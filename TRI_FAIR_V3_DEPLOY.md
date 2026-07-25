# Tri-Fair v3 deployment and experiment guide

This overlay is designed for the current `main` branch of
`vikashmaheshwari97/Tri_Fair`. It adds a new v3 study path without changing the
published v2 optimizer, manifests, results, or figure scripts.

## What is added

- `src/tri_fair_v3.py`: readiness-safe initialization, uncertainty-guarded
  racing, and near-budget archive confirmation.
- `src/fairness/v3_variation.py`: deterministic quality/fairness/cost/balanced
  mutation coverage with duplicate repair.
- `src/config/v3_profiles.py`: frozen, group-complete v3 dataset profiles.
- `scripts/experiment_v3.py`: v3 experiment wrapper; the baseline uses the same
  v3 data manifests and 5M downstream-token budget.
- `scripts/evaluate_checkpoint_fronts.py`: post-hoc development-selected,
  holdout-evaluated nR2/HV/gap checkpoints.
- `scripts/evaluate_initial_pool_v3.py`: separate Initial Instructions baseline,
  selected on development and reported on holdout.
- `analysis/summarize_v3_checkpoints.py`: exact tables.
- `analysis/plot_v3_checkpoint_metrics.py`: exact holdout trajectory figures.
- `jobs/*v3*`: Rocket submission workers and front ends.

## Important study rules

1. Use a new namespace and `data/splits_v3`; never mix v2 and v3 artifacts.
2. Keep `AUTO_EVAL=0` during optimization. Evaluate holdout only after all 5M
   runs finish.
3. Freeze the commit before the 1M stage. The 2M Qwen stage is a technical/budget
   check and is resumed to 5M; do not tune from holdout metrics between stages.
4. Run GPT-OSS-120B from the same frozen commit used for Qwen-3-30B.
5. The changes are designed to improve search efficiency and robustness; they do
   not guarantee that Tri-Fair wins every metric on every dataset.

## Local installation and push

From the local Tri_Fair repository root:

```bash
git checkout main
git pull --ff-only origin main
unzip -o /path/to/tri_fair_v3_code_bundle.zip -d .

python -m py_compile \
  src/config/v3_profiles.py \
  src/fairness/v3_variation.py \
  src/tri_fair_v3.py \
  scripts/experiment_v3.py \
  scripts/evaluate_prompts_v3.py \
  scripts/evaluate_checkpoint_fronts.py \
  scripts/evaluate_initial_pool_v3.py \
  analysis/summarize_v3_checkpoints.py \
  analysis/plot_v3_checkpoint_metrics.py

for file in jobs/*v3*.sh jobs/*v3*.sbatch; do bash -n "$file"; done

git status --short
git add \
  src/config/v3_profiles.py \
  src/fairness/v3_variation.py \
  src/tri_fair_v3.py \
  scripts/experiment_v3.py \
  scripts/evaluate_prompts_v3.py \
  scripts/evaluate_checkpoint_fronts.py \
  scripts/evaluate_initial_pool_v3.py \
  analysis/summarize_v3_checkpoints.py \
  analysis/plot_v3_checkpoint_metrics.py \
  jobs/submit_tri_fair_v3.sh \
  jobs/tri_fair_v3_main.sbatch \
  jobs/run_eval_v3.sbatch \
  jobs/submit_checkpoint_eval_v3.sh \
  jobs/checkpoint_eval_v3.sbatch \
  jobs/submit_initial_eval_v3.sh \
  jobs/initial_eval_v3.sbatch \
  TRI_FAIR_V3_DEPLOY.md

git commit -m "Add readiness-safe Tri-Fair v3 study pipeline"
git push origin main
```

Do not add generated `results/`, `logs/`, `analysis/output/`, manifests, model
files, or `__pycache__/` directories.

## Pull and validate on Rocket

```bash
cd "$HOME/projects/Tri_Fair"
git pull --ff-only origin main
source "$HOME/venvs/tri-fair/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m py_compile \
  src/config/v3_profiles.py src/fairness/v3_variation.py src/tri_fair_v3.py \
  scripts/experiment_v3.py scripts/evaluate_checkpoint_fronts.py \
  scripts/evaluate_initial_pool_v3.py analysis/summarize_v3_checkpoints.py \
  analysis/plot_v3_checkpoint_metrics.py

for file in jobs/*v3*.sh jobs/*v3*.sbatch; do bash -n "$file"; done
```

## Qwen-3-30B: staged 2M to 5M

Use one namespace for both stages so the 5M jobs resume the 2M checkpoints.
First inspect the dry-run matrix:

```bash
TF_MODELS=qwen-3-30b \
TF_DATASETS=bbq,civil_comments,bias_in_bios \
TF_OPTIMIZERS=Tri-Fair-v3,NSGAII-PO-Fair \
TF_SEEDS=42,43,44 \
TF_RESULTS_NAMESPACE=tri_fair_v3_qwen_5m \
MANIFEST_DIR=data/splits_v3 \
BUDGET=2000000 RUN_MODE=fresh AUTO_EVAL=0 \
MAX_OUTPUT_TOKENS=16 META_MAX_OUTPUT_TOKENS=256 \
MAX_CONCURRENT=2 PARTITION=gpu MEMORY=160G TIME_LIMIT=24:00:00 \
DRY_RUN=1 bash jobs/submit_tri_fair_v3.sh
```

Submit the 2M stage by changing `DRY_RUN=1` to `DRY_RUN=0`. This creates 18
array tasks: 3 datasets × 2 methods × 3 seeds.

After all 18 tasks complete successfully, resume the same namespace to 5M:

```bash
TF_MODELS=qwen-3-30b \
TF_DATASETS=bbq,civil_comments,bias_in_bios \
TF_OPTIMIZERS=Tri-Fair-v3,NSGAII-PO-Fair \
TF_SEEDS=42,43,44 \
TF_RESULTS_NAMESPACE=tri_fair_v3_qwen_5m \
MANIFEST_DIR=data/splits_v3 \
BUDGET=5000000 RUN_MODE=resume AUTO_EVAL=0 \
MAX_OUTPUT_TOKENS=16 META_MAX_OUTPUT_TOKENS=256 \
MAX_CONCURRENT=2 PARTITION=gpu MEMORY=160G TIME_LIMIT=24:00:00 \
bash jobs/submit_tri_fair_v3.sh
```

Useful checks:

```bash
squeue -u "$USER"
find results/tri_fair_v3_qwen_5m -name 'stage_2000000.done.json' | wc -l
find results/tri_fair_v3_qwen_5m -name 'stage_5000000.done.json' | wc -l
```

Each count should be 18 after the corresponding stage.

## Qwen exact checkpoint and Initial evaluations

After every 5M optimization task is complete:

```bash
TF_MODELS=qwen-3-30b \
TF_DATASETS=bbq,civil_comments,bias_in_bios \
TF_OPTIMIZERS=Tri-Fair-v3,NSGAII-PO-Fair \
TF_SEEDS=42,43,44 \
TF_RESULTS_NAMESPACE=tri_fair_v3_qwen_5m \
MANIFEST_DIR=data/splits_v3 MAX_OUTPUT_TOKENS=16 \
CHECKPOINTS=2000000,3000000,4000000,5000000 \
MAX_CONCURRENT=2 PARTITION=gpu MEMORY=160G TIME_LIMIT=18:00:00 \
bash jobs/submit_checkpoint_eval_v3.sh

TF_MODELS=qwen-3-30b \
TF_DATASETS=bbq,civil_comments,bias_in_bios \
TF_SEEDS=42,43,44 \
TF_RESULTS_NAMESPACE=tri_fair_v3_qwen_5m \
MANIFEST_DIR=data/splits_v3 N_INIT_PROMPTS=12 MAX_OUTPUT_TOKENS=16 \
MAX_CONCURRENT=2 PARTITION=gpu MEMORY=160G TIME_LIMIT=12:00:00 \
bash jobs/submit_initial_eval_v3.sh
```

The checkpoint jobs evaluate each unique current-archive prompt only once on
holdout, then reuse that evaluation at every checkpoint where it appears.

Build tables and figures:

```bash
python analysis/summarize_v3_checkpoints.py \
  --results-root results/tri_fair_v3_qwen_5m \
  --output-dir analysis/output/v3/qwen_3_30b

python analysis/plot_v3_checkpoint_metrics.py \
  --run-metrics analysis/output/v3/qwen_3_30b/v3_checkpoint_run_metrics.csv \
  --output-dir analysis/output/v3/qwen_3_30b/figures
```

Key outputs:

- `v3_nr2_hv_gap_table.csv`
- `v3_nr2_hv_gap_table.md`
- `v3_checkpoint_run_metrics.csv`
- exact holdout nR2, optimistic-HV, pessimistic-HV, and gap figures per dataset.

## GPT-OSS-120B: same frozen method

Do not change the source files after the Qwen study. Use a new results namespace
but reuse the same immutable manifests:

```bash
TF_MODELS=gpt-oss-120b \
TF_DATASETS=bbq,civil_comments,bias_in_bios \
TF_OPTIMIZERS=Tri-Fair-v3,NSGAII-PO-Fair \
TF_SEEDS=42,43,44 \
TF_RESULTS_NAMESPACE=tri_fair_v3_gptoss_5m \
MANIFEST_DIR=data/splits_v3 \
BUDGET=5000000 RUN_MODE=fresh AUTO_EVAL=0 \
MAX_OUTPUT_TOKENS=96 META_MAX_OUTPUT_TOKENS=256 \
MAX_CONCURRENT=2 PARTITION=gpu MEMORY=300G TIME_LIMIT=48:00:00 \
bash jobs/submit_tri_fair_v3.sh
```

GPT-OSS has a substantially larger output allowance, so the safest main study is
a direct fresh 5M run. The job still writes 2M/3M/4M/5M checkpoints for recovery
and post-hoc exact metrics.

Then submit checkpoint and Initial evaluations with the GPT namespace,
`MAX_OUTPUT_TOKENS=96`, `MEMORY=300G`, and an appropriate time limit. Finally:

```bash
python analysis/summarize_v3_checkpoints.py \
  --results-root results/tri_fair_v3_gptoss_5m \
  --output-dir analysis/output/v3/gpt_oss_120b

python analysis/plot_v3_checkpoint_metrics.py \
  --run-metrics analysis/output/v3/gpt_oss_120b/v3_checkpoint_run_metrics.csv \
  --output-dir analysis/output/v3/gpt_oss_120b/figures
```
