#!/usr/bin/env bash
#SBATCH --job-name=tf-v6-initial
#SBATCH --partition=gpu
#SBATCH --nodelist=pegasus2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tri_fair_v6_initial_%A_%a.out
#SBATCH --error=logs/tri_fair_v6_initial_%A_%a.err

set -Eeuo pipefail
module purge
module --ignore_cache load python/3.11.14
module --ignore_cache load python-venv/1.0
source "$HOME/venvs/tri-fair/bin/activate"
export TF_PYTHON="$HOME/venvs/tri-fair/bin/python"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
source jobs/lib/common.sh
source jobs/lib/model_cache_v3.sh

TF_MODELS="${TF_MODELS:-qwen-3-30b}"
TF_DATASETS="${TF_DATASETS:-bbq,civil_comments,bias_in_bios}"
TF_SEEDS="${TF_SEEDS:-62,63,64}"
TF_RESULTS_NAMESPACE="${TF_RESULTS_NAMESPACE:-tri_fair_v6_qwen_5m}"
MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v6}"
N_INIT_PROMPTS="${N_INIT_PROMPTS:-12}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"

declare -a models datasets seeds
tf_split_csv "$TF_MODELS" models
tf_split_csv "$TF_DATASETS" datasets
tf_split_csv "$TF_SEEDS" seeds
n_models=${#models[@]}; n_datasets=${#datasets[@]}; n_seeds=${#seeds[@]}
idx="${SLURM_ARRAY_TASK_ID:-0}"
seed_idx=$((idx % n_seeds)); idx=$((idx / n_seeds))
dataset_idx=$((idx % n_datasets)); idx=$((idx / n_datasets))
model_idx=$((idx % n_models))
model="${models[$model_idx]}"; dataset="${datasets[$dataset_idx]}"; seed="${seeds[$seed_idx]}"

tf_configure_v3_model_cache "$model"
output_dir="$ROOT/results/$TF_RESULTS_NAMESPACE/initial/$model/$dataset/seed$seed"
mkdir -p "$output_dir"
declare -a PYTHON_CMD
tf_python_command PYTHON_CMD

"${PYTHON_CMD[@]}" scripts/evaluate_initial_pool_v6.py \
  --dataset "$dataset" \
  --model "$model" \
  --random-seed "$seed" \
  --n-init-prompts "$N_INIT_PROMPTS" \
  --manifest-dir "$MANIFEST_DIR" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --output-file "$output_dir/initial_eval.parquet"
