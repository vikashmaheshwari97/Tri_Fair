#!/usr/bin/env bash
#SBATCH --job-name=tf-v3-allstep-holdout
#SBATCH --partition=gpu
#SBATCH --nodelist=firefly1,firefly2,firefly3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tri_fair_v3_allstep_%A_%a.out
#SBATCH --error=logs/tri_fair_v3_allstep_%A_%a.err

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
TF_OPTIMIZERS="${TF_OPTIMIZERS:-Tri-Fair-v3,NSGAII-PO-Fair}"
TF_SEEDS="${TF_SEEDS:-42,43,44}"
TF_RESULTS_NAMESPACE="${TF_RESULTS_NAMESPACE:-tri_fair_v3_qwen_5m}"
MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v3}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"
MIN_ACTUAL_TOKENS="${MIN_ACTUAL_TOKENS:-0}"
MAX_ACTUAL_TOKENS="${MAX_ACTUAL_TOKENS:-5000000}"
ALL_STEP_REPLACE_OUTPUT="${ALL_STEP_REPLACE_OUTPUT:-1}"
ALL_STEP_FORCE="${ALL_STEP_FORCE:-0}"

declare -a models datasets optimizers seeds
tf_split_csv "$TF_MODELS" models
tf_split_csv "$TF_DATASETS" datasets
tf_split_csv "$TF_OPTIMIZERS" optimizers
tf_split_csv "$TF_SEEDS" seeds

n_models=${#models[@]}
n_datasets=${#datasets[@]}
n_optimizers=${#optimizers[@]}
n_seeds=${#seeds[@]}
total=$((n_models * n_datasets * n_optimizers * n_seeds))

idx="${SLURM_ARRAY_TASK_ID:-0}"
((idx < total)) || tf_die "Array task $idx is outside matrix size $total"

seed_idx=$((idx % n_seeds)); idx=$((idx / n_seeds))
optimizer_idx=$((idx % n_optimizers)); idx=$((idx / n_optimizers))
dataset_idx=$((idx % n_datasets)); idx=$((idx / n_datasets))
model_idx=$((idx % n_models))

model="${models[$model_idx]}"
dataset="${datasets[$dataset_idx]}"
optimizer="${optimizers[$optimizer_idx]}"
seed="${seeds[$seed_idx]}"

tf_configure_v3_model_cache "$model"

output_dir="$(tf_run_output_dir "$ROOT" "$model" "$dataset" "$optimizer" "$seed")"
logging_dir="$(tf_read_logging_dir "$output_dir")" || tf_die "Missing run $output_dir"

declare -a PYTHON_CMD
tf_python_command PYTHON_CMD

replace_args=()
tf_is_true "$ALL_STEP_REPLACE_OUTPUT" && replace_args+=(--replace-output)
force_args=()
tf_is_true "$ALL_STEP_FORCE" && force_args+=(--force)

tf_log "All-step holdout model=$model dataset=$dataset optimizer=$optimizer seed=$seed actual_tokens=$MIN_ACTUAL_TOKENS..$MAX_ACTUAL_TOKENS"

"${PYTHON_CMD[@]}" scripts/evaluate_all_logged_fronts.py   --log-path "$logging_dir"   --dataset "$dataset"   --model "$model"   --random-seed "$seed"   --selection current_incumbents   --minimum-actual-tokens "$MIN_ACTUAL_TOKENS"   --maximum-actual-tokens "$MAX_ACTUAL_TOKENS"   --manifest-dir "$MANIFEST_DIR"   --max-output-tokens "$MAX_OUTPUT_TOKENS"   "${replace_args[@]}"   "${force_args[@]}"
