#!/usr/bin/env bash
#SBATCH --job-name=tf-init-5m-v2
#SBATCH --partition=gpu
#SBATCH --nodelist=firefly1,firefly2,firefly3,pegasus2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tri_fair_initial_5m_v2_%A_%a.out
#SBATCH --error=logs/tri_fair_initial_5m_v2_%A_%a.err

# Evaluate one dataset/seed exact initial population for the final 5M v2 study.
# This worker is submitted by jobs/submit_initial_instructions_5m_v2.sh.

set -Eeuo pipefail

module purge
module load python/3.11.14
module load python-venv/1.0

source "$HOME/venvs/tri-fair/bin/activate"
export TF_PYTHON="$HOME/venvs/tri-fair/bin/python"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

TF_DATASETS="${TF_DATASETS:-bbq,bias_in_bios,civil_comments}"
TF_SEEDS="${TF_SEEDS:-42,43,44}"
TF_RESULTS_NAMESPACE="${TF_RESULTS_NAMESPACE:-tri_fair_v2_2}"
MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v2_2}"
SOURCE_BUDGET="${SOURCE_BUDGET:-5000000}"
SOURCE_OPTIMIZERS="${SOURCE_OPTIMIZERS:-Tri-Fair,NSGAII-PO-Fair}"
PREFERRED_OPTIMIZER="${PREFERRED_OPTIMIZER:-Tri-Fair}"
EXPECTED_INITIAL_PROMPTS="${EXPECTED_INITIAL_PROMPTS:-12}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"

IFS=',' read -r -a DATASETS <<< "$TF_DATASETS"
IFS=',' read -r -a SEEDS <<< "$TF_SEEDS"

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
n_datasets="${#DATASETS[@]}"
n_seeds="${#SEEDS[@]}"
total=$((n_datasets * n_seeds))

if ((task_id < 0 || task_id >= total)); then
  echo "Array task $task_id is outside 0-$((total - 1))" >&2
  exit 2
fi

dataset_index=$((task_id / n_seeds))
seed_index=$((task_id % n_seeds))

dataset="${DATASETS[$dataset_index]}"
seed="${SEEDS[$seed_index]}"

dataset="${dataset//[[:space:]]/}"
seed="${seed//[[:space:]]/}"

echo "Initial-baseline task"
echo "  model:        qwen-3-30b"
echo "  dataset:      $dataset"
echo "  seed:         $seed"
echo "  source budget:$SOURCE_BUDGET"
echo "  namespace:    $TF_RESULTS_NAMESPACE"
echo "  manifests:    $MANIFEST_DIR"
echo "  prompts:      $EXPECTED_INITIAL_PROMPTS"

"$TF_PYTHON" -m scripts.evaluate_initial_instructions_5m_v2 \
  --random-seed "$seed" \
  --dataset "$dataset" \
  --model qwen-3-30b \
  --results-root "results/$TF_RESULTS_NAMESPACE" \
  --manifest-dir "$MANIFEST_DIR" \
  --source-budget "$SOURCE_BUDGET" \
  --source-optimizers "$SOURCE_OPTIMIZERS" \
  --preferred-optimizer "$PREFERRED_OPTIMIZER" \
  --expected-initial-prompts "$EXPECTED_INITIAL_PROMPTS" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --log-level INFO
