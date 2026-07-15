#!/bin/bash
#SBATCH --job-name=graphrag-vb
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/graphrag_vb_%A_%a.out
#SBATCH --error=logs/graphrag_vb_%A_%a.err

set -Eeuo pipefail

cd "$HOME/projects/Tri_Fair"

module purge
module load python/3.11.14
module load python-venv/1.0
source "$HOME/venvs/tri-fair/bin/activate"

PROMPTS_ROOT="${PROMPTS_ROOT:?set PROMPTS_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
MODEL_PATH="${MODEL_PATH:-../models/Qwen3-30B}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"

mapfile -t POLICY_DIRS < <(find "$PROMPTS_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
POLICY_DIR="${POLICY_DIRS[$IDX]}"
POLICY_NAME="$(basename "$POLICY_DIR")"

mkdir -p "$OUTPUT_ROOT/$POLICY_NAME"

python -m scripts.graphrag_run_policy_vllm \
  --prompts "$POLICY_DIR/prompts.jsonl" \
  --out "$OUTPUT_ROOT/$POLICY_NAME/predictions.jsonl" \
  --model-path "$MODEL_PATH" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --temperature 0.0 \
  --top-p 1.0 \
  --trust-remote-code
