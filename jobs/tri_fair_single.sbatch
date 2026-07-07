#!/usr/bin/env bash
#SBATCH --job-name=tf-single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tri_fair_single_%j.out
#SBATCH --error=logs/tri_fair_single_%j.err

# Usage:
#   sbatch jobs/tri_fair_single.sbatch MODEL DATASET OPTIMIZER SEED [BUDGET] [RUN_MODE]
# Example smoke test:
#   sbatch jobs/tri_fair_single.sbatch qwen-3-30b bbq Tri-Fair 42 250000 fresh

set -Eeuo pipefail
if (($# < 4 || $# > 6)); then
  echo "Usage: sbatch $0 MODEL DATASET OPTIMIZER SEED [BUDGET] [RUN_MODE]" >&2
  exit 2
fi

export TF_MODELS="$1"
export TF_DATASETS="$2"
export TF_OPTIMIZERS="$3"
export TF_SEEDS="$4"
export BUDGET="${5:-250000}"
export RUN_MODE="${6:-auto}"
export AUTO_EVAL="${AUTO_EVAL:-1}"
export SLURM_ARRAY_TASK_ID=0
export SLURM_ARRAY_TASK_COUNT=1

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs
exec bash jobs/tri_fair_main.sbatch
