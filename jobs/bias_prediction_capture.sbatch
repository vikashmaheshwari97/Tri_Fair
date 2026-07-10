#!/bin/bash
#SBATCH --job-name=bias-pred-capture
#SBATCH --partition=gpu
#SBATCH --nodelist=firefly1,firefly2,firefly3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%2
#SBATCH --output=logs/bias_prediction_capture_%A_%a.out
#SBATCH --error=logs/bias_prediction_capture_%A_%a.err

set -euo pipefail

cd "$HOME/projects/Tri_Fair"

module purge
module load python/3.11.14
module load python-venv/1.0
source "$HOME/venvs/tri-fair/bin/activate"

mkdir -p logs analysis/output/bias_prediction_capture

PAIRS=(
  "Tri-Fair 42"
  "Tri-Fair 43"
  "Tri-Fair 44"
  "NSGAII-PO-Fair 42"
  "NSGAII-PO-Fair 43"
  "NSGAII-PO-Fair 44"
)

read -r OPTIMIZER SEED <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"

echo "Running Bias prediction capture"
echo "Optimizer: $OPTIMIZER"
echo "Seed:      $SEED"
echo "Task:      $SLURM_ARRAY_TASK_ID"

python -m analysis.bias_prediction_capture \
  --dataset bias_in_bios \
  --model qwen-3-30b \
  --optimizer "$OPTIMIZER" \
  --seed "$SEED" \
  --budget 1000000 \
  --manifest-dir data/splits \
  --max-output-tokens 16 \
  --out-dir analysis/output/bias_prediction_capture
