#!/usr/bin/env bash
#SBATCH --job-name=qwen-smoke
#SBATCH --partition=gpu
#SBATCH --nodelist=firefly1,firefly2,firefly3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=logs/qwen_model_smoke_%j.out
#SBATCH --error=logs/qwen_model_smoke_%j.err

set -Eeuo pipefail

cd "$HOME/projects/Tri_Fair"
mkdir -p logs
mkdir -p "$HOME/projects/models/Qwen3-30B"

module purge
module load python/3.11.14
module load python-venv/1.0
source "$HOME/venvs/tri-fair/bin/activate"

export TF_PYTHON="$HOME/venvs/tri-fair/bin/python"

echo "Host: $(hostname)"
echo "Python: $("$TF_PYTHON" --version)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

"$TF_PYTHON" scripts/qwen_model_smoke.py