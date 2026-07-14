#!/bin/bash
#SBATCH --job-name=bias-advanced
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bias_advanced_%j.out
#SBATCH --error=logs/bias_advanced_%j.err

set -Eeuo pipefail

cd "$HOME/projects/Tri_Fair"

module purge
module load python/3.11.14
module load python-venv/1.0
source "$HOME/venvs/tri-fair/bin/activate"

export TF_PYTHON="$HOME/venvs/tri-fair/bin/python"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${MODEL:-qwen-3-30b}"
DATASET="${DATASET:-bias_in_bios}"
OPTIMIZER="${OPTIMIZER:-Tri-Fair}"
SEED="${SEED:-42}"
BUDGET="${BUDGET:-250000}"
BIOS_PROMPT_TIER="${BIOS_PROMPT_TIER:-compact}"
PREDICTION_MODE="${PREDICTION_MODE:-marker}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"
META_MAX_OUTPUT_TOKENS="${META_MAX_OUTPUT_TOKENS:-256}"
N_INIT_PROMPTS="${N_INIT_PROMPTS:-6}"
MAX_STEPS="${MAX_STEPS:-2000}"
MANIFEST_DIR="${MANIFEST_DIR:-data/splits}"
RUN_PRED_CAPTURE="${RUN_PRED_CAPTURE:-1}"
MAKE_BUNDLE="${MAKE_BUNDLE:-1}"

export BIOS_PROMPT_TIER

OUTPUT_ROOT="${OUTPUT_ROOT:-results/tri_fair_bias_advanced/${BIOS_PROMPT_TIER}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL}/${DATASET}/${OPTIMIZER}/seed${SEED}/budget${BUDGET}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PRED_DIR="${PRED_DIR:-analysis/output/bias_advanced_prediction_capture/job_${SLURM_JOB_ID:-local}_${BIOS_PROMPT_TIER}}"

mkdir -p logs "$OUTPUT_DIR" "$PRED_DIR"

echo "============================================================"
echo "Bias advanced run"
echo "model=$MODEL dataset=$DATASET optimizer=$OPTIMIZER seed=$SEED"
echo "budget=$BUDGET tier=$BIOS_PROMPT_TIER"
echo "prediction_mode=$PREDICTION_MODE"
echo "output_root=$OUTPUT_ROOT"
echo "output_dir=$OUTPUT_DIR"
echo "run_id=$RUN_ID"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

"$TF_PYTHON" - <<'PY_CHECK'
from src.config.bios_prompt_guide import get_bios_prompt_guide, get_bios_prompt_tier
tier = get_bios_prompt_tier()
guide = get_bios_prompt_guide()
print(f"Resolved BIOS_PROMPT_TIER={tier}")
print(f"Guide characters={len(guide)}")
print(f"Guide words={len(guide.split())}")
assert "Allowed labels:" in guide
assert "software_engineer" in guide
PY_CHECK

"$TF_PYTHON" scripts/experiment.py \
  --experiment-name "bias_advanced_${BIOS_PROMPT_TIER}_${OPTIMIZER}_${MODEL}_${SEED}_${BUDGET}" \
  --random-seed "$SEED" \
  --budget-per-run "$BUDGET" \
  --budget-checkpoints "$BUDGET" \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_ID" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --optimizer "$OPTIMIZER" \
  --n-init-prompts "$N_INIT_PROMPTS" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --prediction-mode "$PREDICTION_MODE" \
  --meta-max-output-tokens "$META_MAX_OUTPUT_TOKENS" \
  --manifest-dir "$MANIFEST_DIR" \
  --max-steps "$MAX_STEPS"

LOGGING_DIR="$(cat "$OUTPUT_DIR/logging_dir.txt")"
STEP_RESULTS="$LOGGING_DIR/step_results.parquet"

read -r CHOSEN_STEP ACTUAL_TOKENS < <(
  "$TF_PYTHON" - "$STEP_RESULTS" "$BUDGET" <<'PY_STEP'
from pathlib import Path
import sys
import pandas as pd

path = Path(sys.argv[1])
target = int(sys.argv[2])

df = pd.read_parquet(path, columns=["step", "total_tokens_downstream"])
table = (
    df.assign(
        step=pd.to_numeric(df["step"], errors="raise").astype(int),
        total_tokens_downstream=pd.to_numeric(
            df["total_tokens_downstream"], errors="raise"
        ).astype(int),
    )
    .groupby("step")["total_tokens_downstream"]
    .max()
    .sort_index()
)

reached = table[table >= target]
if reached.empty:
    raise SystemExit(f"Target budget {target} not reached; max={int(table.max())}")
print(int(reached.index[0]), int(reached.iloc[0]))
PY_STEP
)

echo "Evaluating chosen_step=$CHOSEN_STEP actual_tokens=$ACTUAL_TOKENS"

"$TF_PYTHON" scripts/evaluate_prompts.py \
  --random-seed "$SEED" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --log-path "$LOGGING_DIR" \
  --incumbents true \
  --step "$CHOSEN_STEP" \
  --manifest-dir "$MANIFEST_DIR" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --prediction-mode "$PREDICTION_MODE" \
  --force

if [[ "$RUN_PRED_CAPTURE" == "1" ]]; then
  echo "Running prediction capture into $PRED_DIR"
  "$TF_PYTHON" -m analysis.bias_prediction_capture \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --optimizer "$OPTIMIZER" \
    --seed "$SEED" \
    --budget "$BUDGET" \
    --manifest-dir "$MANIFEST_DIR" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --prediction-mode "$PREDICTION_MODE" \
    --logging-dir "$LOGGING_DIR" \
    --out-dir "$PRED_DIR"
fi

if [[ "$MAKE_BUNDLE" == "1" ]]; then
  echo "Creating advanced bundle/report"
  "$TF_PYTHON" -m analysis.bias_advanced_bundle \
    --job-id "${SLURM_JOB_ID:-local}" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --optimizer "$OPTIMIZER" \
    --seed "$SEED" \
    --budget "$BUDGET" \
    --tier "$BIOS_PROMPT_TIER" \
    --results-root "$OUTPUT_ROOT" \
    --prediction-dir "$PRED_DIR" \
    --make-tar
fi
