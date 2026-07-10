#!/bin/bash
#SBATCH --job-name=bias-v2-pilot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bias_v2_pilot_%j.out
#SBATCH --error=logs/bias_v2_pilot_%j.err

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
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"
META_MAX_OUTPUT_TOKENS="${META_MAX_OUTPUT_TOKENS:-256}"
N_INIT_PROMPTS="${N_INIT_PROMPTS:-6}"
MAX_STEPS="${MAX_STEPS:-2000}"
MANIFEST_DIR="${MANIFEST_DIR:-data/splits}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/tri_fair_bias_v2}"

OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL}/${DATASET}/${OPTIMIZER}/seed${SEED}/budget${BUDGET}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p logs "$OUTPUT_DIR"

echo "============================================================"
echo "Bias-v2 pilot"
echo "model=$MODEL dataset=$DATASET optimizer=$OPTIMIZER seed=$SEED"
echo "budget=$BUDGET output_dir=$OUTPUT_DIR run_id=$RUN_ID"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"

"$TF_PYTHON" scripts/experiment.py \
  --experiment-name "bias_v2_${OPTIMIZER}_${MODEL}_${SEED}_${BUDGET}" \
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
  --meta-max-output-tokens "$META_MAX_OUTPUT_TOKENS" \
  --manifest-dir "$MANIFEST_DIR" \
  --max-steps "$MAX_STEPS"

LOGGING_DIR="$(cat "$OUTPUT_DIR/logging_dir.txt")"
STEP_RESULTS="$LOGGING_DIR/step_results.parquet"

read -r CHOSEN_STEP ACTUAL_TOKENS < <(
  "$TF_PYTHON" - "$STEP_RESULTS" "$BUDGET" <<'PY_INNER'
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
PY_INNER
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
  --force

"$TF_PYTHON" - "$LOGGING_DIR" <<'PY_INNER'
from pathlib import Path
import sys
import pandas as pd

log = Path(sys.argv[1])
df = pd.read_parquet(log / "eval.parquet")

best = df.sort_values(
    ["test_quality", "test_fairness", "test_cost"],
    ascending=[False, True, True],
).iloc[0]

print("\n=== Bias-v2 pilot result ===")
print(f"log_path:        {log}")
print(f"rows:            {len(df)}")
print(f"best Macro-F1:   {best['test_quality']:.6f}")
print(f"best cost:       {best['test_cost']:.6f}")
print(f"best unfairness: {best['test_fairness']:.6f}")
print(f"chosen_step:     {best.get('chosen_step')}")
PY_INNER
