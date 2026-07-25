#!/usr/bin/env bash
#SBATCH --job-name=tf-v3-eval
#SBATCH --partition=gpu
#SBATCH --nodelist=firefly1,firefly2,firefly3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tri_fair_v3_eval_%j.out
#SBATCH --error=logs/tri_fair_v3_eval_%j.err

set -Eeuo pipefail
module purge
module --ignore_cache load python/3.11.14
module --ignore_cache load python-venv/1.0
source "$HOME/venvs/tri-fair/bin/activate"
export TF_PYTHON="$HOME/venvs/tri-fair/bin/python"

if (($# != 5)); then
  echo "Usage: sbatch $0 MODEL DATASET OPTIMIZER SEED BUDGET" >&2
  exit 2
fi
model="$1"; dataset="$2"; optimizer="$3"; seed="$4"; budget="$5"

tf_configure_v3_model_cache "$model"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=jobs/lib/common.sh
source jobs/lib/common.sh
# shellcheck source=jobs/lib/model_cache_v3.sh
source jobs/lib/model_cache_v3.sh

MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v3}"
TF_RESULTS_NAMESPACE="${TF_RESULTS_NAMESPACE:-tri_fair_v3_qwen_5m}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"
FORCE_EVAL="${FORCE_EVAL:-0}"
if [[ "$model" == "gpt-oss-120b" ]] && ((MAX_OUTPUT_TOKENS < 96)); then
  tf_die "GPT-OSS-120B requires MAX_OUTPUT_TOKENS>=96"
fi

output_dir="$(tf_run_output_dir "$ROOT" "$model" "$dataset" "$optimizer" "$seed")"
logging_dir="$(tf_read_logging_dir "$output_dir")" || tf_die "Missing logging directory"
step_results="$logging_dir/step_results.parquet"
[[ -s "$step_results" ]] || tf_die "Missing $step_results"

declare -a PYTHON_CMD force_args
tf_python_command PYTHON_CMD
force_args=()
tf_is_true "$FORCE_EVAL" && force_args=(--force)

read -r chosen_step actual_tokens < <(
  "${PYTHON_CMD[@]}" - "$step_results" "$budget" <<'PY'
from pathlib import Path
import sys
import pandas as pd
path, target = Path(sys.argv[1]), int(sys.argv[2])
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
eligible = table[table <= target]
if eligible.empty:
    raise SystemExit(f"No completed step exists at or below {target}")
print(int(eligible.index[-1]), int(eligible.iloc[-1]))
PY
)

tf_log "Final v3 holdout evaluation: $model/$dataset/$optimizer/seed$seed step=$chosen_step tokens=$actual_tokens"
"${PYTHON_CMD[@]}" scripts/evaluate_prompts_v3.py \
  --random-seed "$seed" \
  --dataset "$dataset" \
  --model "$model" \
  --log-path "$logging_dir" \
  --selection current \
  --step "$chosen_step" \
  --manifest-dir "$MANIFEST_DIR" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  "${force_args[@]}"

[[ -s "$logging_dir/eval.parquet" ]] || tf_die "Evaluator did not create eval.parquet"
tf_log "Completed final v3 holdout evaluation"
