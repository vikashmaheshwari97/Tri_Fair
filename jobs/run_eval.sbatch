#!/usr/bin/env bash
#SBATCH --job-name=tf-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tri_fair_eval_%j.out
#SBATCH --error=logs/tri_fair_eval_%j.err

# Evaluate one completed optimization stage on the untouched test manifest.
# Usage: sbatch jobs/run_eval.sbatch MODEL DATASET OPTIMIZER SEED BUDGET

set -Eeuo pipefail
if (($# != 5)); then
  echo "Usage: sbatch $0 MODEL DATASET OPTIMIZER SEED BUDGET" >&2
  exit 2
fi

model="$1"
dataset="$2"
optimizer="$3"
seed="$4"
budget="$5"

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=jobs/lib/common.sh
source jobs/lib/common.sh

tf_validate_positive_int BUDGET "$budget"
MANIFEST_DIR="${MANIFEST_DIR:-data/splits}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"
EVAL_INCUMBENTS="${EVAL_INCUMBENTS:-true}"
FORCE_EVAL="${FORCE_EVAL:-0}"

output_dir="$(tf_run_output_dir "$ROOT" "$model" "$dataset" "$optimizer" "$seed")"
status_dir="$output_dir/status"
mkdir -p "$status_dir" logs
logging_dir="$(tf_read_logging_dir "$output_dir")" \
  || tf_die "No valid logging directory found for $output_dir"
step_results="$logging_dir/step_results.parquet"
[[ -s "$step_results" ]] || tf_die "Missing $step_results"

declare -a PYTHON_CMD
tf_python_command PYTHON_CMD

# Checkpoint callbacks run after an optimizer step. The faithful budget mapping
# is therefore the first recorded step whose cumulative downstream tokens are
# at or above the requested checkpoint.
read -r chosen_step actual_tokens < <(
  "${PYTHON_CMD[@]}" - "$step_results" "$budget" <<'PY'
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
    .groupby("step", as_index=True)["total_tokens_downstream"]
    .max()
    .sort_index()
)
reached = table[table >= target]
if reached.empty:
    maximum = int(table.max()) if len(table) else 0
    raise SystemExit(
        f"target budget {target} was not reached; maximum logged tokens={maximum}"
    )
step = int(reached.index[0])
actual = int(reached.iloc[0])
print(step, actual)
PY
)

[[ "$chosen_step" =~ ^[0-9]+$ ]] || tf_die "Could not resolve a checkpoint step"

eval_path="$logging_dir/eval.parquet"
if [[ -s "$eval_path" ]] && ! tf_is_true "$FORCE_EVAL"; then
  if "${PYTHON_CMD[@]}" - "$eval_path" "$chosen_step" <<'PY'
import sys
import pandas as pd
path, step = sys.argv[1], int(sys.argv[2])
df = pd.read_parquet(path, columns=["chosen_step"])
raise SystemExit(0 if (pd.to_numeric(df["chosen_step"], errors="coerce") == step).any() else 1)
PY
  then
    tf_log "Evaluation already exists for chosen_step=$chosen_step; skipping idempotently"
    exit 0
  fi
fi

started_at="$(date -Is)"
running_marker="$status_dir/eval_${budget}.running.json"
done_marker="$status_dir/eval_${budget}.done.json"
failed_marker="$status_dir/eval_${budget}.failed.json"
rm -f "$done_marker" "$failed_marker"
tf_write_status_json "$running_marker" \
  state running started_at "$started_at" job_id "${SLURM_JOB_ID:-local}" \
  model "$model" dataset "$dataset" optimizer "$optimizer" seed "$seed" \
  requested_budget "$budget" chosen_step "$chosen_step" actual_tokens "$actual_tokens"

on_exit() {
  local code=$?
  local finished_at
  finished_at="$(date -Is)"
  rm -f "$running_marker"
  if ((code == 0)); then
    tf_write_status_json "$done_marker" \
      state complete started_at "$started_at" finished_at "$finished_at" \
      job_id "${SLURM_JOB_ID:-local}" model "$model" dataset "$dataset" \
      optimizer "$optimizer" seed "$seed" requested_budget "$budget" \
      chosen_step "$chosen_step" actual_tokens "$actual_tokens"
  else
    tf_write_status_json "$failed_marker" \
      state failed exit_code "$code" started_at "$started_at" finished_at "$finished_at" \
      job_id "${SLURM_JOB_ID:-local}" model "$model" dataset "$dataset" \
      optimizer "$optimizer" seed "$seed" requested_budget "$budget" \
      chosen_step "$chosen_step" actual_tokens "$actual_tokens"
  fi
  exit "$code"
}
trap on_exit EXIT

tf_log "Evaluating $model/$dataset/$optimizer/seed$seed"
tf_log "requested_budget=$budget chosen_step=$chosen_step actual_tokens=$actual_tokens"
tf_log "logging_dir=$logging_dir"

"${PYTHON_CMD[@]}" scripts/evaluate_prompts.py \
  --random-seed "$seed" \
  --dataset "$dataset" \
  --model "$model" \
  --log-path "$logging_dir" \
  --incumbents "$EVAL_INCUMBENTS" \
  --step "$chosen_step" \
  --manifest-dir "$MANIFEST_DIR" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS"

[[ -s "$eval_path" ]] || tf_die "Evaluator completed without creating $eval_path"
tf_log "Holdout evaluation completed: $eval_path"
