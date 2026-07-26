#!/usr/bin/env bash
#SBATCH --job-name=tri-fair-v4
#SBATCH --partition=gpu
#SBATCH --nodelist=firefly1,firefly2,firefly3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tri_fair_v4_%A_%a.out
#SBATCH --error=logs/tri_fair_v4_%A_%a.err

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
TF_OPTIMIZERS="${TF_OPTIMIZERS:-Tri-Fair-v4,NSGAII-PO-Fair}"
TF_SEEDS="${TF_SEEDS:-52,53,54}"
BUDGET="${BUDGET:-5000000}"
RUN_MODE="${RUN_MODE:-fresh}"
AUTO_EVAL="${AUTO_EVAL:-0}"
MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v4}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"
META_MAX_OUTPUT_TOKENS="${META_MAX_OUTPUT_TOKENS:-256}"
N_INIT_PROMPTS="${N_INIT_PROMPTS:-12}"
TF_RESULTS_NAMESPACE="${TF_RESULTS_NAMESPACE:-tri_fair_v4_qwen_5m}"
MAX_STEPS="${MAX_STEPS:-4000}"
FORCE_FRESH="${FORCE_FRESH:-0}"
STRICT_TOKEN_BUDGET="${STRICT_TOKEN_BUDGET:-1}"
NEAR_BUDGET_FRACTION="${NEAR_BUDGET_FRACTION:-0.90}"
MIN_BUDGET_UTILIZATION="${MIN_BUDGET_UTILIZATION:-0.95}"
OUTPUT_TOKEN_RESERVE="${OUTPUT_TOKEN_RESERVE:-2}"

case "$RUN_MODE" in
  auto|fresh|resume) ;;
  *) tf_die "RUN_MODE must be auto, fresh, or resume; got '$RUN_MODE'" ;;
esac

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
task_id="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$task_id" =~ ^[0-9]+$ ]] || tf_die "Invalid SLURM_ARRAY_TASK_ID '$task_id'"
((task_id < total)) || tf_die "Array task $task_id is outside matrix size $total"

idx=$task_id
seed_idx=$((idx % n_seeds)); idx=$((idx / n_seeds))
optimizer_idx=$((idx % n_optimizers)); idx=$((idx / n_optimizers))
dataset_idx=$((idx % n_datasets)); idx=$((idx / n_datasets))
model_idx=$((idx % n_models))

model="${models[$model_idx]}"
dataset="${datasets[$dataset_idx]}"
optimizer="${optimizers[$optimizer_idx]}"
seed="${seeds[$seed_idx]}"

tf_configure_v3_model_cache "$model"

tf_contains "$dataset" bbq civil_comments bias_in_bios \
  || tf_die "Unsupported fairness dataset '$dataset'"
tf_contains "$optimizer" Tri-Fair-v4 NSGAII-PO-Fair \
  || tf_die "Unsupported v4 optimizer '$optimizer'"
if [[ "$model" == "gpt-oss-120b" ]] && ((MAX_OUTPUT_TOKENS < 96)); then
  tf_die "GPT-OSS-120B requires MAX_OUTPUT_TOKENS>=96"
fi

checkpoints="$(tf_stage_checkpoints "$BUDGET")"
output_dir="$(tf_run_output_dir "$ROOT" "$model" "$dataset" "$optimizer" "$seed")"
status_dir="$output_dir/status"
mkdir -p "$output_dir" "$status_dir" logs

if command -v flock >/dev/null 2>&1; then
  exec 9>"$output_dir/.run.lock"
  flock -n 9 || tf_die "Another process holds $output_dir/.run.lock"
fi

logging_dir=""
latest_checkpoint=""
if logging_dir="$(tf_read_logging_dir "$output_dir" 2>/dev/null)"; then
  latest_checkpoint="$logging_dir/checkpoints/latest.pkl"
fi
resolved_mode="$RUN_MODE"
if [[ "$RUN_MODE" == "auto" ]]; then
  if [[ -n "$logging_dir" && -f "$latest_checkpoint" ]]; then
    resolved_mode=resume
  else
    resolved_mode=fresh
  fi
fi

resume_args=()
if [[ "$resolved_mode" == "resume" ]]; then
  [[ -f "$latest_checkpoint" ]] || tf_die "Missing checkpoint $latest_checkpoint"
  resume_args=(--resume-from "$latest_checkpoint")
elif [[ -n "$logging_dir" && -e "$logging_dir/step_results.parquet" ]] \
     && ! tf_is_true "$FORCE_FRESH"; then
  tf_die "Existing run found at $logging_dir; use resume or FORCE_FRESH=1"
fi

strict_budget_args=(--strict-token-budget)
if ! tf_is_true "$STRICT_TOKEN_BUDGET"; then
  strict_budget_args=(--no-strict-token-budget)
fi

started_at="$(date -Is)"
running_marker="$status_dir/stage_${BUDGET}.running.json"
done_marker="$status_dir/stage_${BUDGET}.done.json"
failed_marker="$status_dir/stage_${BUDGET}.failed.json"
rm -f "$done_marker" "$failed_marker"
tf_write_status_json "$running_marker" \
  state running started_at "$started_at" job_id "${SLURM_JOB_ID:-local}" \
  model "$model" dataset "$dataset" optimizer "$optimizer" seed "$seed" budget "$BUDGET"

on_exit() {
  local code=$?
  local finished_at
  finished_at="$(date -Is)"
  rm -f "$running_marker"
  if ((code == 0)); then
    tf_write_status_json "$done_marker" state complete finished_at "$finished_at" \
      model "$model" dataset "$dataset" optimizer "$optimizer" seed "$seed" budget "$BUDGET"
  else
    tf_write_status_json "$failed_marker" state failed exit_code "$code" finished_at "$finished_at" \
      model "$model" dataset "$dataset" optimizer "$optimizer" seed "$seed" budget "$BUDGET"
  fi
  exit "$code"
}
trap on_exit EXIT

declare -a PYTHON_CMD
tf_python_command PYTHON_CMD

tf_log "v4 task model=$model dataset=$dataset optimizer=$optimizer seed=$seed budget=$BUDGET"
"${PYTHON_CMD[@]}" scripts/experiment_v4.py \
  --experiment-name "tri_fair_v4_${optimizer}_${model}_${dataset}_${seed}_${BUDGET}" \
  --random-seed "$seed" \
  --budget-per-run "$BUDGET" \
  --budget-checkpoints "$checkpoints" \
  --output-dir "$output_dir" \
  --dataset "$dataset" \
  --model "$model" \
  --optimizer "$optimizer" \
  --n-init-prompts "$N_INIT_PROMPTS" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --meta-max-output-tokens "$META_MAX_OUTPUT_TOKENS" \
  --manifest-dir "$MANIFEST_DIR" \
  --max-steps "$MAX_STEPS" \
  --near-budget-fraction "$NEAR_BUDGET_FRACTION" \
  --minimum-budget-utilization "$MIN_BUDGET_UTILIZATION" \
  --output-token-reserve "$OUTPUT_TOKEN_RESERVE" \
  "${strict_budget_args[@]}" \
  "${resume_args[@]}"

logging_dir="$(tf_read_logging_dir "$output_dir")" \
  || tf_die "Experiment ended without logging_dir.txt"
[[ -s "$logging_dir/step_results.parquet" ]] || tf_die "Missing step_results.parquet"
[[ -s "$logging_dir/checkpoints/latest.pkl" ]] || tf_die "Missing latest checkpoint"

tf_log "Optimization completed: $logging_dir"
if tf_is_true "$AUTO_EVAL"; then
  tf_log "AUTO_EVAL is intentionally disabled for v4 main-study runs; use the all-step holdout job after all optimization tasks complete."
fi
