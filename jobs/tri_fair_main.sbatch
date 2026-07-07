#!/usr/bin/env bash

#SBATCH --partition=gpu
#SBATCH --nodelist=firefly1,firefly2,firefly3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=160G

# Array bounds are supplied by jobs/submit_tri_fair.sh. This prevents the
# model/dataset/method matrix from silently diverging from a hard-coded #SBATCH
# --array line.

module purge
module load python/3.11.14
module load python-venv/1.0

source "$HOME/venvs/tri-fair/bin/activate"
export TF_PYTHON="$HOME/venvs/tri-fair/bin/python"

echo "Python executable: $TF_PYTHON"
"$TF_PYTHON" --version

set -Eeuo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=jobs/lib/common.sh
source jobs/lib/common.sh

TF_MODELS="${TF_MODELS:-qwen-3-30b}"
TF_DATASETS="${TF_DATASETS:-bbq,bias_in_bios,civil_comments}"
TF_OPTIMIZERS="${TF_OPTIMIZERS:-Tri-Fair,NSGAII-PO-Fair}"
TF_SEEDS="${TF_SEEDS:-42,43,44}"
BUDGET="${BUDGET:-1000000}"
RUN_MODE="${RUN_MODE:-auto}"
AUTO_EVAL="${AUTO_EVAL:-1}"
MANIFEST_DIR="${MANIFEST_DIR:-data/splits}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"
META_MAX_OUTPUT_TOKENS="${META_MAX_OUTPUT_TOKENS:-256}"
N_INIT_PROMPTS="${N_INIT_PROMPTS:-6}"
MAX_STEPS="${MAX_STEPS:-2000}"
FORCE_FRESH="${FORCE_FRESH:-0}"

case "$RUN_MODE" in
  auto|fresh|resume) ;;
  *) tf_die "RUN_MODE must be auto, fresh, or resume; got '$RUN_MODE'" ;;
esac
for pair in \
  "BUDGET:$BUDGET" \
  "MAX_OUTPUT_TOKENS:$MAX_OUTPUT_TOKENS" \
  "META_MAX_OUTPUT_TOKENS:$META_MAX_OUTPUT_TOKENS" \
  "N_INIT_PROMPTS:$N_INIT_PROMPTS" \
  "MAX_STEPS:$MAX_STEPS"; do
  tf_validate_positive_int "${pair%%:*}" "${pair#*:}"
done

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
((task_id < total)) || tf_die "Array task $task_id is outside the computed matrix size $total"

idx=$task_id
seed_idx=$((idx % n_seeds)); idx=$((idx / n_seeds))
optimizer_idx=$((idx % n_optimizers)); idx=$((idx / n_optimizers))
dataset_idx=$((idx % n_datasets)); idx=$((idx / n_datasets))
model_idx=$((idx % n_models))

model="${models[$model_idx]}"
dataset="${datasets[$dataset_idx]}"
optimizer="${optimizers[$optimizer_idx]}"
seed="${seeds[$seed_idx]}"

tf_contains "$dataset" bbq bias_in_bios civil_comments \
  || tf_die "Unsupported Tri-Fair dataset '$dataset'"
tf_contains "$optimizer" Tri-Fair NSGAII-PO-Fair \
  || tf_die "Unsupported main-study optimizer '$optimizer'"
tf_contains "$seed" 42 43 44 \
  || tf_warn "Seed $seed is outside the planned main-study seeds 42,43,44"

checkpoints="$(tf_stage_checkpoints "$BUDGET")"
output_dir="$(tf_run_output_dir "$ROOT" "$model" "$dataset" "$optimizer" "$seed")"
status_dir="$output_dir/status"
mkdir -p "$output_dir" "$status_dir" logs

# Prevent duplicate array submissions from writing the same run concurrently.
if command -v flock >/dev/null 2>&1; then
  exec 9>"$output_dir/.run.lock"
  flock -n 9 || tf_die "Another process holds $output_dir/.run.lock"
else
  tf_warn "flock is unavailable; duplicate-run protection is disabled"
fi

started_at="$(date -Is)"
running_marker="$status_dir/stage_${BUDGET}.running.json"
done_marker="$status_dir/stage_${BUDGET}.done.json"
failed_marker="$status_dir/stage_${BUDGET}.failed.json"
rm -f "$done_marker" "$failed_marker"
tf_write_status_json "$running_marker" \
  state running started_at "$started_at" job_id "${SLURM_JOB_ID:-local}" \
  array_task "$task_id" model "$model" dataset "$dataset" optimizer "$optimizer" \
  seed "$seed" budget "$BUDGET" run_mode "$RUN_MODE"

on_exit() {
  local code=$?
  local finished_at
  finished_at="$(date -Is)"
  rm -f "$running_marker"
  if ((code == 0)); then
    tf_write_status_json "$done_marker" \
      state complete started_at "$started_at" finished_at "$finished_at" \
      job_id "${SLURM_JOB_ID:-local}" model "$model" dataset "$dataset" \
      optimizer "$optimizer" seed "$seed" budget "$BUDGET"
  else
    tf_write_status_json "$failed_marker" \
      state failed exit_code "$code" started_at "$started_at" finished_at "$finished_at" \
      job_id "${SLURM_JOB_ID:-local}" model "$model" dataset "$dataset" \
      optimizer "$optimizer" seed "$seed" budget "$BUDGET"
  fi
  exit "$code"
}
trap on_exit EXIT

logging_dir=""
latest_checkpoint=""
if logging_dir="$(tf_read_logging_dir "$output_dir" 2>/dev/null)"; then
  latest_checkpoint="$logging_dir/checkpoints/latest.pkl"
fi

resolved_mode="$RUN_MODE"
if [[ "$RUN_MODE" == "auto" ]]; then
  if [[ -n "$logging_dir" && -f "$latest_checkpoint" ]]; then
    resolved_mode=resume
  elif ((BUDGET > 1000000)); then
    tf_die "Budget $BUDGET requires a prior checkpoint, but none exists for $output_dir"
  else
    resolved_mode=fresh
  fi
fi

resume_args=()
if [[ "$resolved_mode" == "resume" ]]; then
  [[ -n "$logging_dir" ]] || tf_die "Cannot resume: logging_dir.txt is missing or empty in $output_dir"
  [[ -f "$latest_checkpoint" ]] || tf_die "Cannot resume: $latest_checkpoint does not exist"
  resume_args=(--resume-from "$latest_checkpoint")
else
  if [[ -n "$logging_dir" && -e "$logging_dir/step_results.parquet" ]] \
     && ! tf_is_true "$FORCE_FRESH"; then
    tf_die "A run already exists at $logging_dir. Use RUN_MODE=resume or FORCE_FRESH=1 deliberately."
  fi
fi

declare -a PYTHON_CMD
tf_python_command PYTHON_CMD

tf_log "============================================================"
tf_log "Tri-Fair optimization task"
tf_log "job=${SLURM_JOB_ID:-local} array_task=$task_id/$total"
tf_log "model=$model dataset=$dataset optimizer=$optimizer seed=$seed"
tf_log "budget=$BUDGET checkpoints=$checkpoints mode=$resolved_mode"
tf_log "output_dir=$output_dir"
tf_log "============================================================"

"${PYTHON_CMD[@]}" scripts/experiment.py \
  --experiment-name "tri_fair_${optimizer}_${model}_${dataset}_${seed}_${BUDGET}" \
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
  "${resume_args[@]}"

logging_dir="$(tf_read_logging_dir "$output_dir")" \
  || tf_die "Experiment ended without a valid $output_dir/logging_dir.txt"
[[ -s "$logging_dir/step_results.parquet" ]] \
  || tf_die "Experiment ended without $logging_dir/step_results.parquet"
[[ -s "$logging_dir/checkpoints/latest.pkl" ]] \
  || tf_die "Experiment ended without $logging_dir/checkpoints/latest.pkl"

tf_log "Optimization stage completed successfully: $logging_dir"

# Holdout evaluation is a separate GPU job. This keeps a failed evaluation from
# invalidating an otherwise valid optimization run and makes reruns idempotent.
if tf_is_true "$AUTO_EVAL"; then
  export MANIFEST_DIR MAX_OUTPUT_TOKENS
  if eval_job="$(sbatch --parsable jobs/run_eval.sbatch \
      "$model" "$dataset" "$optimizer" "$seed" "$BUDGET" 2>/dev/null)"; then
    tf_log "Submitted holdout evaluation job $eval_job for budget $BUDGET"
  else
    tf_warn "Could not submit holdout evaluation automatically. Run jobs/submit_evaluations.sh later."
  fi
fi
