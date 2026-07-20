#!/usr/bin/env bash
# Recovery helper: submit missing holdout evaluations for a completed stage.

set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=jobs/lib/common.sh
source jobs/lib/common.sh

TF_MODELS="${TF_MODELS:-qwen-3-30b}"
TF_DATASETS="${TF_DATASETS:-bbq,bias_in_bios,civil_comments}"
TF_OPTIMIZERS="${TF_OPTIMIZERS:-Tri-Fair,NSGAII-PO-Fair}"
TF_SEEDS="${TF_SEEDS:-42,43,44}"
BUDGET="${BUDGET:-1000000}"
FORCE_EVAL="${FORCE_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
export FORCE_EVAL MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v2}" \
  MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}" \
  TF_RESULTS_NAMESPACE="${TF_RESULTS_NAMESPACE:-tri_fair_v2}"
NODELIST="${NODELIST:-firefly1,firefly2,firefly3,pegasus2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-}"
ACCOUNT="${ACCOUNT:-}"
GRES="${GRES:-}"
TIME_LIMIT="${TIME_LIMIT:-}"
MEMORY="${MEMORY:-}"

tf_validate_positive_int BUDGET "$BUDGET"
tf_validate_positive_int MAX_OUTPUT_TOKENS "$MAX_OUTPUT_TOKENS"
declare -a models datasets optimizers seeds
tf_split_csv "$TF_MODELS" models
tf_split_csv "$TF_DATASETS" datasets
tf_split_csv "$TF_OPTIMIZERS" optimizers
tf_split_csv "$TF_SEEDS" seeds

for model in "${models[@]}"; do
  if [[ "$model" == "gpt-oss-120b" ]] && ((MAX_OUTPUT_TOKENS < 96)); then
    tf_die "GPT-OSS-120B evaluation requires MAX_OUTPUT_TOKENS>=96; got $MAX_OUTPUT_TOKENS"
  fi
done

mkdir -p logs

SBATCH_ARGS=(--parsable)
[[ -n "${PARTITION:-}" ]] && SBATCH_ARGS+=("--partition=$PARTITION")
[[ -n "${QOS:-}" ]] && SBATCH_ARGS+=("--qos=$QOS")
[[ -n "${ACCOUNT:-}" ]] && SBATCH_ARGS+=("--account=$ACCOUNT")
[[ -n "${GRES:-}" ]] && SBATCH_ARGS+=("--gres=$GRES")
[[ -n "${TIME_LIMIT:-}" ]] && SBATCH_ARGS+=("--time=$TIME_LIMIT")
[[ -n "${MEMORY:-}" ]] && SBATCH_ARGS+=("--mem=$MEMORY")
[[ -n "${CPUS_PER_TASK:-}" ]] && SBATCH_ARGS+=("--cpus-per-task=$CPUS_PER_TASK")
[[ -n "${NODELIST:-}" ]] && SBATCH_ARGS+=("--nodelist=$NODELIST")

submitted=0
skipped=0
missing=0
for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for optimizer in "${optimizers[@]}"; do
      for seed in "${seeds[@]}"; do
        output_dir="$(tf_run_output_dir "$ROOT" "$model" "$dataset" "$optimizer" "$seed")"
        if ! logging_dir="$(tf_read_logging_dir "$output_dir" 2>/dev/null)" \
           || [[ ! -s "$logging_dir/step_results.parquet" ]]; then
          tf_warn "Skipping missing optimization run: $model/$dataset/$optimizer/seed$seed"
          ((missing+=1))
          continue
        fi
        marker="$output_dir/status/eval_${BUDGET}.done.json"
        if [[ -s "$marker" ]] && ! tf_is_true "$FORCE_EVAL"; then
          ((skipped+=1))
          continue
        fi
        cmd=(sbatch "${SBATCH_ARGS[@]}" jobs/run_eval.sbatch "$model" "$dataset" "$optimizer" "$seed" "$BUDGET")
        if tf_is_true "$DRY_RUN"; then
          printf 'DRY RUN:'; printf ' %q' "${cmd[@]}"; printf '\n'
        else
          job_id="$("${cmd[@]}")"
          tf_log "Submitted eval $job_id for $model/$dataset/$optimizer/seed$seed"
        fi
        ((submitted+=1))
      done
    done
  done
done

tf_log "Evaluation submission summary: submitted=$submitted skipped=$skipped missing_runs=$missing"
