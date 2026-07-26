#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source jobs/lib/common.sh

export TF_MODELS="${TF_MODELS:-qwen-3-30b}"
export TF_DATASETS="${TF_DATASETS:-bbq,civil_comments,bias_in_bios}"
export TF_OPTIMIZERS="${TF_OPTIMIZERS:-Tri-Fair-v3,NSGAII-PO-Fair}"
export TF_SEEDS="${TF_SEEDS:-42,43,44}"
export TF_RESULTS_NAMESPACE="${TF_RESULTS_NAMESPACE:-tri_fair_v3_qwen_5m}"
export MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v3}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"
export MIN_ACTUAL_TOKENS="${MIN_ACTUAL_TOKENS:-0}"
export MAX_ACTUAL_TOKENS="${MAX_ACTUAL_TOKENS:-5000000}"
export ALL_STEP_REPLACE_OUTPUT="${ALL_STEP_REPLACE_OUTPUT:-1}"
export ALL_STEP_FORCE="${ALL_STEP_FORCE:-0}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
export NODELIST="${NODELIST:-firefly1,firefly2,firefly3}"

for model in ${TF_MODELS//,/ }; do
  if [[ "$model" == "gpt-oss-120b" ]] && ((MAX_OUTPUT_TOKENS < 96)); then
    tf_die "GPT-OSS-120B requires MAX_OUTPUT_TOKENS>=96"
  fi
done

declare -a models datasets optimizers seeds
tf_split_csv "$TF_MODELS" models
tf_split_csv "$TF_DATASETS" datasets
tf_split_csv "$TF_OPTIMIZERS" optimizers
tf_split_csv "$TF_SEEDS" seeds

total=$((${#models[@]} * ${#datasets[@]} * ${#optimizers[@]} * ${#seeds[@]}))
array_spec="0-$((total - 1))%$MAX_CONCURRENT"

mkdir -p logs
SBATCH_ARGS=(--parsable "--array=$array_spec")
[[ -n "${PARTITION:-}" ]] && SBATCH_ARGS+=("--partition=$PARTITION")
[[ -n "${QOS:-}" ]] && SBATCH_ARGS+=("--qos=$QOS")
[[ -n "${ACCOUNT:-}" ]] && SBATCH_ARGS+=("--account=$ACCOUNT")
[[ -n "${GRES:-}" ]] && SBATCH_ARGS+=("--gres=$GRES")
[[ -n "${TIME_LIMIT:-}" ]] && SBATCH_ARGS+=("--time=$TIME_LIMIT")
[[ -n "${MEMORY:-}" ]] && SBATCH_ARGS+=("--mem=$MEMORY")
[[ -n "${CPUS_PER_TASK:-}" ]] && SBATCH_ARGS+=("--cpus-per-task=$CPUS_PER_TASK")
[[ -n "${NODELIST:-}" ]] && SBATCH_ARGS+=("--nodelist=$NODELIST")

tf_log "All-step holdout protocol: actual_tokens=$MIN_ACTUAL_TOKENS..$MAX_ACTUAL_TOKENS replace_output=$ALL_STEP_REPLACE_OUTPUT force=$ALL_STEP_FORCE"
job_id="$(sbatch "${SBATCH_ARGS[@]}" jobs/all_step_holdout_eval_v3.sbatch)"
tf_log "Submitted v3 all-step holdout array $job_id ($array_spec)"
