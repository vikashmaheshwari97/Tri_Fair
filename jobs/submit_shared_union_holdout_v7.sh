#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source jobs/lib/common.sh

export TF_MODELS="${TF_MODELS:-qwen-3-30b}"
export TF_DATASETS="${TF_DATASETS:-bbq,civil_comments,bias_in_bios}"
export TF_SEEDS="${TF_SEEDS:-72,73,74}"
export TF_OPTIMIZERS="${TF_OPTIMIZERS:-Tri-Fair-v7,NSGAII-PO-Fair}"
export TF_RESULTS_NAMESPACE="${TF_RESULTS_NAMESPACE:-tri_fair_v7_qwen_5m}"
export MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v7}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"
export MIN_ACTUAL_TOKENS="${MIN_ACTUAL_TOKENS:-0}"
export MAX_ACTUAL_TOKENS="${MAX_ACTUAL_TOKENS:-5000000}"
export UNION_FORCE="${UNION_FORCE:-0}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
export NODELIST="${NODELIST:-pegasus2}"

declare -a models datasets seeds
tf_split_csv "$TF_MODELS" models
tf_split_csv "$TF_DATASETS" datasets
tf_split_csv "$TF_SEEDS" seeds
total=$((${#models[@]} * ${#datasets[@]} * ${#seeds[@]}))
array_spec="0-$((total - 1))%$MAX_CONCURRENT"

SBATCH_ARGS=(--parsable "--array=$array_spec")
[[ -n "${PARTITION:-}" ]] && SBATCH_ARGS+=("--partition=$PARTITION")
[[ -n "${TIME_LIMIT:-}" ]] && SBATCH_ARGS+=("--time=$TIME_LIMIT")
[[ -n "${MEMORY:-}" ]] && SBATCH_ARGS+=("--mem=$MEMORY")
[[ -n "${CPUS_PER_TASK:-}" ]] && SBATCH_ARGS+=("--cpus-per-task=$CPUS_PER_TASK")
[[ -n "${NODELIST:-}" ]] && SBATCH_ARGS+=("--nodelist=$NODELIST")

job_id="$(sbatch "${SBATCH_ARGS[@]}" jobs/shared_union_holdout_v7.sbatch)"
tf_log "Submitted shared-union v7 holdout array $job_id ($array_spec)"
