#!/usr/bin/env bash
# Submit the 3 datasets × 3 seeds initial-population evaluation matrix.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=jobs/lib/common.sh
source jobs/lib/common.sh

export TF_DATASETS="${TF_DATASETS:-bbq,bias_in_bios,civil_comments}"
export TF_SEEDS="${TF_SEEDS:-42,43,44}"
export TF_RESULTS_NAMESPACE="${TF_RESULTS_NAMESPACE:-tri_fair_v2_2}"
export MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v2_2}"
export SOURCE_BUDGET="${SOURCE_BUDGET:-5000000}"
export SOURCE_OPTIMIZERS="${SOURCE_OPTIMIZERS:-Tri-Fair,NSGAII-PO-Fair}"
export PREFERRED_OPTIMIZER="${PREFERRED_OPTIMIZER:-Tri-Fair}"
export EXPECTED_INITIAL_PROMPTS="${EXPECTED_INITIAL_PROMPTS:-12}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
export DRY_RUN="${DRY_RUN:-0}"

NODELIST="${NODELIST:-firefly1,firefly2,firefly3,pegasus2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
PARTITION="${PARTITION:-}"
QOS="${QOS:-}"
ACCOUNT="${ACCOUNT:-}"
GRES="${GRES:-}"
TIME_LIMIT="${TIME_LIMIT:-}"
MEMORY="${MEMORY:-}"

tf_validate_positive_int SOURCE_BUDGET "$SOURCE_BUDGET"
tf_validate_positive_int EXPECTED_INITIAL_PROMPTS "$EXPECTED_INITIAL_PROMPTS"
tf_validate_positive_int MAX_CONCURRENT "$MAX_CONCURRENT"

declare -a datasets seeds
tf_split_csv "$TF_DATASETS" datasets
tf_split_csv "$TF_SEEDS" seeds

total=$((${#datasets[@]} * ${#seeds[@]}))
((total > 0)) || tf_die "Computed zero initial-baseline tasks"

mkdir -p logs "results/$TF_RESULTS_NAMESPACE/submissions"

array_spec="0-$((total - 1))%$MAX_CONCURRENT"

SBATCH_ARGS=(
  --parsable
  "--array=$array_spec"
  --job-name=tf-init-5m-v2
)

[[ -n "$PARTITION" ]] && SBATCH_ARGS+=("--partition=$PARTITION")
[[ -n "$QOS" ]] && SBATCH_ARGS+=("--qos=$QOS")
[[ -n "$ACCOUNT" ]] && SBATCH_ARGS+=("--account=$ACCOUNT")
[[ -n "$GRES" ]] && SBATCH_ARGS+=("--gres=$GRES")
[[ -n "$TIME_LIMIT" ]] && SBATCH_ARGS+=("--time=$TIME_LIMIT")
[[ -n "$MEMORY" ]] && SBATCH_ARGS+=("--mem=$MEMORY")
[[ -n "$CPUS_PER_TASK" ]] && SBATCH_ARGS+=("--cpus-per-task=$CPUS_PER_TASK")
[[ -n "$NODELIST" ]] && SBATCH_ARGS+=("--nodelist=$NODELIST")

printf 'Initial Instructions 5M v2 evaluation plan\n'
printf '  datasets:       %s\n' "$TF_DATASETS"
printf '  seeds:          %s\n' "$TF_SEEDS"
printf '  source budget:  %s\n' "$SOURCE_BUDGET"
printf '  namespace:      %s\n' "$TF_RESULTS_NAMESPACE"
printf '  manifest dir:   %s\n' "$MANIFEST_DIR"
printf '  source methods: %s\n' "$SOURCE_OPTIMIZERS"
printf '  initial prompts:%s\n' "$EXPECTED_INITIAL_PROMPTS"
printf '  array:          %s (%d tasks)\n' "$array_spec" "$total"

if tf_is_true "$DRY_RUN"; then
  printf 'DRY RUN: sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}" jobs/run_initial_instructions_5m_v2.sbatch
  printf '\n'
  exit 0
fi

job_id="$(sbatch "${SBATCH_ARGS[@]}" jobs/run_initial_instructions_5m_v2.sbatch)"
submitted_at="$(date -Is)"
manifest="results/${TF_RESULTS_NAMESPACE}/submissions/${submitted_at//[:+]/-}_initial_5m_v2_job${job_id}.json"

tf_write_status_json "$manifest" \
  job_id "$job_id" \
  submitted_at "$submitted_at" \
  task_type initial_instructions_5m_v2 \
  datasets "$TF_DATASETS" \
  seeds "$TF_SEEDS" \
  source_budget "$SOURCE_BUDGET" \
  source_optimizers "$SOURCE_OPTIMIZERS" \
  expected_initial_prompts "$EXPECTED_INITIAL_PROMPTS" \
  array "$array_spec" \
  max_concurrent "$MAX_CONCURRENT" \
  results_namespace "$TF_RESULTS_NAMESPACE" \
  manifest_dir "$MANIFEST_DIR"

tf_log "Submitted initial-instruction array job $job_id"
tf_log "Submission manifest: $manifest"
