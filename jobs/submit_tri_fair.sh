#!/usr/bin/env bash
# Dynamic submission front-end for the Tri-Fair experiment array.
# Run this from the repository root; do not submit this helper with sbatch.

set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=jobs/lib/common.sh
source jobs/lib/common.sh

# First full stage defaults: one model x three datasets x two methods x three seeds = 18 tasks.
export TF_MODELS="${TF_MODELS:-qwen-3-30b}"
export TF_DATASETS="${TF_DATASETS:-bbq,bias_in_bios,civil_comments}"
export TF_OPTIMIZERS="${TF_OPTIMIZERS:-Tri-Fair,NSGAII-PO-Fair}"
export TF_SEEDS="${TF_SEEDS:-42,43,44}"
export BUDGET="${BUDGET:-1000000}"
export RUN_MODE="${RUN_MODE:-auto}"       # auto | fresh | resume
export AUTO_EVAL="${AUTO_EVAL:-1}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
export MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v2_2}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"
export META_MAX_OUTPUT_TOKENS="${META_MAX_OUTPUT_TOKENS:-256}"
export N_INIT_PROMPTS="${N_INIT_PROMPTS:-12}"
export TF_RESULTS_NAMESPACE="${TF_RESULTS_NAMESPACE:-tri_fair_v2_3}"
export MAX_STEPS="${MAX_STEPS:-2000}"
export STRICT_TOKEN_BUDGET="${STRICT_TOKEN_BUDGET:-1}"
export NEAR_BUDGET_FRACTION="${NEAR_BUDGET_FRACTION:-0.90}"
export MIN_BUDGET_UTILIZATION="${MIN_BUDGET_UTILIZATION:-0.95}"
export OUTPUT_TOKEN_RESERVE="${OUTPUT_TOKEN_RESERVE:-2}"
export FORCE_FRESH="${FORCE_FRESH:-0}"
export DRY_RUN="${DRY_RUN:-0}"
export ALLOW_PARTIAL_RESUME="${ALLOW_PARTIAL_RESUME:-0}"
export NODELIST="${NODELIST:-firefly1,firefly2,firefly3,pegasus2}"
export EVAL_NODELIST="${EVAL_NODELIST:-$NODELIST}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-32}"

case "$RUN_MODE" in
  auto|fresh|resume) ;;
  *) tf_die "RUN_MODE must be auto, fresh, or resume; got '$RUN_MODE'" ;;
esac

tf_validate_positive_int BUDGET "$BUDGET"
tf_validate_positive_int MAX_CONCURRENT "$MAX_CONCURRENT"
tf_validate_positive_int N_INIT_PROMPTS "$N_INIT_PROMPTS"
tf_validate_positive_int MAX_STEPS "$MAX_STEPS"

declare -a models datasets optimizers seeds
tf_split_csv "$TF_MODELS" models
tf_split_csv "$TF_DATASETS" datasets
tf_split_csv "$TF_OPTIMIZERS" optimizers
tf_split_csv "$TF_SEEDS" seeds

total=$((${#models[@]} * ${#datasets[@]} * ${#optimizers[@]} * ${#seeds[@]}))
((total > 0)) || tf_die "Computed zero array tasks"

[[ "$TF_RESULTS_NAMESPACE" =~ ^[A-Za-z0-9._-]+$ ]] \
  || tf_die "Invalid TF_RESULTS_NAMESPACE: $TF_RESULTS_NAMESPACE"
mkdir -p logs "results/$TF_RESULTS_NAMESPACE/submissions"

# A resumed stage must have a valid checkpoint for every requested configuration.
# Failing early is much cheaper than submitting an array that mostly exits.
if [[ "$RUN_MODE" == "resume" || ("$RUN_MODE" == "auto" && "$BUDGET" -gt 1000000) ]]; then
  missing=()
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      for optimizer in "${optimizers[@]}"; do
        for seed in "${seeds[@]}"; do
          output_dir="$(tf_run_output_dir "$ROOT" "$model" "$dataset" "$optimizer" "$seed")"
          if ! logging_dir="$(tf_read_logging_dir "$output_dir")" \
             || [[ ! -f "$logging_dir/checkpoints/latest.pkl" ]]; then
            missing+=("$model/$dataset/$optimizer/seed$seed")
          fi
        done
      done
    done
  done
  if ((${#missing[@]} > 0)); then
    printf 'Missing resumable checkpoints for %d configuration(s):\n' "${#missing[@]}" >&2
    printf '  %s\n' "${missing[@]}" >&2
    tf_is_true "$ALLOW_PARTIAL_RESUME" \
      || tf_die "Resume preflight failed. Set ALLOW_PARTIAL_RESUME=1 only for deliberate partial recovery."
    tf_warn "Partial resume enabled; missing configurations will fail inside the array worker."
  fi
fi

array_spec="0-$((total - 1))%$MAX_CONCURRENT"
SBATCH_ARGS=(
  --parsable
  "--array=$array_spec"
  "--job-name=tfv23-${BUDGET}"
)
[[ -n "${PARTITION:-}" ]] && SBATCH_ARGS+=("--partition=$PARTITION")
[[ -n "${QOS:-}" ]] && SBATCH_ARGS+=("--qos=$QOS")
[[ -n "${ACCOUNT:-}" ]] && SBATCH_ARGS+=("--account=$ACCOUNT")
[[ -n "${GRES:-}" ]] && SBATCH_ARGS+=("--gres=$GRES")
[[ -n "${TIME_LIMIT:-}" ]] && SBATCH_ARGS+=("--time=$TIME_LIMIT")
[[ -n "${MEMORY:-}" ]] && SBATCH_ARGS+=("--mem=$MEMORY")
[[ -n "${CPUS_PER_TASK:-}" ]] && SBATCH_ARGS+=("--cpus-per-task=$CPUS_PER_TASK")
[[ -n "${NODELIST:-}" ]] && SBATCH_ARGS+=("--nodelist=$NODELIST")

printf 'Tri-Fair submission plan\n'
printf '  models:       %s\n' "$TF_MODELS"
printf '  datasets:     %s\n' "$TF_DATASETS"
printf '  optimizers:   %s\n' "$TF_OPTIMIZERS"
printf '  seeds:        %s\n' "$TF_SEEDS"
printf '  budget:       %s\n' "$BUDGET"
printf '  namespace:    %s\n' "$TF_RESULTS_NAMESPACE"
printf '  init prompts: %s\n' "$N_INIT_PROMPTS"
printf '  mode:         %s\n' "$RUN_MODE"
printf '  auto eval:    %s\n' "$AUTO_EVAL"
printf '  strict budget:%s\n' "$STRICT_TOKEN_BUDGET"
printf '  near mode:    %s\n' "$NEAR_BUDGET_FRACTION"
printf '  min utilize:  %s\n' "$MIN_BUDGET_UTILIZATION"
printf '  output reserve:%s\n' "$OUTPUT_TOKEN_RESERVE"
printf '  array:        %s (%d tasks)\n' "$array_spec" "$total"

if tf_is_true "$DRY_RUN"; then
  printf 'DRY RUN: sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}" jobs/tri_fair_main.sbatch
  printf '\n'
  exit 0
fi

job_id="$(sbatch "${SBATCH_ARGS[@]}" jobs/tri_fair_main.sbatch)"
submitted_at="$(date -Is)"
manifest="results/${TF_RESULTS_NAMESPACE}/submissions/${submitted_at//[:+]/-}_budget${BUDGET}_job${job_id}.json"
tf_write_status_json "$manifest" \
  job_id "$job_id" submitted_at "$submitted_at" budget "$BUDGET" \
  run_mode "$RUN_MODE" models "$TF_MODELS" datasets "$TF_DATASETS" \
  optimizers "$TF_OPTIMIZERS" seeds "$TF_SEEDS" array "$array_spec" \
  max_concurrent "$MAX_CONCURRENT" auto_eval "$AUTO_EVAL" \
  results_namespace "$TF_RESULTS_NAMESPACE" n_init_prompts "$N_INIT_PROMPTS" \
  strict_token_budget "$STRICT_TOKEN_BUDGET" \
  near_budget_fraction "$NEAR_BUDGET_FRACTION" \
  minimum_budget_utilization "$MIN_BUDGET_UTILIZATION" \
  output_token_reserve "$OUTPUT_TOKEN_RESERVE"

tf_log "Submitted Tri-Fair array job $job_id"
tf_log "Submission manifest: $manifest"
