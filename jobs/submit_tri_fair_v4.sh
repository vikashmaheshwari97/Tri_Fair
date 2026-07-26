#!/usr/bin/env bash
# Submit Tri-Fair-v4 versus the unchanged full-development NSGA-II baseline.

set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source jobs/lib/common.sh

export TF_MODELS="${TF_MODELS:-qwen-3-30b}"
export TF_DATASETS="${TF_DATASETS:-bbq,civil_comments,bias_in_bios}"
export TF_OPTIMIZERS="${TF_OPTIMIZERS:-Tri-Fair-v4,NSGAII-PO-Fair}"
export TF_SEEDS="${TF_SEEDS:-52,53,54}"
export BUDGET="${BUDGET:-5000000}"
export RUN_MODE="${RUN_MODE:-fresh}"
export AUTO_EVAL="${AUTO_EVAL:-0}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
export MANIFEST_DIR="${MANIFEST_DIR:-data/splits_v4}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-16}"
export META_MAX_OUTPUT_TOKENS="${META_MAX_OUTPUT_TOKENS:-256}"
export N_INIT_PROMPTS="${N_INIT_PROMPTS:-12}"
export TF_RESULTS_NAMESPACE="${TF_RESULTS_NAMESPACE:-tri_fair_v4_qwen_5m}"
export MAX_STEPS="${MAX_STEPS:-4000}"
export STRICT_TOKEN_BUDGET="${STRICT_TOKEN_BUDGET:-1}"
export NEAR_BUDGET_FRACTION="${NEAR_BUDGET_FRACTION:-0.90}"
export MIN_BUDGET_UTILIZATION="${MIN_BUDGET_UTILIZATION:-0.95}"
export OUTPUT_TOKEN_RESERVE="${OUTPUT_TOKEN_RESERVE:-2}"
export FORCE_FRESH="${FORCE_FRESH:-0}"
export DRY_RUN="${DRY_RUN:-0}"
export NODELIST="${NODELIST:-firefly1,firefly2,firefly3}"
export EVAL_NODELIST="${EVAL_NODELIST:-$NODELIST}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-32}"

case "$RUN_MODE" in
  auto|fresh|resume) ;;
  *) tf_die "RUN_MODE must be auto, fresh, or resume; got '$RUN_MODE'" ;;
esac

for pair in \
  "BUDGET:$BUDGET" \
  "MAX_CONCURRENT:$MAX_CONCURRENT" \
  "N_INIT_PROMPTS:$N_INIT_PROMPTS" \
  "MAX_STEPS:$MAX_STEPS" \
  "MAX_OUTPUT_TOKENS:$MAX_OUTPUT_TOKENS"; do
  tf_validate_positive_int "${pair%%:*}" "${pair#*:}"
done

declare -a models datasets optimizers seeds
tf_split_csv "$TF_MODELS" models
tf_split_csv "$TF_DATASETS" datasets
tf_split_csv "$TF_OPTIMIZERS" optimizers
tf_split_csv "$TF_SEEDS" seeds

for model in "${models[@]}"; do
  if [[ "$model" == "gpt-oss-120b" ]] && ((MAX_OUTPUT_TOKENS < 96)); then
    tf_die "GPT-OSS-120B requires MAX_OUTPUT_TOKENS>=96; got $MAX_OUTPUT_TOKENS"
  fi
done
for optimizer in "${optimizers[@]}"; do
  tf_contains "$optimizer" Tri-Fair-v4 NSGAII-PO-Fair \
    || tf_die "Unsupported v4 optimizer '$optimizer'"
done

total=$((${#models[@]} * ${#datasets[@]} * ${#optimizers[@]} * ${#seeds[@]}))
array_spec="0-$((total - 1))%$MAX_CONCURRENT"
mkdir -p logs "results/$TF_RESULTS_NAMESPACE/submissions"

if [[ "$RUN_MODE" == "resume" ]]; then
  missing=()
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      for optimizer in "${optimizers[@]}"; do
        for seed in "${seeds[@]}"; do
          output_dir="$(tf_run_output_dir "$ROOT" "$model" "$dataset" "$optimizer" "$seed")"
          if ! logging_dir="$(tf_read_logging_dir "$output_dir" 2>/dev/null)" \
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
    tf_die "Resume preflight failed; complete or restart the missing configurations"
  fi
fi

SBATCH_ARGS=(
  --parsable
  "--array=$array_spec"
  "--job-name=tfv4-${BUDGET}"
)
[[ -n "${PARTITION:-}" ]] && SBATCH_ARGS+=("--partition=$PARTITION")
[[ -n "${QOS:-}" ]] && SBATCH_ARGS+=("--qos=$QOS")
[[ -n "${ACCOUNT:-}" ]] && SBATCH_ARGS+=("--account=$ACCOUNT")
[[ -n "${GRES:-}" ]] && SBATCH_ARGS+=("--gres=$GRES")
[[ -n "${TIME_LIMIT:-}" ]] && SBATCH_ARGS+=("--time=$TIME_LIMIT")
[[ -n "${MEMORY:-}" ]] && SBATCH_ARGS+=("--mem=$MEMORY")
[[ -n "${CPUS_PER_TASK:-}" ]] && SBATCH_ARGS+=("--cpus-per-task=$CPUS_PER_TASK")
[[ -n "${NODELIST:-}" ]] && SBATCH_ARGS+=("--nodelist=$NODELIST")

printf 'Tri-Fair-v4 submission plan\n'
printf '  models:       %s\n' "$TF_MODELS"
printf '  datasets:     %s\n' "$TF_DATASETS"
printf '  optimizers:   %s\n' "$TF_OPTIMIZERS"
printf '  seeds:        %s\n' "$TF_SEEDS"
printf '  budget:       %s\n' "$BUDGET"
printf '  namespace:    %s\n' "$TF_RESULTS_NAMESPACE"
printf '  manifest dir: %s\n' "$MANIFEST_DIR"
printf '  array:        %s (%d tasks)\n' "$array_spec" "$total"

if tf_is_true "$DRY_RUN"; then
  printf 'DRY RUN: sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}" jobs/tri_fair_v4_main.sbatch
  printf '\n'
  exit 0
fi

job_id="$(sbatch "${SBATCH_ARGS[@]}" jobs/tri_fair_v4_main.sbatch)"
submitted_at="$(date -Is)"
manifest="results/${TF_RESULTS_NAMESPACE}/submissions/${submitted_at//[:+]/-}_budget${BUDGET}_job${job_id}.json"
tf_write_status_json "$manifest" \
  job_id "$job_id" submitted_at "$submitted_at" budget "$BUDGET" \
  run_mode "$RUN_MODE" models "$TF_MODELS" datasets "$TF_DATASETS" \
  optimizers "$TF_OPTIMIZERS" seeds "$TF_SEEDS" array "$array_spec" \
  max_concurrent "$MAX_CONCURRENT" auto_eval "$AUTO_EVAL" \
  results_namespace "$TF_RESULTS_NAMESPACE" manifest_dir "$MANIFEST_DIR"

tf_log "Submitted Tri-Fair-v4 array job $job_id"
tf_log "Submission manifest: $manifest"
