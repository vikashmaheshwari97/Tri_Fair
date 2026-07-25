#!/usr/bin/env bash
# Resolve complete local model snapshots for Rocket's offline compute nodes.
# This file is sourced by v3 workers.

tf_configure_v3_model_cache() {
  local model="$1"
  local qwen_revision="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
  local gptoss_revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
  local candidate=""

  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

  case "$model" in
    qwen-3-30b)
      local qwen_cache="${QWEN_HF_CACHE:-$HOME/projects/models/Qwen3-30B}"
      local -a qwen_candidates=(
        "${QWEN_LOCAL_SNAPSHOT:-}"
        "$qwen_cache/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/$qwen_revision"
        "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/$qwen_revision"
      )
      for candidate in "${qwen_candidates[@]}"; do
        [[ -n "$candidate" ]] || continue
        if [[ -f "$candidate/config.json" ]] \
           && [[ -f "$candidate/tokenizer.json" || -f "$candidate/tokenizer_config.json" ]] \
           && { [[ -f "$candidate/model.safetensors.index.json" ]] \
                || compgen -G "$candidate/*.safetensors" >/dev/null; }; then
          export QWEN_LOCAL_SNAPSHOT="$candidate"
          export HF_HUB_CACHE="$(dirname "$(dirname "$(dirname "$candidate")")")"
          export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
          tf_log "Using Qwen local snapshot: $QWEN_LOCAL_SNAPSHOT"
          return 0
        fi
      done
      tf_die "No complete Qwen snapshot found. Set QWEN_LOCAL_SNAPSHOT explicitly."
      ;;

    gpt-oss-120b)
      local -a gptoss_candidates=(
        "${GPT_OSS_LOCAL_SNAPSHOT:-}"
        "$HOME/models/gpt-oss-120b/models--openai--gpt-oss-120b/snapshots/$gptoss_revision"
        "$HOME/projects/models/gpt-oss-120b/models--openai--gpt-oss-120b/snapshots/$gptoss_revision"
      )
      for candidate in "${gptoss_candidates[@]}"; do
        [[ -n "$candidate" ]] || continue
        if [[ -f "$candidate/config.json" ]] \
           && [[ -f "$candidate/tokenizer.json" || -f "$candidate/tokenizer_config.json" ]] \
           && [[ -f "$candidate/model.safetensors.index.json" ]]; then
          export GPT_OSS_LOCAL_SNAPSHOT="$candidate"
          tf_log "Using GPT-OSS local snapshot: $GPT_OSS_LOCAL_SNAPSHOT"
          return 0
        fi
      done
      tf_die "No complete GPT-OSS snapshot found. Set GPT_OSS_LOCAL_SNAPSHOT explicitly."
      ;;

    *)
      tf_die "No v3 model-cache rule for '$model'"
      ;;
  esac
}
