# Tri-Fair v3 Rocket model-cache hotfix

This hotfix fixes the failure:

`huggingface_hub.errors.LocalEntryNotFoundError`

The original v3 workers started Rocket in offline mode but allowed vLLM to resolve
the Qwen tokenizer through the default `~/.cache/huggingface/hub` path. The model
weights are stored beneath the project model cache, so all array tasks failed
before optimization began.

Files:
- `src/helpers/llm_creation.py`
- `jobs/lib/model_cache_v3.sh`
- `jobs/tri_fair_v3_main.sbatch`
- `jobs/run_eval_v3.sbatch`
- `jobs/checkpoint_eval_v3.sbatch`
- `jobs/initial_eval_v3.sbatch`

The helper locates the complete Qwen or GPT-OSS snapshot and the Python loader
passes the absolute local snapshot path to vLLM. Evaluation workers use the same
resolution logic.

After extracting at the repository root:

```bash
python -m py_compile src/helpers/llm_creation.py
for f in jobs/tri_fair_v3_main.sbatch jobs/run_eval_v3.sbatch \
         jobs/checkpoint_eval_v3.sbatch jobs/initial_eval_v3.sbatch \
         jobs/lib/model_cache_v3.sh; do
  bash -n "$f"
done
```
