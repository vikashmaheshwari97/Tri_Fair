# Tri-Fair implementation patch

This directory is an overlay for `mo374z/mo-capo` (the implementation reviewed at commit `265848bb6e0482ab7ef11028d0b8483f6560e724`). Copy its contents into a clean checkout of that repository, preserving paths.

The implementation adds two optimizers:

- **Tri-Fair** — MO-CAPO with three objectives and group-aware intensification.
- **NSGA-II-PO-Fair** — the same three objectives, but full-development evaluation without racing.

The maximize-all optimizer vector is:

```text
[quality, -weighted_inference_cost, -unfairness]
```

Fairness is recomputed over the complete union of evaluated blocks. It is not averaged across blocks, because the group metrics are non-decomposable.

## Included files

The requested files are included, plus support files for optimizer configs, resumable checkpoints, package initialization, and tests.

```text
src/config/base_config.py
src/config/dataset_configs.py
src/config/optimizer_configs.py
src/helpers/task_creation.py
src/tasks/fairness_task.py
src/tasks/__init__.py
src/fairness/base.py
src/fairness/bbq.py
src/fairness/bios.py
src/fairness/civilcomments.py
src/fairness/block_sampler.py
src/fairness/__init__.py
src/checkpointing.py
src/tri_fair.py
src/nsgaii_po_fair.py
scripts/experiment.py
scripts/evaluate_prompts.py
src/callbacks.py
analysis/tri_fair_pipeline.py
jobs/tri_fair_main.sbatch
tests/test_fairness_metrics.py
tests/test_block_sampler.py
```

## Dependencies

The base project already uses most required packages. Ensure these are explicit in the environment:

```bash
uv add pandas pyarrow scikit-learn huggingface-hub
uv add --optional analysis pymoo matplotlib scipy
uv add --dev pytest
```

Civil Comments identities are loaded directly from the pinned raw CSV in the Hugging Face dataset repository; the `wilds` package is not required.

## Pinned datasets

- BBQ: `5d6faae52070aa5eb71b46d1c0723d3ba7930209`
- Bias in Bios: `052f01de644dba841176e0449528b41f27d94a61`
- Civil Comments WILDS: `3fbfeca80bad0f3aec37e72fa07eff222b6e752f`

Generated manifests also record the revision and refuse to load when dataset, seed, or revision differs.

### Official BBQ target metadata

BBQ bias scoring uses the official directional and ambiguity-adjusted formula. The loader also attempts to download the commit-pinned `analysis_scripts/additional_metadata.csv` from the official NYU BBQ repository and caches it at:

```text
data/external/bbq/additional_metadata.csv
```

For final publication runs, stage this file on the cluster and change `require_official_metadata` to `True` in `src/config/dataset_configs.py`. When that flag is false and the file cannot be downloaded, the code logs a warning and uses a conservative target-location inference from the Hugging Face metadata. That fallback is suitable for smoke tests, not the final reported experiment.

## Pilot configuration

| Dataset | Development | Few-shot | Block | Holdout sample |
|---|---:|---:|---:|---:|
| BBQ | 220 | 88 | 44 | 500 |
| Bias in Bios | 336 | 112 | 112 | 500 |
| Civil Comments | 288 | 96 | 96 | 500 |

Optimizer defaults:

```text
initial population:        6
crossovers per iteration:  2
maximum few-shot examples: 3
downstream output limit:   16 tokens
meta-LLM output limit:     256 tokens
seeds:                     42, 43, 44
```

The 16-token downstream limit is configurable with `--max-output-tokens`. The meta-LLM retains a larger limit because it must write complete candidate instructions.

## Immutable split manifests

The first run creates:

```text
data/splits/bbq_seed42.json
data/splits/bias_in_bios_seed42.json
data/splits/civil_comments_seed42.json
```

Every optimizer using that seed loads the same manifest. Do not use `--regenerate-manifest` after results have been produced unless you intentionally invalidate all affected runs.

BBQ sampling keeps four related polarity/context variants together. Bias-in-Bios blocks contain equal quotas from all profession × gender cells. Civil Comments uses greedy quota coverage for overlapping identity × toxicity groups.

## Stage 1: 1M tokens

Run a single local configuration:

```bash
export PYTHONPATH="$PWD"
uv run scripts/experiment.py \
  --experiment-name tri_fair_qwen_bbq_42_1m \
  --random-seed 42 \
  --budget-per-run 1000000 \
  --budget-checkpoints 250000,500000,750000,1000000 \
  --output-dir results/tri_fair/qwen-3-30b/bbq/Tri-Fair/seed42/ \
  --dataset bbq \
  --model qwen-3-30b \
  --optimizer Tri-Fair \
  --n-init-prompts 6 \
  --max-output-tokens 16 \
  --manifest-dir data/splits
```

Evaluate the development-selected prompts on holdout data:

```bash
LOG_DIR=$(cat results/tri_fair/qwen-3-30b/bbq/Tri-Fair/seed42/logging_dir.txt)
uv run scripts/evaluate_prompts.py \
  --random-seed 42 \
  --dataset bbq \
  --model qwen-3-30b \
  --log-path "$LOG_DIR" \
  --incumbents true \
  --step -1 \
  --manifest-dir data/splits
```

## Stage 2: resume from 1M to 5M

The checkpoint excludes model weights, then restores prompt populations, incumbent archive, prediction caches, token counts, Python/NumPy/Torch RNG states, run history, fairness records, and completed-step count after the model is reconstructed. This is stateful algorithmic continuation; bitwise-identical decoding across a new vLLM process is not guaranteed by vLLM itself.

```bash
LOG_DIR=$(cat results/tri_fair/qwen-3-30b/bbq/Tri-Fair/seed42/logging_dir.txt)
uv run scripts/experiment.py \
  --experiment-name tri_fair_qwen_bbq_42_5m \
  --random-seed 42 \
  --budget-per-run 5000000 \
  --budget-checkpoints 2000000,3000000,4000000,5000000 \
  --resume-from "$LOG_DIR/checkpoints/latest.pkl" \
  --output-dir results/tri_fair/qwen-3-30b/bbq/Tri-Fair/seed42/ \
  --dataset bbq \
  --model qwen-3-30b \
  --optimizer Tri-Fair \
  --n-init-prompts 6 \
  --manifest-dir data/splits
```

Only load checkpoint files produced by your own runs; the format is Python pickle.

## Stage 3: resume from 5M to 7.5M

Use the same command with:

```text
--budget-per-run 7500000
--budget-checkpoints 6000000,7000000,7500000
--resume-from <same logging directory>/checkpoints/latest.pkl
```

## SLURM array

The supplied job defines all:

```text
3 models × 3 datasets × 2 optimizers × 3 seeds = 54 tasks
```

For the first stage:

```bash
BUDGET=1000000 RESUME=0 sbatch jobs/tri_fair_main.sbatch
```

For the stateful continuation to 5M:

```bash
BUDGET=5000000 RESUME=1 sbatch jobs/tri_fair_main.sbatch
```

Then:

```bash
BUDGET=7500000 RESUME=1 sbatch jobs/tri_fair_main.sbatch
```

The script intentionally leaves Rocket-specific partition, QoS, account, and GPU-type directives unset. Add those according to the Rocket allocation.

For a one-model pilot, submit only the corresponding array-index range or temporarily reduce the `models` array before submission.

## Analysis

```bash
uv sync --extra analysis
uv run analysis/tri_fair_pipeline_v0.py \
  --results-root results/tri_fair \
  --output-dir analysis/tri_fair_output
```

Outputs include per-run and aggregated CSVs, normalization bounds, 3-D Pareto plots, and pairwise quality–cost–fairness projections.

## Tests

```bash
uv run pytest tests/test_fairness_metrics.py tests/test_block_sampler.py
python -m py_compile $(find src scripts analysis -name '*.py')
```

## Important accounting convention

`--budget-per-run` limits **downstream/evaluation-model tokens**, matching MO-CAPO. Meta-LLM tokens are logged separately. Holdout evaluation is also logged separately and is not silently included in the optimization budget. Like the upstream callback, the stop check occurs after an optimization step, so the actual final count can exceed the requested threshold by one step; the logged actual count is the value used in analysis.
