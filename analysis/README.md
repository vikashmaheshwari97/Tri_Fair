# Tri-Fair analysis folder

This folder is the Tri-Fair replacement for the uploaded MO-CAPO analysis notebooks and scripts.
Copy the complete `analysis/` directory into the root of the Tri-Fair repository.

## What from the MO-CAPO analysis folder is needed?

| MO-CAPO file | Tri-Fair decision | Tri-Fair replacement |
|---|---|---|
| `analysis_pipeline.py` | **Essential**, but it must become three-objective and budget-aware | `analysis_pipeline.py` plus `io.py`, `objectives.py`, `metrics.py`, and `plots.py` |
| `experiment_status.ipynb` | **Essential operationally** | `experiment_status.py` |
| `eval_analysis.ipynb` | Useful, but its single “best score” logic is invalid for three objectives | `inspect_results.py` |
| `step_results_analysis.ipynb` | Useful for debugging and convergence | `trajectory_metrics.csv` from `analysis_pipeline.py`, plus `inspect_results.py` |
| `pareto_front_criterions.ipynb` | Needed for final method comparison, but the notebook is ad hoc | `compare_fronts.py` |
| `make_paper_tables.py` | Needed after results exist | New dynamic `make_paper_tables.py` |
| `figure1.ipynb` | Not required for the first run; useful for the paper | `plot_representative_front.py` |
| `cd_plot.ipynb` | **Do not use as-is.** Initially there are only two optimizers | `statistical_tests.py` uses paired Wilcoxon; it adds Friedman/rank plots only after 3+ methods exist |
| `global_normalization_bounds_all.json` | **Do not copy.** It contains bounds for old datasets and only two objectives | Generate `analysis/normalization_bounds.json` from 1M development results and freeze it |

## Why the old normalization file must not be reused

The uploaded JSON contains performance and cost bounds for AG News, GSM8K, MBPP, and Subj. Tri-Fair instead uses BBQ, Bias in Bios, and Civil Comments with three coordinates:

1. quality loss, `1 - quality`;
2. inference cost;
3. unfairness.

At the first complete 1M stage, the pipeline creates bounds from **development data only**. Quality loss and unfairness use semantic `[0, 1]` bounds; cost uses a model/dataset-specific development maximum with a small margin. Keep that generated JSON unchanged for the later 5M and 7.5M analyses. Otherwise the metric scale would move with the budget.

## Expected result files

The loader expects the schema produced by the supplied Tri-Fair implementation:

```text
results/tri_fair/<model>/<dataset>/<optimizer>/seed<seed>/<run-id>/
├── args.json
├── step_results.parquet
├── eval.parquet
├── runhistory.json
└── checkpoints/
```

`step_results.parquet` must contain at least:

```text
step, prompt, quality, cost, fairness, fairness_ready,
is_incumbent, total_tokens_downstream, time
```

`eval.parquet` must contain at least:

```text
chosen_step, prompt,
dev_quality, dev_cost, dev_fairness, dev_fairness_ready,
test_quality, test_cost, test_fairness, test_fairness_ready
```

## Correct analysis order

### 1. Audit the 1M experiment grid

Use the actual model aliases in `--models`:

```bash
python -m analysis.experiment_status \
  --results-root results/tri_fair \
  --models qwen-3-30b \
  --target-budget 1000000
```

Outputs:

```text
analysis/status/experiment_status.csv
analysis/status/run_or_resume_missing.sh
analysis/status/evaluate_missing_checkpoints.sh
```

Run the generated checkpoint-evaluation commands before the main analysis. The pipeline cannot calculate holdout nR2 or approximation gap for a checkpoint that has never been evaluated.

### 2. Create the 1M analysis and freeze bounds

```bash
python -m analysis.analysis_pipeline \
  --results-root results/tri_fair \
  --output-dir analysis/output \
  --bounds-file analysis/normalization_bounds.json \
  --bounds-budget 1000000 \
  --plot-fronts
```

The first invocation creates `analysis/normalization_bounds.json`. Commit or safely archive this file.

Main outputs:

```text
analysis/output/all_evaluations.parquet
analysis/output/run_metrics.csv
analysis/output/summary.csv
analysis/output/trajectory_metrics.csv
analysis/output/plots/
```

### 3. Inspect representative prompts

Tri-Fair has no single unconditional “best prompt.” This command reports quality-first, cost-first, fairness-first, and balanced prompts selected on development objectives:

```bash
python -m analysis.inspect_results \
  --run-dir results/tri_fair/<model>/<dataset>/<optimizer>/seed42/<run-id> \
  --bounds-file analysis/normalization_bounds.json \
  --budget 1000000 \
  --show-prompts
```

### 4. Compare Tri-Fair with NSGA-II-PO-Fair

```bash
python -m analysis.compare_fronts \
  --results-root results/tri_fair \
  --bounds-file analysis/normalization_bounds.json \
  --optimizer-a Tri-Fair \
  --optimizer-b NSGAII-PO-Fair
```

This produces the optimistic/pessimistic hypervolume-separation criterion and the pessimistic-front-versus-optimistic-front dominance criterion for every matched model, dataset, seed, and budget.

### 5. Statistical comparison

For the initial two-method study:

```bash
python -m analysis.statistical_tests \
  --run-metrics analysis/output/run_metrics.csv \
  --metric noisy_r2_3d \
  --budget 1000000 \
  --optimizers Tri-Fair,NSGAII-PO-Fair
```

This runs a paired Wilcoxon test over complete model-dataset-seed blocks. A critical-difference diagram is not meaningful with only two methods. When a third baseline is added, the same script performs a Friedman test, Holm-corrected pairwise Wilcoxon tests, and an average-rank plot.

### 6. Generate paper tables

```bash
python -m analysis.make_paper_tables \
  --summary analysis/output/summary.csv \
  --budgets 1000000,5000000,7500000
```

Missing future budgets appear as unavailable until those runs are completed.

## Continuing to 5M and 7.5M

After resuming experiments and evaluating their checkpoints, rerun the pipeline **without** `--rebuild-bounds`:

```bash
python -m analysis.analysis_pipeline \
  --results-root results/tri_fair \
  --output-dir analysis/output \
  --bounds-file analysis/normalization_bounds.json
```

Do not regenerate bounds at 5M or 7.5M. The frozen 1M scale is what makes budget trajectories comparable.

## Metrics produced

- exact three-dimensional development hypervolume;
- optimistic holdout hypervolume;
- pessimistic holdout hypervolume;
- three-dimensional approximation gap;
- three-dimensional noisy R2 with Dirichlet preference vectors;
- development-to-test fairness generalization gap;
- balanced, quality-first, cost-first, and fairness-first development-selected prompts;
- per-step development hypervolume and objective trajectories;
- pairwise robust-front criteria across methods.

## Test-data discipline

Prompt selection is always based on development objectives. Test objectives are used only after a prompt or Pareto set has been selected. The normalization file is built from development results, not test results.

## Local validation

```bash
python -m compileall analysis
python -m unittest discover -s analysis/tests -v
```
