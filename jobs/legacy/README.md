# Legacy MO-CAPO job files

Do not copy the original MO-CAPO job scripts into the active Tri_Fair jobs directory.
They encode the original datasets, methods, 7.5M-only budget, and two-objective
analysis/evaluation assumptions.

Files intentionally retired from the active Tri_Fair workflow:

- `main_experiments.sbatch`: replaced by `submit_tri_fair.sh` plus `tri_fair_main.sbatch`.
- `main_experiment_single.sbatch`: replaced by `tri_fair_single.sbatch`.
- `run_eval.sbatch`: replaced by the Tri-Fair-aware `run_eval.sbatch`.
- `evaluate_initial_prompts.sbatch`: defer until the corresponding Python evaluator
  is upgraded to emit quality, cost, fairness, group support, and fixed-manifest metadata.
- `cost_ablation.sbatch`: defer until after the main 1M study.
- `hp_sens.sbatch`: defer until after the main 1M study; fairness readiness and
  dataset-specific block settings require a new sensitivity design.
- `tournament_ablation.sbatch`: do not reuse. Its MO-CAPO-only selection ablations
  should later become explicit Tri-Fair ablations.

The old files may be preserved here for provenance, but they must not be submitted
for Tri_Fair experiments.
