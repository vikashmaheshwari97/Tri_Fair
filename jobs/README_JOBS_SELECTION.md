# MO-CAPO jobs → Tri_Fair jobs selection

| MO-CAPO file | Tri_Fair decision | Active replacement |
|---|---|---|
| `main_experiments.sbatch` | Replace | `submit_tri_fair.sh` + `tri_fair_main.sbatch` |
| `main_experiment_single.sbatch` | Replace | `tri_fair_single.sbatch` |
| `run_eval.sbatch` | Replace | `run_eval.sbatch` |
| `evaluate_initial_prompts.sbatch` | Defer until scripts phase | Not active yet |
| `cost_ablation.sbatch` | Defer until after main 1M | Future three-objective ablation |
| `hp_sens.sbatch` | Defer until after main 1M | Future fairness-aware sensitivity study |
| `tournament_ablation.sbatch` | Retire | Future explicit Tri-Fair selection ablations |
| current `tri_fair_main.sbatch` | Archive as v0, then replace | upgraded worker in this package |

The active first-stage jobs folder should therefore contain five executable
workflow files plus the shared helper and documentation, not every MO-CAPO job.
