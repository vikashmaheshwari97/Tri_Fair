# Tri-Fair v4: progressive shared-fidelity racing

## Why this is a new method version

The completed Civil Comments v3 holdout was inspected before this revision was
designed. Therefore the code must not silently replace v3. It is installed as
`Tri-Fair-v4`, uses a new results namespace, and should be evaluated from fresh
runs. The unchanged `NSGAII-PO-Fair` baseline must also be rerun under the same
seeds, manifests, model, and 5M downstream-token budget.

The recommended confirmatory seeds are `52,53,54`, not the already inspected
`42,43,44`. This also creates fresh split manifests under `data/splits_v4`.

## Diagnosis from Civil Comments v3

The problem was not the holdout evaluator or the figures. At the final real
logged state:

- Tri-Fair-v3 had slightly higher development HV on average, but lower holdout
  optimistic and pessimistic HV, higher nR2, and a larger approximation gap.
- Tri-Fair-v3 was often better on unfairness, especially at cost-first and
  fairness-first operating points.
- NSGA-II-PO-Fair was stronger on the quality edge and produced a broader,
  more stable holdout archive.
- Tri-Fair-v3 final archives contained 7/11/7 candidates versus 9/11/12 for the
  baseline.

This pattern indicates development-ranking instability rather than a missing
fairness objective. The v3 archive could be selected on two common blocks, while
Civil Comments has six development blocks. Full-development NSGA-II receives a
more stable ranking before every environmental selection.

## Algorithmic changes

1. **Progressive shared fidelity**
   - 0–30% budget: at least 2 common blocks
   - 30–60%: at least 3
   - 60–85%: at least 4
   - 85–100%: at least 5

2. **Interleaved confirmation**
   Archive confirmation begins during search and is limited to two consecutive
   confirmation actions before another search iteration, except in the final
   phase.

3. **Less aggressive early rejection**
   The quality and fairness uncertainty margins are reduced so small real
   quality improvements are not discarded simply because an incumbent is
   slightly cheaper or fairer.

4. **All-incumbent robust comparison**
   A challenger is checked against every comparable incumbent, not only one
   geometrically closest incumbent.

5. **Reference-direction offspring**
   Every iteration deliberately generates quality, fairness, cost, and balanced
   candidates from objective-specific archive elites and lexically diverse
   partners.

6. **Signed Civil error-cell targeting**
   Fairness few-shots use the sign of the largest TPR/FPR gap to target the
   confusion-matrix cell that moves the disparity toward zero. V3 used only the
   largest absolute gap and could select the wrong label direction.

7. **Matched offspring count**
   Tri-Fair-v4 uses four offspring per search iteration, matching the baseline.
   V3 used eight and spent more budget on shallow races and meta-model calls.

## Install locally

From the repository root:

```bash
unzip -o ~/Downloads/tri_fair_v4_progressive_racing_patch_20260726.zip -d .

python -m py_compile \
  src/fairness/v4_variation.py \
  src/tri_fair_v4.py \
  src/config/v4_profiles.py \
  scripts/experiment_v4.py

bash -n \
  jobs/submit_tri_fair_v4.sh \
  jobs/tri_fair_v4_main.sbatch

git add \
  src/fairness/v4_variation.py \
  src/tri_fair_v4.py \
  src/config/v4_profiles.py \
  scripts/experiment_v4.py \
  jobs/submit_tri_fair_v4.sh \
  jobs/tri_fair_v4_main.sbatch \
  TRI_FAIR_V4_DEPLOY.md

git commit -m "Add progressive shared-fidelity Tri-Fair v4"
git push origin main
```

## Freeze before running

On Rocket:

```bash
cd "$HOME/projects/Tri_Fair"
git pull --ff-only origin main
source "$HOME/venvs/tri-fair/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

git rev-parse HEAD | tee TRI_FAIR_V4_FROZEN_COMMIT.txt
```

Do not change the method after the first v4 holdout result is inspected.

## Dry-run Civil Comments

```bash
NODELIST=pegasus2 \
QWEN_LOCAL_SNAPSHOT="$QWEN_LOCAL_SNAPSHOT" \
TF_HF_HOME="$TF_HF_HOME" \
TF_HF_HUB_CACHE="$TF_HF_HUB_CACHE" \
TF_HF_DATASETS_CACHE="$TF_HF_DATASETS_CACHE" \
TF_MODELS=qwen-3-30b \
TF_DATASETS=civil_comments \
TF_OPTIMIZERS=Tri-Fair-v4,NSGAII-PO-Fair \
TF_SEEDS=52,53,54 \
TF_RESULTS_NAMESPACE=tri_fair_v4_qwen_5m \
MANIFEST_DIR=data/splits_v4 \
BUDGET=5000000 \
RUN_MODE=fresh \
AUTO_EVAL=0 \
MAX_OUTPUT_TOKENS=16 \
META_MAX_OUTPUT_TOKENS=256 \
N_INIT_PROMPTS=12 \
MAX_STEPS=4000 \
MAX_CONCURRENT=2 \
PARTITION=gpu \
MEMORY=160G \
CPUS_PER_TASK=32 \
TIME_LIMIT=24:00:00 \
DRY_RUN=1 \
bash jobs/submit_tri_fair_v4.sh
```

Change `DRY_RUN=1` to `DRY_RUN=0` only after reviewing the six-task matrix.

## Holdout protocol

After all six optimizations finish, use the existing all-real-step evaluator:

```bash
NODELIST=pegasus2 \
QWEN_LOCAL_SNAPSHOT="$QWEN_LOCAL_SNAPSHOT" \
TF_HF_HOME="$TF_HF_HOME" \
TF_HF_HUB_CACHE="$TF_HF_HUB_CACHE" \
TF_HF_DATASETS_CACHE="$TF_HF_DATASETS_CACHE" \
TF_MODELS=qwen-3-30b \
TF_DATASETS=civil_comments \
TF_OPTIMIZERS=Tri-Fair-v4,NSGAII-PO-Fair \
TF_SEEDS=52,53,54 \
TF_RESULTS_NAMESPACE=tri_fair_v4_qwen_5m \
MANIFEST_DIR=data/splits_v4 \
MIN_ACTUAL_TOKENS=0 \
MAX_ACTUAL_TOKENS=5000000 \
MAX_OUTPUT_TOKENS=16 \
ALL_STEP_REPLACE_OUTPUT=1 \
ALL_STEP_FORCE=0 \
MAX_CONCURRENT=2 \
PARTITION=gpu \
MEMORY=160G \
CPUS_PER_TASK=32 \
TIME_LIMIT=24:00:00 \
bash jobs/submit_all_step_holdout_eval_v3.sh
```

The filename contains `v3` because it is the generic fairness-profile evaluator;
it accepts `Tri-Fair-v4` through `TF_OPTIMIZERS` and uses the same frozen data
profile.

## Important interpretation

The changes are intended to improve quality-front discovery and development to
holdout stability while retaining Tri-Fair's fairness gains. No algorithm can
honestly guarantee that one method will win accuracy, cost, unfairness, nR2,
optimistic HV, pessimistic HV, and gap on every seed. Such a guarantee would
require manipulating the baseline, the evaluation, or the reported data.
