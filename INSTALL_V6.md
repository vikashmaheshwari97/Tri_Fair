# Install the Tri-Fair v6 overlay

Copy the overlay into the repository root:

```bash
cp -R tri_fair_v6_overlay/. ~/PycharmProjects/Tri_Fair/
cd ~/PycharmProjects/Tri_Fair
chmod +x jobs/*v6*.sh jobs/*v6*.sbatch tools/plan_legacy_cleanup.sh
python -m py_compile \
  src/config/v6_profiles.py \
  src/fairness/v6_variation.py \
  src/tri_fair_v6.py \
  scripts/experiment_v6.py \
  scripts/evaluate_initial_pool_v6.py
git add \
  src/config/v6_profiles.py \
  src/fairness/v6_variation.py \
  src/tri_fair_v6.py \
  scripts/experiment_v6.py \
  scripts/evaluate_initial_pool_v6.py \
  jobs/tri_fair_v6_main.sbatch \
  jobs/submit_tri_fair_v6.sh \
  jobs/all_step_holdout_eval_v6.sbatch \
  jobs/submit_all_step_holdout_eval_v6.sh \
  jobs/initial_eval_v6.sbatch \
  jobs/submit_initial_eval_v6.sh \
  tools/plan_legacy_cleanup.sh \
  TRI_FAIR_V6_PROTOCOL.md
git commit -m "Add advanced Tri-Fair v6 matched-study overlay"
git push origin main
```

On Rocket:

```bash
cd "$HOME/projects/Tri_Fair"
git pull origin main
chmod +x jobs/*v6*.sh jobs/*v6*.sbatch
```

Run the smoke-test ladder in `TRI_FAIR_V6_PROTOCOL.md` before the full study.


Legacy cleanup is intentionally read-only because v6 still depends on shared v3/v4 modules.
