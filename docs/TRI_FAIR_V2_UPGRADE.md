# Tri-Fair v2: statistical, uncertainty-aware fairness optimization

Method version: `2.0-statistical-objective-aware`

## Changes

- Coverage-valid BBQ unfairness so always-unknown prompts are not treated as fair.
- Civil Comments equalized-odds objective with TPR/FPR identity gaps.
- Bias-in-Bios TPR-gap objective with Wilson-interval readiness.
- Statistical readiness based on support and confidence-interval width.
- Fairness-uncertainty-aware block selection for challenger racing and incumbent advancement.
- Fixed `[quality, -cost, -unfairness]` normalization bounds for Tri-Fair distance calculations.
- Objective-aware mutation shared by Tri-Fair and NSGA-II-PO-Fair.
- Twelve initial prompts, four crossover children, and at most three few-shot examples.
- Separate `results/tri_fair_v2` namespace and `data/splits_v2` manifests, including evaluation recovery jobs.

## Important

These objectives, manifests and population settings are incompatible with the old checkpoints.
Start fresh in the v2 namespace. Run a fresh 1M validation stage before resuming to 5M.
