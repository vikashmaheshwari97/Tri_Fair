# Tri-Fair v5 study protocol

## Scientific comparison

The main comparison uses the same:

- Qwen-3-30B model snapshot;
- 5,000,000 downstream-token budget;
- v5 split manifest for each dataset/seed;
- 20-instruction shared raw seed pool, with 12 sampled by seed;
- development, few-shot, and test partitions;
- cost weights and output-token limit;
- seeds 52, 53, and 54.

Tri-Fair-v5 then performs four additional method-internal warm-start proposals
(quality, fairness, cost, balanced).  Those proposals are evaluated on development
data and their downstream calls consume the same 5M budget.  This is not a hidden
stronger initial pool; it is an explicit optimizer component.  The Initial
Instructions baseline evaluates only the shared raw pool.

## Why v5 is needed

The v4 variation goals were Civil-Comments-specific and contained toxicity,
`toxic/non_toxic`, TPR/FPR, and equalized-odds language for every dataset.  That is
appropriate for Civil Comments but incorrect for BBQ and Bias-in-Bios.  V5 selects
mutation goals from the active fairness metric.

V5 additionally introduces:

1. a larger shared and semantically diverse initial pool;
2. four counted smart-start search directions;
3. dataset-specific quality/fairness/cost/balanced mutations;
4. contrastive few-shot pairs from the frozen few-shot split;
5. a verified parent pool;
6. an eight-point champion-preserving archive;
7. dynamic progressive fidelity: Civil reaches all six blocks; BBQ and Bios reach
   eight of ten common development blocks late in the run;
8. all-incumbent uncertainty-guarded racing inherited from v4.

## No result guarantee

These changes are designed to improve frontier quality and holdout stability.  No
valid multi-objective method can guarantee superiority in every metric and seed.
Do not edit result files, weaken the baseline, or tune again after inspecting v5
holdout results.

## Freeze rule

Use new seeds and manifests because v3 Civil holdout has already been inspected.
After the v5 smoke test passes, freeze one commit and run all three datasets from
that commit.  Do not change source code between BBQ, Civil Comments, and
Bias-in-Bios.
