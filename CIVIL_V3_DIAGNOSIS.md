# Civil Comments v3 diagnosis (uploaded 5M Qwen bundle)

These numbers describe the already completed v3 experiment. They are not changed
by the v4 patch.

## Final real logged state, mean ± sample standard deviation over seeds 42/43/44

| Metric | Tri-Fair-v3 | NSGA-II-PO-Fair | Better |
|---|---:|---:|---|
| Development HV ↑ | 0.84109 ± 0.01287 | 0.83815 ± 0.02306 | Tri-Fair, slight |
| Holdout optimistic HV ↑ | 0.87445 ± 0.01010 | 0.88392 ± 0.02291 | NSGA-II |
| Holdout pessimistic HV ↑ | 0.84749 ± 0.02258 | 0.87246 ± 0.03049 | NSGA-II |
| Approximation gap ↓ | 0.02696 ± 0.01485 | 0.01146 ± 0.00763 | NSGA-II |
| Holdout nR2 ↓ | 0.10018 ± 0.00425 | 0.09634 ± 0.00592 | NSGA-II |
| Final archive size | 8.33 ± 2.31 | 10.67 ± 1.53 | NSGA-II broader |

## Representative operating points

| Operating point | Metric | Tri-Fair-v3 | NSGA-II-PO-Fair | Better |
|---|---|---:|---:|---|
| Quality-first | Accuracy ↑ | 0.79467 ± 0.00808 | 0.83133 ± 0.00833 | NSGA-II |
| Quality-first | Cost ↓ | 48.9553 ± 18.8352 | 41.5795 ± 17.5465 | NSGA-II |
| Quality-first | Unfairness ↓ | 0.13944 ± 0.04587 | 0.19561 ± 0.08401 | Tri-Fair |
| Cost-first | Accuracy ↑ | 0.67000 ± 0.10440 | 0.69800 ± 0.12347 | NSGA-II, noisy |
| Cost-first | Cost ↓ | 13.69884 ± 0.25264 | 13.69279 ± 0.24818 | Tied |
| Cost-first | Unfairness ↓ | 0.11911 ± 0.02562 | 0.40999 ± 0.51113 | Tri-Fair |
| Fairness-first | Accuracy ↑ | 0.79200 ± 0.03857 | 0.74800 ± 0.08558 | Tri-Fair |
| Fairness-first | Cost ↓ | 35.2412 ± 22.6943 | 26.6579 ± 7.0968 | NSGA-II |
| Fairness-first | Unfairness ↓ | 0.11852 ± 0.05083 | 0.12837 ± 0.00951 | Tri-Fair, slight |
| Balanced | Accuracy ↑ | 0.76800 ± 0.06416 | 0.76733 ± 0.03443 | Tied |
| Balanced | Cost ↓ | 16.49047 ± 1.04284 | 16.01489 ± 0.39760 | NSGA-II |
| Balanced | Unfairness ↓ | 0.10921 ± 0.03029 | 0.13559 ± 0.02676 | Tri-Fair |

## Interpretation

NSGA-II is not better in every objective. Tri-Fair-v3 has consistent fairness
advantages, and its development hypervolume is marginally higher. The failure is
that those development-selected archives do not transfer as reliably to
holdout. The quality edge is weak, the final archive is often smaller, and the
optimistic–pessimistic gap is larger.

The code-level mechanism matching this pattern is shallow common-block evidence:
Tri-Fair-v3 can initialize and race at two common blocks, whereas Civil Comments
has six development blocks. Archive confirmation starts only at 85% budget. In
contrast, the baseline fully evaluates offspring before rank-and-crowding
selection. The v4 patch addresses this mechanism rather than altering results or
weakening the baseline.
