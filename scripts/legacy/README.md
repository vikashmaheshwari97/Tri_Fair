# Legacy scripts

These files are snapshots of the original MO-CAPO entry points and the first
Tri-Fair draft. They are retained only for provenance and code comparison.

Do not call them from SLURM jobs. In particular, the original evaluators:

- do not compute fairness,
- do not preserve structured few-shot prompts reliably,
- use non-fixed or incomplete dataset handling,
- do not emit the canonical three-objective evaluation schema.
