"""Launch one optimizer-parameter sensitivity run through ``experiment.py``.

The main 1M experiment must use the frozen values in optimizer_configs.py.
This wrapper is reserved for later sensitivity analysis.  Dataset block sizes
are deliberately not exposed: changing a fairness block requires rebuilding
and revalidating the immutable manifest, not merely changing an integer flag.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Tri-Fair hyperparameter sensitivity configuration."
    )
    parser.add_argument("--crossovers-per-iteration", type=int, required=True)
    parser.add_argument("--max-few-shot-examples", type=int, required=True)
    args, experiment_args = parser.parse_known_args()
    if args.crossovers_per_iteration <= 0:
        parser.error("--crossovers-per-iteration must be positive")
    if args.max_few_shot_examples < 0:
        parser.error("--max-few-shot-examples cannot be negative")
    if not experiment_args:
        parser.error(
            "Pass the normal scripts/experiment.py arguments after the sensitivity flags"
        )

    experiment_script = Path(__file__).with_name("experiment.py")
    command = [
        sys.executable,
        str(experiment_script),
        *experiment_args,
        "--crossovers-per-iteration",
        str(args.crossovers_per_iteration),
        "--max-few-shot-examples",
        str(args.max_few_shot_examples),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
