"""Launch one cost-definition sensitivity run through the canonical experiment script.

This is intentionally a thin wrapper.  It is not part of the first 1M main
study.  Each invocation runs exactly one variant so variants can be scheduled
as independent SLURM jobs without sharing an output directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

VARIANTS = {
    "original": (1.0, 1.0),
    "input_zero": (0.0, 1.0),
    "output_zero": (1.0, 0.0),
    "both_zero": (0.0, 0.0),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Tri-Fair cost sensitivity variant.",
        add_help=True,
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    args, experiment_args = parser.parse_known_args()
    if not experiment_args:
        parser.error("Pass the normal scripts/experiment.py arguments after --variant")

    input_multiplier, output_multiplier = VARIANTS[args.variant]
    experiment_script = Path(__file__).with_name("experiment.py")
    command = [
        sys.executable,
        str(experiment_script),
        *experiment_args,
        "--input-cost-multiplier",
        str(input_multiplier),
        "--output-cost-multiplier",
        str(output_multiplier),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
