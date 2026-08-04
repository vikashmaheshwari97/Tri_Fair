"""Generate Mistral-Small-3.2-24B 5M publication figures for Bias-in-Bios."""
from __future__ import annotations
import sys
from analysis.make_mistral_5m_figures import main

if __name__ == "__main__":
    main(["--dataset", "bias_in_bios", *sys.argv[1:]])
