from __future__ import annotations

import os
from typing import Any, Sequence

from promptolution.predictors import MarkerBasedPredictor

from src.predictors.bios_label_score import BiosLabelScorePredictor


PREDICTION_MODES = (
    "marker",
    "bios_label_score",
    "bios_label_score_calibrated",
)


def create_predictor(
    mode: str,
    llm: Any,
    classes: Sequence[str],
    *,
    dataset: str,
):
    resolved = str(mode).strip().casefold()

    if resolved == "marker":
        return MarkerBasedPredictor(llm, classes)

    if resolved in {"bios_label_score", "bios_label_score_calibrated"}:
        if dataset != "bias_in_bios":
            raise ValueError(
                f"{resolved} is only supported for bias_in_bios, got dataset={dataset!r}"
            )
        return BiosLabelScorePredictor(
            llm,
            classes,
            calibrated=(resolved == "bios_label_score_calibrated"),
            calibration_alpha=float(os.environ.get("BIOS_LABEL_SCORE_CALIBRATION_ALPHA", "1.0")),
            candidate_batch_size=int(os.environ.get("BIOS_LABEL_SCORE_BATCH_SIZE", "128")),
        )

    raise ValueError(
        f"Unknown prediction mode {mode!r}. Valid modes: {', '.join(PREDICTION_MODES)}"
    )
