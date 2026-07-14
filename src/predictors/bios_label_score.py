from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

from src.config.bios_prompt_guide import (
    BIOS_ALLOWED_LABELS,
    BIOS_LABEL_SET,
    bios_label_completion,
    get_bios_label_completions,
    validate_bios_label,
)


@dataclass(frozen=True)
class BiosCandidateScore:
    label: str
    completion: str
    raw_score: float
    calibrated_score: float
    input_tokens: int
    completion_tokens: int


class BiosLabelScorePredictor:
    """Constrained Bias-in-Bios predictor based on label likelihood scoring.

    For every biography, this predictor scores all 28 allowed profession labels
    and returns the label with the best length-normalized log probability.

    It implements the same minimal interface used by Promptolution predictors:
      preds, sequences = predictor.predict(prompts=[...], xs=[...])
    """

    def __init__(
        self,
        llm: Any,
        classes: Sequence[str],
        *,
        calibrated: bool = False,
        calibration_alpha: float = 1.0,
        candidate_batch_size: int | None = None,
        prompt_logprobs: int = 1,
    ) -> None:
        self.llm = llm
        self.classes = [str(value).casefold() for value in classes]
        self.calibrated = bool(calibrated)
        self.calibration_alpha = float(calibration_alpha)
        self.candidate_batch_size = int(
            candidate_batch_size
            or os.environ.get("BIOS_LABEL_SCORE_BATCH_SIZE", "128")
        )
        self.prompt_logprobs = int(
            os.environ.get("BIOS_LABEL_SCORE_PROMPT_LOGPROBS", str(prompt_logprobs))
        )

        missing = set(BIOS_ALLOWED_LABELS) - set(self.classes)
        if missing:
            raise ValueError(
                "BiosLabelScorePredictor requires the Bias in Bios 28-label class set. "
                f"Missing labels: {sorted(missing)}"
            )

        self.engine = getattr(llm, "llm", None)
        if self.engine is None:
            raise TypeError(
                "BiosLabelScorePredictor expects the Promptolution VLLM wrapper "
                "with an underlying .llm vLLM engine."
            )

        self.tokenizer = self._resolve_tokenizer()
        self.completions = get_bios_label_completions()

        self._prior_cache: dict[str, dict[str, BiosCandidateScore]] = {}
        self._usage_cache: dict[tuple[str, str], dict[str, int]] = {}
        self._last_score_cache: dict[tuple[str, str], dict[str, float]] = {}

        manual = getattr(self.llm, "_tri_fair_manual_token_count", None)
        if not isinstance(manual, dict):
            manual = {"input_tokens": 0, "output_tokens": 0}
            setattr(self.llm, "_tri_fair_manual_token_count", manual)
        self._manual_counts = manual

    def _resolve_tokenizer(self) -> Any:
        if hasattr(self.engine, "get_tokenizer"):
            return self.engine.get_tokenizer()
        tokenizer = getattr(self.engine, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
        raise TypeError("Could not resolve tokenizer from the vLLM engine.")

    def _encode(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "encode"):
            try:
                return list(self.tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                return list(self.tokenizer.encode(text))
        raise TypeError("Resolved tokenizer does not expose encode().")

    def _token_count(self, text: str) -> int:
        return len(self._encode(text))

    def _prediction_context(self, prompt: str, x: str) -> str:
        return f"{str(prompt).rstrip()}\n{str(x).strip()}\n"

    def _neutral_context(self, prompt: str) -> str:
        return (
            f"{str(prompt).rstrip()}\n"
            "This person has a profession, but no job-specific evidence is provided.\n"
        )

    def _extract_logprob(self, entry: Any, token_id: int) -> float | None:
        if entry is None:
            return None

        if isinstance(entry, dict):
            value = entry.get(token_id)
            if value is None:
                value = entry.get(str(token_id))
            if value is None:
                for key, candidate in entry.items():
                    try:
                        if int(key) == int(token_id):
                            value = candidate
                            break
                    except Exception:
                        continue
            if value is None:
                return None
            if hasattr(value, "logprob"):
                return float(value.logprob)
            return float(value)

        if hasattr(entry, "logprob"):
            return float(entry.logprob)

        try:
            return float(entry)
        except Exception:
            return None

    def _completion_start(
        self,
        full_ids: list[int],
        completion_ids: list[int],
        context: str,
    ) -> int:
        if completion_ids and len(full_ids) >= len(completion_ids):
            if full_ids[-len(completion_ids):] == completion_ids:
                return len(full_ids) - len(completion_ids)
        context_len = len(self._encode(context))
        return min(context_len, len(full_ids))

    def _make_sampling_params(self) -> Any:
        try:
            from vllm import SamplingParams
        except Exception as error:
            raise RuntimeError(
                "vLLM is required for Bias label scoring. Run on the GPU cluster."
            ) from error

        return SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=self.prompt_logprobs,
        )

    def _engine_generate(self, texts: list[str]) -> list[Any]:
        if not texts:
            return []
        params = self._make_sampling_params()
        outputs: list[Any] = []
        for start in range(0, len(texts), self.candidate_batch_size):
            chunk = texts[start : start + self.candidate_batch_size]
            try:
                chunk_outputs = self.engine.generate(
                    chunk,
                    params,
                    use_tqdm=False,
                )
            except TypeError:
                chunk_outputs = self.engine.generate(chunk, params)
            outputs.extend(chunk_outputs)
        return outputs

    def _score_context(self, context: str) -> dict[str, BiosCandidateScore]:
        labels = list(BIOS_ALLOWED_LABELS)
        completions = [self.completions[label] for label in labels]
        full_texts = [context + completion for completion in completions]

        outputs = self._engine_generate(full_texts)
        if len(outputs) != len(labels):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} scoring outputs for {len(labels)} labels"
            )

        scores: dict[str, BiosCandidateScore] = {}

        for label, completion, full_text, output in zip(
            labels,
            completions,
            full_texts,
            outputs,
        ):
            full_ids = list(
                getattr(output, "prompt_token_ids", None)
                or self._encode(full_text)
            )
            completion_ids = self._encode(completion)
            start = self._completion_start(full_ids, completion_ids, context)
            prompt_logprobs = getattr(output, "prompt_logprobs", None)

            token_logprobs: list[float] = []
            if prompt_logprobs is not None:
                for pos in range(start, len(full_ids)):
                    if pos >= len(prompt_logprobs):
                        continue
                    value = self._extract_logprob(prompt_logprobs[pos], full_ids[pos])
                    if value is not None and math.isfinite(value):
                        token_logprobs.append(float(value))

            if token_logprobs:
                raw_score = float(sum(token_logprobs) / len(token_logprobs))
            else:
                raw_score = float("-inf")

            scores[label] = BiosCandidateScore(
                label=label,
                completion=completion,
                raw_score=raw_score,
                calibrated_score=raw_score,
                input_tokens=len(full_ids),
                completion_tokens=len(completion_ids),
            )

        return scores

    def _prior_scores(self, prompt: str) -> dict[str, BiosCandidateScore]:
        if prompt not in self._prior_cache:
            self._prior_cache[prompt] = self._score_context(self._neutral_context(prompt))
        return self._prior_cache[prompt]

    def _score_example(self, prompt: str, x: str) -> tuple[str, str, dict[str, int]]:
        context = self._prediction_context(prompt, x)
        scores = self._score_context(context)

        if self.calibrated:
            priors = self._prior_scores(prompt)
            calibrated: dict[str, BiosCandidateScore] = {}
            for label, score in scores.items():
                prior = priors[label].raw_score
                calibrated_score = score.raw_score - self.calibration_alpha * prior
                calibrated[label] = BiosCandidateScore(
                    label=score.label,
                    completion=score.completion,
                    raw_score=score.raw_score,
                    calibrated_score=calibrated_score,
                    input_tokens=score.input_tokens,
                    completion_tokens=score.completion_tokens,
                )
            scores = calibrated

        best = max(
            scores.values(),
            key=lambda item: (
                item.calibrated_score,
                item.raw_score,
                -BIOS_ALLOWED_LABELS.index(item.label),
            ),
        )

        predicted_label = validate_bios_label(best.label)
        completion = bios_label_completion(predicted_label)
        sequence = f"{x}\n{completion}"

        input_tokens = int(sum(score.input_tokens for score in scores.values()))
        output_tokens = int(best.completion_tokens)

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "candidate_count": len(scores),
        }

        self._usage_cache[(str(prompt), str(x))] = usage
        self._last_score_cache[(str(prompt), str(x))] = {
            label: float(score.calibrated_score) for label, score in scores.items()
        }

        self._manual_counts["input_tokens"] = int(self._manual_counts.get("input_tokens", 0)) + input_tokens
        self._manual_counts["output_tokens"] = int(self._manual_counts.get("output_tokens", 0)) + output_tokens

        return predicted_label, sequence, usage

    def token_usage(
        self,
        prompt: str,
        x: str,
        raw_output: str | None = None,
    ) -> dict[str, int]:
        del raw_output
        return dict(
            self._usage_cache.get(
                (str(prompt), str(x)),
                {"input_tokens": 0, "output_tokens": 0, "candidate_count": 0},
            )
        )

    def label_scores(self, prompt: str, x: str) -> dict[str, float]:
        return dict(self._last_score_cache.get((str(prompt), str(x)), {}))

    def predict(
        self,
        prompts: Sequence[str],
        xs: Sequence[str],
        system_prompts: Any = None,
    ) -> tuple[list[str], list[str]]:
        if system_prompts is not None:
            # Current project does not use system prompts for these fairness tasks.
            pass

        if len(prompts) != len(xs):
            raise ValueError(
                f"prompts/xs length mismatch: {len(prompts)} prompts vs {len(xs)} inputs"
            )

        preds: list[str] = []
        sequences: list[str] = []

        for prompt, x in zip(prompts, xs):
            pred, sequence, _usage = self._score_example(str(prompt), str(x))
            preds.append(pred)
            sequences.append(sequence)

        return preds, sequences


def is_bios_label_score_mode(mode: str) -> bool:
    return str(mode).strip().casefold() in {
        "bios_label_score",
        "bios_label_score_calibrated",
    }
