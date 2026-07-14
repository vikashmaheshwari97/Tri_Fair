from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

from src.config.bios_prompt_guide import (
    BIOS_ALLOWED_LABELS,
    BIOS_LABEL_SET,
    bios_confusion_cluster_for_labels,
    bios_label_completion,
    get_bios_label_completions,
    get_bios_rerank_label_cues,
    validate_bios_label,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().casefold() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class BiosCandidateScore:
    label: str
    completion: str
    raw_score: float
    calibrated_score: float
    input_tokens: int
    completion_tokens: int


class BiosLabelScorePredictor:
    """Bias-in-Bios predictor using constrained label scoring plus cluster reranking."""

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

        # Promptolution optimizers use this attribute when they build meta-prompts
        # for prompt mutation/crossover.  Prediction is still done by constrained
        # label scoring; this text only tells the meta-optimizer what output
        # format a good prompt should request.
        self.extraction_description = (
            "Extract exactly one Bias in Bios profession label. "
            "The answer must be one of the allowed profession labels and should be "
            "written as <final_answer>label</final_answer> with no explanation. "
            "Allowed labels are: " + ", ".join(BIOS_ALLOWED_LABELS) + "."
        )

        self.rerank_enabled = _env_bool("BIOS_LABEL_SCORE_RERANK", True)
        self.rerank_always_for_cluster = _env_bool(
            "BIOS_LABEL_SCORE_RERANK_ALWAYS_FOR_CLUSTER",
            True,
        )
        self.rerank_margin = float(os.environ.get("BIOS_LABEL_SCORE_RERANK_MARGIN", "0.15"))
        self.rerank_top_k = int(os.environ.get("BIOS_LABEL_SCORE_RERANK_TOP_K", "4"))
        self.rerank_max_candidates = int(
            os.environ.get("BIOS_LABEL_SCORE_RERANK_MAX_CANDIDATES", "8")
        )
        self.rerank_mix_weight = float(
            os.environ.get("BIOS_LABEL_SCORE_RERANK_MIX_WEIGHT", "0.25")
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
        self._last_debug_cache: dict[tuple[str, str], dict[str, Any]] = {}

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

    def _prediction_context(self, prompt: str, x: str) -> str:
        return f"{str(prompt).rstrip()}\n{str(x).strip()}\n"

    def _neutral_context(self, prompt: str) -> str:
        return (
            f"{str(prompt).rstrip()}\n"
            "This person has a profession, but no job-specific evidence is provided.\n"
        )

    def _rerank_context(self, prompt: str, x: str, labels: Sequence[str]) -> str:
        label_tuple = tuple(validate_bios_label(label) for label in labels)
        cue_text = get_bios_rerank_label_cues(label_tuple)
        return "\n\n".join(
            [
                str(prompt).rstrip(),
                str(x).strip(),
                "The first-stage label scorer found a close profession ambiguity.",
                "Choose only from these candidate labels:\n" + ", ".join(label_tuple),
                "Distinguishing evidence:\n" + cue_text,
                (
                    "Prefer the most specific supported profession over a generic label. "
                    "For example, choose dentist for teeth/oral evidence, chiropractor "
                    "for spine/back/manual-adjustment evidence, surgeon for operations, "
                    "and nurse for RN/bedside nursing evidence."
                ),
                "Use biography evidence only. Do not use gender, names, or stereotypes.",
                "Answer:",
            ]
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
            vllm_module = importlib.import_module("vllm")
            sampling_params_cls = getattr(vllm_module, "SamplingParams")
        except Exception as error:
            raise RuntimeError(
                "vLLM is required for Bias label scoring. Run this mode on the Linux GPU cluster."
            ) from error

        return sampling_params_cls(
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

    def _score_context(
        self,
        context: str,
        labels: Sequence[str] | None = None,
    ) -> dict[str, BiosCandidateScore]:
        label_list = [validate_bios_label(label) for label in (labels or BIOS_ALLOWED_LABELS)]
        completions = [self.completions[label] for label in label_list]
        full_texts = [context + completion for completion in completions]

        outputs = self._engine_generate(full_texts)
        if len(outputs) != len(label_list):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} scoring outputs for {len(label_list)} labels"
            )

        scores: dict[str, BiosCandidateScore] = {}

        for label, completion, full_text, output in zip(
            label_list,
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

    def _apply_calibration(
        self,
        prompt: str,
        scores: dict[str, BiosCandidateScore],
    ) -> dict[str, BiosCandidateScore]:
        if not self.calibrated:
            return scores

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
        return calibrated

    def _ranked(self, scores: dict[str, BiosCandidateScore]) -> list[BiosCandidateScore]:
        return sorted(
            scores.values(),
            key=lambda item: (
                item.calibrated_score,
                item.raw_score,
                -BIOS_ALLOWED_LABELS.index(item.label),
            ),
            reverse=True,
        )

    def _select_rerank_labels(
        self,
        scores: dict[str, BiosCandidateScore],
    ) -> tuple[str, ...]:
        if not self.rerank_enabled:
            return ()

        ranked = self._ranked(scores)
        if len(ranked) < 2:
            return ()

        top = ranked[0]
        top_score = float(top.calibrated_score)

        close_labels = [
            item.label
            for item in ranked[: max(2, self.rerank_top_k)]
            if top_score - float(item.calibrated_score) <= self.rerank_margin
        ]

        top_has_cluster = len(bios_confusion_cluster_for_labels([top.label])) > 1
        if len(close_labels) < 2 and not (
            self.rerank_always_for_cluster and top_has_cluster
        ):
            return ()

        seed_labels = close_labels or [item.label for item in ranked[: self.rerank_top_k]]
        expanded = bios_confusion_cluster_for_labels(seed_labels)

        ranked_position = {item.label: index for index, item in enumerate(ranked)}
        expanded_sorted = sorted(
            expanded,
            key=lambda label: (
                0 if label in ranked_position else 1,
                ranked_position.get(label, len(BIOS_ALLOWED_LABELS)),
                BIOS_ALLOWED_LABELS.index(label),
            ),
        )

        selected = tuple(expanded_sorted[: max(2, self.rerank_max_candidates)])
        return selected if len(selected) >= 2 else ()

    def _combine_rerank_scores(
        self,
        first_scores: dict[str, BiosCandidateScore],
        rerank_scores: dict[str, BiosCandidateScore],
    ) -> dict[str, BiosCandidateScore]:
        combined: dict[str, BiosCandidateScore] = {}
        for label, score in rerank_scores.items():
            first = first_scores[label]
            mixed = (
                float(score.calibrated_score)
                + self.rerank_mix_weight * float(first.calibrated_score)
            )
            combined[label] = BiosCandidateScore(
                label=score.label,
                completion=score.completion,
                raw_score=score.raw_score,
                calibrated_score=mixed,
                input_tokens=score.input_tokens,
                completion_tokens=score.completion_tokens,
            )
        return combined

    def _score_example(self, prompt: str, x: str) -> tuple[str, str, dict[str, int]]:
        context = self._prediction_context(prompt, x)
        first_scores = self._score_context(context)
        first_scores = self._apply_calibration(prompt, first_scores)
        first_ranked = self._ranked(first_scores)

        input_tokens = int(sum(score.input_tokens for score in first_scores.values()))
        output_tokens = int(len(first_scores))
        candidate_count = int(len(first_scores))
        rerank_used = False
        rerank_labels: tuple[str, ...] = ()
        final_scores = first_scores

        rerank_labels = self._select_rerank_labels(first_scores)
        if rerank_labels:
            rerank_context = self._rerank_context(prompt, x, rerank_labels)
            rerank_scores = self._score_context(rerank_context, labels=rerank_labels)
            rerank_scores = self._apply_calibration(prompt, rerank_scores)
            final_scores = self._combine_rerank_scores(first_scores, rerank_scores)

            input_tokens += int(sum(score.input_tokens for score in rerank_scores.values()))
            output_tokens += int(len(rerank_scores))
            candidate_count += int(len(rerank_scores))
            rerank_used = True

        best = self._ranked(final_scores)[0]
        predicted_label = validate_bios_label(best.label)
        completion = bios_label_completion(predicted_label)
        sequence = f"{x}\n{completion}"

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "candidate_count": candidate_count,
            "rerank_used": int(rerank_used),
            "rerank_candidate_count": len(rerank_labels),
        }

        cache_key = (str(prompt), str(x))
        self._usage_cache[cache_key] = usage
        self._last_score_cache[cache_key] = {
            label: float(score.calibrated_score) for label, score in final_scores.items()
        }
        self._last_debug_cache[cache_key] = {
            "first_stage_top": first_ranked[0].label,
            "first_stage_top_score": float(first_ranked[0].calibrated_score),
            "final_top": predicted_label,
            "rerank_used": bool(rerank_used),
            "rerank_labels": list(rerank_labels),
        }

        self._manual_counts["input_tokens"] = (
            int(self._manual_counts.get("input_tokens", 0)) + input_tokens
        )
        self._manual_counts["output_tokens"] = (
            int(self._manual_counts.get("output_tokens", 0)) + output_tokens
        )

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
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "candidate_count": 0,
                    "rerank_used": 0,
                    "rerank_candidate_count": 0,
                },
            )
        )

    def label_scores(self, prompt: str, x: str) -> dict[str, float]:
        return dict(self._last_score_cache.get((str(prompt), str(x)), {}))

    def label_debug(self, prompt: str, x: str) -> dict[str, Any]:
        return dict(self._last_debug_cache.get((str(prompt), str(x)), {}))

    def predict(
        self,
        prompts: Sequence[str],
        xs: Sequence[str],
        system_prompts: Any = None,
    ) -> tuple[list[str], list[str]]:
        if system_prompts is not None:
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
