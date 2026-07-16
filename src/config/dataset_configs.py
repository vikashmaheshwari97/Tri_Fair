"""Dataset configurations for original MO-CAPO and Tri-Fair experiments."""

from __future__ import annotations

import re
import signal
from typing import Dict

from sklearn.metrics import accuracy_score

from src.config.base_config import (
    DatasetConfig,
    DatasetType,
    FairnessConfig,
    SplitConfig,
)
from src.config.initial_prompts import INITIAL_PROMPTS
from src.config.task_descriptions import TASK_DESCRIPTIONS


# ---------------------------------------------------------------------------
# Original MO-CAPO datasets
# ---------------------------------------------------------------------------

_AGNEWS_CONFIG = DatasetConfig(
    name="SetFit/ag_news",
    alias="agnews",
    revision="ca5ba619eb034211db5f70932b6702efd21e7c73",
    input="text",
    target="label_text",
    initial_prompts=INITIAL_PROMPTS["agnews"],
    task_description=TASK_DESCRIPTIONS["agnews"],
    task_type="Classification",
    metric_func=accuracy_score,
)

_SUBJ_CONFIG = DatasetConfig(
    name="SetFit/subj",
    alias="subj",
    revision="f3c1162e678417f664d76b21864fdb87b0615fcf",
    input="text",
    target="label_text",
    initial_prompts=INITIAL_PROMPTS["subj"],
    task_description=TASK_DESCRIPTIONS["subj"],
    task_type="Classification",
    metric_func=accuracy_score,
)

_GSM8K_CONFIG = DatasetConfig(
    name="openai/gsm8k",
    alias="gsm8k",
    revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
    input="question",
    target=lambda frame: frame["answer"].str.extract(r"#### (.*)")[0],
    names=SplitConfig(train="main", test="main"),
    initial_prompts=INITIAL_PROMPTS["gsm8k"],
    task_description=TASK_DESCRIPTIONS["gsm8k"],
    task_type="Classification",
    metric_func=accuracy_score,
)


def get_input_mbpp(frame):
    """Build MBPP inputs while explicitly preserving the required function name."""

    inputs: list[str] = []
    for _, row in frame.iterrows():
        match = re.search(
            r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
            str(row["code"]),
            re.MULTILINE,
        )
        if match is None:
            raise ValueError(
                "Could not infer expected function name from MBPP reference code"
            )
        inputs.append(f"{row['text']}\n\nExpected function name is '{match.group(1)}'.")
    return inputs


def mbpp_reward_func(generated_code, test_list=None):
    """Execute MBPP tests using the original research-baseline semantics."""

    tests = list(test_list or [])
    if not tests:
        return 0.0
    code = str(generated_code).replace("```python", "").replace("```", "")
    passed = 0

    def timeout_handler(signum, frame):  # pragma: no cover - POSIX-only safety path
        raise TimeoutError("Code execution timed out")

    for test in tests:
        try:
            if hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
            exec(code + "\n" + str(test), {})  # noqa: S102 - preserved baseline behavior
            passed += 1
        except Exception:
            pass
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
    return passed / len(tests)


_MBPP_CONFIG = DatasetConfig(
    name="google-research-datasets/mbpp",
    alias="mbpp",
    revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    input=get_input_mbpp,
    target="code",
    initial_prompts=INITIAL_PROMPTS["mbpp"],
    task_description=TASK_DESCRIPTIONS["mbpp"],
    splits=SplitConfig(train="all", test="all"),
    task_type="Reward",
    metric_func=mbpp_reward_func,
    reward_columns=["test_list"],
)


# ---------------------------------------------------------------------------
# Tri-Fair datasets
# ---------------------------------------------------------------------------

BBQ_SUBSETS = (
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_SES",
    "Race_x_gender",
    "Religion",
    "SES",
    "Sexual_orientation",
)

BIOS_PROFESSIONS = (
    "accountant",
    "architect",
    "attorney",
    "chiropractor",
    "comedian",
    "composer",
    "dentist",
    "dietitian",
    "dj",
    "filmmaker",
    "interior_designer",
    "journalist",
    "model",
    "nurse",
    "painter",
    "paralegal",
    "pastor",
    "personal_trainer",
    "photographer",
    "physician",
    "poet",
    "professor",
    "psychologist",
    "rapper",
    "software_engineer",
    "surgeon",
    "teacher",
    "yoga_teacher",
)

CIVIL_IDENTITIES = (
    "male",
    "female",
    "lgbtq",
    "christian",
    "muslim",
    "other_religions",
    "black",
    "white",
)

_BBQ_CONFIG = DatasetConfig(
    name="heegyu/bbq",
    alias="bbq",
    revision="5d6faae52070aa5eb71b46d1c0723d3ba7930209",
    input="input",
    target="target",
    task_type="Fairness",
    metric_func=None,
    loader="bbq_multi_config",
    subsets=BBQ_SUBSETS,
    initial_prompts=INITIAL_PROMPTS["bbq"],
    task_description=TASK_DESCRIPTIONS["bbq"],
    fairness=FairnessConfig(
        metric_name="bbq_bias_statistical",
        quality_metric="accuracy",
        protected_columns=("category",),
        metadata_columns=(
            "category",
            "question_polarity",
            "context_condition",
            "target_loc",
            "unknown_loc",
            "label_index",
            "template_id",
        ),
        class_names=("a", "b", "c"),
        dev_size=220,
        fs_size=88,
        test_size=500,
        block_size=44,
        min_group_count=4,
        dev_source_split="test",
        fewshot_source_split="test",
        test_source_split="test",
        id_columns=("category", "example_id"),
        group_unit_column="template_id",
        stratify_columns=("category",),
        fairness_kwargs={
            "require_all_contexts": True,
            "official_metadata_cache": "data/external/bbq/additional_metadata.csv",
            "official_metadata_url": (
                "https://raw.githubusercontent.com/nyu-mll/BBQ/"
                "bea11bd97d79217245b5871acd247b9d6eb24598/"
                "analysis_scripts/additional_metadata.csv"
            ),
            "require_official_metadata": True,
            "confidence": 0.95,
            "max_accuracy_ci_width": 0.80,
            "min_disambig_coverage": 0.70,
            "min_disambig_coverage_lower": 0.50,
            "normalization_cost_upper_bound": 100.0,
        },
    ),
)

_BIAS_IN_BIOS_CONFIG = DatasetConfig(
    name="LabHC/bias_in_bios",
    alias="bias_in_bios",
    revision="052f01de644dba841176e0449528b41f27d94a61",
    input="input",
    target="target",
    task_type="Fairness",
    metric_func=None,
    loader="datasets",
    splits=SplitConfig(train="train", test="test"),
    initial_prompts=INITIAL_PROMPTS["bias_in_bios"],
    task_description=TASK_DESCRIPTIONS["bias_in_bios"],
    fairness=FairnessConfig(
        metric_name="bios_tpr_gap_statistical",
        quality_metric="macro_f1",
        protected_columns=("gender",),
        metadata_columns=("gender", "profession_id"),
        class_names=BIOS_PROFESSIONS,
        dev_size=336,
        fs_size=112,
        test_size=500,
        block_size=112,
        min_group_count=4,
        dev_source_split="dev",
        fewshot_source_split="train",
        test_source_split="test",
        id_columns=("_source_index",),
        stratify_columns=("profession_id", "gender"),
        fairness_kwargs={
            "min_valid_professions": 28,
            "confidence": 0.95,
            "max_rate_ci_width": 0.85,
            "smoothing": 0.5,
            "min_macro_recall": 0.20,
            "normalization_cost_upper_bound": 120.0,
        },
    ),
)

_CIVIL_COMMENTS_CONFIG = DatasetConfig(
    name="shlomihod/civil-comments-wilds",
    alias="civil_comments",
    revision="3fbfeca80bad0f3aec37e72fa07eff222b6e752f",
    input="input",
    target="target",
    task_type="Fairness",
    metric_func=None,
    loader="civil_comments_raw_csv",
    initial_prompts=INITIAL_PROMPTS["civil_comments"],
    task_description=TASK_DESCRIPTIONS["civil_comments"],
    fairness=FairnessConfig(
        metric_name="civil_equalized_odds",
        quality_metric="accuracy",
        protected_columns=CIVIL_IDENTITIES,
        metadata_columns=CIVIL_IDENTITIES + ("label_index",),
        class_names=("non_toxic", "toxic"),
        dev_size=288,
        fs_size=96,
        test_size=500,
        block_size=96,
        min_group_count=8,
        dev_source_split="val",
        fewshot_source_split="train",
        test_source_split="test",
        id_columns=("_source_index",),
        stratify_columns=("target",),
        fairness_kwargs={
            "min_valid_identities": 8,
            "confidence": 0.95,
            "max_rate_ci_width": 0.70,
            "smoothing": 0.5,
            "min_class_recall": 0.50,
            "normalization_cost_upper_bound": 100.0,
        },
    ),
)

ALL_DATASETS: Dict[DatasetType, DatasetConfig] = {
    "agnews": _AGNEWS_CONFIG,
    "subj": _SUBJ_CONFIG,
    "gsm8k": _GSM8K_CONFIG,
    "mbpp": _MBPP_CONFIG,
    "bbq": _BBQ_CONFIG,
    "bias_in_bios": _BIAS_IN_BIOS_CONFIG,
    "civil_comments": _CIVIL_COMMENTS_CONFIG,
}

for _dataset_config in ALL_DATASETS.values():
    _dataset_config.validate()
