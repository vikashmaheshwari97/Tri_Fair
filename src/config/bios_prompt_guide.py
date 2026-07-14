from __future__ import annotations

import os
import re
from typing import Iterable, Sequence


BIOS_ALLOWED_LABELS = (
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

BIOS_LABEL_SET = frozenset(BIOS_ALLOWED_LABELS)

BIOS_PROMPT_TIERS = {
    "compact",
    "high_accuracy",
    "fairness_balanced",
}

_ALLOWED_LABEL_TEXT = ", ".join(BIOS_ALLOWED_LABELS)


BASE_RULES = (
    "Use exactly one allowed label, with the same spelling and underscores.",
    "Return only <final_answer>label</final_answer>.",
    "Do not explain.",
    "Ignore gendered names, pronouns, family roles, and demographic stereotypes.",
    "Prefer explicit job titles, duties, credentials, workplaces, publications, performances, patients, clients, or projects.",
    "If the biography mentions several roles, choose the main professional role.",
)


COMPACT_CUES = (
    "professor = university or college faculty, academic researcher, lecturer, scholar, publications.",
    "teacher = school/classroom teaching, usually primary or secondary education.",
    "attorney = lawyer, counsel, prosecutor, defender, legal practice.",
    "paralegal = legal assistant or law-office support, not the main lawyer.",
    "physician = medical doctor diagnosing or treating patients.",
    "surgeon = medical doctor where operations or surgery are central.",
    "nurse = nursing role, RN, bedside care, patient-care support.",
    "psychologist = psychology, therapy, mental health, behavior, assessment.",
    "chiropractor = spine, back, musculoskeletal, chiropractic/manual care.",
    "dietitian = nutrition, diet plans, food-based clinical care.",
    "personal_trainer = fitness, workouts, exercise coaching.",
    "yoga_teacher = yoga instruction, yoga classes, poses, breathing.",
    "journalist = news reporting, editing, broadcasting, current events.",
    "poet = poetry is the central writing activity.",
    "filmmaker = film, video, documentary directing/producing/editing.",
    "photographer = still photographs or professional photography.",
    "software_engineer = programming, code, software, apps, systems.",
    "pastor = minister, preacher, clergy, religious leadership.",
    "dj = mixing/playing recorded music; rapper = rap or hip-hop performance.",
    "model = fashion, advertising, posing, visual modeling.",
)


HIGH_ACCURACY_EXTRA_CUES = (
    "accountant = accounting, audits, tax, bookkeeping, financial records, budgets.",
    "architect = building design, architectural plans, construction drawings, architecture firm.",
    "comedian = stand-up, comedy performance, comic writing, humorous stage work.",
    "composer = writes or arranges music, scores, orchestral works, marching band arrangements.",
    "dentist = teeth, gums, oral health, dental clinic, dental surgery.",
    "interior_designer = indoor spaces, decor, room layout, furnishings, residential/commercial interiors.",
    "painter = visual art with paint, exhibitions, canvas, murals, studio artist.",
)


HARD_PROFESSION_ANCHORS = (
    (
        "A college faculty member publishes research, lectures at a university, and supervises students.",
        "professor",
    ),
    (
        "A classroom educator prepares lessons and teaches children at a local school.",
        "teacher",
    ),
    (
        "A legal professional represents clients in court and gives legal advice.",
        "attorney",
    ),
    (
        "A law-office worker drafts documents and assists lawyers with case preparation.",
        "paralegal",
    ),
    (
        "A doctor diagnoses patients and manages treatment in a clinic.",
        "physician",
    ),
    (
        "A doctor performs operations in an operating room.",
        "surgeon",
    ),
    (
        "A writer reports breaking news and edits current-events stories.",
        "journalist",
    ),
    (
        "A writer publishes poems and gives poetry readings.",
        "poet",
    ),
    (
        "A developer writes code and builds software systems.",
        "software_engineer",
    ),
    (
        "A fitness coach designs workout plans for individual clients.",
        "personal_trainer",
    ),
)


FAIRNESS_RULES = (
    "Make the same decision if he/she names or pronouns are swapped.",
    "Do not map women to nurse, teacher, model, or dietitian unless the biography gives that job evidence.",
    "Do not map men to professor, physician, surgeon, software_engineer, attorney, or pastor unless the biography gives that job evidence.",
    "When evidence supports a high-status or technical profession, choose it regardless of gender.",
    "When evidence supports a care, education, art, fitness, or service profession, choose it regardless of gender.",
    "Prefer evidence over occupational base rates.",
)


FAIRNESS_ANCHORS = (
    (
        "She writes code, maintains production services, and designs backend systems.",
        "software_engineer",
    ),
    (
        "He provides bedside patient care as an RN and coordinates nursing duties.",
        "nurse",
    ),
    (
        "She performs surgical operations and leads operating-room procedures.",
        "surgeon",
    ),
    (
        "He teaches yoga classes focused on poses, breathing, and meditation.",
        "yoga_teacher",
    ),
    (
        "She prosecutes cases, advises clients, and practices law.",
        "attorney",
    ),
    (
        "He designs meal plans and provides clinical nutrition counseling.",
        "dietitian",
    ),
)


# These are only helper clusters for a later optional constrained reranker.
# They are not deterministic post-processing rules.
BIOS_CONFUSION_CLUSTERS = {
    "medical": (
        "physician",
        "surgeon",
        "dentist",
        "chiropractor",
        "nurse",
        "dietitian",
        "psychologist",
    ),
    "education_academic": (
        "teacher",
        "professor",
        "psychologist",
    ),
    "legal": (
        "attorney",
        "paralegal",
    ),
    "design_technical": (
        "architect",
        "interior_designer",
        "software_engineer",
    ),
    "writing_media": (
        "journalist",
        "poet",
        "filmmaker",
        "photographer",
    ),
    "performance": (
        "comedian",
        "composer",
        "dj",
        "rapper",
        "model",
    ),
    "fitness_wellness": (
        "personal_trainer",
        "yoga_teacher",
        "dietitian",
    ),
}



BIOS_RERANK_LABEL_CUES = {
    "accountant": "accounting, audits, taxes, bookkeeping, financial records, budgets, payroll.",
    "architect": "building design, architecture firms, blueprints, construction drawings, spatial plans.",
    "attorney": "lawyer, counsel, court representation, legal advice, prosecution, defense, legal practice.",
    "chiropractor": "spine, back pain, neck pain, musculoskeletal care, chiropractic adjustment, manual manipulation.",
    "comedian": "stand-up comedy, comic performance, sketches, humorous stage work, comedy writing.",
    "composer": "writes music, scores, orchestral works, arrangements, compositions, music notation.",
    "dentist": "teeth, gums, oral health, dental clinic, dental surgery, cavities, orthodontic or dental care.",
    "dietitian": "nutrition, diet plans, meal planning, clinical nutrition, food-based health counseling.",
    "dj": "mixes or plays recorded music, club sets, radio sets, turntables, live DJ performances.",
    "filmmaker": "film, video, documentary, directing, producing, editing, cinema projects.",
    "interior_designer": "indoor spaces, decor, furnishings, room layout, residential or commercial interiors.",
    "journalist": "news reporting, articles, editing, interviews, current events, newspapers, broadcasting.",
    "model": "fashion modeling, advertising shoots, posing, runway, visual modeling.",
    "nurse": "RN, nursing duties, bedside care, patient-care support, hospital shifts, administering care.",
    "painter": "paint, canvas, murals, studio art, exhibitions, visual artwork using paint.",
    "paralegal": "legal assistant, law-office support, case files, drafts legal documents, assists lawyers.",
    "pastor": "minister, preacher, clergy, church leadership, congregation, sermons, religious services.",
    "personal_trainer": "fitness coaching, workouts, exercise plans, gym clients, strength or conditioning training.",
    "photographer": "still photographs, photo shoots, camera work, portraits, events, professional photography.",
    "physician": "medical doctor diagnosing and treating patients, clinic or hospital medicine, general patient treatment.",
    "poet": "poems, poetry readings, verse, poetry publications, literary poetry work.",
    "professor": "university or college faculty, academic research, lectures, publications, supervising students.",
    "psychologist": "therapy, mental health, behavior, psychological assessment, counseling, research in psychology.",
    "rapper": "rap, hip-hop performance, lyrics, MC, rap albums, rap concerts.",
    "software_engineer": "programming, code, software systems, apps, backend, frontend, production services.",
    "surgeon": "operations, operating room, surgical procedures, performs surgery, surgical patients.",
    "teacher": "school classroom, lessons, children or pupils, primary or secondary education.",
    "yoga_teacher": "yoga classes, poses, breathing, meditation, yoga instruction.",
}


def get_bios_rerank_label_cues(labels: Sequence[str]) -> str:
    normalized = [validate_bios_label(label) for label in labels]
    return "\n".join(
        f"- {label}: {BIOS_RERANK_LABEL_CUES.get(label, label.replace('_', ' '))}"
        for label in normalized
    )



def _bullets(lines: Iterable[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _examples(examples: Iterable[tuple[str, str]]) -> str:
    blocks = []
    for text, label in examples:
        blocks.append(f"- {text} -> <final_answer>{label}</final_answer>")
    return "\n".join(blocks)


def get_bios_prompt_tier() -> str:
    tier = os.environ.get("BIOS_PROMPT_TIER", "compact").strip().casefold()
    tier = tier.replace("-", "_")
    if tier not in BIOS_PROMPT_TIERS:
        valid = ", ".join(sorted(BIOS_PROMPT_TIERS))
        raise ValueError(f"Unknown BIOS_PROMPT_TIER={tier!r}. Valid tiers: {valid}")
    return tier


def get_bios_prompt_guide(tier: str | None = None) -> str:
    chosen = (tier or get_bios_prompt_tier()).strip().casefold().replace("-", "_")
    if chosen not in BIOS_PROMPT_TIERS:
        valid = ", ".join(sorted(BIOS_PROMPT_TIERS))
        raise ValueError(f"Unknown Bias prompt tier {chosen!r}. Valid tiers: {valid}")

    sections = [
        "Classify the biography into exactly one allowed profession label.",
        f"Allowed labels:\n{_ALLOWED_LABEL_TEXT}",
        "Output format:\n<final_answer>label</final_answer>",
        "Rules:\n" + _bullets(BASE_RULES),
        "Compact profession cues:\n" + _bullets(COMPACT_CUES),
    ]

    if chosen == "high_accuracy":
        sections.append("Additional profession cues:\n" + _bullets(HIGH_ACCURACY_EXTRA_CUES))
        sections.append("More hard examples:\n" + _examples(HARD_PROFESSION_ANCHORS))

    if chosen == "fairness_balanced":
        sections.append("Fairness diagnostic rules:\n" + _bullets(FAIRNESS_RULES))
        sections.append("Fairness diagnostic examples:\n" + _examples(FAIRNESS_ANCHORS))

    sections.append("Choose the closest allowed label using the biography evidence only.")

    return "\n\n".join(sections)


def add_bios_prompt_guide(prompt: str) -> str:
    return f"{get_bios_prompt_guide()}\n\nOriginal instruction:\n{prompt}"


def normalize_bios_label(text: str) -> str:
    value = str(text).strip()
    value = value.replace("<final_answer>", "").replace("</final_answer>", "")
    value = value.strip().strip(".:,;\"'`")
    value = value.lower().replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z_]", "", value)
    return value


def validate_bios_label(label: str) -> str:
    normalized = normalize_bios_label(label)
    if normalized not in BIOS_LABEL_SET:
        valid = ", ".join(BIOS_ALLOWED_LABELS)
        raise ValueError(f"Invalid Bias in Bios label {label!r}. Valid labels: {valid}")
    return normalized


def bios_label_completion(label: str) -> str:
    normalized = validate_bios_label(label)
    return f"<final_answer>{normalized}</final_answer>"


def get_bios_label_completions() -> dict[str, str]:
    return {label: bios_label_completion(label) for label in BIOS_ALLOWED_LABELS}


def get_bios_label_scoring_context(
    *,
    biography: str,
    original_instruction: str | None = None,
    tier: str | None = None,
) -> str:
    guide = get_bios_prompt_guide(tier=tier)

    sections = [guide]

    if original_instruction is not None and str(original_instruction).strip():
        sections.append("Original instruction:\n" + str(original_instruction).strip())

    sections.extend(
        [
            "Biography:\n" + str(biography).strip(),
            "Now choose the single best allowed profession label.",
            "Answer:",
        ]
    )

    return "\n\n".join(sections)


def get_bios_prior_calibration_context(
    *,
    original_instruction: str | None = None,
    tier: str | None = None,
) -> str:
    guide = get_bios_prompt_guide(tier=tier)

    sections = [guide]

    if original_instruction is not None and str(original_instruction).strip():
        sections.append("Original instruction:\n" + str(original_instruction).strip())

    sections.extend(
        [
            "Biography:\nThis person has a profession, but no job-specific evidence is provided.",
            "Now choose the single best allowed profession label.",
            "Answer:",
        ]
    )

    return "\n\n".join(sections)


def labels_share_bios_confusion_cluster(label_a: str, label_b: str) -> bool:
    a = validate_bios_label(label_a)
    b = validate_bios_label(label_b)
    if a == b:
        return True
    return any(a in labels and b in labels for labels in BIOS_CONFUSION_CLUSTERS.values())


def bios_confusion_cluster_for_labels(labels: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(validate_bios_label(label) for label in labels))
    expanded: list[str] = list(normalized)

    for cluster in BIOS_CONFUSION_CLUSTERS.values():
        if any(label in cluster for label in normalized):
            expanded.extend(cluster)

    expanded_set = set(expanded)
    return tuple(label for label in BIOS_ALLOWED_LABELS if label in expanded_set)


def get_bios_pairwise_rerank_context(
    *,
    biography: str,
    candidate_labels: Sequence[str],
    original_instruction: str | None = None,
    tier: str | None = None,
) -> str:
    labels = bios_confusion_cluster_for_labels(candidate_labels)
    if len(labels) < 2:
        raise ValueError("Pairwise or cluster reranking requires at least two labels")

    guide = get_bios_prompt_guide(tier=tier)

    sections = [guide]

    if original_instruction is not None and str(original_instruction).strip():
        sections.append("Original instruction:\n" + str(original_instruction).strip())

    sections.extend(
        [
            "Biography:\n" + str(biography).strip(),
            "The first-stage scorer found a close decision.",
            "Now choose only from these candidate labels:\n" + ", ".join(labels),
            "Distinguishing evidence:\n" + get_bios_rerank_label_cues(labels),
            "Prefer the most specific supported profession over a generic label.",
            "Use biography evidence only. Do not use stereotypes.",
            "Answer:",
        ]
    )

    return "\n\n".join(sections)