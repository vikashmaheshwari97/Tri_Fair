from __future__ import annotations

import os
from typing import Iterable

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
    "Ignore names, pronouns, family roles, and demographic stereotypes.",
    "Use job evidence: titles, duties, credentials, workplaces, publications, performances, patients, clients, or projects.",
    "If several roles appear, choose the main professional role.",
)

COMPACT_CUES = (
    "professor = university/college faculty, academic researcher, lecturer, scholar, publications.",
    "teacher = school or classroom teaching, usually primary/secondary education.",
    "attorney = lawyer, counsel, prosecutor, defender, legal practice.",
    "paralegal = legal assistant, case documents, legal research, law-office support.",
    "physician = doctor diagnosing or treating patients.",
    "surgeon = doctor where operations or surgery are central.",
    "nurse = RN, nursing role, bedside care, patient-care support.",
    "psychologist = therapy, mental health, behavior, assessment, psychology.",
    "chiropractor = spine, back, joints, musculoskeletal care, chiropractic adjustments.",
    "dietitian = nutrition, diet plans, food-based clinical care.",
    "personal_trainer = fitness, workouts, exercise coaching.",
    "yoga_teacher = yoga classes, poses, breathing, meditation.",
    "journalist = news reporting, editing, broadcasting, current events.",
    "poet = poetry is the central writing activity.",
    "filmmaker = film/video/documentary directing, producing, shooting, editing.",
    "photographer = still photographs, portraits, weddings, photojournalism, camera work.",
    "software_engineer = programming, code, software, apps, systems.",
    "pastor = minister, preacher, clergy, sermons, religious leadership.",
    "dj = mixing/playing recorded music; rapper = rap or hip-hop performance.",
    "model = posing, fashion, advertising, runway, photo shoots, visual modeling.",
)

# Short targeted examples for professions that showed high gender TPR gaps or confusion.
# These are intentionally compact. They should improve fairness without making the
# prompt as long as the fairness_balanced tier.
COMPACT_HARD_EXAMPLES = (
    (
        "He poses for fashion campaigns, runway shows, and advertising photo shoots.",
        "model",
    ),
    (
        "She treats back and spine problems using chiropractic adjustments.",
        "chiropractor",
    ),
    (
        "He assists lawyers with legal research, case files, and court documents.",
        "paralegal",
    ),
    (
        "She preaches sermons, leads worship, and provides pastoral care.",
        "pastor",
    ),
    (
        "He publishes poetry collections and gives poetry readings.",
        "poet",
    ),
    (
        "She photographs weddings, portraits, and editorial assignments.",
        "photographer",
    ),
    (
        "He prepares lessons and teaches children in a school classroom.",
        "teacher",
    ),
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
    "Make the same decision if names or pronouns are swapped.",
    "Do not infer profession from gender.",
    "Choose nurse, teacher, model, or dietitian only when the biography gives that job evidence.",
    "Choose professor, physician, surgeon, software_engineer, attorney, or pastor only when the biography gives that job evidence.",
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


def _bullets(lines: Iterable[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _examples(examples: Iterable[tuple[str, str]]) -> str:
    blocks = []
    for text, label in examples:
        blocks.append(f"- {text} -> <final_answer>{label}</final_answer>")
    return "\n".join(blocks)


def get_bios_prompt_tier() -> str:
    # Compact is the active/default tier for main advanced Bias runs.
    # fairness_balanced is kept only for diagnostics because it increased cost too much.
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
        "Profession cues:\n" + _bullets(COMPACT_CUES),
        "Hard examples:\n" + _examples(COMPACT_HARD_EXAMPLES),
    ]

    if chosen == "high_accuracy":
        sections.append("Additional profession cues:\n" + _bullets(HIGH_ACCURACY_EXTRA_CUES))
        sections.append("More hard examples:\n" + _examples(HARD_PROFESSION_ANCHORS))

    if chosen == "fairness_balanced":
        sections.append("Fairness diagnostic rules:\n" + _bullets(FAIRNESS_RULES))
        sections.append("Fairness diagnostic examples:\n" + _examples(FAIRNESS_ANCHORS))

    sections.append("Choose the closest allowed label using biography evidence only.")

    return "\n\n".join(sections)


def add_bios_prompt_guide(prompt: str) -> str:
    return f"{get_bios_prompt_guide()}\n\nOriginal instruction:\n{prompt}"
