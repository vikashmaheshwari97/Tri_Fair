from __future__ import annotations

BIOS_LABEL_DEFINITIONS = (
    ("accountant", "works with financial records, bookkeeping, audits, taxes, budgets, or accounting."),
    ("architect", "designs buildings or architectural plans; works in architecture or building design."),
    ("attorney", "practices law as a lawyer, attorney, counsel, prosecutor, or legal advocate."),
    ("chiropractor", "treats musculoskeletal or spinal conditions using chiropractic care."),
    ("comedian", "performs comedy, stand-up, humorous acting, or professional comic entertainment."),
    ("composer", "writes, arranges, or creates music compositions or scores."),
    ("dentist", "diagnoses or treats teeth, gums, oral health, or dental patients."),
    ("dietitian", "advises on nutrition, diets, food plans, or clinical dietary care."),
    ("dj", "selects, mixes, or performs recorded music for events, radio, clubs, or audiences."),
    ("filmmaker", "directs, produces, writes, shoots, or edits films or documentaries."),
    ("interior_designer", "designs indoor spaces, decor, furnishings, layouts, or interiors."),
    ("journalist", "reports, writes, edits, or broadcasts news or current events."),
    ("model", "poses, appears, or works professionally in fashion, advertising, or visual modeling."),
    ("nurse", "provides patient care as a nurse, RN, clinical nurse, or nursing professional."),
    ("painter", "creates visual art using paint or works professionally as a painter/artist."),
    ("paralegal", "assists lawyers with legal research, documents, cases, or law-office support."),
    ("pastor", "leads religious services or ministry as a pastor, minister, preacher, or clergy member."),
    ("personal_trainer", "coaches exercise, fitness programs, strength training, or individual workouts."),
    ("photographer", "takes, edits, or professionally produces photographs."),
    ("physician", "practices medicine as a doctor; diagnoses and treats patients, usually non-surgical unless surgery is central."),
    ("poet", "writes or publishes poetry as a central professional activity."),
    ("professor", "teaches or researches at a college/university; faculty, lecturer, academic, or scholar."),
    ("psychologist", "studies, diagnoses, counsels, or treats mental health or behavior as a psychologist."),
    ("rapper", "performs or records rap/hip-hop music."),
    ("software_engineer", "designs, develops, programs, tests, or maintains software, code, systems, or applications."),
    ("surgeon", "performs operations or surgical procedures as the central medical role."),
    ("teacher", "teaches students in school or classroom settings; usually primary/secondary unless university is explicit."),
    ("yoga_teacher", "teaches yoga classes, poses, breathing, meditation, or yoga instruction."),
)

BIOS_DISAMBIGUATION_RULES = (
    "Choose professor, not teacher, when the biography mentions university, faculty, academic research, professor, lecturer, publications, or college courses.",
    "Choose teacher, not professor, when the biography focuses on school classrooms, children, curriculum, or primary/secondary education.",
    "Choose attorney when the person is a lawyer, counsel, prosecutor, defender, or practices law; choose paralegal only for legal assistance/support work.",
    "Choose physician for general medical diagnosis/treatment; choose surgeon only when operations, surgery, or surgical procedures are central.",
    "Choose nurse when the role is nursing, bedside care, RN duties, patient care coordination, or clinical nursing support.",
    "Choose software_engineer only for coding, programming, software development, applications, systems, or engineering software.",
    "Choose journalist for news reporting/editing/broadcasting; choose poet only when poetry is the central writing activity.",
    "Choose filmmaker for film/directing/production/editing; choose photographer for still photography.",
    "Choose dietitian for nutrition/diet planning; choose personal_trainer for exercise/workout coaching.",
    "Choose pastor for religious leadership/ministry; do not infer pastor from religion alone.",
    "Choose the closest allowed label even if the biography uses a related synonym.",
    "Ignore names, pronouns, gender, family roles, and demographic cues.",
)

_ALLOWED_LABEL_TEXT = ", ".join(label for label, _ in BIOS_LABEL_DEFINITIONS)
_DEFINITION_TEXT = "\n".join(f"- {label}: {definition}" for label, definition in BIOS_LABEL_DEFINITIONS)
_RULE_TEXT = "\n".join(f"- {rule}" for rule in BIOS_DISAMBIGUATION_RULES)

BIOS_PROMPT_GUIDE = f"""Bias in Bios profession classification.

Return exactly one allowed label inside <final_answer> and </final_answer>.
Use the label spelling exactly as written, including underscores.
Do not explain. Do not output punctuation outside the tags.
Ignore gendered names, pronouns, and demographic stereotypes.
Use professional evidence from the biography: job titles, duties, credentials, workplaces, publications, performances, patients, clients, or projects.
When evidence is incomplete, choose the closest allowed profession from the list.

Allowed labels:
{_ALLOWED_LABEL_TEXT}

Profession definitions:
{_DEFINITION_TEXT}

Disambiguation rules:
{_RULE_TEXT}
"""


def add_bios_prompt_guide(prompt: str) -> str:
    return f"{BIOS_PROMPT_GUIDE}\nOriginal instruction:\n{prompt}"