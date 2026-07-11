from __future__ import annotations

# Compact Bias in Bios guide.
#
# Goal:
# - Keep the full allowed-label list.
# - Keep only the most useful profession cues and disambiguation rules.
# - Reduce prompt cost compared with the long v2 guide.

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

_ALLOWED_LABEL_TEXT = ", ".join(BIOS_ALLOWED_LABELS)

BIOS_PROMPT_GUIDE = f"""Classify the biography into exactly one allowed profession label.

Allowed labels:
{_ALLOWED_LABEL_TEXT}

Output format:
<final_answer>label</final_answer>

Rules:
- Use exactly one allowed label, with the same spelling and underscores.
- Do not explain.
- Ignore gendered names, pronouns, family roles, and demographic stereotypes.
- Prefer explicit job titles, duties, credentials, workplaces, publications, performances, patients, clients, or projects.
- If the biography mentions several roles, choose the main professional role.

Compact profession cues:
- professor = university or college faculty, academic researcher, lecturer, scholar, publications.
- teacher = school/classroom teaching, usually primary or secondary education.
- attorney = lawyer, counsel, prosecutor, defender, legal practice.
- paralegal = legal assistant or law-office support, not the main lawyer.
- physician = medical doctor diagnosing or treating patients.
- surgeon = medical doctor where operations or surgery are central.
- nurse = nursing role, RN, bedside care, patient-care support.
- psychologist = psychology, therapy, mental health, behavior, assessment.
- chiropractor = spine, back, musculoskeletal, chiropractic/manual care.
- dietitian = nutrition, diet plans, food-based clinical care.
- personal_trainer = fitness, workouts, exercise coaching.
- yoga_teacher = yoga instruction, yoga classes, poses, breathing.
- journalist = news reporting, editing, broadcasting, current events.
- poet = poetry is the central writing activity.
- filmmaker = film, video, documentary directing/producing/editing.
- photographer = still photographs or professional photography.
- software_engineer = programming, code, software, apps, systems.
- pastor = minister, preacher, clergy, religious leadership.
- dj = mixing/playing recorded music; rapper = rap or hip-hop performance.
- model = fashion, advertising, posing, visual modeling.

Choose the closest allowed label using the biography evidence only.
"""


def add_bios_prompt_guide(prompt: str) -> str:
    return f"{BIOS_PROMPT_GUIDE}\nOriginal instruction:\n{prompt}"
