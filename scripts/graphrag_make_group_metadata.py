#!/usr/bin/env python
"""Create group metadata for Tri-Fair-GR diagnostics.

Supported schemes:
  topic        : question-topic diagnostic groups
  region_proxy : entity/culture/region proxy groups from question + reasoning paths
  west_proxy   : western_proxy / non_western_proxy / mixed_region / unknown_region

These are metadata-generation utilities. They do not claim perfect demographic
labels. For paper-grade claims, document the grouping scheme and report unknowns.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REGION_TERMS: dict[str, list[str]] = {
    "north_america_western": [
        "united states", "u.s.", "usa", "american", "america",
        "canada", "canadian",
    ],
    "western_europe": [
        "england", "english", "britain", "british", "scotland", "scottish",
        "wales", "welsh", "ireland", "irish", "france", "french",
        "germany", "german", "italy", "italian", "spain", "spanish",
        "netherlands", "dutch", "belgium", "belgian", "switzerland",
        "swiss", "austria", "austrian", "portugal", "portuguese",
        "sweden", "swedish", "norway", "norwegian", "denmark", "danish",
        "finland", "finnish", "iceland", "icelandic",
    ],
    "eastern_europe_russia": [
        "russia", "russian", "ukraine", "ukrainian", "poland", "polish",
        "romania", "romanian", "bulgaria", "bulgarian", "serbia",
        "serbian", "croatia", "croatian", "czech", "slovak",
        "hungary", "hungarian", "estonia", "estonian", "latvia",
        "latvian", "lithuania", "lithuanian",
    ],
    "latin_america": [
        "mexico", "mexican", "brazil", "brazilian", "argentina",
        "argentine", "chile", "chilean", "peru", "peruvian",
        "colombia", "colombian", "venezuela", "venezuelan",
        "bolivia", "bolivian", "ecuador", "ecuadorian",
        "guatemala", "guatemalan", "costa rica", "cuban", "cuba",
    ],
    "caribbean": [
        "jamaica", "jamaican", "haiti", "haitian", "dominican",
        "barbados", "barbadian", "trinidad", "tobago", "caribbean",
        "puerto rico", "puerto rican",
    ],
    "africa": [
        "africa", "african", "nigeria", "nigerian", "ghana", "ghanaian",
        "kenya", "kenyan", "ethiopia", "ethiopian", "somalia", "somali",
        "south africa", "south african", "egypt", "egyptian", "morocco",
        "moroccan", "algeria", "algerian", "tunisia", "tunisian",
        "uganda", "ugandan", "tanzania", "tanzanian", "sudan", "sudanese",
    ],
    "middle_east_north_africa": [
        "israel", "israeli", "palestine", "palestinian", "arab", "arabic",
        "saudi", "iran", "iranian", "iraq", "iraqi", "syria", "syrian",
        "lebanon", "lebanese", "turkey", "turkish", "qatar", "emirati",
        "jordan", "jordanian", "yemen", "yemeni",
    ],
    "south_asia": [
        "india", "indian", "pakistan", "pakistani", "bangladesh",
        "bangladeshi", "sri lanka", "sri lankan", "nepal", "nepali",
        "hindi", "urdu", "bengali", "tamil", "punjabi",
    ],
    "east_southeast_asia": [
        "china", "chinese", "japan", "japanese", "korea", "korean",
        "vietnam", "vietnamese", "thailand", "thai", "indonesia",
        "indonesian", "malaysia", "malaysian", "philippines", "filipino",
        "singapore", "singaporean", "taiwan", "taiwanese", "mongolia",
        "mongolian",
    ],
    "oceania_western": [
        "australia", "australian", "new zealand", "zealand",
    ],
}


WESTERN_REGIONS = {
    "north_america_western",
    "western_europe",
    "oceania_western",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--scheme", choices=["topic", "region_proxy", "west_proxy"], required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def extract_reasoning_paths(prompt_input: str) -> list[str]:
    text = str(prompt_input)
    marker = "Reasoning Paths:"
    if marker not in text:
        return []

    after = text.split(marker, 1)[1]
    for stop in ["\n\nQuestion:", "\nQuestion:"]:
        if stop in after:
            after = after.split(stop, 1)[0]
            break

    return [line.strip(" -\t") for line in after.splitlines() if line.strip()]


def looks_like_relation(part: str) -> bool:
    value = part.strip()
    return "." in value and " " not in value


def extract_surface_entities(prompt_input: str) -> list[str]:
    entities: list[str] = []
    for path in extract_reasoning_paths(prompt_input):
        parts = [p.strip() for p in path.split("->") if p.strip()]
        for part in parts:
            if not looks_like_relation(part):
                entities.append(part)
    return entities


def topic_group(question: str) -> str:
    q = question.lower()
    if re.search(r"\blanguage|speak|spoken\b", q):
        return "language"
    if re.search(r"\breligion|religious\b", q):
        return "religion"
    if re.search(r"\bnationality|country|countries|where|located|born|from\b", q):
        return "location_nationality"
    if re.search(r"\bteam|sport|sports|series|championship|mascot|game\b", q):
        return "sports"
    if re.search(r"\bwhen|year|date|time\b", q):
        return "time"
    if re.search(r"\bwho|person|people|leader|artist|actor|author|president\b", q):
        return "person"
    return "other"


def matched_regions(question: str, prompt_input: str) -> set[str]:
    entities = extract_surface_entities(prompt_input)
    blob = " ".join([question, prompt_input, *entities]).lower()

    found: set[str] = set()
    for region, terms in REGION_TERMS.items():
        for term in terms:
            if re.search(rf"\b{re.escape(term.lower())}\b", blob):
                found.add(region)
                break
    return found


def region_proxy_group(question: str, prompt_input: str) -> str:
    regions = matched_regions(question, prompt_input)
    if not regions:
        return "unknown_region"
    if len(regions) == 1:
        return next(iter(regions))
    return "multiple_regions"


def west_proxy_group(question: str, prompt_input: str) -> str:
    regions = matched_regions(question, prompt_input)
    if not regions:
        return "unknown_region"

    western = regions & WESTERN_REGIONS
    non_western = regions - WESTERN_REGIONS

    if western and non_western:
        return "mixed_region"
    if western:
        return "western_proxy"
    return "non_western_proxy"


def main() -> None:
    args = parse_args()
    pred_path = Path(args.predictions)

    rows = []
    with pred_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = str(row.get("id", ""))
            question = str(row.get("question", ""))
            prompt_input = str(row.get("input", ""))

            if args.scheme == "topic":
                group = topic_group(question)
            elif args.scheme == "region_proxy":
                group = region_proxy_group(question, prompt_input)
            elif args.scheme == "west_proxy":
                group = west_proxy_group(question, prompt_input)
            else:
                raise ValueError(args.scheme)

            rows.append(
                {
                    "id": qid,
                    "question": question,
                    "protected_group": group,
                    "group_scheme": args.scheme,
                    "surface_entities": " | ".join(extract_surface_entities(prompt_input)[:12]),
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).drop_duplicates("id")
    df.to_csv(out, index=False)

    print(out, "rows=", len(df))
    print(df["protected_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
