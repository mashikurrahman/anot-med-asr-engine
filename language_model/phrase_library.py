#!/usr/bin/env python3
"""
Anot Medical ASR — Multilingual Medical Phrase Library
======================================================
Stage 24: Turns the curated, categorized medical vocabulary in
`medical_phrases.json` into language-aware Whisper biasing:
  - a capped, prioritized HOTWORD string (terms + lay patient expressions), and
  - a short INITIAL PROMPT that sets clinical context in the right language.

Design notes:
  - Whisper's prompt/hotword budget is small, so we cap and prioritize rather
    than dumping the whole library.
  - Clinic-specific terms mined from local notes (medical_hotwords.txt) are
    merged FIRST so the clinic's actual drugs/terms take priority.
  - English accents (US/CA/UK/AU) share one list; UK spelling variants are added
    so both forms are recognised. Bengali is best-effort (see JSON note).
"""

import json
import pathlib
from functools import lru_cache
from typing import List, Optional

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PHRASES = _ROOT / "language_model" / "medical_phrases.json"
_MINED_HOTWORDS = _ROOT / "language_model" / "models" / "medical_hotwords.txt"

# English categories, round-robined so every category (incl. lay patient
# expressions) is represented within the hotword budget.
_EN_ORDER = [
    "medications", "symptoms_lay", "symptoms_clinical", "conditions",
    "procedures", "anatomy",
]
# Cap on clinic-mined single-word terms so curated phrases still fit.
_MINED_CAP = 45

_PROMPTS = {
    "en": ("Clinical encounter. Patient and clinician discuss symptoms, "
           "medications, dosages, vital signs, diagnoses, and procedures."),
    "bn": ("চিকিৎসা পরামর্শ। রোগী ও ডাক্তার উপসর্গ, ওষুধ, প্রেসার, ডায়াবেটিস ও "
           "পরীক্ষা নিয়ে কথা বলছেন।"),
}


@lru_cache(maxsize=1)
def _load() -> dict:
    try:
        return json.loads(_PHRASES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _mined_terms() -> List[str]:
    try:
        if _MINED_HOTWORDS.exists():
            return _MINED_HOTWORDS.read_text(encoding="utf-8").split()
    except OSError:
        pass
    return []


def _normalize_lang(language: Optional[str]) -> str:
    """Map a language code / 'auto' to a library key ('en' or 'bn')."""
    if not language or language == "auto":
        return "en"
    lang = language.lower()
    if lang.startswith("bn") or lang.startswith("ben"):
        return "bn"
    return "en"


def build_hotwords(language: Optional[str] = "en", max_terms: int = 110) -> Optional[str]:
    """
    Build a capped, prioritized, de-duplicated hotword string for `language`.
    Clinic-mined terms come first, then curated library terms by category order.
    """
    lib = _load()
    key = _normalize_lang(language)
    seen: set = set()
    ordered: List[str] = []

    def add(term: str):
        t = term.strip()
        tl = t.lower()
        if t and tl not in seen:
            seen.add(tl)
            ordered.append(t)

    section = lib.get(key, {})

    # 1. Clinic-specific terms mined from local notes take priority (capped so
    #    curated phrases still fit the budget).
    if key == "en":
        for t in _mined_terms()[:_MINED_CAP]:
            add(t)
        categories = _EN_ORDER
        # UK spelling variants appended last so both US/UK forms are recognised.
        tail = list(section.get("spelling_variants_uk", {}).values())
    else:  # bn: English code-switch terms first, then Bangla terms/expressions.
        categories = ("english_terms_used", "bangla_terms", "patient_expressions_bn")
        tail = []

    # 2. Round-robin across categories so every category is represented.
    lists = [list(section.get(cat, [])) for cat in categories]
    idx = 0
    while any(idx < len(lst) for lst in lists) and len(ordered) < max_terms:
        for lst in lists:
            if idx < len(lst):
                add(lst[idx])
        idx += 1

    for t in tail:
        add(t)

    ordered = ordered[:max_terms]
    return " ".join(ordered) if ordered else None


def build_initial_prompt(language: Optional[str] = "en") -> str:
    return _PROMPTS.get(_normalize_lang(language), _PROMPTS["en"])


def supported_languages() -> List[str]:
    return [k for k in _load().keys() if not k.startswith("_")]
