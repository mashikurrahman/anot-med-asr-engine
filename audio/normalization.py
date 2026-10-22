#!/usr/bin/env python3
"""
Anot Medical ASR — Deterministic Medical Normalization Layer
============================================================
Stage 8H: Medical Normalization Layer for Anot Medical ASR.

Standardizes common clinical representations (blood pressures, dosages, units,
frequencies, vitals, abbreviations) WITHOUT changing facts, values, or meaning.

CORE SAFETY PRINCIPLES:
1. NORMALIZE REPRESENTATION ONLY: Never alter clinical numbers or facts.
   "blood pressure one fifty two over ninety two" -> "blood pressure 152/92" (ALLOWED)
   "152/92" -> "120/80" (STRICTLY PROHIBITED)
2. PRESERVE UNCERTAINTY: Ambiguous or isolated numbers without clinical context
   are preserved verbatim. Never guess.
3. PRESERVE STRUCTURE & TRACEABILITY: Preserves speaker tags, timestamps,
   chronological turn order, and raw transcripts.
4. AUDITABILITY: Produces structured normalization event records for each change.
5. 100% LOCAL & DETERMINISTIC: Zero external APIs, zero LLMs, zero cloud services.
"""

import re
import json
import pathlib
from typing import Dict, Any, List, Optional, Tuple, Union

# Word-to-number dictionary for deterministic conversion
NUMBER_WORDS: Dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

# Ordered from longest to shortest to prevent prefix collision (e.g. 'nine' matching before 'ninety')
NUM_WORDS_ORDERED = [
    "thousand", "hundred",
    "seventeen", "fourteen", "nineteen", "thirteen", "eighteen", "sixteen", "fifteen",
    "twelve", "eleven",
    "seventy", "eighty", "ninety", "thirty", "twenty", "fifty", "sixty", "forty",
    "eight", "seven", "three", "nine", "four", "five", "six", "zero", "two", "one", "ten",
    "and", "point"
]

NUM_TOKEN_PATTERN = r"(?:\d+(?:\.\d+)?|\b(?:" + "|".join(NUM_WORDS_ORDERED) + r")\b)"
NUM_PHRASE_PATTERN = rf"(?:{NUM_TOKEN_PATTERN}(?:\s+{NUM_TOKEN_PATTERN})*)"

# Canonical clinical unit mappings
CLINICAL_UNITS: Dict[str, str] = {
    "milligrams": "mg",
    "milligram": "mg",
    "mg": "mg",
    "micrograms": "mcg",
    "microgram": "mcg",
    "mcg": "mcg",
    "ug": "mcg",
    "milliliters": "mL",
    "milliliter": "mL",
    "ml": "mL",
    "cc": "mL",
    "kilograms": "kg",
    "kilogram": "kg",
    "kg": "kg",
    "grams": "g",
    "gram": "g",
    "g": "g",
    "centimeters": "cm",
    "centimeter": "cm",
    "cm": "cm",
    "millimeters": "mm",
    "millimeter": "mm",
    "mm": "mm",
    "percent": "%",
    "beats per minute": "bpm",
    "bpm": "bpm",
}

UNIT_PATTERN = (
    r"(?:milligrams?|micrograms?|milliliters?|kilograms?|grams?|centimeters?|millimeters?|percent|%)"
)

# Medication frequency canonical mappings
FREQUENCY_MAPPINGS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(two|2)\s+times\s+(a\s+day|daily)\b", re.IGNORECASE), "twice daily"),
    (re.compile(r"\btwice\s+a\s+day\b", re.IGNORECASE), "twice daily"),
    (re.compile(r"\b(one|1)\s+time\s+(a\s+day|daily)\b", re.IGNORECASE), "once daily"),
    (re.compile(r"\bonce\s+a\s+day\b", re.IGNORECASE), "once daily"),
    (re.compile(r"\b(three|3)\s+times\s+(a\s+day|daily)\b", re.IGNORECASE), "three times daily"),
    (re.compile(r"\b(four|4)\s+times\s+(a\s+day|daily)\b", re.IGNORECASE), "four times daily"),
    (re.compile(r"\bevery\s+(four|4)\s+hours\b", re.IGNORECASE), "every 4 hours"),
    (re.compile(r"\bevery\s+(six|6)\s+hours\b", re.IGNORECASE), "every 6 hours"),
    (re.compile(r"\bevery\s+(eight|8)\s+hours\b", re.IGNORECASE), "every 8 hours"),
    (re.compile(r"\bevery\s+(twelve|12)\s+hours\b", re.IGNORECASE), "every 12 hours"),
    (re.compile(r"\bbefore\s+bed\b", re.IGNORECASE), "at bedtime"),
]

# Explicit clinical terminology abbreviations (unambiguous only)
ABBREVIATION_MAPPINGS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\belectrocardiograms?\b", re.IGNORECASE), "ECG"),
    (re.compile(r"\bbeats\s+per\s+minute\b", re.IGNORECASE), "bpm"),
]


def parse_spoken_number(phrase: str) -> Optional[Union[int, float]]:
    """
    Deterministically convert spoken number phrases to numeric values.
    Supports standard numbers ('five hundred', 'twenty five'),
    colloquial clinical readings ('one fifty two', 'one thirty eight', 'ninety two'),
    and decimal numbers ('two point five', 'ninety eight point six').
    Returns None if phrase is not a valid spoken number.
    """
    clean = phrase.strip().lower().replace("-", " ")
    words = clean.split()
    if not words:
        return None

    # Check for direct integer/float string
    if len(words) == 1:
        if words[0].isdigit():
            return int(words[0])
        try:
            return float(words[0])
        except ValueError:
            pass

    # Decimal handling
    if "point" in words:
        p_idx = words.index("point")
        int_phrase = " ".join(words[:p_idx])
        dec_words = words[p_idx + 1 :]
        int_val = parse_spoken_number(int_phrase)
        if int_val is not None and dec_words:
            dec_digits = []
            for w in dec_words:
                if w in NUMBER_WORDS and NUMBER_WORDS[w] < 10:
                    dec_digits.append(str(NUMBER_WORDS[w]))
                elif w.isdigit():
                    dec_digits.append(w)
                else:
                    return None
            if dec_digits:
                return float(f"{int_val}.{''.join(dec_digits)}")
        return None

    # Colloquial clinical hundred-expressions: e.g. "one fifty two" -> 152
    if (
        len(words) >= 2
        and words[0] in ("one", "two", "three")
        and words[1] in ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    ):
        hundreds = NUMBER_WORDS[words[0]] * 100
        remainder = parse_spoken_number(" ".join(words[1:]))
        if remainder is not None:
            return hundreds + remainder

    # General standard number accumulation
    total = 0
    current = 0
    i = 0
    n = len(words)

    while i < n:
        w = words[i]
        if w == "and":
            i += 1
            continue

        if w.isdigit():
            current += int(w)
        elif w in NUMBER_WORDS:
            val = NUMBER_WORDS[w]
            current += val
        elif w == "hundred":
            if current == 0:
                current = 1
            current *= 100
        elif w == "thousand":
            if current == 0:
                current = 1
            total += current * 1000
            current = 0
        else:
            return None
        i += 1

    return total + current


# Canonical unit-phrase equivalences for symmetric WER normalization.
# Spoken long-form <-> clinical short-form. Applied to BOTH reference and
# hypothesis so a clinically-identical unit is not scored as an error.
UNIT_PHRASE_EQUIVALENTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:millimeters?|mm)\s+of\s+mercury\b", re.IGNORECASE), "mmhg"),
    (re.compile(r"\bmmhg\b", re.IGNORECASE), "mmhg"),
    (re.compile(r"\b(?:milligrams?|mg)\s+per\s+deciliter\b", re.IGNORECASE), "mg/dl"),
    (re.compile(r"\bmg\s*/\s*dl\b", re.IGNORECASE), "mg/dl"),
    (re.compile(r"\bbeats\s+per\s+minute\b", re.IGNORECASE), "bpm"),
    (re.compile(r"\bbreaths\s+per\s+minute\b", re.IGNORECASE), "breaths/min"),
    (re.compile(r"\bdegrees\s+fahrenheit\b", re.IGNORECASE), "degf"),
    (re.compile(r"\bdegrees\s+celsius\b", re.IGNORECASE), "degc"),
]


def normalize_spoken_numbers(text: str) -> str:
    """
    Convert ALL spoken number phrases to canonical digit form, for symmetric
    WER normalization only (NOT used in the production transcript path).

    Handles multi-word integers ("forty five" -> 45, "one hundred fourteen" ->
    114) and spoken decimals ("ninety eight point six" -> 98.6). It is purely a
    representation change: it can never mask a value error, because two different
    spoken values map to two different numbers (98.6 != 98.5).
    """
    def repl(match: re.Match) -> str:
        phrase = match.group(0)
        tokens = phrase.lower().replace("-", " ").split()
        # Strip pure joiner tokens ('and'/'point') from the edges; require a
        # genuine number token so stray 'and'/'point' are never turned into 0.
        core = [t for t in tokens if t not in ("and", "point")]
        if not core:
            return phrase
        has_number = any(t.isdigit() or t in NUMBER_WORDS or t in ("hundred", "thousand")
                         for t in core)
        if not has_number:
            return phrase
        val = parse_spoken_number(phrase)
        if val is None:
            return phrase
        if val == 0 and "zero" not in tokens:
            return phrase
        return f"{int(val)}" if float(val).is_integer() else f"{val}"

    return re.sub(NUM_PHRASE_PATTERN, repl, text, flags=re.IGNORECASE)


def canonicalize_unit_phrases(text: str) -> str:
    """Map clinically-equivalent unit phrases to a single canonical token."""
    out = text
    for pat, canonical in UNIT_PHRASE_EQUIVALENTS:
        out = pat.sub(canonical, out)
    return out


class MedicalNormalizer:
    """
    Deterministic Medical Normalization engine.
    Transforms raw ASR clinical text into standardized representations
    with comprehensive event auditing.
    """

    def __init__(
        self,
        normalize_vitals: bool = True,
        normalize_dosages: bool = True,
        normalize_frequencies: bool = True,
        normalize_units: bool = True,
        normalize_abbreviations: bool = True,
    ):
        self.normalize_vitals = normalize_vitals
        self.normalize_dosages = normalize_dosages
        self.normalize_frequencies = normalize_frequencies
        self.normalize_units = normalize_units
        self.normalize_abbreviations = normalize_abbreviations

    def normalize_text(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Normalize a single line or paragraph of clinical text.
        Returns: (normalized_text, list_of_normalization_events)
        """
        normalized = text

        # Clean CTC artifacts if present (e.g. </s>, {comma}, {period})
        if "</s>" in normalized or "{" in normalized:
            clean_ctc = re.sub(r"</s>|\{[^}]+\}", "", normalized).strip()
            normalized = clean_ctc
        events: List[Dict[str, Any]] = []

        # 1. Blood Pressure Normalization
        if self.normalize_vitals:
            normalized = self._normalize_blood_pressure(normalized, events)

        # 2. Oxygen Saturation (SpO2) Normalization
        if self.normalize_vitals:
            normalized = self._normalize_oxygen_saturation(normalized, events)

        # 3. Pulse / Heart Rate Normalization
        if self.normalize_vitals:
            normalized = self._normalize_pulse(normalized, events)

        # 4. Medication Dosages & Clinical Units
        if self.normalize_dosages or self.normalize_units:
            normalized = self._normalize_dosages_and_units(normalized, events)

        # 5. Medication Frequencies
        if self.normalize_frequencies:
            normalized = self._normalize_frequencies(normalized, events)

        # 6. Clinical Abbreviations
        if self.normalize_abbreviations:
            normalized = self._normalize_abbreviations(normalized, events)

        return normalized, events

    def normalize_turns(
        self, turns: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Normalize structured conversation turns while strictly preserving
        speaker labels, start/end timestamps, duration, and turn sequence.
        """
        normalized_turns = []
        all_events = []

        for turn in turns:
            raw_text = turn.get("transcript", "")
            norm_text, turn_events = self.normalize_text(raw_text)

            new_turn = dict(turn)
            new_turn["raw_transcript"] = raw_text
            new_turn["normalized_transcript"] = norm_text
            new_turn["transcript"] = norm_text  # Active transcript updated to normalized
            new_turn["normalization_events_count"] = len(turn_events)

            # Add turn-level context to events
            for ev in turn_events:
                ev["turn_id"] = turn.get("turn_id", "")
                ev["speaker"] = turn.get("speaker", "UNKNOWN")
                ev["start_time"] = turn.get("start", 0.0)
                all_events.append(ev)

            normalized_turns.append(new_turn)

        return normalized_turns, all_events

    def _normalize_blood_pressure(self, text: str, events: List[Dict[str, Any]]) -> str:
        """
        Normalize spoken blood pressure expressions.
        Matches:
          'blood pressure 152 over 92' -> 'blood pressure 152/92'
          'blood pressure one fifty two over ninety two' -> 'blood pressure 152/92'
          '138 over 84' -> '138/84'
        """
        pattern = re.compile(
            rf"\b((?:blood\s+pressure|bp)(?:\s+(?:today\s+is|is|of))?\s+)?({NUM_PHRASE_PATTERN})\s+over\s+({NUM_PHRASE_PATTERN})\b",
            re.IGNORECASE,
        )

        def repl(match: re.Match) -> str:
            prefix = match.group(1) or ""
            sys_phrase = match.group(2).strip()
            dia_phrase = match.group(3).strip()

            sys_num = parse_spoken_number(sys_phrase)
            dia_num = parse_spoken_number(dia_phrase)

            # Physiological validation: systolic 60-260, diastolic 30-150
            if (
                sys_num is not None
                and dia_num is not None
                and 60 <= sys_num <= 260
                and 30 <= dia_num <= 150
            ):
                formatted_bp = f"{int(sys_num)}/{int(dia_num)}"
                result = f"{prefix}{formatted_bp}"
                events.append({
                    "original": match.group(0),
                    "normalized": result,
                    "category": "blood_pressure",
                    "confidence": "deterministic",
                    "details": {"systolic": int(sys_num), "diastolic": int(dia_num)},
                })
                return result

            return match.group(0)

        return pattern.sub(repl, text)

    def _normalize_oxygen_saturation(self, text: str, events: List[Dict[str, Any]]) -> str:
        """
        Normalize oxygen saturation expressions.
        'oxygen saturation ninety eight percent' -> 'oxygen saturation 98%'
        'o2 sat 97 percent' -> 'o2 sat 97%'
        """
        pattern = re.compile(
            rf"\b(oxygen\s+saturation|o2\s+sat(?:uration)?|spo2)(?:\s+(?:is|of))?\s+({NUM_PHRASE_PATTERN})\s*(?:percent|%)?",
            re.IGNORECASE,
        )

        def repl(match: re.Match) -> str:
            prefix = match.group(1)
            val_phrase = match.group(2).strip()
            val = parse_spoken_number(val_phrase)

            if val is not None and 50 <= val <= 100:
                val_formatted = f"{int(val)}%" if val == int(val) else f"{val}%"
                result = f"{prefix} {val_formatted}"
                events.append({
                    "original": match.group(0),
                    "normalized": result,
                    "category": "oxygen_saturation",
                    "confidence": "deterministic",
                    "details": {"value": val, "unit": "%"},
                })
                return result
            return match.group(0)

        return pattern.sub(repl, text)

    def _normalize_pulse(self, text: str, events: List[Dict[str, Any]]) -> str:
        """
        Normalize pulse / heart rate readings.
        'pulse seventy four beats per minute' -> 'pulse 74 bpm'
        'heart rate 82 beats per minute' -> 'heart rate 82 bpm'
        """
        pattern = re.compile(
            rf"\b(pulse|heart\s+rate)(?:\s+(?:is|of))?\s+({NUM_PHRASE_PATTERN})\s*(?:beats\s+per\s+minute|bpm)?\b",
            re.IGNORECASE,
        )

        def repl(match: re.Match) -> str:
            prefix = match.group(1)
            val_phrase = match.group(2).strip()
            val = parse_spoken_number(val_phrase)

            if val is not None and 30 <= val <= 250:
                result = f"{prefix} {int(val)} bpm"
                events.append({
                    "original": match.group(0),
                    "normalized": result,
                    "category": "pulse",
                    "confidence": "deterministic",
                    "details": {"pulse_rate": int(val), "unit": "bpm"},
                })
                return result
            return match.group(0)

        return pattern.sub(repl, text)

    def _normalize_dosages_and_units(self, text: str, events: List[Dict[str, Any]]) -> str:
        """
        Normalize medication dosages and clinical units.
        'five hundred milligrams' -> '500 mg'
        '10 milligrams' -> '10 mg'
        'twenty micrograms' -> '20 mcg'
        'five milliliters' -> '5 mL'
        'ninety seven percent' -> '97%'
        """
        pattern = re.compile(
            rf"\b({NUM_PHRASE_PATTERN})\s+({UNIT_PATTERN})\b",
            re.IGNORECASE,
        )

        def repl(match: re.Match) -> str:
            num_phrase = match.group(1).strip()
            unit_str = match.group(2).strip().lower()

            val = parse_spoken_number(num_phrase)
            if val is None:
                return match.group(0)

            canonical_unit = CLINICAL_UNITS.get(unit_str, unit_str)
            num_str = f"{int(val)}" if val == int(val) else f"{val}"

            if canonical_unit == "%":
                result = f"{num_str}%"
            else:
                result = f"{num_str} {canonical_unit}"

            events.append({
                "original": match.group(0),
                "normalized": result,
                "category": "medication_dosage" if canonical_unit in ("mg", "mcg", "g", "mL") else "clinical_unit",
                "confidence": "deterministic",
                "details": {"value": val, "unit": canonical_unit},
            })
            return result

        return pattern.sub(repl, text)

    def _normalize_frequencies(self, text: str, events: List[Dict[str, Any]]) -> str:
        """Normalize medication frequency expressions."""
        normalized = text
        for pat, canonical in FREQUENCY_MAPPINGS:
            def make_repl(target_canonical: str):
                def repl(match: re.Match) -> str:
                    orig = match.group(0)
                    if orig.lower() != target_canonical.lower():
                        events.append({
                            "original": orig,
                            "normalized": target_canonical,
                            "category": "frequency",
                            "confidence": "deterministic",
                        })
                    return target_canonical
                return repl

            normalized = pat.sub(make_repl(canonical), normalized)
        return normalized

    def _normalize_abbreviations(self, text: str, events: List[Dict[str, Any]]) -> str:
        """Normalize explicit unambiguous medical abbreviations."""
        normalized = text
        for pat, canonical in ABBREVIATION_MAPPINGS:
            def make_repl(target_canonical: str):
                def repl(match: re.Match) -> str:
                    orig = match.group(0)
                    events.append({
                        "original": orig,
                        "normalized": target_canonical,
                        "category": "medical_abbreviation",
                        "confidence": "deterministic",
                    })
                    return target_canonical
                return repl

            normalized = pat.sub(make_repl(canonical), normalized)
        return normalized


# Module-level convenience functions
_DEFAULT_NORMALIZER = MedicalNormalizer()


def normalize_text(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Convenience functional API to normalize a text string."""
    return _DEFAULT_NORMALIZER.normalize_text(text)


def normalize_conversation_turns(
    turns: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience functional API to normalize structured conversation turns."""
    return _DEFAULT_NORMALIZER.normalize_turns(turns)
