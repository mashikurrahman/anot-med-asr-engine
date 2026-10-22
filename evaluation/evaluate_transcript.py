#!/usr/bin/env python3
"""
Anot Medical ASR Evaluation Engine
===================================
A reusable evaluation framework for benchmarking medical ASR output against
ground-truth clinical reference transcripts.

Features:
- General ASR metrics (Word Error Rate, Substitutions, Deletions, Insertions)
- Configurable medical entity vocabulary matching
- Critical numeric entity extraction & comparison (age, vitals, dosages, durations)
- Medication regimen extraction (drug name, dosage, normalized unit equivalence, frequency)
- False-start / stutter acoustic disfluency analysis
- Potentially clinically significant error classification
- Separate metric reporting (NO single overall accuracy score)

IMPORTANT DISCLAIMER:
This evaluation engine is for preliminary development and research benchmarking only.
It is NOT clinically comprehensive, certified, or intended for autonomous clinical decision-making.
"""

import os
import re
import sys
import json
import argparse
import pathlib
from typing import Dict, List, Tuple, Any, Optional

# Ensure the project root is importable when this module is run as a script
# (e.g. `python evaluation/evaluate_transcript.py`), so sibling packages such as
# `audio` (the deterministic clinical normalizer) resolve correctly.
_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Word-to-number mapping for normalizing numerals
WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100"
}


def load_medical_entities(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the medical entities configuration from JSON."""
    if config_path is None or not os.path.exists(config_path):
        default_path = pathlib.Path(__file__).parent / "medical_entities.json"
        config_path = str(default_path)
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_unit(unit_str: str, dosage_units_cfg: Dict[str, List[str]]) -> str:
    """Normalize a dosage unit string to its canonical equivalent (e.g., 'milligrams' -> 'mg')."""
    if not unit_str:
        return ""
    u_clean = unit_str.strip().lower()
    for canonical, aliases in dosage_units_cfg.items():
        if u_clean == canonical or u_clean in aliases:
            return canonical
    return u_clean


def normalize_frequency(freq_str: str, freq_cfg: Dict[str, List[str]]) -> str:
    """Normalize a frequency string to canonical form."""
    if not freq_str:
        return ""
    f_clean = freq_str.strip().lower()
    for canonical, aliases in freq_cfg.items():
        if f_clean == canonical or f_clean in aliases:
            return canonical
    return f_clean


def normalize_for_wer(text: str, normalize_numbers: bool = True) -> str:
    """
    Normalize text strictly for WER computation:
    - lowercase
    - strip CTC artifacts (e.g. </s>, {comma})
    - remove punctuation (converting hyphens/slashes to spaces)
    - optionally normalize common number words (e.g. 'two' -> '2')
    - normalize whitespace
    """
    t = re.sub(r"</s>|\{[^}]+\}", " ", text)
    t = t.lower()
    # Replace slashes and hyphens with spaces to treat compound words cleanly
    t = re.sub(r"[\/\-_]", " ", t)
    # Remove remaining punctuation
    t = re.sub(r"[^\w\s]", " ", t)
    words = t.split()
    
    if normalize_numbers:
        norm_words = []
        for w in words:
            if w in WORD_TO_NUM:
                norm_words.append(WORD_TO_NUM[w])
            else:
                norm_words.append(w)
        words = norm_words
        
    return " ".join(words)


def compute_wer_and_alignment(ref_text: str, hyp_text: str, normalize_numbers: bool = True) -> Dict[str, Any]:
    """Compute standard Levenshtein-distance WER, alignment, and S/D/I counts."""
    ref_norm = normalize_for_wer(ref_text, normalize_numbers=normalize_numbers)
    hyp_norm = normalize_for_wer(hyp_text, normalize_numbers=normalize_numbers)
    
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    
    r_len = len(ref_words)
    h_len = len(hyp_words)
    
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j
        
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j])
                
    # Backtrack alignment
    i, j = r_len, h_len
    alignment = []
    substitutions = 0
    deletions = 0
    insertions = 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            alignment.append(("OK", ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append(("SUB", ref_words[i - 1], hyp_words[j - 1]))
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            alignment.append(("INS", "<epsilon>", hyp_words[j - 1]))
            insertions += 1
            j -= 1
        else:
            alignment.append(("DEL", ref_words[i - 1], "<epsilon>"))
            deletions += 1
            i -= 1
            
    alignment.reverse()
    total_errors = substitutions + deletions + insertions
    wer = total_errors / r_len if r_len > 0 else 0.0
    
    return {
        "reference_word_count": r_len,
        "hypothesis_word_count": h_len,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_errors": total_errors,
        "wer": wer,
        "alignment": alignment,
        "reference_normalized": ref_norm,
        "hypothesis_normalized": hyp_norm
    }


def strip_abandoned_false_starts(text: str) -> str:
    """
    Remove abandoned acoustic false-starts from a hypothesis, preserving the
    corrected target word.

    A clinician who says "isupro- lisinopril" produced one clinical token
    (lisinopril); the abandoned fragment "isupro-" is a verbatim disfluency, not
    a transcription error. This removes only the trailing-hyphen fragment and
    keeps the corrected word, so downstream scoring is not penalised for the
    speaker's own self-correction.

    This NEVER removes or alters numbers, units, or complete words. Only the
    `prefix-` fragment immediately preceding a longer word is dropped.
    """
    out = text
    for fs in detect_false_starts(text):
        prefix = fs["prefix"][:-1]  # fragment without the trailing hyphen
        out = re.sub(r"\b" + re.escape(prefix) + r"-\s+", "", out, count=1)
    return re.sub(r"\s{2,}", " ", out).strip()


def compute_clinical_normalized_wer(
    ref_text: str,
    hyp_text: str,
    strip_disfluencies: bool = True,
) -> Dict[str, Any]:
    """
    Compute a *clinically-normalized* Word Error Rate.

    This is reported ALONGSIDE (never instead of) the raw WER. It applies the
    same deterministic clinical normalization to BOTH the reference and the
    hypothesis before scoring, so that clinically-identical surface forms are
    treated as matches rather than errors:

      - dosage units:   "500 milligrams"  == "500 mg"
      - blood pressure: "138 over 84"      == "138/84"
      - oxygen sat:     "97 percent"       == "97%"
      - number words:   "type two"         == "type 2"

    Facts are never changed: the clinical value must be identical on both sides
    for it to count as a match. Optionally removes the speaker's own abandoned
    false-starts from the hypothesis (verbatim disfluencies, not errors).

    Returns the same shape as `compute_wer_and_alignment` plus a
    `methodology` field documenting the transforms applied.
    """
    # Lazy import keeps the deterministic clinical normalizer optional and
    # avoids any import cost for callers that only need raw WER.
    from audio.normalization import (
        MedicalNormalizer,
        normalize_spoken_numbers,
        canonicalize_unit_phrases,
    )

    normalizer = MedicalNormalizer()

    def _clinical_canon(text: str) -> str:
        # 1. Contextual vitals/dosage/frequency normalization.
        out, _ = normalizer.normalize_text(text)
        # 2. Canonicalize clinically-equivalent unit phrases (mmHg, mg/dL, ...).
        out = canonicalize_unit_phrases(out)
        # 3. Convert any remaining spoken numbers/decimals to digit form.
        out = normalize_spoken_numbers(out)
        return out

    hyp_source = strip_abandoned_false_starts(hyp_text) if strip_disfluencies else hyp_text

    ref_norm = _clinical_canon(ref_text)
    hyp_norm = _clinical_canon(hyp_source)

    result = compute_wer_and_alignment(ref_norm, hyp_norm, normalize_numbers=True)
    result["methodology"] = {
        "symmetric_clinical_normalization": True,
        "spoken_number_normalization": True,
        "unit_phrase_canonicalization": True,
        "disfluency_stripping": strip_disfluencies,
        "normalizer": "audio.normalization.MedicalNormalizer (deterministic, fact-preserving)",
    }
    return result


def detect_false_starts(text: str) -> List[Dict[str, Any]]:
    """
    Detect acoustic false-starts, partial stutter repetitions, and trailing hyphens.
    Examples:
      - 'isupro- lisinopril' -> false-start prefix 'isupro-'
      - 'ator- atorvastatin' -> stutter prefix 'ator-'
    """
    false_starts = []
    pattern = r'\b([a-zA-Z]{2,})-\s+([a-zA-Z]{3,})\b'
    for match in re.finditer(pattern, text):
        prefix = match.group(1).lower()
        following = match.group(2).lower()
        full_match = match.group(0)
        
        is_stutter = following.startswith(prefix)
        category = "stutter_repetition" if is_stutter else "abandoned_false_start"
        
        false_starts.append({
            "matched_text": full_match,
            "prefix": prefix + "-",
            "target_word": following,
            "category": category,
            "span": match.span()
        })
        
    return false_starts


def parse_numeric_entities(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Detect and extract critical numeric entities:
    - patient_age
    - blood_pressure
    - pulse
    - oxygen_saturation
    - temperature
    - weight
    - height
    - clinical_duration
    """
    results: Dict[str, List[Dict[str, Any]]] = {
        "patient_age": [],
        "blood_pressure": [],
        "pulse": [],
        "oxygen_saturation": [],
        "temperature": [],
        "weight": [],
        "height": [],
        "clinical_duration": []
    }
    
    t_lower = text.lower()
    
    # 1. Patient Age (e.g., '58-year-old', '50-year-old', '58 year old')
    age_pattern = r'\b(?:a\s+)?(\d+|[a-z]+)[\s-]*(?:year[\s-]*old|yo|y\.?o\.?|years?\s+of\s+age)\b'
    for match in re.finditer(age_pattern, t_lower):
        val_str = match.group(1)
        val_num = WORD_TO_NUM.get(val_str, val_str)
        results["patient_age"].append({
            "raw_text": match.group(0),
            "value": str(val_num),
            "unit": "years"
        })
        
    # 2. Blood Pressure (e.g., '138 over 84', '138/84')
    bp_pattern = r'\b(\d{2,3})\s*(?:\/|over)\s*(\d{2,3})\b'
    for match in re.finditer(bp_pattern, t_lower):
        systolic = match.group(1)
        diastolic = match.group(2)
        results["blood_pressure"].append({
            "raw_text": match.group(0),
            "value": f"{systolic}/{diastolic}",
            "systolic": int(systolic),
            "diastolic": int(diastolic)
        })
        
    # 3. Pulse / Heart Rate (e.g., 'pulse is 78 beats per minute', '78 bpm')
    pulse_pattern = r'\b(?:pulse(?:\s+is)?\s+|heart\s+rate(?:\s+is)?\s+)?(\d{2,3})\s*(?:beats?\s+per\s+minute|bpm)\b'
    for match in re.finditer(pulse_pattern, t_lower):
        val = match.group(1)
        results["pulse"].append({
            "raw_text": match.group(0),
            "value": str(val),
            "unit": "bpm"
        })
        
    # 4. Oxygen Saturation (e.g., 'oxygen saturation is 97 percent', '97%')
    o2_pattern = r'\b(?:oxygen\s+saturation(?:\s+is)?\s+|o2\s+sat(?:\s+is)?\s+)?(\d{2,3})\s*(?:%|\bpercent\b)'
    for match in re.finditer(o2_pattern, t_lower):
        val = match.group(1)
        results["oxygen_saturation"].append({
            "raw_text": match.group(0),
            "value": str(val),
            "unit": "%"
        })
        
    # 5. Temperature (e.g., '98.6 degrees', '37.0 c', '101 f')
    temp_pattern = r'\b(\d{2,3}(?:\.\d+)?)\s*(?:degrees?(?:\s*(?:fahrenheit|celsius|f|c))?|°[fc]|deg\s*[fc])\b'
    for match in re.finditer(temp_pattern, t_lower):
        results["temperature"].append({
            "raw_text": match.group(0),
            "value": match.group(1),
            "unit": "deg"
        })
        
    # 6. Weight (e.g., '70 kg', '154 lbs', '70 kilograms')
    weight_pattern = r'\b(\d+(?:\.\d+)?)\s*(?:kg|kilograms?|lbs?|pounds?)\b'
    for match in re.finditer(weight_pattern, t_lower):
        results["weight"].append({
            "raw_text": match.group(0),
            "value": match.group(1),
            "unit": "weight_unit"
        })

    # 7. Height (e.g., '5 feet 10 inches', '175 cm')
    height_pattern = r'\b(\d+)\s*(?:feet|ft)\s*(\d+)?(?:\s*(?:inches|in))?\b|\b(\d+)\s*cm\b'
    for match in re.finditer(height_pattern, t_lower):
        results["height"].append({
            "raw_text": match.group(0),
            "value": match.group(0)
        })

    # 8. Clinical Duration (e.g., 'three-week', 'two weeks', but not '58-year-old' or 'today')
    dur_pattern = r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|several|few)[\s-]+(weeks?|months?|days?|hours?)(?![\s-]*old)\b'
    for match in re.finditer(dur_pattern, t_lower):
        num_raw = match.group(1)
        num_val = WORD_TO_NUM.get(num_raw, num_raw)
        unit = match.group(2)
        results["clinical_duration"].append({
            "raw_text": match.group(0),
            "value": str(num_val),
            "unit": unit
        })
        
    return results


def compare_numeric_entities(ref_numerics: Dict[str, List[Dict[str, Any]]],
                             hyp_numerics: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Compare numeric entities between reference and hypothesis.
    Separate identical values, equivalent notations, and CRITICAL NUMERIC MISMATCHES.
    """
    comparison = {
        "matches": [],
        "equivalent_notations": [],
        "critical_mismatches": [],
        "reference_only": [],
        "hypothesis_only": []
    }
    
    all_categories = sorted(list(set(ref_numerics.keys()) | set(hyp_numerics.keys())))
    
    for cat in all_categories:
        ref_items = ref_numerics.get(cat, [])
        hyp_items = hyp_numerics.get(cat, [])
        
        min_len = min(len(ref_items), len(hyp_items))
        for idx in range(min_len):
            r = ref_items[idx]
            h = hyp_items[idx]
            
            r_val = r["value"]
            h_val = h["value"]
            
            if r_val == h_val:
                # Check if representation differed (e.g. '138 over 84' vs '138/84', '97 percent' vs '97%')
                if r["raw_text"].strip() != h["raw_text"].strip():
                    comparison["equivalent_notations"].append({
                        "category": cat,
                        "reference_raw": r["raw_text"],
                        "hypothesis_raw": h["raw_text"],
                        "normalized_value": r_val
                    })
                else:
                    comparison["matches"].append({
                        "category": cat,
                        "value": r_val,
                        "raw_text": r["raw_text"]
                    })
            else:
                # Value differs -> CRITICAL NUMERIC MISMATCH!
                comparison["critical_mismatches"].append({
                    "category": cat,
                    "reference": r_val,
                    "hypothesis": h_val,
                    "reference_raw": r["raw_text"],
                    "hypothesis_raw": h["raw_text"]
                })
                
        if len(ref_items) > min_len:
            for r in ref_items[min_len:]:
                comparison["reference_only"].append({"category": cat, "item": r})
        if len(hyp_items) > min_len:
            for h in hyp_items[min_len:]:
                comparison["hypothesis_only"].append({"category": cat, "item": h})
                
    return comparison


def extract_medications(text: str, entities_cfg: Dict[str, Any], false_starts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract structured medication mentions:
    - Drug name
    - Dosage amount
    - Dosage unit (and normalized equivalent form)
    - Frequency
    - Preceding false starts
    """
    medications_list = entities_cfg.get("medications", [])
    dosage_units_cfg = entities_cfg.get("dosage_units", {})
    freq_cfg = entities_cfg.get("frequency_expressions", {})
    
    detected = []
    t_lower = text.lower()
    
    all_units = []
    for canonical, aliases in dosage_units_cfg.items():
        all_units.append(canonical)
        all_units.extend(aliases)
    all_units = sorted(list(set(all_units)), key=len, reverse=True)
    units_regex = r'(?:' + '|'.join(re.escape(u) for u in all_units) + r')'
    
    all_freqs = []
    for canonical, aliases in freq_cfg.items():
        all_freqs.append(canonical)
        all_freqs.extend(aliases)
    all_freqs = sorted(list(set(all_freqs)), key=len, reverse=True)
    freq_regex = r'(?:' + '|'.join(re.escape(f) for f in all_freqs) + r')'
    
    for med in medications_list:
        med_pattern = r'\b' + re.escape(med) + r'\b'
        for match in re.finditer(med_pattern, t_lower):
            start_pos = match.start()
            end_pos = match.end()
            
            preceding_false_start = None
            for fs in false_starts:
                if fs["target_word"] == med:
                    preceding_false_start = fs["prefix"]
                    
            window = t_lower[end_pos:end_pos + 70]
            
            # Extract dosage: number + unit (supports optional leading prepositions like 'to' or 'at')
            dose_val = None
            dose_unit_raw = None
            norm_unit = None
            dose_pattern = r'^\s*(?:to\s+|at\s+|:\s*)?(\d+(?:\.\d+)?)\s*(' + units_regex + r')\b'
            dose_match = re.search(dose_pattern, window)
            if dose_match:
                dose_val = dose_match.group(1)
                dose_unit_raw = dose_match.group(2)
                norm_unit = normalize_unit(dose_unit_raw, dosage_units_cfg)
                
            # Extract frequency
            freq_raw = None
            norm_freq = None
            freq_match = re.search(r'\b(' + freq_regex + r')\b', window)
            if freq_match:
                freq_raw = freq_match.group(1)
                norm_freq = normalize_frequency(freq_raw, freq_cfg)
                
            detected.append({
                "drug_name": med,
                "dosage_value": dose_val,
                "dosage_unit_raw": dose_unit_raw,
                "dosage_unit_normalized": norm_unit,
                "frequency_raw": freq_raw,
                "frequency_normalized": norm_freq,
                "preceding_false_start": preceding_false_start,
                "span": (start_pos, end_pos)
            })
            
    # Sort detected by position in text
    detected.sort(key=lambda x: x["span"][0])
    return detected


def compare_medications(ref_meds: List[Dict[str, Any]],
                        hyp_meds: List[Dict[str, Any]],
                        dosage_units_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare medications between reference and hypothesis:
    - Drug name matching
    - Dosage value matching vs critical dosage discrepancy
    - Dosage unit equivalence (e.g. milligrams vs mg -> EQUIVALENT, NOT an error)
    - Frequency matching
    - False-start association
    """
    results = {
        "matched_regimens": [],
        "equivalent_dosage_units": [],
        "dosage_mismatches": [],
        "frequency_mismatches": [],
        "omitted_medications": [],
        "added_medications": []
    }
    
    hyp_matched_indices = set()
    
    for r in ref_meds:
        match_idx = None
        for idx, h in enumerate(hyp_meds):
            if idx not in hyp_matched_indices and h["drug_name"] == r["drug_name"]:
                match_idx = idx
                break
                
        if match_idx is None:
            results["omitted_medications"].append(r)
            continue
            
        hyp_matched_indices.add(match_idx)
        h = hyp_meds[match_idx]
        
        # Compare dosage
        if r["dosage_value"] != h["dosage_value"]:
            results["dosage_mismatches"].append({
                "drug_name": r["drug_name"],
                "reference_dosage": f"{r['dosage_value']} {r['dosage_unit_raw']}",
                "hypothesis_dosage": f"{h['dosage_value']} {h['dosage_unit_raw']}"
            })
        elif r["dosage_unit_normalized"] == h["dosage_unit_normalized"]:
            if r["dosage_unit_raw"] != h["dosage_unit_raw"]:
                results["equivalent_dosage_units"].append({
                    "drug_name": r["drug_name"],
                    "reference": f"{r['dosage_value']} {r['dosage_unit_raw']}",
                    "hypothesis": f"{h['dosage_value']} {h['dosage_unit_raw']}",
                    "normalized": f"{r['dosage_value']} {r['dosage_unit_normalized']}",
                    "note": f"Equivalent dosage unit ('{r['dosage_unit_raw']}' == '{h['dosage_unit_raw']}'). NOT a dosage error."
                })
                
        # Compare frequency
        if r["frequency_normalized"] != h["frequency_normalized"]:
            results["frequency_mismatches"].append({
                "drug_name": r["drug_name"],
                "reference_frequency": r["frequency_raw"],
                "hypothesis_frequency": h["frequency_raw"]
            })
            
        results["matched_regimens"].append({
            "drug_name": r["drug_name"],
            "reference": r,
            "hypothesis": h,
            "false_start_detected": h.get("preceding_false_start")
        })
        
    for idx, h in enumerate(hyp_meds):
        if idx not in hyp_matched_indices:
            results["added_medications"].append(h)
            
    return results


def find_entity_spans(text_norm: str, entity_list: List[str]) -> List[Tuple[str, int, int]]:
    """
    Find entity matches in normalized text using longest-match first to prevent
    redundant substring sub-matches (e.g. 'persistent cough' superseding 'cough').
    """
    # Sort entities longest first
    sorted_entities = sorted(entity_list, key=lambda x: len(normalize_for_wer(x)), reverse=True)
    matched_spans: List[Tuple[str, int, int]] = []
    
    for term in sorted_entities:
        norm_term = normalize_for_wer(term, normalize_numbers=True)
        if not norm_term:
            continue
        pattern = r'\b' + re.escape(norm_term) + r'\b'
        for match in re.finditer(pattern, text_norm):
            s, e = match.span()
            # Check overlap with existing spans
            overlaps = any(not (e <= existing_s or s >= existing_e) for _, existing_s, existing_e in matched_spans)
            if not overlaps:
                matched_spans.append((term, s, e))
                
    return matched_spans


def extract_and_compare_medical_terms(ref_text: str, hyp_text: str,
                                      entities_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and compare non-medication medical entities across all configured categories:
    - diagnoses
    - symptoms
    - procedures
    - anatomy
    - laboratory_tests
    - vital_signs
    """
    categories = [
        "diagnoses", "symptoms", "procedures",
        "anatomy", "laboratory_tests", "vital_signs"
    ]
    
    ref_norm = normalize_for_wer(ref_text, normalize_numbers=True)
    hyp_norm = normalize_for_wer(hyp_text, normalize_numbers=True)
    
    term_results = {}
    
    for cat in categories:
        entity_list = entities_cfg.get(cat, [])
        ref_matches = {term for term, _, _ in find_entity_spans(ref_norm, entity_list)}
        hyp_matches = {term for term, _, _ in find_entity_spans(hyp_norm, entity_list)}
        
        matches = sorted(list(ref_matches & hyp_matches))
        omissions = sorted(list(ref_matches - hyp_matches))
        additions = sorted(list(hyp_matches - ref_matches))
        
        term_results[cat] = {
            "matches": matches,
            "omissions": omissions,
            "additions": additions
        }
        
    return term_results


def identify_clinical_significance(numeric_comp: Dict[str, Any],
                                   med_comp: Dict[str, Any],
                                   terms_comp: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Synthesize all discrepancies into potentially clinically significant findings:
    - Demographic / Patient Age errors
    - Dosage errors (number differences)
    - Medication omissions or additions
    - Critical vital sign value errors
    - Missing major diagnoses or procedures
    """
    significant_errors = []
    
    # 1. Critical numeric mismatches (e.g. Patient Age 58 -> 50)
    for mm in numeric_comp.get("critical_mismatches", []):
        cat = mm["category"]
        if cat == "patient_age":
            significant_errors.append({
                "severity": "HIGH",
                "category": "patient_demographics",
                "title": "Patient Age Discrepancy",
                "description": f"Age transcribed as {mm['hypothesis']} instead of {mm['reference']} ({mm['hypothesis_raw']} vs {mm['reference_raw']}). Risk of patient record demographic mismatch."
            })
        elif cat in ("blood_pressure", "pulse", "oxygen_saturation"):
            significant_errors.append({
                "severity": "HIGH",
                "category": "vital_signs",
                "title": f"Vital Sign Discrepancy ({cat})",
                "description": f"Reference: {mm['reference']} vs Hypothesis: {mm['hypothesis']}"
            })
        else:
            significant_errors.append({
                "severity": "MEDIUM",
                "category": cat,
                "title": f"Numeric Discrepancy ({cat})",
                "description": f"Reference: {mm['reference']} vs Hypothesis: {mm['hypothesis']}"
            })
            
    # 2. Medication dosage value discrepancies
    for dm in med_comp.get("dosage_mismatches", []):
        significant_errors.append({
            "severity": "CRITICAL",
            "category": "medication_dosage",
            "title": f"Dosage Value Error: {dm['drug_name']}",
            "description": f"Reference dosage '{dm['reference_dosage']}' transcribed as '{dm['hypothesis_dosage']}'."
        })
        
    # 3. Frequency discrepancies
    for fm in med_comp.get("frequency_mismatches", []):
        significant_errors.append({
            "severity": "MEDIUM",
            "category": "medication_frequency",
            "title": f"Dosing Frequency Error: {fm['drug_name']}",
            "description": f"Reference '{fm['reference_frequency']}' transcribed as '{fm['hypothesis_frequency']}'."
        })
        
    # 4. Medication omissions
    for om in med_comp.get("omitted_medications", []):
        significant_errors.append({
            "severity": "CRITICAL",
            "category": "medication_omission",
            "title": f"Omitted Medication: {om['drug_name']}",
            "description": f"Medication '{om['drug_name']}' present in reference was missing from transcription."
        })
        
    # 5. Medication additions
    for am in med_comp.get("added_medications", []):
        significant_errors.append({
            "severity": "HIGH",
            "category": "medication_addition",
            "title": f"Added / Unmatched Medication: {am['drug_name']}",
            "description": f"Medication '{am['drug_name']}' transcribed but not found in reference."
        })
        
    # 6. Diagnosis omissions
    for d_om in terms_comp.get("diagnoses", {}).get("omissions", []):
        significant_errors.append({
            "severity": "HIGH",
            "category": "diagnosis_omission",
            "title": f"Omitted Diagnosis: {d_om}",
            "description": f"Diagnosis '{d_om}' not recognized in transcript."
        })
        
    return significant_errors


def format_evaluation_report(
    ref_text: str,
    hyp_text: str,
    wer_res: Dict[str, Any],
    false_starts: List[Dict[str, Any]],
    numeric_comp: Dict[str, Any],
    med_comp: Dict[str, Any],
    terms_comp: Dict[str, Any],
    clinical_sig: List[Dict[str, str]],
    clinical_wer_res: Optional[Dict[str, Any]] = None
) -> str:
    """Format the complete multi-metric clinical evaluation report."""
    lines = []
    sep = "=" * 78
    sub_sep = "-" * 78
    
    lines.append(sep)
    lines.append("        ANOT HEALTH — MEDICAL ASR TRANSCRIPT EVALUATION REPORT")
    lines.append(sep)
    lines.append("")
    lines.append("IMPORTANT DISCLAIMER:")
    lines.append("This evaluation report contains separate, objective clinical metrics for development")
    lines.append("benchmarking. It does NOT compute a single aggregate 'accuracy score'. This test")
    lines.append("does NOT constitute clinical certification or validation of safety.")
    lines.append("")
    
    # 1. General ASR Metrics
    lines.append(sub_sep)
    lines.append("1. GENERAL ASR METRICS (Word Error Rate & Edit Counts)")
    lines.append(sub_sep)
    lines.append(f"  Reference Word Count:     {wer_res['reference_word_count']}")
    lines.append(f"  Hypothesis Word Count:    {wer_res['hypothesis_word_count']}")
    lines.append(f"  Substitutions (S):        {wer_res['substitutions']}")
    lines.append(f"  Deletions (D):            {wer_res['deletions']}")
    lines.append(f"  Insertions (I):           {wer_res['insertions']}")
    lines.append(f"  Total Word Errors (S+D+I):{wer_res['total_errors']}")
    lines.append(f"  Word Error Rate (WER):    {wer_res['wer']:.4f} ({wer_res['wer']*100:.2f}%)")
    lines.append("")

    # 1B. Clinically-Normalized ASR Metrics (reported ALONGSIDE raw WER)
    if clinical_wer_res is not None:
        lines.append(sub_sep)
        lines.append("1B. CLINICALLY-NORMALIZED ASR METRICS (unit/vital/disfluency-equivalent)")
        lines.append(sub_sep)
        lines.append("  Applies the SAME deterministic clinical normalization to reference and")
        lines.append("  hypothesis so clinically-identical surface forms (500 milligrams == 500 mg,")
        lines.append("  138 over 84 == 138/84, 97 percent == 97%) and the speaker's own abandoned")
        lines.append("  false-starts are not counted as recognition errors. Clinical VALUES must")
        lines.append("  still match exactly; this never masks a changed number, drug, or dose.")
        lines.append("")
        lines.append(f"  Reference Word Count:     {clinical_wer_res['reference_word_count']}")
        lines.append(f"  Hypothesis Word Count:    {clinical_wer_res['hypothesis_word_count']}")
        lines.append(f"  Substitutions (S):        {clinical_wer_res['substitutions']}")
        lines.append(f"  Deletions (D):            {clinical_wer_res['deletions']}")
        lines.append(f"  Insertions (I):           {clinical_wer_res['insertions']}")
        lines.append(f"  Total Word Errors (S+D+I):{clinical_wer_res['total_errors']}")
        lines.append(f"  Clinically-Normalized WER:{clinical_wer_res['wer']:.4f} ({clinical_wer_res['wer']*100:.2f}%)")
        lines.append(f"  Clinically-Normalized Acc:{(1 - clinical_wer_res['wer'])*100:.2f}%")
        lines.append("")

    # 2. Word-Level Alignment Differences
    lines.append(sub_sep)
    lines.append("2. WORD-LEVEL ALIGNMENT DIFFERENCES")
    lines.append(sub_sep)
    diff_count = 0
    for op, rw, hw in wer_res["alignment"]:
        if op != "OK":
            diff_count += 1
            lines.append(f"  [{op:<3}] Ref: \"{rw}\"  <--->  Hyp: \"{hw}\"")
    if diff_count == 0:
        lines.append("  (No word-level differences detected)")
    lines.append("")
    
    # 3. Medical Terminology Comparison
    lines.append(sub_sep)
    lines.append("3. MEDICAL TERMINOLOGY COMPARISON")
    lines.append(sub_sep)
    for cat, data in terms_comp.items():
        cat_title = cat.replace("_", " ").title()
        lines.append(f"  [{cat_title}]")
        lines.append(f"    Matches ({len(data['matches'])}):    {', '.join(data['matches']) if data['matches'] else 'None'}")
        lines.append(f"    Omissions ({len(data['omissions'])}):  {', '.join(data['omissions']) if data['omissions'] else 'None'}")
        lines.append(f"    Additions ({len(data['additions'])}):  {', '.join(data['additions']) if data['additions'] else 'None'}")
    lines.append("")
    
    # 4. Medication Comparison
    lines.append(sub_sep)
    lines.append("4. MEDICATION COMPARISON")
    lines.append(sub_sep)
    lines.append(f"  Matched Medications ({len(med_comp['matched_regimens'])}):")
    for m in med_comp["matched_regimens"]:
        fs_note = f" (Acoustic false-start prefix: '{m['false_start_detected']}')" if m['false_start_detected'] else ""
        lines.append(f"    - {m['drug_name'].capitalize()}{fs_note}")
    if med_comp["omitted_medications"]:
        lines.append(f"  Omitted Medications ({len(med_comp['omitted_medications'])}):")
        for m in med_comp["omitted_medications"]:
            lines.append(f"    - {m['drug_name']}")
    if med_comp["added_medications"]:
        lines.append(f"  Added / Unmatched Medications ({len(med_comp['added_medications'])}):")
        for m in med_comp["added_medications"]:
            lines.append(f"    - {m['drug_name']}")
    lines.append("")
    
    # 5. Dosage Comparison
    lines.append(sub_sep)
    lines.append("5. DOSAGE COMPARISON & UNIT EQUIVALENCE")
    lines.append(sub_sep)
    if med_comp["equivalent_dosage_units"]:
        lines.append("  Equivalent Dosage Notations (Normalized equivalent units, NOT an error):")
        for eq in med_comp["equivalent_dosage_units"]:
            lines.append(f"    - {eq['drug_name'].capitalize()}: Ref: '{eq['reference']}' <---> Hyp: '{eq['hypothesis']}'")
            lines.append(f"      Normalized Standard: {eq['normalized']} [{eq['note']}]")
    if med_comp["dosage_mismatches"]:
        lines.append("  CRITICAL DOSAGE MISMATCHES:")
        for dm in med_comp["dosage_mismatches"]:
            lines.append(f"    - {dm['drug_name'].capitalize()}: Ref: '{dm['reference_dosage']}' vs Hyp: '{dm['hypothesis_dosage']}'")
    if not med_comp["equivalent_dosage_units"] and not med_comp["dosage_mismatches"]:
        lines.append("  (No dosage discrepancies detected)")
    lines.append("")
    
    # 6. Frequency Comparison
    lines.append(sub_sep)
    lines.append("6. FREQUENCY COMPARISON")
    lines.append(sub_sep)
    if med_comp["frequency_mismatches"]:
        lines.append("  Frequency Discrepancies:")
        for fm in med_comp["frequency_mismatches"]:
            lines.append(f"    - {fm['drug_name'].capitalize()}: Ref: '{fm['reference_frequency']}' vs Hyp: '{fm['hypothesis_frequency']}'")
    else:
        lines.append("  All detected medication dosing frequencies matched the reference:")
        for m in med_comp["matched_regimens"]:
            ref_f = m["reference"].get("frequency_raw")
            hyp_f = m["hypothesis"].get("frequency_raw")
            lines.append(f"    - {m['drug_name'].capitalize()}: '{ref_f}' matched '{hyp_f}'")
    lines.append("")
    
    # 7. Critical Numeric Comparison
    lines.append(sub_sep)
    lines.append("7. CRITICAL NUMERIC COMPARISON")
    lines.append(sub_sep)
    if numeric_comp["critical_mismatches"]:
        lines.append("  CRITICAL NUMERIC MISMATCHES:")
        for cm in numeric_comp["critical_mismatches"]:
            lines.append(f"    [!] Category:       {cm['category']}")
            lines.append(f"        Reference:      {cm['reference']} (raw: \"{cm['reference_raw']}\")")
            lines.append(f"        Hypothesis:     {cm['hypothesis']} (raw: \"{cm['hypothesis_raw']}\")")
    if numeric_comp["equivalent_notations"]:
        lines.append("  Equivalent Numeric Notations (Clinically identical values):")
        for en in numeric_comp["equivalent_notations"]:
            lines.append(f"    - {en['category']}: Ref \"{en['reference_raw']}\" <---> Hyp \"{en['hypothesis_raw']}\" (Value: {en['normalized_value']})")
    if numeric_comp["matches"]:
        lines.append("  Identical Numeric Matches:")
        for nm in numeric_comp["matches"]:
            lines.append(f"    - {nm['category']}: {nm['value']} (\"{nm['raw_text']}\")")
    lines.append("")
    
    # 8. Potentially Clinically Significant Mismatches
    lines.append(sub_sep)
    lines.append("8. POTENTIALLY CLINICALLY SIGNIFICANT MISMATCHES")
    lines.append(sub_sep)
    if clinical_sig:
        for idx, cs in enumerate(clinical_sig, 1):
            lines.append(f"  {idx}. [{cs['severity']}] {cs['title']}")
            lines.append(f"     Category:    {cs['category']}")
            lines.append(f"     Description: {cs['description']}")
    else:
        lines.append("  (No clinically significant mismatches flagged)")
    lines.append("")
    
    # 9. False-Start / Stutter Artifacts
    lines.append(sub_sep)
    lines.append("9. FALSE-START / STUTTER ARTIFACTS")
    lines.append(sub_sep)
    if false_starts:
        lines.append(f"  Detected Acoustic Disfluencies ({len(false_starts)}):")
        for fs in false_starts:
            lines.append(f"    - Artifact: \"{fs['matched_text']}\" (prefix '{fs['prefix']}' before '{fs['target_word']}')")
            lines.append(f"      Classification: {fs['category']}")
            lines.append(f"      Handling: Preserved in raw transcript; recognized as acoustic disfluency rather than drug hallucination.")
    else:
        lines.append("  (No false-start or stutter disfluencies detected)")
    lines.append("")
    
    # 10. Raw Hypothesis Transcript
    lines.append(sub_sep)
    lines.append("10. OVERALL RAW HYPOTHESIS TRANSCRIPT")
    lines.append(sub_sep)
    lines.append(hyp_text.strip())
    lines.append("")
    
    # 11. Overall Reference Transcript
    lines.append(sub_sep)
    lines.append("11. OVERALL REFERENCE TRANSCRIPT")
    lines.append(sub_sep)
    lines.append(ref_text.strip())
    lines.append("")
    lines.append(sep)
    lines.append("                          END OF EVALUATION REPORT")
    lines.append(sep)
    
    return "\n".join(lines)


def evaluate(ref_text: str, hyp_text: str, entities_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Execute complete evaluation pipeline and return structured results and formatted report."""
    # 1. General ASR metrics & alignment (raw)
    wer_res = compute_wer_and_alignment(ref_text, hyp_text, normalize_numbers=True)

    # 1B. Clinically-normalized ASR metrics (reported alongside raw WER)
    clinical_wer_res = compute_clinical_normalized_wer(ref_text, hyp_text)

    # 2. False starts & stutters
    false_starts = detect_false_starts(hyp_text)
    
    # 3. Numeric extraction & comparison
    ref_num = parse_numeric_entities(ref_text)
    hyp_num = parse_numeric_entities(hyp_text)
    num_comp = compare_numeric_entities(ref_num, hyp_num)
    
    # 4. Medication extraction & comparison
    ref_meds = extract_medications(ref_text, entities_cfg, [])
    hyp_meds = extract_medications(hyp_text, entities_cfg, false_starts)
    med_comp = compare_medications(ref_meds, hyp_meds, entities_cfg.get("dosage_units", {}))
    
    # 5. Medical terminology comparison
    terms_comp = extract_and_compare_medical_terms(ref_text, hyp_text, entities_cfg)
    
    # 6. Potentially clinically significant errors
    clinical_sig = identify_clinical_significance(num_comp, med_comp, terms_comp)
    
    # 7. Format complete report
    report_text = format_evaluation_report(
        ref_text=ref_text,
        hyp_text=hyp_text,
        wer_res=wer_res,
        clinical_wer_res=clinical_wer_res,
        false_starts=false_starts,
        numeric_comp=num_comp,
        med_comp=med_comp,
        terms_comp=terms_comp,
        clinical_sig=clinical_sig
    )

    results = {
        "wer_metrics": wer_res,
        "clinical_normalized_wer_metrics": clinical_wer_res,
        "false_starts": false_starts,
        "numeric_comparison": num_comp,
        "medication_comparison": med_comp,
        "terminology_comparison": terms_comp,
        "clinical_significance": clinical_sig
    }
    
    return results, report_text


def main():
    parser = argparse.ArgumentParser(
        description="Anot Medical ASR Evaluation Engine: Compare ASR hypothesis against clinical reference transcript."
    )
    parser.add_argument("--reference", "-r", required=True, help="Path to ground truth reference transcript file.")
    parser.add_argument("--hypothesis", "-y", required=True, help="Path to ASR hypothesis transcript file.")
    parser.add_argument("--output", "-o", default=None, help="Path to output report file. If omitted, prints to stdout.")
    parser.add_argument("--entities", "-e", default=None, help="Optional path to custom medical_entities.json file.")
    parser.add_argument("--json-output", "-j", default=None, help="Optional path to save structured metrics as JSON.")
    
    args = parser.parse_args()
    
    with open(args.reference, "r", encoding="utf-8") as f:
        ref_text = f.read()
        
    with open(args.hypothesis, "r", encoding="utf-8") as f:
        hyp_text = f.read()
        
    entities_cfg = load_medical_entities(args.entities)
    
    results, report_text = evaluate(ref_text, hyp_text, entities_cfg)
    
    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[Evaluation Report Saved]: {out_path}")
    else:
        print(report_text)
        
    if args.json_output:
        json_path = pathlib.Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Structured JSON Metrics Saved]: {json_path}")


if __name__ == "__main__":
    main()
