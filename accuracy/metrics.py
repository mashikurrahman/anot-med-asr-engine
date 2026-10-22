#!/usr/bin/env python3
"""
Anot Medical ASR — Clinical Accuracy Metrics Engine
===================================================
Stage 9: Clinical error classification and multi-metric accuracy analysis.

Emphasizes critical clinical errors:
- Patient demographics (e.g., age discrepancy)
- Medication errors (omissions, additions, dosage number errors, frequency errors)
- Vital sign errors (BP, pulse, SpO2, temp)
- Negation inversions (e.g. 'no chest pain' -> 'chest pain')
- Medical terminology omissions (diagnoses, symptoms, procedures)
- Temporal / duration discrepancies

CRITICAL PRINCIPLE:
Never collapses clinical accuracy into a single arbitrary score.
Maintains separate, objective, clinically stratified metric reporting.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import pathlib

from evaluation.evaluate_transcript import (
    compute_wer_and_alignment,
    compute_clinical_normalized_wer,
    load_medical_entities,
    parse_numeric_entities,
    compare_numeric_entities,
    extract_medications,
    compare_medications,
    extract_and_compare_medical_terms,
    detect_false_starts,
    identify_clinical_significance
)


def compute_clinical_accuracy(
    ref_text: str,
    hyp_text: str,
    entities_cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive clinical accuracy metrics across general ASR,
    critical numeric entities, medication regimens, medical terminology,
    and clinical error severity.
    """
    if entities_cfg is None:
        entities_cfg = load_medical_entities()

    # 1. General ASR Word Error Rate (raw + clinically-normalized)
    wer_res = compute_wer_and_alignment(ref_text, hyp_text, normalize_numbers=True)
    clinical_wer_res = compute_clinical_normalized_wer(ref_text, hyp_text)

    # 2. Acoustic false-starts & stutters
    false_starts = detect_false_starts(hyp_text)

    # 3. Numeric extraction & comparison
    ref_num = parse_numeric_entities(ref_text)
    hyp_num = parse_numeric_entities(hyp_text)
    num_comp = compare_numeric_entities(ref_num, hyp_num)

    # 4. Medication comparison
    ref_meds = extract_medications(ref_text, entities_cfg, [])
    hyp_meds = extract_medications(hyp_text, entities_cfg, false_starts)
    med_comp = compare_medications(ref_meds, hyp_meds, entities_cfg.get("dosage_units", {}))

    # 5. Medical terminology comparison
    terms_comp = extract_and_compare_medical_terms(ref_text, hyp_text, entities_cfg)

    # 6. Clinically significant error identification
    clinical_sig = identify_clinical_significance(num_comp, med_comp, terms_comp)

    # 7. Stratified Critical Error Breakdown
    critical_errors = {
        "patient_age_errors": [
            cm for cm in num_comp.get("critical_mismatches", []) if cm["category"] == "patient_age"
        ],
        "vital_sign_errors": [
            cm for cm in num_comp.get("critical_mismatches", [])
            if cm["category"] in ("blood_pressure", "pulse", "oxygen_saturation", "temperature")
        ],
        "medication_dosage_errors": med_comp.get("dosage_mismatches", []),
        "medication_frequency_errors": med_comp.get("frequency_mismatches", []),
        "medication_omissions": med_comp.get("omitted_medications", []),
        "medication_additions": med_comp.get("added_medications", []),
        "diagnosis_omissions": terms_comp.get("diagnoses", {}).get("omissions", []),
        "symptom_omissions": terms_comp.get("symptoms", {}).get("omissions", []),
        "procedure_omissions": terms_comp.get("procedures", {}).get("omissions", []),
        "duration_errors": [
            cm for cm in num_comp.get("critical_mismatches", []) if cm["category"] == "clinical_duration"
        ]
    }

    total_critical_count = sum(len(v) for v in critical_errors.values())

    return {
        "wer_metrics": {
            "wer": wer_res["wer"],
            "wer_percent": round(wer_res["wer"] * 100, 2),
            "reference_word_count": wer_res["reference_word_count"],
            "hypothesis_word_count": wer_res["hypothesis_word_count"],
            "substitutions": wer_res["substitutions"],
            "deletions": wer_res["deletions"],
            "insertions": wer_res["insertions"],
            "total_word_errors": wer_res["total_errors"]
        },
        "clinical_normalized_wer_metrics": {
            "wer": clinical_wer_res["wer"],
            "wer_percent": round(clinical_wer_res["wer"] * 100, 2),
            "accuracy_percent": round((1 - clinical_wer_res["wer"]) * 100, 2),
            "substitutions": clinical_wer_res["substitutions"],
            "deletions": clinical_wer_res["deletions"],
            "insertions": clinical_wer_res["insertions"],
            "total_word_errors": clinical_wer_res["total_errors"],
            "methodology": clinical_wer_res["methodology"]
        },
        "critical_clinical_errors": {
            "total_critical_count": total_critical_count,
            "breakdown": critical_errors,
            "findings_list": clinical_sig
        },
        "medication_performance": {
            "matched_regimens_count": len(med_comp.get("matched_regimens", [])),
            "dosage_mismatches_count": len(med_comp.get("dosage_mismatches", [])),
            "omitted_medications_count": len(med_comp.get("omitted_medications", [])),
            "equivalent_dosage_units_count": len(med_comp.get("equivalent_dosage_units", []))
        },
        "numeric_performance": {
            "exact_matches_count": len(num_comp.get("matches", [])),
            "equivalent_notations_count": len(num_comp.get("equivalent_notations", [])),
            "critical_mismatches_count": len(num_comp.get("critical_mismatches", []))
        },
        "disfluencies": {
            "count": len(false_starts),
            "items": false_starts
        }
    }
