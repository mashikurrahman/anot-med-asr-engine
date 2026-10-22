#!/usr/bin/env python3
"""
Anot Medical ASR — LM Rescoring Comparison Engine
===================================================
Stage 9B: Compare greedy, beam, and beam+LM rescoring runs.

Extends compare_runs.py with LM-specific metrics:
- Per-run LM configuration
- Changed token analysis (medical vs numeric vs general)
- Acoustic vs LM score contribution
- 58→50 critical numeric error tracking
"""

from typing import Dict, Any, List, Optional
from accuracy.metrics import compute_clinical_accuracy


def compare_lm_runs(
    ref_text: str,
    runs: Dict[str, Dict[str, Any]],
    entities_cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compare transcription runs with LM rescoring metadata.
    
    Args:
        ref_text: Ground truth reference transcript
        runs: Dict of {run_name: {"text": hypothesis, "lm_config": {...}, "timing": {...}}}
        entities_cfg: Optional medical entity configuration
        
    Returns:
        Structured comparison with clinical accuracy metrics per run
    """
    results = {}
    for name, run_data in runs.items():
        hyp_text = run_data.get("text", "")
        metrics = compute_clinical_accuracy(ref_text, hyp_text, entities_cfg=entities_cfg)
        
        entry = {
            "metrics": metrics,
            "hypothesis_text": hyp_text,
        }
        
        if "lm_config" in run_data:
            entry["lm_config"] = run_data["lm_config"]
        if "timing" in run_data:
            entry["timing"] = run_data["timing"]
        if "word_count" in run_data:
            entry["word_count"] = run_data["word_count"]
        if "baseline_comparison" in run_data:
            entry["baseline_comparison"] = run_data["baseline_comparison"]
            
        results[name] = entry

    # Build comparison table
    table_rows = []
    for name, data in results.items():
        m = data["metrics"]
        wer_m = m["wer_metrics"]
        crit_m = m["critical_clinical_errors"]
        med_m = m["medication_performance"]
        num_m = m["numeric_performance"]
        dis_m = m["disfluencies"]

        row = {
            "run_name": name,
            "wer_percent": wer_m["wer_percent"],
            "substitutions": wer_m["substitutions"],
            "deletions": wer_m["deletions"],
            "insertions": wer_m["insertions"],
            "total_errors": wer_m["total_word_errors"],
            "critical_numeric_errors": num_m["critical_mismatches_count"],
            "medication_errors": med_m["dosage_mismatches_count"],
            "dosage_errors": med_m["dosage_mismatches_count"],
            "medication_omissions": med_m["omitted_medications_count"],
            "terminology_errors": sum(
                len(v.get("omissions", []))
                for v in m.get("critical_clinical_errors", {}).get("breakdown", {}).values()
                if isinstance(v, list)
            ),
            "disfluencies": dis_m["count"],
        }
        
        # Add LM config if available
        if "lm_config" in data:
            row["lm_weight"] = data["lm_config"].get("lm_weight", 0)
            row["medical_weight"] = data["lm_config"].get("medical_weight", 0)
            row["beam_width"] = data["lm_config"].get("beam_width", 0)
        
        # Add timing if available
        if "timing" in data:
            row["decode_time"] = data["timing"].get("decode_time", 0)
            row["rtf"] = data["timing"].get("rtf", 0)
            
        table_rows.append(row)

    return {
        "comparison_table": table_rows,
        "runs": results,
    }


def format_lm_comparison_table(comparison: Dict[str, Any]) -> str:
    """Format LM comparison into a readable ASCII table."""
    rows = comparison.get("comparison_table", [])
    if not rows:
        return "No comparison runs available."

    lines = [
        "=" * 130,
        "  ANOT MEDICAL ASR — LM RESCORING COMPARISON MATRIX (Stage 9B)",
        "=" * 130,
        f"{'Configuration':<30} | {'WER%':<7} | {'S':<4} | {'D':<4} | {'I':<4} | "
        f"{'Errs':<5} | {'CritNum':<7} | {'MedErr':<6} | {'DoseErr':<7} | "
        f"{'LM_W':<6} | {'Med_W':<6} | {'BW':<4}",
        "-" * 130,
    ]

    for r in rows:
        lm_w = f"{r.get('lm_weight', '-')}" if 'lm_weight' in r else "-"
        med_w = f"{r.get('medical_weight', '-')}" if 'medical_weight' in r else "-"
        bw = f"{r.get('beam_width', '-')}" if 'beam_width' in r else "-"
        
        lines.append(
            f"{r['run_name']:<30} | "
            f"{r['wer_percent']:<7.2f} | "
            f"{r['substitutions']:<4} | "
            f"{r['deletions']:<4} | "
            f"{r['insertions']:<4} | "
            f"{r['total_errors']:<5} | "
            f"{r['critical_numeric_errors']:<7} | "
            f"{r['medication_errors']:<6} | "
            f"{r['dosage_errors']:<7} | "
            f"{lm_w:<6} | "
            f"{med_w:<6} | "
            f"{bw:<4}"
        )

    lines.append("=" * 130)
    return "\n".join(lines)


def format_change_analysis(comparison: Dict[str, Any]) -> str:
    """Format per-change analysis across runs."""
    lines = [
        "=" * 90,
        "  CHANGE ANALYSIS: BASELINE vs LM-RESCORED TRANSCRIPTS",
        "=" * 90,
    ]
    
    for name, data in comparison.get("runs", {}).items():
        bc = data.get("baseline_comparison", {})
        if not bc:
            continue
            
        lines.append(f"\n--- {name} ---")
        lines.append(f"  Total changes: {bc.get('total_changes', 0)}")
        lines.append(f"  Changed by medical vocab:  {bc.get('changed_by_medical_vocab', 0)}")
        lines.append(f"  Changed by numeric:        {bc.get('changed_by_numeric', 0)}")
        lines.append(f"  Changed by general vocab:  {bc.get('changed_by_general_vocab', 0)}")
        
        for change in bc.get("changes", []):
            lines.append(
                f"    [{change.get('change_source', '?'):<20}] "
                f"'{change.get('baseline', '')}' → '{change.get('lm_decoded', '')}'"
            )
    
    lines.append("=" * 90)
    return "\n".join(lines)
