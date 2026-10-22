#!/usr/bin/env python3
"""
Anot Medical ASR — Multi-Run Accuracy Comparison Engine
========================================================
Stage 9: Reusable framework for side-by-side comparison of transcription runs.

Compares:
- MedASR Greedy vs Beam Search (+ optional LM)
- Baseline MedASR vs Fine-Tuned MedASR
- Alternative local ASR pipelines

Reports differences across WER, substitutions, deletions, insertions,
critical numeric errors, medication discrepancies, and latency metrics.
"""

from typing import Dict, Any, List, Optional
import json
import pathlib
import sys

from accuracy.metrics import compute_clinical_accuracy


def compare_transcription_runs(
    ref_text: str,
    runs: Dict[str, str],
    timings: Optional[Dict[str, Dict[str, float]]] = None,
    entities_cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compare multiple transcription outputs against a common reference transcript.
    
    Args:
        ref_text: Ground truth reference transcript
        runs: Dict of {run_name: hypothesis_transcript}
        timings: Optional dict of {run_name: {"inference_time": float, "decode_time": float, "rtf": float}}
        entities_cfg: Optional entity configuration dictionary
    
    Returns:
        Structured comparison dictionary
    """
    results = {}
    for name, hyp_text in runs.items():
        metrics = compute_clinical_accuracy(ref_text, hyp_text, entities_cfg=entities_cfg)
        run_data = {
            "metrics": metrics,
            "hypothesis_text": hyp_text
        }
        if timings and name in timings:
            run_data["timing"] = timings[name]
        results[name] = run_data

    # Compute differential table
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
            "critical_errors": crit_m["total_critical_count"],
            "dosage_mismatches": med_m["dosage_mismatches_count"],
            "numeric_mismatches": num_m["critical_mismatches_count"],
            "disfluencies": dis_m["count"]
        }
        if "timing" in data:
            row.update(data["timing"])
        table_rows.append(row)

    return {
        "comparison_table": table_rows,
        "runs": results
    }


def format_comparison_table(comparison_result: Dict[str, Any]) -> str:
    """Format comparison result into an ASCII comparison table."""
    rows = comparison_result.get("comparison_table", [])
    if not rows:
        return "No comparison runs available."

    lines = [
        "=" * 105,
        "  ANOT MEDICAL ASR — TRANSCRIPTION ACCURACY COMPARISON MATRIX",
        "=" * 105,
        f"{'Run Configuration':<24} | {'WER (%)':<8} | {'S':<4} | {'D':<4} | {'I':<4} | {'Errors':<6} | {'Crit Err':<8} | {'Dose Err':<8} | {'Num Err':<8}",
        "-" * 105,
    ]

    for r in rows:
        lines.append(
            f"{r['run_name']:<24} | "
            f"{r['wer_percent']:<8.2f} | "
            f"{r['substitutions']:<4} | "
            f"{r['deletions']:<4} | "
            f"{r['insertions']:<4} | "
            f"{r['total_errors']:<6} | "
            f"{r['critical_errors']:<8} | "
            f"{r['dosage_mismatches']:<8} | "
            f"{r['numeric_mismatches']:<8}"
        )

    lines.append("=" * 105)
    return "\n".join(lines)
