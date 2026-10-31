#!/usr/bin/env python3
"""
Run clinical evaluation engine across all 29 real encounters for both
Whisper large-v3 and Parakeet-TDT 0.6B.

Aggregates:
1. Medical Terminology Recall (diagnoses, symptoms, procedures, anatomy, lab tests)
2. Medication & Dosage Accuracy (matched, omitted, added, dosage mismatches)
3. Critical Numbers & Vitals (matched, critical mismatches, equivalent notations)
4. Raw WER vs Clinically-Normalized WER
5. Clinically Significant Discrepancies (severity breakdown)
"""
import glob
import json
import os
import pathlib
import sys

# Ensure root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluate_transcript import evaluate, load_medical_entities
from scratch_score_real import docx_to_plain_reference, AUDIO_DIR, HYPOTHESES_PATH

REPORTS_DIR = os.path.join("evaluation", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def main():
    if not os.path.exists(HYPOTHESES_PATH):
        print(f"ERROR: {HYPOTHESES_PATH} not found.")
        sys.exit(1)

    with open(HYPOTHESES_PATH, "r", encoding="utf-8") as f:
        hyps = json.load(f)

    entities_cfg = load_medical_entities()
    docx_files = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(AUDIO_DIR, "*.docx"))
    }

    engines = ["whisper", "parakeet"]
    report_data = {eng: {} for eng in engines}
    summary = {
        eng: {
            "encounters_evaluated": 0,
            "ref_words_total": 0,
            "raw_wer": {"errors": 0, "ref_words": 0},
            "clinical_wer": {"errors": 0, "ref_words": 0},
            "medications": {
                "matched": 0,
                "omitted": 0,
                "added": 0,
                "dosage_mismatches": 0,
                "dosage_equivalents": 0,
            },
            "numerics": {
                "matched": 0,
                "critical_mismatches": 0,
                "equivalent_notations": 0,
                "omissions": 0,
                "additions": 0,
            },
            "terms": {
                "diagnoses": {"matched": 0, "omitted": 0, "added": 0},
                "symptoms": {"matched": 0, "omitted": 0, "added": 0},
                "procedures": {"matched": 0, "omitted": 0, "added": 0},
                "anatomy": {"matched": 0, "omitted": 0, "added": 0},
                "lab_tests": {"matched": 0, "omitted": 0, "added": 0},
                "vitals": {"matched": 0, "omitted": 0, "added": 0},
            },
            "clinical_significance": {
                "HIGH": 0,
                "MEDIUM": 0,
                "LOW": 0,
            },
            "false_starts_detected": 0,
        }
        for eng in engines
    }

    per_encounter_rows = []

    for m4a_name in sorted(hyps.keys()):
        stem = os.path.splitext(m4a_name)[0]
        docx_path = docx_files.get(stem)
        if not docx_path:
            continue

        try:
            ref_text = docx_to_plain_reference(docx_path)
        except Exception:
            continue

        if not ref_text.strip():
            continue

        row = {"file": stem}

        for eng in engines:
            hyp_text = hyps[m4a_name].get(eng, "")
            if hyp_text.startswith("ERROR") or not hyp_text.strip():
                continue

            res, report_txt = evaluate(ref_text, hyp_text, entities_cfg)
            report_data[eng][stem] = res

            s = summary[eng]
            s["encounters_evaluated"] += 1

            # WER
            w_raw = res["wer_metrics"]
            s["raw_wer"]["errors"] += w_raw["total_errors"]
            s["raw_wer"]["ref_words"] += w_raw["reference_word_count"]

            w_clin = res["clinical_normalized_wer_metrics"]
            s["clinical_wer"]["errors"] += w_clin["total_errors"]
            s["clinical_wer"]["ref_words"] += w_clin["reference_word_count"]

            # Medications
            med = res["medication_comparison"]
            s["medications"]["matched"] += len(med.get("matched_regimens", []))
            s["medications"]["omitted"] += len(med.get("omitted_medications", []))
            s["medications"]["added"] += len(med.get("added_medications", []))
            s["medications"]["dosage_mismatches"] += len(med.get("dosage_mismatches", []))
            s["medications"]["dosage_equivalents"] += len(med.get("equivalent_dosage_units", []))

            # Numerics
            num = res["numeric_comparison"]
            s["numerics"]["matched"] += len(num.get("matches", []))
            s["numerics"]["critical_mismatches"] += len(num.get("critical_mismatches", []))
            s["numerics"]["equivalent_notations"] += len(num.get("equivalent_notations", []))
            s["numerics"]["omissions"] += len(num.get("omissions", []))
            s["numerics"]["additions"] += len(num.get("additions", []))

            # Terminology
            terms = res["terminology_comparison"]
            for cat, cat_data in terms.items():
                if cat in s["terms"]:
                    s["terms"][cat]["matched"] += len(cat_data.get("matches", []))
                    s["terms"][cat]["omitted"] += len(cat_data.get("omissions", []))
                    s["terms"][cat]["added"] += len(cat_data.get("additions", []))

            # Clinical severity
            for cs in res["clinical_significance"]:
                sev = cs.get("severity", "LOW")
                if sev in s["clinical_significance"]:
                    s["clinical_significance"][sev] += 1

            # False starts
            s["false_starts_detected"] += len(res.get("false_starts", []))

            row[f"{eng}_raw_wer"] = round(w_raw["wer"] * 100, 1)
            row[f"{eng}_clin_wer"] = round(w_clin["wer"] * 100, 1)
            row[f"{eng}_med_matched"] = len(med.get("matched_regimens", []))
            row[f"{eng}_med_omitted"] = len(med.get("omitted_medications", []))
            row[f"{eng}_num_matched"] = len(num.get("matches", []))
            row[f"{eng}_num_mismatch"] = len(num.get("critical_mismatches", []))

        per_encounter_rows.append(row)

    # Compute aggregate rates
    for eng in engines:
        s = summary[eng]
        raw_tot = s["raw_wer"]["ref_words"]
        s["raw_wer"]["wer_pct"] = round(100.0 * s["raw_wer"]["errors"] / max(raw_tot, 1), 2)
        s["raw_wer"]["accuracy_pct"] = round(100.0 - s["raw_wer"]["wer_pct"], 2)

        clin_tot = s["clinical_wer"]["ref_words"]
        s["clinical_wer"]["wer_pct"] = round(100.0 * s["clinical_wer"]["errors"] / max(clin_tot, 1), 2)
        s["clinical_wer"]["accuracy_pct"] = round(100.0 - s["clinical_wer"]["wer_pct"], 2)

        med_ref_tot = s["medications"]["matched"] + s["medications"]["omitted"]
        s["medications"]["recall_pct"] = round(100.0 * s["medications"]["matched"] / max(med_ref_tot, 1), 1)

        num_ref_tot = s["numerics"]["matched"] + s["numerics"]["omissions"] + s["numerics"]["critical_mismatches"]
        s["numerics"]["recall_pct"] = round(100.0 * s["numerics"]["matched"] / max(num_ref_tot, 1), 1)

    # Save JSON summary
    out_json = os.path.join(REPORTS_DIR, "real_clinical_benchmark_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_encounter": per_encounter_rows}, f, indent=2)

    # Save human-readable formatted report
    out_txt = os.path.join(REPORTS_DIR, "real_clinical_benchmark_report.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("       ANOT HEALTH — 29 REAL ENCOUNTERS CLINICAL BENCHMARK REPORT\n")
        f.write("=" * 80 + "\n\n")

        for eng in engines:
            s = summary[eng]
            f.write(f"\n--- {eng.upper()} LARGE / TDT AGGREGATE PERFORMANCE ---\n")
            f.write(f"Encounters Evaluated:             {s['encounters_evaluated']}\n")
            f.write(f"Raw WER / Accuracy:               {s['raw_wer']['wer_pct']}% / {s['raw_wer']['accuracy_pct']}%\n")
            f.write(f"Clinical-Normalized WER / Acc:    {s['clinical_wer']['wer_pct']}% / {s['clinical_wer']['accuracy_pct']}%\n\n")

            f.write("Medication Regimen Recognition:\n")
            f.write(f"  Matched Medications:            {s['medications']['matched']}\n")
            f.write(f"  Omitted Medications:            {s['medications']['omitted']}\n")
            f.write(f"  Added / Unmatched Medications:  {s['medications']['added']}\n")
            f.write(f"  Medication Recall:              {s['medications']['recall_pct']}%\n")
            f.write(f"  Dosage Mismatches:              {s['medications']['dosage_mismatches']}\n")
            f.write(f"  Normalized Equivalent Dosages:  {s['medications']['dosage_equivalents']}\n\n")

            f.write("Critical Numeric & Vitals Recognition:\n")
            f.write(f"  Exact / Equivalent Matches:     {s['numerics']['matched']}\n")
            f.write(f"  Critical Numeric Mismatches:    {s['numerics']['critical_mismatches']}\n")
            f.write(f"  Omitted Numeric Values:         {s['numerics']['omissions']}\n")
            f.write(f"  Numeric Recall:                 {s['numerics']['recall_pct']}%\n\n")

            f.write("Medical Vocabulary by Domain (Matched / Omitted / Added):\n")
            for cat, t in s["terms"].items():
                tot = t["matched"] + t["omitted"]
                rec = round(100.0 * t["matched"] / max(tot, 1), 1)
                f.write(f"  {cat:15s}  Matched: {t['matched']:3d} | Omitted: {t['omitted']:3d} | Added: {t['added']:3d} | Recall: {rec:5.1f}%\n")

            f.write("\nClinically Significant Error Flags:\n")
            f.write(f"  HIGH Severity:                  {s['clinical_significance']['HIGH']}\n")
            f.write(f"  MEDIUM Severity:                {s['clinical_significance']['MEDIUM']}\n")
            f.write(f"  LOW Severity:                   {s['clinical_significance']['LOW']}\n")
            f.write("-" * 80 + "\n")

    print(f"Clinical benchmark complete. Saved to:")
    print(f"  -> {out_txt}")
    print(f"  -> {out_json}")


if __name__ == "__main__":
    main()

