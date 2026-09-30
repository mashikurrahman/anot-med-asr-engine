#!/usr/bin/env python3
"""
Unit and integration tests for Anot Medical ASR Evaluation Engine.
"""

import unittest
import pathlib
import sys
import os

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_transcript import (
    load_medical_entities,
    normalize_for_wer,
    compute_wer_and_alignment,
    detect_false_starts,
    parse_numeric_entities,
    compare_numeric_entities,
    extract_medications,
    compare_medications,
    extract_and_compare_medical_terms,
    evaluate
)


class TestMedicalASREvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.entities_cfg = load_medical_entities()

    def test_wer_exact_match(self):
        ref = "Patient has hypertension and diabetes."
        hyp = "Patient has hypertension and diabetes."
        res = compute_wer_and_alignment(ref, hyp)
        self.assertEqual(res["wer"], 0.0)
        self.assertEqual(res["substitutions"], 0)
        self.assertEqual(res["deletions"], 0)
        self.assertEqual(res["insertions"], 0)
        self.assertEqual(res["total_errors"], 0)

    def test_wer_substitutions_deletions_insertions(self):
        ref = "patient takes metformin daily"
        hyp = "patient takes lisinopril twice daily extra"
        # ref words: patient takes metformin daily (4)
        # hyp words: patient takes lisinopril twice daily extra (6)
        # substitutions: metformin -> lisinopril (1)
        # insertions: twice, extra (2)
        # deletions: 0
        res = compute_wer_and_alignment(ref, hyp)
        self.assertGreater(res["wer"], 0.0)
        self.assertIn("alignment", res)
        self.assertEqual(res["reference_word_count"], 4)

    def test_number_normalization_in_wer(self):
        # Spoken numerals normalized to digits
        ref = "type 2 diabetes"
        hyp = "type two diabetes"
        res = compute_wer_and_alignment(ref, hyp, normalize_numbers=True)
        self.assertEqual(res["wer"], 0.0)
        self.assertEqual(res["substitutions"], 0)

    def test_detect_false_starts_and_stutters(self):
        text = "medications include isupro- lisinopril and ator- atorvastatin"
        false_starts = detect_false_starts(text)
        self.assertEqual(len(false_starts), 2)
        
        # Test isupro- lisinopril (abandoned false start)
        fs1 = false_starts[0]
        self.assertEqual(fs1["prefix"], "isupro-")
        self.assertEqual(fs1["target_word"], "lisinopril")
        self.assertEqual(fs1["category"], "abandoned_false_start")
        
        # Test ator- atorvastatin (stutter repetition)
        fs2 = false_starts[1]
        self.assertEqual(fs2["prefix"], "ator-")
        self.assertEqual(fs2["target_word"], "atorvastatin")
        self.assertEqual(fs2["category"], "stutter_repetition")

    def test_critical_numeric_mismatch_patient_age(self):
        ref = "The patient is a 58-year-old male."
        hyp = "The patient is a 50-year-old male."
        ref_num = parse_numeric_entities(ref)
        hyp_num = parse_numeric_entities(hyp)
        comp = compare_numeric_entities(ref_num, hyp_num)
        
        self.assertEqual(len(comp["critical_mismatches"]), 1)
        mm = comp["critical_mismatches"][0]
        self.assertEqual(mm["category"], "patient_age")
        self.assertEqual(mm["reference"], "58")
        self.assertEqual(mm["hypothesis"], "50")

    def test_equivalent_vitals_notations(self):
        ref = "Blood pressure is 138 over 84 and oxygen saturation is 97 percent."
        hyp = "Blood pressure is 138/84 and oxygen saturation is 97%."
        ref_num = parse_numeric_entities(ref)
        hyp_num = parse_numeric_entities(hyp)
        comp = compare_numeric_entities(ref_num, hyp_num)
        
        # Should NOT be critical mismatches
        self.assertEqual(len(comp["critical_mismatches"]), 0)
        
        # Should be classified as equivalent notations
        eq_cats = [eq["category"] for eq in comp["equivalent_notations"]]
        self.assertIn("blood_pressure", eq_cats)
        self.assertIn("oxygen_saturation", eq_cats)

    def test_equivalent_dosage_units_not_dosage_error(self):
        ref = "metformin 500 milligrams twice daily"
        hyp = "metformin 500 mg twice daily"
        ref_meds = extract_medications(ref, self.entities_cfg, [])
        hyp_meds = extract_medications(hyp, self.entities_cfg, [])
        comp = compare_medications(ref_meds, hyp_meds, self.entities_cfg.get("dosage_units", {}))
        
        # Must NOT be classified as dosage mismatch
        self.assertEqual(len(comp["dosage_mismatches"]), 0)
        # Must be in equivalent_dosage_units
        self.assertEqual(len(comp["equivalent_dosage_units"]), 1)
        eq = comp["equivalent_dosage_units"][0]
        self.assertEqual(eq["drug_name"], "metformin")
        self.assertEqual(eq["normalized"], "500 mg")

    def test_critical_dosage_amount_mismatch(self):
        ref = "metformin 500 mg twice daily"
        hyp = "metformin 250 mg twice daily"
        ref_meds = extract_medications(ref, self.entities_cfg, [])
        hyp_meds = extract_medications(hyp, self.entities_cfg, [])
        comp = compare_medications(ref_meds, hyp_meds, self.entities_cfg.get("dosage_units", {}))
        
        # MUST be classified as a critical dosage mismatch
        self.assertEqual(len(comp["dosage_mismatches"]), 1)
        dm = comp["dosage_mismatches"][0]
        self.assertEqual(dm["drug_name"], "metformin")
        self.assertIn("500", dm["reference_dosage"])
        self.assertIn("250", dm["hypothesis_dosage"])

    def test_frequency_mismatch_detected(self):
        ref = "lisinopril 10 mg once daily"
        hyp = "lisinopril 10 mg twice daily"
        ref_meds = extract_medications(ref, self.entities_cfg, [])
        hyp_meds = extract_medications(hyp, self.entities_cfg, [])
        comp = compare_medications(ref_meds, hyp_meds, self.entities_cfg.get("dosage_units", {}))
        
        self.assertEqual(len(comp["frequency_mismatches"]), 1)
        fm = comp["frequency_mismatches"][0]
        self.assertEqual(fm["drug_name"], "lisinopril")

    def test_stage6_end_to_end_evaluation(self):
        ref_path = PROJECT_ROOT / "results" / "anot_test_01_reference.txt"
        hyp_path = PROJECT_ROOT / "results" / "anot_test_01_raw.txt"
        
        self.assertTrue(ref_path.exists(), "Reference file missing")
        self.assertTrue(hyp_path.exists(), "Hypothesis file missing")
        
        with open(ref_path, "r", encoding="utf-8") as f:
            ref_text = f.read()
        with open(hyp_path, "r", encoding="utf-8") as f:
            hyp_text = f.read()
            
        results, report_text = evaluate(ref_text, hyp_text, self.entities_cfg)
        
        # 1. Verify patient age mismatch (58 -> 50)
        num_comp = results["numeric_comparison"]
        age_mismatches = [
            cm for cm in num_comp["critical_mismatches"]
            if cm["category"] == "patient_age"
        ]
        self.assertEqual(len(age_mismatches), 1)
        self.assertEqual(age_mismatches[0]["reference"], "58")
        self.assertEqual(age_mismatches[0]["hypothesis"], "50")
        
        # 2. Verify dosage unit equivalence (500 milligrams -> 500 mg is NOT a dosage error)
        med_comp = results["medication_comparison"]
        self.assertEqual(len(med_comp["dosage_mismatches"]), 0)
        self.assertEqual(len(med_comp["equivalent_dosage_units"]), 3)
        
        metformin_eq = next(eq for eq in med_comp["equivalent_dosage_units"] if eq["drug_name"] == "metformin")
        self.assertEqual(metformin_eq["normalized"], "500 mg")
        
        # 3. Verify false starts detected
        self.assertEqual(len(results["false_starts"]), 2)
        fs_prefixes = [fs["prefix"] for fs in results["false_starts"]]
        self.assertIn("isupro-", fs_prefixes)
        self.assertIn("ator-", fs_prefixes)
        
        # 4. Verify no single aggregate score
        self.assertNotIn("accuracy_score", results)
        self.assertNotIn("overall_score", results)
        self.assertNotIn("single_quality_score", results)


if __name__ == "__main__":
    unittest.main()
