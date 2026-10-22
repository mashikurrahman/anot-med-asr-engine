#!/usr/bin/env python3
"""
Unit Tests for Accuracy Comparison Framework
=============================================
Stage 9: Tests for metrics calculation and differential multi-run comparison.
"""

import unittest

from accuracy.metrics import compute_clinical_accuracy
from accuracy.compare_runs import compare_transcription_runs, format_comparison_table


class TestAccuracyFramework(unittest.TestCase):
    """Test suite for clinical accuracy metrics and run comparison."""

    def test_exact_match_perfect_metrics(self):
        """Identical reference and hypothesis should have 0% WER and 0 critical errors."""
        ref = "Patient is taking lisinopril 10 mg once daily with blood pressure 120/80."
        hyp = "Patient is taking lisinopril 10 mg once daily with blood pressure 120/80."

        res = compute_clinical_accuracy(ref, hyp)
        self.assertEqual(res["wer_metrics"]["wer"], 0.0)
        self.assertEqual(res["wer_metrics"]["total_word_errors"], 0)
        self.assertEqual(res["critical_clinical_errors"]["total_critical_count"], 0)
        self.assertEqual(res["medication_performance"]["dosage_mismatches_count"], 0)

    def test_critical_dosage_error_detection(self):
        """Detect medication dosage difference (e.g., 10 mg -> 20 mg)."""
        ref = "Patient is taking lisinopril 10 mg once daily."
        hyp = "Patient is taking lisinopril 20 mg once daily."

        res = compute_clinical_accuracy(ref, hyp)
        self.assertGreater(res["wer_metrics"]["wer"], 0.0)
        dosage_errors = res["critical_clinical_errors"]["breakdown"]["medication_dosage_errors"]
        self.assertGreaterEqual(len(dosage_errors), 1)
        self.assertEqual(dosage_errors[0]["drug_name"], "lisinopril")

    def test_critical_vital_error_detection(self):
        """Detect blood pressure value discrepancies (138/84 -> 150/90)."""
        ref = "Blood pressure is 138 over 84."
        hyp = "Blood pressure is 150/90."

        res = compute_clinical_accuracy(ref, hyp)
        vital_errors = res["critical_clinical_errors"]["breakdown"]["vital_sign_errors"]
        self.assertGreaterEqual(len(vital_errors), 1)

    def test_compare_runs_matrix(self):
        """Test multi-run side-by-side comparison engine."""
        ref = "Patient has hypertension and takes amlodipine 5 mg daily."
        run_a = "Patient has hypertension and takes amlodipine 5 mg daily."
        run_b = "Patient has hypertension and takes amlodipine 10 mg daily."

        comparison = compare_transcription_runs(
            ref_text=ref,
            runs={"Run_A_Greedy": run_a, "Run_B_Beam": run_b}
        )

        self.assertIn("comparison_table", comparison)
        self.assertEqual(len(comparison["comparison_table"]), 2)

        table_str = format_comparison_table(comparison)
        self.assertIn("Run_A_Greedy", table_str)
        self.assertIn("Run_B_Beam", table_str)
        self.assertIn("WER (%)", table_str)


if __name__ == "__main__":
    unittest.main()
