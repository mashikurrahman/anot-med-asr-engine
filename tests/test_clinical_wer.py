#!/usr/bin/env python3
"""
Tests for the clinically-normalized WER metric
(evaluation/evaluate_transcript.compute_clinical_normalized_wer).

Focus: clinically-identical surface forms are not penalised, the speaker's own
abandoned false-starts are not counted, but genuine fact changes (wrong age,
wrong dose) are STILL counted as errors.
"""

import unittest

from evaluation.evaluate_transcript import (
    compute_wer_and_alignment,
    compute_clinical_normalized_wer,
    strip_abandoned_false_starts,
)


class TestClinicalNormalizedWER(unittest.TestCase):
    def test_unit_equivalence_not_penalised(self):
        ref = "metformin 500 milligrams twice daily"
        hyp = "metformin 500 mg twice daily"
        raw = compute_wer_and_alignment(ref, hyp)
        clin = compute_clinical_normalized_wer(ref, hyp)
        self.assertGreater(raw["total_errors"], 0)       # raw penalises mg vs milligrams
        self.assertEqual(clin["total_errors"], 0)        # clinically identical

    def test_blood_pressure_and_percent_equivalence(self):
        ref = "blood pressure 138 over 84 oxygen saturation 97 percent"
        hyp = "blood pressure 138/84 oxygen saturation 97%"
        clin = compute_clinical_normalized_wer(ref, hyp)
        self.assertEqual(clin["total_errors"], 0)

    def test_disfluency_stripped(self):
        ref = "start lisinopril 10 milligrams"
        hyp = "start isupro- lisinopril 10 mg"
        clin = compute_clinical_normalized_wer(ref, hyp)
        self.assertEqual(clin["total_errors"], 0)

    def test_wrong_age_still_counts_as_error(self):
        # Fact change must NEVER be masked by normalization.
        ref = "the patient is a 58 year old male"
        hyp = "the patient is a 50 year old male"
        clin = compute_clinical_normalized_wer(ref, hyp)
        self.assertGreaterEqual(clin["substitutions"], 1)
        self.assertGreater(clin["total_errors"], 0)

    def test_wrong_dose_still_counts_as_error(self):
        ref = "lisinopril 10 milligrams"
        hyp = "lisinopril 20 mg"
        clin = compute_clinical_normalized_wer(ref, hyp)
        self.assertGreater(clin["total_errors"], 0)

    def test_strip_abandoned_false_starts_keeps_target(self):
        out = strip_abandoned_false_starts("ator- atorvastatin 20 mg")
        self.assertNotIn("ator-", out)
        self.assertIn("atorvastatin", out)

    def test_spoken_decimal_equivalence(self):
        # Spoken decimals must match their digit form (model usually emits digits).
        ref = "temperature is ninety eight point six degrees fahrenheit"
        hyp = "temperature is 98.6 degrees fahrenheit"
        clin = compute_clinical_normalized_wer(ref, hyp)
        self.assertEqual(clin["total_errors"], 0)

    def test_multiword_integer_equivalence(self):
        ref = "patient is a forty five year old male"
        hyp = "patient is a 45-year-old male"
        clin = compute_clinical_normalized_wer(ref, hyp)
        self.assertEqual(clin["total_errors"], 0)

    def test_unit_phrase_equivalence(self):
        ref = "blood glucose is one hundred fourteen milligrams per deciliter"
        hyp = "blood glucose is 114 mg/dL"
        clin = compute_clinical_normalized_wer(ref, hyp)
        self.assertEqual(clin["total_errors"], 0)

    def test_wrong_decimal_value_still_counts(self):
        # A genuine value error must survive number normalization.
        ref = "temperature is ninety eight point six degrees"
        hyp = "temperature is 98.5 degrees"
        clin = compute_clinical_normalized_wer(ref, hyp)
        self.assertGreater(clin["total_errors"], 0)

    def test_methodology_is_reported(self):
        clin = compute_clinical_normalized_wer("97 percent", "97%")
        self.assertIn("methodology", clin)
        self.assertTrue(clin["methodology"]["symmetric_clinical_normalization"])


if __name__ == "__main__":
    unittest.main()
