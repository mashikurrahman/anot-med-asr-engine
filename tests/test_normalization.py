#!/usr/bin/env python3
"""
Unit tests for Anot Medical ASR Medical Normalization Layer.
Tests deterministic standardization of vitals, dosages, frequencies, units,
and speaker/timestamp preservation without altering facts or meaning.
"""

import unittest
from audio.normalization import (
    parse_spoken_number,
    normalize_text,
    normalize_conversation_turns,
    MedicalNormalizer,
)


class TestMedicalNormalization(unittest.TestCase):
    """Unit tests for Stage 8H Medical Normalization Layer."""

    def setUp(self):
        self.normalizer = MedicalNormalizer()

    def test_spoken_number_parsing(self):
        """Verify spoken numbers parse accurately into integers and floats."""
        self.assertEqual(parse_spoken_number("one fifty two"), 152)
        self.assertEqual(parse_spoken_number("ninety two"), 92)
        self.assertEqual(parse_spoken_number("five hundred"), 500)
        self.assertEqual(parse_spoken_number("ten"), 10)
        self.assertEqual(parse_spoken_number("twenty"), 20)
        self.assertEqual(parse_spoken_number("one thirty eight"), 138)
        self.assertEqual(parse_spoken_number("eighty four"), 84)
        self.assertEqual(parse_spoken_number("two point five"), 2.5)
        self.assertEqual(parse_spoken_number("98"), 98)
        self.assertIsNone(parse_spoken_number("apple banana"))

    def test_blood_pressure_normalization(self):
        """1. Test blood pressure spoken phrases convert to systolic/diastolic format."""
        text = "Her blood pressure today is one fifty two over ninety two compared with previous visit."
        norm, events = self.normalizer.normalize_text(text)
        self.assertIn("152/92", norm)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["category"], "blood_pressure")
        self.assertEqual(events[0]["details"]["systolic"], 152)
        self.assertEqual(events[0]["details"]["diastolic"], 92)

        # Isolated '138 over 84'
        norm2, _ = self.normalizer.normalize_text("The reading was 138 over 84 mmHg.")
        self.assertIn("138/84", norm2)

    def test_oxygen_saturation_normalization(self):
        """2. Test oxygen saturation expressions normalize to percentage format."""
        text = "Oxygen saturation is ninety eight percent on room air."
        norm, events = self.normalizer.normalize_text(text)
        self.assertIn("oxygen saturation 98%", norm.lower())
        self.assertTrue(any(e["category"] == "oxygen_saturation" for e in events))

        text2 = "O2 sat 97 percent."
        norm2, _ = self.normalizer.normalize_text(text2)
        self.assertIn("97%", norm2)

    def test_pulse_normalization(self):
        """3. Test pulse rate expressions normalize with bpm unit."""
        text = "Her pulse is seventy four beats per minute."
        norm, events = self.normalizer.normalize_text(text)
        self.assertIn("pulse 74 bpm", norm)
        self.assertTrue(any(e["category"] == "pulse" for e in events))

    def test_medication_dosage_normalization(self):
        """4. Test medication dosages standardize number and unit."""
        cases = [
            ("metformin five hundred milligrams", "metformin 500 mg"),
            ("amlodipine ten milligrams", "amlodipine 10 mg"),
            ("twenty micrograms of levothyroxine", "20 mcg of levothyroxine"),
            ("albuterol five milliliters", "albuterol 5 mL"),
            ("aspirin eighty one milligrams", "aspirin 81 mg"),
            ("prednisone 2.5 milligrams", "prednisone 2.5 mg"),
        ]
        for raw, expected in cases:
            norm, events = self.normalizer.normalize_text(raw)
            self.assertEqual(norm, expected)
            self.assertTrue(len(events) > 0)
            self.assertEqual(events[0]["category"], "medication_dosage")

    def test_frequency_normalization(self):
        """5. Test medication frequencies normalize to canonical forms."""
        cases = [
            ("take metformin twice a day", "take metformin twice daily"),
            ("take lisinopril two times a day", "take lisinopril twice daily"),
            ("amlodipine once a day", "amlodipine once daily"),
            ("amoxicillin three times a day", "amoxicillin three times daily"),
            ("take at before bed", "take at at bedtime"),
            ("every four hours as needed", "every 4 hours as needed"),
        ]
        for raw, expected in cases:
            norm, events = self.normalizer.normalize_text(raw)
            self.assertEqual(norm, expected)
            self.assertTrue(len(events) > 0)
            self.assertEqual(events[0]["category"], "frequency")

    def test_clinical_units_normalization(self):
        """6. Test clinical units normalize to standard medical abbreviations."""
        norm, _ = self.normalizer.normalize_text("weight 75 kilograms height 175 centimeters")
        self.assertIn("75 kg", norm)
        self.assertIn("175 cm", norm)

    def test_ambiguous_number_preservation(self):
        """7. Verify isolated numbers and non-vital numbers are preserved verbatim without guessing."""
        raw = "The patient reports 78 on the scale and returns in four weeks."
        norm, events = self.normalizer.normalize_text(raw)
        # 78 and four must not be converted to vitals or altered
        self.assertEqual(norm, raw)
        self.assertEqual(len(events), 0)

    def test_speaker_and_timestamp_preservation(self):
        """8 & 9. Verify turns preserve speaker labels, timestamps, and order."""
        turns = [
            {
                "turn_id": "turn_000",
                "speaker": "SPEAKER_00",
                "start": 0.5,
                "end": 4.2,
                "duration": 3.7,
                "transcript": "Her blood pressure today is one fifty two over ninety two.",
            },
            {
                "turn_id": "turn_001",
                "speaker": "SPEAKER_01",
                "start": 4.5,
                "end": 5.8,
                "duration": 1.3,
                "transcript": "Yes doctor.",
            },
        ]
        norm_turns, events = normalize_conversation_turns(turns)
        self.assertEqual(len(norm_turns), 2)
        # Check Turn 0
        self.assertEqual(norm_turns[0]["speaker"], "SPEAKER_00")
        self.assertEqual(norm_turns[0]["start"], 0.5)
        self.assertEqual(norm_turns[0]["end"], 4.2)
        self.assertIn("152/92", norm_turns[0]["transcript"])
        # Check Turn 1
        self.assertEqual(norm_turns[1]["speaker"], "SPEAKER_01")
        self.assertEqual(norm_turns[1]["transcript"], "Yes doctor.")
        self.assertEqual(len(events), 1)

    def test_raw_transcript_preservation(self):
        """10. Verify raw_transcript field is preserved in structured turn."""
        turns = [
            {"turn_id": "turn_000", "speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "transcript": "five hundred milligrams"}
        ]
        norm_turns, _ = normalize_conversation_turns(turns)
        self.assertEqual(norm_turns[0]["raw_transcript"], "five hundred milligrams")
        self.assertEqual(norm_turns[0]["normalized_transcript"], "500 mg")

    def test_multiple_normalizations_in_single_turn(self):
        """11. Test multiple distinct normalization categories in a single turn."""
        text = (
            "Blood pressure is one fifty two over ninety two, pulse seventy four beats per minute, "
            "oxygen saturation ninety eight percent, taking amlodipine ten milligrams once a day."
        )
        norm, events = self.normalizer.normalize_text(text)
        self.assertIn("152/92", norm)
        self.assertIn("74 bpm", norm)
        self.assertIn("98%", norm)
        self.assertIn("10 mg", norm)
        self.assertIn("once daily", norm)
        # At least 4 distinct events recorded
        categories = set(e["category"] for e in events)
        self.assertTrue(len(categories) >= 4)

    def test_no_accidental_fact_changes(self):
        """12. Verify numeric facts are strictly preserved (152/92 never becomes 120/80; 500mg never 250mg)."""
        raw = "blood pressure 152/92 metformin 500 mg"
        norm, _ = self.normalizer.normalize_text(raw)
        self.assertIn("152/92", norm)
        self.assertIn("500 mg", norm)
        self.assertNotIn("120/80", norm)
        self.assertNotIn("250 mg", norm)

    def test_empty_input(self):
        """13. Test empty string and whitespace."""
        self.assertEqual(self.normalizer.normalize_text("")[0], "")
        self.assertEqual(self.normalizer.normalize_text("   ")[0], "   ")

    def test_punctuation_handling(self):
        """14. Test normalization with commas, periods, and question marks."""
        raw = "Is your blood pressure one thirty eight over eighty four? Yes, pulse is 74 beats per minute."
        norm, _ = self.normalizer.normalize_text(raw)
        self.assertIn("138/84?", norm)
        self.assertIn("74 bpm.", norm)

    def test_already_normalized_text(self):
        """15. Test already normalized text is preserved without double-normalization corruption."""
        raw = "Patient taking amlodipine 10 mg twice daily with blood pressure 138/84."
        norm, _ = self.normalizer.normalize_text(raw)
        self.assertEqual(norm, raw)


if __name__ == "__main__":
    unittest.main()
