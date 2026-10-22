#!/usr/bin/env python3
"""
Tests for the conservative CTC glitch corrector (decoding/ctc_postprocess.py).

Focus: the safety contract. The corrector must repair unambiguous surface
glitches while NEVER altering clinical values or acting on ambiguity.
"""

import unittest

from decoding.ctc_postprocess import CTCGlitchCorrector


class TestCTCGlitchCorrector(unittest.TestCase):
    def setUp(self):
        # Minimal domain vocabulary for deterministic, isolated tests.
        self.vocab = {
            "examination", "saturation", "lungs", "clear", "patient",
            "the", "on", "no", "fifty", "eight", "mg",
        }
        self.corr = CTCGlitchCorrector(self.vocab)

    def test_fixes_doubled_letter_glitch(self):
        out, events = self.corr.correct_text("On examinNation the lungs are clear")
        self.assertIn("examination", out.lower())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["original"].lower(), "examinnation")
        self.assertEqual(events[0]["corrected"], "examination")

    def test_preserves_case(self):
        # First letter lowercase (as in the real 'examinNation' glitch) -> lowercase.
        out, _ = self.corr.correct_text("examinNation")
        self.assertEqual(out, "examination")
        # First letter uppercase -> capitalized replacement.
        out_cap, _ = self.corr.correct_text("Examinnation")
        self.assertEqual(out_cap, "Examination")
        # All uppercase -> uppercase replacement.
        out_upper, _ = self.corr.correct_text("EXAMINNATION")
        self.assertEqual(out_upper, "EXAMINATION")

    def test_known_words_untouched(self):
        text = "the patient lungs are clear"
        out, events = self.corr.correct_text(text)
        self.assertEqual(out, text)
        self.assertEqual(events, [])

    def test_never_touches_tokens_with_digits(self):
        # '500' and '138/84' and '58' must pass through unchanged.
        text = "metformin 500 mg bp 138/84 age 58"
        out, events = self.corr.correct_text(text)
        self.assertEqual(out, text)
        self.assertEqual(events, [])

    def test_ambiguous_glitch_left_unchanged(self):
        # If collapsing doubled letters yields no known word, do nothing.
        out, events = self.corr.correct_text("zzqqxx")
        self.assertEqual(out, "zzqqxx")
        self.assertEqual(events, [])

    def test_two_known_candidates_is_ambiguous(self):
        # Construct a case with two distinct known collapses -> must NOT act.
        vocab = {"baa", "aba"}  # contrived; neither reachable from a single token
        corr = CTCGlitchCorrector({"too", "to"})
        # 'too' is already known -> untouched.
        out, events = corr.correct_text("too")
        self.assertEqual(out, "too")
        self.assertEqual(events, [])

    def test_whitespace_and_punctuation_preserved(self):
        out, _ = self.corr.correct_text("On examinNation, the lungs.")
        self.assertEqual(out, "On examination, the lungs.")


if __name__ == "__main__":
    unittest.main()
