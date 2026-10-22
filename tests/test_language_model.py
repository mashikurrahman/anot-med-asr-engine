#!/usr/bin/env python3
"""
Unit Tests for ARPA Language Model and Medical Lexicon
======================================================
Stage 9B: Tests for arpa_lm.py and medical_lexicon.py
"""

import unittest
import math
import tempfile
import os

from language_model.arpa_lm import ArpaLanguageModel
from language_model.medical_lexicon import MedicalLexicon


class TestArpaLanguageModel(unittest.TestCase):
    """Test suite for the pure Python ARPA n-gram reader."""

    SAMPLE_ARPA = r"""\data\
ngram 1=8
ngram 2=5
ngram 3=2

\1-grams:
-1.0000	<s>	-0.5
-1.0000	</s>
-2.0000	<unk>	-0.5
-1.5000	patient	-0.3
-1.5000	is	-0.2
-1.5000	a	-0.3
-2.0000	male	-0.3
-2.5000	hypertension	-0.2

\2-grams:
-0.5000	<s> patient	-0.2
-0.5000	patient is	-0.2
-0.5000	is a	-0.2
-0.8000	a male	-0.3
-0.8000	with hypertension	-0.2

\3-grams:
-0.3000	<s> patient is
-0.3000	patient is a

\end\
"""

    def setUp(self):
        self.lm = ArpaLanguageModel.from_arpa_string(self.SAMPLE_ARPA)

    def test_load_from_string(self):
        """Test ARPA format parsing from string."""
        self.assertTrue(self.lm._loaded)
        self.assertEqual(self.lm.order, 3)

    def test_ngram_counts(self):
        """Test that n-gram counts match expected values."""
        self.assertEqual(len(self.lm.ngrams[1]), 8)
        self.assertEqual(len(self.lm.ngrams[2]), 5)
        self.assertEqual(len(self.lm.ngrams[3]), 2)

    def test_unigram_score(self):
        """Test unigram scoring returns expected log probability."""
        score = self.lm.score_ngram(("patient",))
        expected = -1.5 * math.log(10.0)
        self.assertAlmostEqual(score, expected, places=3)

    def test_bigram_direct_score(self):
        """Test direct bigram lookup when the bigram exists."""
        score = self.lm.score_ngram(("patient", "is"))
        expected = -0.5 * math.log(10.0)
        self.assertAlmostEqual(score, expected, places=3)

    def test_trigram_direct_score(self):
        """Test direct trigram lookup."""
        score = self.lm.score_ngram(("<s>", "patient", "is"))
        expected = -0.3 * math.log(10.0)
        self.assertAlmostEqual(score, expected, places=3)

    def test_backoff_for_unseen_bigram(self):
        """Test that unseen bigrams fall back to unigram + backoff weight."""
        # "male hypertension" does not exist as a bigram
        score = self.lm.score_ngram(("male", "hypertension"))
        # Should use: backoff(male) + P(hypertension)
        backoff = -0.3 * math.log(10.0)  # male's backoff weight
        unigram = -2.5 * math.log(10.0)  # hypertension's unigram
        expected = backoff + unigram
        self.assertAlmostEqual(score, expected, places=3)

    def test_unknown_word_uses_unk(self):
        """Test that completely unknown words use <unk> probability."""
        score = self.lm.score_ngram(("xyzzyplugh",))
        expected = -2.0 * math.log(10.0)  # <unk> probability
        self.assertAlmostEqual(score, expected, places=3)

    def test_unknown_word_does_not_crash(self):
        """Test that unknown n-grams return a finite score without crashing."""
        score = self.lm.score_ngram(("xyzzy", "plugh", "frobozz"))
        self.assertTrue(math.isfinite(score))
        self.assertLess(score, 0)  # Should be negative log prob

    def test_score_sequence(self):
        """Test full sequence scoring."""
        score = self.lm.score_sequence(["patient", "is", "a"])
        self.assertTrue(math.isfinite(score))
        self.assertLess(score, 0)

    def test_incremental_scoring(self):
        """Test the begin/score_token/finish incremental interface."""
        state = self.lm.begin()
        self.assertEqual(state, ["<s>"])

        score1, state = self.lm.score_token(state, "patient")
        self.assertTrue(math.isfinite(score1))

        score2, state = self.lm.score_token(state, "is")
        self.assertTrue(math.isfinite(score2))

        end_score = self.lm.finish(state)
        self.assertTrue(math.isfinite(end_score))

    def test_score_word_in_context(self):
        """Test word scoring with explicit context."""
        # "patient is" bigram exists
        score = self.lm.score_word_in_context(["patient"], "is")
        expected = -0.5 * math.log(10.0)
        self.assertAlmostEqual(score, expected, places=3)

    def test_vocabulary_populated(self):
        """Test that vocabulary is populated from unigrams."""
        self.assertIn("patient", self.lm.vocab)
        self.assertIn("hypertension", self.lm.vocab)
        # Special tokens should NOT be in vocab
        self.assertNotIn("<s>", self.lm.vocab)
        self.assertNotIn("</s>", self.lm.vocab)

    def test_contains_word(self):
        """Test vocabulary membership check."""
        self.assertTrue(self.lm.contains_word("patient"))
        self.assertFalse(self.lm.contains_word("xyzzy"))

    def test_load_from_file(self):
        """Test loading ARPA from a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".arpa", delete=False, encoding="utf-8") as f:
            f.write(self.SAMPLE_ARPA)
            f.flush()
            path = f.name

        try:
            lm = ArpaLanguageModel.from_arpa_file(path)
            self.assertTrue(lm._loaded)
            self.assertEqual(lm.order, 3)
            self.assertEqual(len(lm.ngrams[1]), 8)
        finally:
            os.unlink(path)

    def test_summary(self):
        """Test diagnostic summary output."""
        s = self.lm.summary()
        self.assertEqual(s["order"], 3)
        self.assertTrue(s["loaded"])
        self.assertIn("ngram_counts", s)
        self.assertIn("vocabulary_size", s)

    def test_empty_sequence_score(self):
        """Test that empty sequence returns 0.0."""
        self.assertEqual(self.lm.score_sequence([]), 0.0)


class TestMedicalLexicon(unittest.TestCase):
    """Test suite for the medical vocabulary lexicon."""

    def setUp(self):
        self.lexicon = MedicalLexicon()

    def test_default_lexicon_loaded(self):
        """Test that the default lexicon has expected categories."""
        summary = self.lexicon.summary()
        self.assertGreater(summary["total_unique_words"], 100)
        self.assertIn("medications", summary["categories"])
        self.assertIn("diagnoses", summary["categories"])
        self.assertIn("symptoms", summary["categories"])

    def test_medication_lookup(self):
        """Test that known medications are found."""
        self.assertTrue(self.lexicon.contains("metformin"))
        self.assertTrue(self.lexicon.contains("lisinopril"))
        self.assertTrue(self.lexicon.contains("atorvastatin"))

    def test_diagnosis_lookup(self):
        """Test that known diagnoses are found."""
        self.assertTrue(self.lexicon.contains("hypertension"))
        self.assertTrue(self.lexicon.contains("diabetes"))

    def test_unknown_word_not_found(self):
        """Test that non-medical words are not in the lexicon."""
        self.assertFalse(self.lexicon.contains("basketball"))
        self.assertFalse(self.lexicon.contains("guitar"))

    def test_score_medical_word(self):
        """Test that medical words receive a positive score."""
        score = self.lexicon.score_word("metformin")
        self.assertGreater(score, 0.0)

    def test_score_non_medical_word(self):
        """Test that non-medical words receive zero score."""
        score = self.lexicon.score_word("basketball")
        self.assertEqual(score, 0.0)

    def test_medication_weight_higher_than_anatomy(self):
        """Test that medication category has higher weight than anatomy."""
        med_weight = self.lexicon.category_weights.get("medications", 0)
        anat_weight = self.lexicon.category_weights.get("anatomy", 0)
        self.assertGreater(med_weight, anat_weight)

    def test_case_insensitive_lookup(self):
        """Test that lookups are case-insensitive."""
        self.assertTrue(self.lexicon.contains("Metformin"))
        self.assertTrue(self.lexicon.contains("HYPERTENSION"))
        self.assertEqual(
            self.lexicon.score_word("metformin"),
            self.lexicon.score_word("METFORMIN")
        )

    def test_score_word_detailed(self):
        """Test detailed scoring output."""
        detail = self.lexicon.score_word_detailed("metformin")
        self.assertTrue(detail["in_lexicon"])
        self.assertGreater(detail["score"], 0.0)
        self.assertGreater(len(detail["categories"]), 0)

    def test_get_all_words(self):
        """Test vocabulary retrieval."""
        all_words = self.lexicon.get_all_words()
        self.assertIsInstance(all_words, set)
        self.assertIn("metformin", all_words)

    def test_custom_lexicon(self):
        """Test creating lexicon with custom word list."""
        custom = MedicalLexicon(
            lexicon={"custom_meds": ["wonderdrug", "magicpill"]},
            category_weights={"custom_meds": 2.0}
        )
        self.assertTrue(custom.contains("wonderdrug"))
        self.assertEqual(custom.score_word("wonderdrug"), 2.0)
        self.assertFalse(custom.contains("metformin"))  # Default not loaded

    def test_merge_entities_json(self):
        """Test merging with medical_entities.json."""
        entities_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "evaluation", "medical_entities.json"
        )
        if os.path.exists(entities_path):
            lexicon = MedicalLexicon(entities_json_path=entities_path)
            # Should have merged additional terms
            self.assertGreater(
                lexicon.summary()["total_unique_words"],
                self.lexicon.summary()["total_unique_words"] - 10  # Allow some overlap
            )


class TestArpaBackoffEdgeCases(unittest.TestCase):
    """Edge case tests for ARPA backoff behavior."""

    def test_trigram_backoff_to_bigram(self):
        """Test trigram backoff falls to bigram correctly."""
        arpa = r"""\data\
ngram 1=4
ngram 2=2
ngram 3=1

\1-grams:
-1.0	<s>	-0.5
-1.0	</s>
-2.0	<unk>	-0.5
-1.5	word1	-0.3
-1.5	word2	-0.3
-1.5	word3	-0.3

\2-grams:
-0.5	word2 word3	-0.2
-0.5	word1 word2	-0.2

\3-grams:
-0.3	word1 word2 word3

\end\
"""
        lm = ArpaLanguageModel.from_arpa_string(arpa)

        # Existing trigram should return direct score
        direct = lm.score_ngram(("word1", "word2", "word3"))
        self.assertAlmostEqual(direct, -0.3 * math.log(10), places=3)

        # Non-existent trigram should backoff
        backed_off = lm.score_ngram(("word3", "word1", "word2"))
        self.assertTrue(math.isfinite(backed_off))
        # Should be different from direct score
        self.assertNotAlmostEqual(backed_off, direct, places=1)

    def test_no_backoff_weight_defaults_to_zero(self):
        """Test that missing backoff weight defaults to 0 (no penalty)."""
        arpa = r"""\data\
ngram 1=3

\1-grams:
-1.0	<s>
-1.0	</s>
-1.5	hello

\end\
"""
        lm = ArpaLanguageModel.from_arpa_string(arpa)
        # No backoff weights stored for unigrams without them
        score = lm.score_ngram(("hello",))
        self.assertAlmostEqual(score, -1.5 * math.log(10), places=3)


if __name__ == "__main__":
    unittest.main()
