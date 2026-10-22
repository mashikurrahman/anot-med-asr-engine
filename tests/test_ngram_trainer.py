#!/usr/bin/env python3
"""
Tests for the pure-Python n-gram trainer and ARPA round-trip
(language_model/ngram_trainer.py -> language_model/arpa_lm.py).
"""

import unittest

from language_model.ngram_trainer import NGramTrainer, train_arpa
from language_model.arpa_lm import ArpaLanguageModel


class TestNGramTrainer(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            ["the", "patient", "has", "hypertension"],
            ["the", "patient", "has", "diabetes"],
            ["the", "patient", "denies", "chest", "pain"],
            ["blood", "pressure", "is", "normal"],
            ["the", "patient", "has", "hypertension"],
        ]

    def test_summary_counts(self):
        tr = NGramTrainer(order=3, min_count=1).fit(self.corpus)
        s = tr.summary()
        self.assertEqual(s["order"], 3)
        self.assertGreater(s["vocab_size"], 0)
        self.assertGreater(s["ngrams_2"], 0)

    def test_arpa_roundtrip_loads(self):
        arpa = train_arpa(self.corpus, order=3, min_count=1)
        self.assertIn("\\data\\", arpa)
        self.assertIn("\\1-grams:", arpa)
        self.assertIn("\\end\\", arpa)
        lm = ArpaLanguageModel.from_arpa_string(arpa)
        self.assertTrue(lm._loaded)
        self.assertEqual(lm.order, 3)
        # A frequent word should be in vocab.
        self.assertIn("patient", lm.vocab)

    def test_frequent_sequence_scores_higher(self):
        arpa = train_arpa(self.corpus, order=3, min_count=1)
        lm = ArpaLanguageModel.from_arpa_string(arpa)
        # "the patient has" (seen 3x) should score >= a rare/unseen sequence.
        seen = lm.score_sequence(["the", "patient", "has", "hypertension"])
        unseen = lm.score_sequence(["chest", "pain", "diabetes", "normal"])
        self.assertGreater(seen, unseen)

    def test_probabilities_are_negative_logs(self):
        arpa = train_arpa(self.corpus, order=2, min_count=1)
        lm = ArpaLanguageModel.from_arpa_string(arpa)
        # ln prob of any sequence must be <= 0.
        self.assertLessEqual(lm.score_sequence(["the", "patient"]), 0.0)


if __name__ == "__main__":
    unittest.main()
