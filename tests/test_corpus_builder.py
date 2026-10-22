#!/usr/bin/env python3
"""
Tests for the de-identifying corpus builder and PHI-safe hotword miner
(language_model/corpus_builder.py).
"""

import unittest

from language_model.corpus_builder import deidentify, tokenize_sentences, mine_hotwords
from language_model.medical_lexicon import MedicalLexicon


class TestCorpusBuilder(unittest.TestCase):
    def test_deidentify_removes_phi(self):
        txt = "Patient John Smith, SSN 123-45-6789, phone 555-123-4567, seen on 12/19/2021."
        out = deidentify(txt)
        self.assertNotIn("123-45-6789", out)
        self.assertNotIn("555-123-4567", out)
        self.assertNotIn("12/19/2021", out)
        self.assertNotIn("Smith", out)  # name after 'Patient' label removed

    def test_tokenize_sentences(self):
        sents = tokenize_sentences("Blood pressure is normal. Lungs clear bilaterally.")
        self.assertTrue(all(isinstance(s, list) for s in sents))
        self.assertIn("bilaterally", [w for s in sents for w in s])

    def test_mine_hotwords_only_medical_terms(self):
        lex = MedicalLexicon()
        corpus = [
            ["patient", "has", "hypertension", "and", "diabetes"],
            ["prescribed", "metformin", "and", "lisinopril", "today"],
            ["lungs", "clear", "bilaterally", "no", "wheezing"],
            ["patient", "has", "hypertension"],
            ["start", "metformin"],
        ]
        hot = mine_hotwords(corpus, medical_lexicon=lex, top_n=20, min_count=1)
        # Every returned hotword must be a real medical term (no PHI/junk).
        known = lex.get_all_words()
        for w in hot:
            self.assertIn(w, known)
        # Frequent clinical terms should appear.
        self.assertIn("hypertension", hot)
        self.assertIn("metformin", hot)
        # Non-medical frequent words (e.g. 'patient'/'and'/'has') must NOT leak in
        # unless they are genuinely in the lexicon.
        self.assertNotIn("and", hot)


if __name__ == "__main__":
    unittest.main()
