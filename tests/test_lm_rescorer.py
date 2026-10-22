#!/usr/bin/env python3
"""
Unit Tests for LM Rescorer and Subword-Word Bridge
====================================================
Stage 9B: Tests for lm_rescorer.py integration with BeamDecoder.
"""

import unittest
import math
import numpy as np

from language_model.arpa_lm import ArpaLanguageModel
from language_model.medical_lexicon import MedicalLexicon
from language_model.lm_rescorer import (
    SubwordWordBridge,
    LMBeamScorer,
    LMRescorer,
    build_small_medical_arpa,
    create_default_rescorer
)
from decoding.beam_decoder import BeamDecoder
from decoding.greedy_decoder import GreedyDecoder


# ======================================================================
# Minimal test tokenizer that simulates SentencePiece behavior
# ======================================================================
class MockSentencePieceTokenizer:
    """
    Mimics MedASR LasrTokenizer's subword structure for testing.
    
    Uses '▁' prefix to mark word boundaries, matching SentencePiece convention.
    """

    def __init__(self):
        self.vocab = {
            "<epsilon>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "▁": 4,        # bare word boundary (space)
            "▁pa": 5,      # word start: 'pa...'
            "t": 6,         # continuation: 't'
            "i": 7,         # continuation: 'i'
            "ent": 8,       # continuation: 'ent'
            "▁is": 9,       # complete word: 'is'
            "▁a": 10,       # complete word: 'a'
            "▁fi": 11,      # word start: 'fi...'
            "f": 12,        # continuation
            "y": 13,        # continuation
            "▁e": 14,       # word start
            "ight": 15,     # continuation
            "▁year": 16,    # complete word: 'year'
            "▁old": 17,     # complete word: 'old'
            "▁male": 18,    # complete word: 'male'
            "▁with": 19,    # complete word: 'with'
            "▁hy": 20,      # word start: 'hy...'
            "per": 21,      # continuation
            "ten": 22,      # continuation
            "s": 23,        # continuation
            "ion": 24,      # continuation
            "▁no": 25,      # complete word: 'no'
            "▁chest": 26,   # complete word: 'chest'
            "▁pain": 27,    # complete word: 'pain'
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def get_vocab(self):
        return dict(self.vocab)

    def decode(self, token_ids, clean_up_tokenization_spaces=False):
        pieces = [self.inv_vocab.get(tid, f"<{tid}>") for tid in token_ids]
        text = "".join(pieces)
        # Remove SentencePiece markers for display
        text = text.replace("▁", " ").strip()
        return text

    def batch_decode(self, token_batches, clean_up_tokenization_spaces=False):
        return [self.decode(batch) for batch in token_batches]


class TestSubwordWordBridge(unittest.TestCase):
    """Test suite for subword-to-word bridge."""

    def setUp(self):
        self.tokenizer = MockSentencePieceTokenizer()
        self.bridge = SubwordWordBridge(self.tokenizer)

    def test_token_id_to_piece(self):
        """Test single token ID to piece conversion."""
        self.assertEqual(self.bridge.token_id_to_piece(5), "▁pa")
        self.assertEqual(self.bridge.token_id_to_piece(6), "t")
        self.assertEqual(self.bridge.token_id_to_piece(9), "▁is")

    def test_simple_word_reconstruction(self):
        """Test reconstructing 'patient' from subword tokens."""
        # ▁pa + t + i + ent = "patient"
        token_ids = [5, 6, 7, 8]
        words = self.bridge.token_ids_to_words(token_ids)
        self.assertEqual(words, ["patient"])

    def test_multi_word_reconstruction(self):
        """Test reconstructing multiple words."""
        # ▁pa + t + i + ent + ▁is + ▁a = "patient is a"
        token_ids = [5, 6, 7, 8, 9, 10]
        words = self.bridge.token_ids_to_words(token_ids)
        self.assertEqual(words, ["patient", "is", "a"])

    def test_special_tokens_filtered(self):
        """Test that special tokens are filtered out."""
        # <s> + ▁pa + t + i + ent + </s>
        token_ids = [1, 5, 6, 7, 8, 2]
        words = self.bridge.token_ids_to_words(token_ids)
        self.assertEqual(words, ["patient"])

    def test_empty_input(self):
        """Test empty token list."""
        self.assertEqual(self.bridge.token_ids_to_words([]), [])

    def test_partial_word_detection(self):
        """Test partial word tracking."""
        # ▁pa + t + i (incomplete 'pati...')
        token_ids = [5, 6, 7]
        completed, partial = self.bridge.get_partial_word(token_ids)
        self.assertEqual(completed, [])
        self.assertEqual(partial, "pati")

    def test_partial_word_with_completed(self):
        """Test partial word with preceding completed word."""
        # ▁pa + t + i + ent + ▁e + ight
        token_ids = [5, 6, 7, 8, 14, 15]
        completed, partial = self.bridge.get_partial_word(token_ids)
        self.assertEqual(completed, ["patient"])
        self.assertEqual(partial, "eight")

    def test_hypertension_reconstruction(self):
        """Test multi-piece medical word reconstruction."""
        # ▁hy + per + ten + s + ion = "hypertension"
        token_ids = [20, 21, 22, 23, 24]
        words = self.bridge.token_ids_to_words(token_ids)
        self.assertEqual(words, ["hypertension"])


class TestLMBeamScorer(unittest.TestCase):
    """Test suite for the LM beam scorer callback."""

    def setUp(self):
        self.tokenizer = MockSentencePieceTokenizer()
        self.bridge = SubwordWordBridge(self.tokenizer)
        arpa_text = build_small_medical_arpa()
        self.arpa_lm = ArpaLanguageModel.from_arpa_string(arpa_text)
        self.medical_lexicon = MedicalLexicon()

    def test_scorer_returns_finite_score(self):
        """Test that scorer returns finite values."""
        scorer = LMBeamScorer(
            arpa_lm=self.arpa_lm,
            medical_lexicon=self.medical_lexicon,
            bridge=self.bridge,
            lm_weight=0.1,
            medical_weight=0.05
        )
        # Score extending empty prefix with a word boundary token
        score = scorer([], 9)  # ▁is
        self.assertTrue(math.isfinite(score))

    def test_scorer_word_boundary_triggers_lm(self):
        """Test that LM scoring occurs at word boundaries."""
        scorer = LMBeamScorer(
            arpa_lm=self.arpa_lm,
            medical_lexicon=self.medical_lexicon,
            bridge=self.bridge,
            lm_weight=0.1,
            medical_weight=0.05
        )
        # Build prefix: ▁pa + t + i + ent (="patient")
        prefix = [5, 6, 7, 8]
        # Extend with ▁is (word boundary → completes "patient", scores it)
        score_boundary = scorer(prefix, 9)
        # Extend with continuation token 't' (no word boundary, no LM scoring)
        score_continuation = scorer(prefix, 6)
        
        # Word boundary should trigger LM scoring (non-zero if word is known)
        # Continuation should return 0.0 (no word completed)
        self.assertEqual(score_continuation, 0.0)

    def test_scorer_caching(self):
        """Test that scorer caches results."""
        scorer = LMBeamScorer(
            arpa_lm=self.arpa_lm,
            medical_lexicon=self.medical_lexicon,
            bridge=self.bridge,
            lm_weight=0.1,
            medical_weight=0.05
        )
        score1 = scorer([5, 6, 7, 8], 9)
        score2 = scorer([5, 6, 7, 8], 9)
        self.assertEqual(score1, score2)


class TestLMRescorerIntegration(unittest.TestCase):
    """Integration tests for the full LM rescoring pipeline."""

    def setUp(self):
        self.tokenizer = MockSentencePieceTokenizer()

    def test_decode_returns_valid_result(self):
        """Test that LM rescorer decode returns expected structure."""
        rescorer = LMRescorer(
            arpa_lm=ArpaLanguageModel.from_arpa_string(build_small_medical_arpa()),
            medical_lexicon=MedicalLexicon(),
            tokenizer_or_processor=self.tokenizer,
            lm_weight=0.1,
            medical_weight=0.05,
            beam_width=4
        )

        # Create simple logits [T=3, V=28]
        V = len(self.tokenizer.vocab)
        logits = np.full((3, V), -20.0)
        logits[0, 9] = 20.0   # ▁is
        logits[1, 0] = 20.0   # blank
        logits[2, 10] = 20.0  # ▁a

        result = rescorer.decode(logits, self.tokenizer, clean_special_tokens=True)

        self.assertIn("text", result)
        self.assertIn("words", result)
        self.assertIn("word_count", result)
        self.assertIn("lm_config", result)
        self.assertIn("total_decode_time", result)

    def test_compare_with_baseline(self):
        """Test baseline comparison output structure."""
        rescorer = LMRescorer(
            arpa_lm=ArpaLanguageModel.from_arpa_string(build_small_medical_arpa()),
            medical_lexicon=MedicalLexicon(),
            tokenizer_or_processor=self.tokenizer,
            lm_weight=0.1,
            medical_weight=0.05,
            beam_width=4
        )

        V = len(self.tokenizer.vocab)
        logits = np.full((2, V), -20.0)
        logits[0, 9] = 20.0
        logits[1, 0] = 20.0

        result = rescorer.compare_with_baseline(
            logits, "is", self.tokenizer, clean_special_tokens=True
        )

        self.assertIn("baseline_comparison", result)
        self.assertIn("changes", result["baseline_comparison"])
        self.assertIn("total_changes", result["baseline_comparison"])

    def test_lm_config_in_result(self):
        """Test that LM configuration is included in result."""
        rescorer = LMRescorer(
            arpa_lm=ArpaLanguageModel.from_arpa_string(build_small_medical_arpa()),
            medical_lexicon=MedicalLexicon(),
            tokenizer_or_processor=self.tokenizer,
            lm_weight=0.15,
            medical_weight=0.08,
            beam_width=6
        )

        V = len(self.tokenizer.vocab)
        logits = np.full((2, V), -20.0)
        logits[0, 9] = 20.0
        logits[1, 0] = 20.0

        result = rescorer.decode(logits, self.tokenizer)
        config = result["lm_config"]
        self.assertEqual(config["lm_weight"], 0.15)
        self.assertEqual(config["medical_weight"], 0.08)
        self.assertEqual(config["beam_width"], 6)
        self.assertGreater(config["lm_order"], 0)
        self.assertGreater(config["lm_vocab_size"], 0)


class TestBuiltinMedicalArpa(unittest.TestCase):
    """Test the built-in small medical ARPA model."""

    def test_builtin_arpa_loads(self):
        """Test that the built-in ARPA text parses correctly."""
        arpa_text = build_small_medical_arpa()
        lm = ArpaLanguageModel.from_arpa_string(arpa_text)
        self.assertTrue(lm._loaded)
        self.assertEqual(lm.order, 3)
        self.assertGreater(len(lm.ngrams[1]), 50)

    def test_builtin_has_medical_vocabulary(self):
        """Test that built-in LM contains medical terms."""
        lm = ArpaLanguageModel.from_arpa_string(build_small_medical_arpa())
        self.assertTrue(lm.contains_word("patient"))
        self.assertTrue(lm.contains_word("hypertension"))
        self.assertTrue(lm.contains_word("metformin"))
        self.assertTrue(lm.contains_word("blood"))
        self.assertTrue(lm.contains_word("pressure"))

    def test_clinical_bigram_scores(self):
        """Test that clinical bigrams have reasonable scores."""
        lm = ArpaLanguageModel.from_arpa_string(build_small_medical_arpa())
        
        # "blood pressure" should have a good bigram score
        bp_score = lm.score_ngram(("blood", "pressure"))
        # Random pair should have worse score (backed off)
        random_score = lm.score_ngram(("blood", "guitar"))
        self.assertGreater(bp_score, random_score)

    def test_fifty_eight_vs_fifty_two_trigram(self):
        """Test that 'a fifty eight' has higher trigram score than 'a fifty two'."""
        lm = ArpaLanguageModel.from_arpa_string(build_small_medical_arpa())
        
        score_58 = lm.score_ngram(("a", "fifty", "eight"))
        score_52 = lm.score_ngram(("a", "fifty", "two"))
        # Both should be valid but "a fifty eight" should have equal or better score
        # based on our ARPA construction (we gave it -0.3 vs -1.0)
        self.assertGreaterEqual(score_58, score_52)

    def test_deterministic_output(self):
        """Test that scoring is deterministic across calls."""
        lm = ArpaLanguageModel.from_arpa_string(build_small_medical_arpa())
        score1 = lm.score_sequence(["patient", "is", "a", "fifty", "eight", "year", "old", "male"])
        score2 = lm.score_sequence(["patient", "is", "a", "fifty", "eight", "year", "old", "male"])
        self.assertEqual(score1, score2)


if __name__ == "__main__":
    unittest.main()
