#!/usr/bin/env python3
"""
Unit Tests for MedASR Decoding Subsystem
========================================
Stage 9: Tests for GreedyDecoder and BeamDecoder.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

from decoding.greedy_decoder import GreedyDecoder
from decoding.beam_decoder import BeamDecoder


class DummyTokenizer:
    """Deterministic dummy tokenizer for testing decoder token conversions."""

    def __init__(self):
        self.vocab = {
            "<epsilon>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "hello": 4,
            "world": 5,
            "med": 6,
            "asr": 7
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def decode(self, token_ids, clean_up_tokenization_spaces=False):
        words = [self.inv_vocab.get(i, f"[{i}]") for i in token_ids]
        return " ".join(words)

    def batch_decode(self, token_batches, clean_up_tokenization_spaces=False):
        return [self.decode(batch) for batch in token_batches]


class TestDecodingSubsystem(unittest.TestCase):
    """Test suite for Greedy and Beam CTC decoders."""

    def setUp(self):
        self.greedy_decoder = GreedyDecoder(blank_id=0)
        self.beam_decoder = BeamDecoder(blank_id=0, beam_width=4)
        self.tokenizer = DummyTokenizer()

    def test_greedy_token_collapse_and_blank_removal(self):
        """Test CTC token collapse and blank elimination in GreedyDecoder."""
        raw_ids = [0, 4, 4, 0, 5, 5, 5, 0, 0, 6, 7, 7, 0]
        # Expected: [4, 5, 6, 7]
        collapsed = self.greedy_decoder.decode_token_ids(raw_ids)
        self.assertEqual(collapsed, [4, 5, 6, 7])

    def test_greedy_empty_sequence(self):
        """Test greedy decoding on all-blank frame sequence."""
        all_blanks = [0, 0, 0, 0]
        collapsed = self.greedy_decoder.decode_token_ids(all_blanks)
        self.assertEqual(collapsed, [])

    def test_greedy_decode_logits(self):
        """Test full greedy logits decoding pipeline."""
        # Create synthetic logits [T=5, V=8]
        # Frames:
        # t=0: token 4 ("hello")
        # t=1: token 4 ("hello" - should collapse)
        # t=2: token 0 (blank)
        # t=3: token 5 ("world")
        # t=4: token 0 (blank)
        logits = np.full((5, 8), -10.0)
        logits[0, 4] = 10.0
        logits[1, 4] = 10.0
        logits[2, 0] = 10.0
        logits[3, 5] = 10.0
        logits[4, 0] = 10.0

        res = self.greedy_decoder.decode_logits(logits, self.tokenizer)
        self.assertEqual(res["token_ids"], [4, 5])
        self.assertIn("hello", res["text"])
        self.assertIn("world", res["text"])
        self.assertEqual(res["frame_count"], 5)

    def test_beam_search_identical_to_greedy_on_unambiguous_logits(self):
        """On high-confidence unambiguous logits, beam search must match greedy."""
        logits = np.full((4, 8), -20.0)
        logits[0, 4] = 20.0  # hello
        logits[1, 0] = 20.0  # blank
        logits[2, 5] = 20.0  # world
        logits[3, 0] = 20.0  # blank

        g_res = self.greedy_decoder.decode_logits(logits, self.tokenizer)
        b_res = self.beam_decoder.decode_logits(logits, self.tokenizer, beam_width=4)

        self.assertEqual(g_res["token_ids"], b_res["token_ids"])
        self.assertEqual(g_res["token_ids"], [4, 5])
        self.assertIn("hello", b_res["text"])
        self.assertIn("world", b_res["text"])

    def test_beam_search_repeated_token_with_blank_separation(self):
        """Verify that identical tokens separated by a blank are not collapsed."""
        # t=0: token 4, t=1: blank 0, t=2: token 4
        logits = np.full((3, 8), -20.0)
        logits[0, 4] = 20.0
        logits[1, 0] = 20.0
        logits[2, 4] = 20.0

        b_res = self.beam_decoder.decode_logits(logits, self.tokenizer, beam_width=4)
        # Should be [4, 4] (not collapsed because separated by blank)
        self.assertEqual(b_res["token_ids"], [4, 4])

    def test_beam_search_top_beams_generation(self):
        """Verify that beam search returns multiple scored hypothesis beams."""
        logits = np.full((2, 8), 0.0)
        logits[0, 4] = 5.0
        logits[0, 6] = 4.0
        logits[1, 5] = 5.0

        b_res = self.beam_decoder.decode_logits(logits, self.tokenizer, beam_width=4)
        self.assertIn("top_beams", b_res)
        self.assertGreater(len(b_res["top_beams"]), 1)
        # Top beam score must be >= second beam score
        self.assertGreaterEqual(b_res["top_beams"][0]["score"], b_res["top_beams"][1]["score"])

    def test_beam_search_lm_scoring_callback(self):
        """Test language model scorer integration in beam search."""
        # Biased scoring callback that strongly prefers token 7 ("asr") over token 5 ("world")
        def mock_lm_scorer(prefix, next_token):
            if next_token == 7:
                return 10.0
            return 0.0

        beam_lm = BeamDecoder(blank_id=0, beam_width=4, lm_weight=2.0, lm_scorer=mock_lm_scorer)
        logits = np.full((2, 8), 0.0)
        logits[0, 5] = 2.0  # world
        logits[0, 7] = 1.9  # asr (slightly lower acoustic logit, but LM gives +10)
        logits[1, 0] = 5.0  # blank

        b_res = beam_lm.decode_logits(logits, self.tokenizer)
        self.assertEqual(b_res["token_ids"], [7])


if __name__ == "__main__":
    unittest.main()
