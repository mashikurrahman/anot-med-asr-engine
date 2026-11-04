#!/usr/bin/env python3
"""Tests for language_model/gap_recovery.py (find loud stretches the first pass left untranscribed)."""

import unittest

from language_model.gap_recovery import find_gaps, filter_recovered


def level(n, loud=()):
    """n seconds of quiet (0.001) with the given (start, end) ranges loud (0.02)."""
    out = [0.001] * n
    for a, b in loud:
        for i in range(a, b):
            out[i] = 0.02
    return out


class TestFindGaps(unittest.TestCase):
    def test_loud_uncovered_stretch_is_found(self):
        segs = [{"start": 0.0, "end": 20.0}, {"start": 60.0, "end": 80.0}]
        lv = level(80, loud=[(0, 20), (25, 55), (60, 80)])
        self.assertEqual(find_gaps(segs, lv, 80.0), [(25.0, 55.0)])

    def test_quiet_gap_is_ignored(self):
        segs = [{"start": 0.0, "end": 20.0}, {"start": 60.0, "end": 80.0}]
        self.assertEqual(find_gaps(segs, level(80, loud=[(0, 20), (60, 80)]), 80.0), [])

    def test_short_loud_burst_is_ignored(self):
        segs = [{"start": 0.0, "end": 20.0}, {"start": 60.0, "end": 80.0}]
        lv = level(80, loud=[(0, 20), (30, 33), (60, 80)])
        self.assertEqual(find_gaps(segs, lv, 80.0), [])

    def test_small_dip_inside_a_loud_stretch_does_not_split_it(self):
        segs = [{"start": 0.0, "end": 20.0}]
        lv = level(80, loud=[(0, 20), (30, 40), (42, 60)])
        self.assertEqual(find_gaps(segs, lv, 80.0), [(30.0, 60.0)])

    def test_nothing_transcribed_means_no_reference_level(self):
        self.assertEqual(find_gaps([], level(30, loud=[(0, 30)]), 30.0), [])


class TestFilterRecovered(unittest.TestCase):
    def test_drops_likely_noise_and_repeats(self):
        segs = [
            {"text": "Open your mouth.", "no_speech_prob": 0.1, "avg_logprob": -0.4},
            {"text": "Thanks for watching.", "no_speech_prob": 0.9, "avg_logprob": -0.4},
            {"text": "mm mm mm", "no_speech_prob": 0.1, "avg_logprob": -1.8},
            {"text": "Open your mouth.", "no_speech_prob": 0.1, "avg_logprob": -0.4},
        ]
        self.assertEqual([s["text"] for s in filter_recovered(segs)], ["Open your mouth."])

    def test_kept_segments_are_flagged_recovered_and_low_confidence(self):
        out = filter_recovered([{"text": "Come here.", "no_speech_prob": 0.1, "avg_logprob": -0.5}])
        self.assertTrue(out[0]["recovered"] and out[0]["low_confidence"])


if __name__ == "__main__":
    unittest.main()
