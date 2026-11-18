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

    def test_known_noise_phrases_are_dropped(self):
        segs = [{"text": "Thank you for watching!", "no_speech_prob": 0.1, "avg_logprob": -0.3},
                {"text": "Don't forget to subscribe.", "no_speech_prob": 0.1, "avg_logprob": -0.3}]
        self.assertEqual(filter_recovered(segs), [])

    def test_kept_segments_are_flagged_recovered_and_low_confidence(self):
        out = filter_recovered([{"text": "Come here.", "no_speech_prob": 0.1, "avg_logprob": -0.5}])
        self.assertTrue(out[0]["recovered"] and out[0]["low_confidence"])


class TestKeepRecovered(unittest.TestCase):
    def test_keeps_confident_lines_from_main_speakers_only(self):
        from language_model.gap_recovery import keep_recovered
        segs = [
            {"start": 0, "end": 60, "text": "doctor talks", "speaker": "Speaker 1"},
            {"start": 60, "end": 100, "text": "patient talks", "speaker": "Speaker 2"},
            {"start": 100, "end": 102, "text": "side chat", "speaker": "Speaker 3"},
            {"start": 110, "end": 112, "text": "quiet patient", "speaker": "Speaker 2", "recovered": True, "avg_logprob": -0.4},
            {"start": 120, "end": 122, "text": "unsure words", "speaker": "Speaker 2", "recovered": True, "avg_logprob": -0.9},
            {"start": 130, "end": 132, "text": "background voice", "speaker": "Speaker 3", "recovered": True, "avg_logprob": -0.2},
            {"start": 140, "end": 142, "text": "new voice", "speaker": "Speaker 4", "recovered": True, "avg_logprob": -0.2},
        ]
        self.assertEqual([s["text"] for s in keep_recovered(segs)],
                         ["doctor talks", "patient talks", "side chat", "quiet patient"])

    def test_without_speaker_labels_only_confidence_counts(self):
        from language_model.gap_recovery import keep_recovered
        segs = [{"start": 0, "end": 10, "text": "a"},
                {"start": 20, "end": 22, "text": "b", "recovered": True, "avg_logprob": -0.3},
                {"start": 30, "end": 32, "text": "c", "recovered": True, "avg_logprob": -1.1}]
        self.assertEqual([s["text"] for s in keep_recovered(segs)], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
