#!/usr/bin/env python3
"""Tests for language_model/language_spans.py (English / non-English span planning)."""

import unittest

from language_model.language_spans import build_spans, english_flags, format_marker


class TestEnglishFlags(unittest.TestCase):
    def test_threshold(self):
        self.assertEqual(english_flags([0.9, 0.9, 0.05, 0.05], threshold=0.4), [True, True, False, False])

    def test_lone_non_english_window_inside_english_is_absorbed(self):
        self.assertEqual(english_flags([0.9, 0.9, 0.1, 0.9, 0.9]), [True] * 5)

    def test_lone_english_window_inside_non_english_is_absorbed(self):
        self.assertEqual(english_flags([0.1, 0.1, 0.9, 0.1, 0.1]), [False] * 5)

    def test_quiet_windows_inherit_neighbour(self):
        self.assertEqual(english_flags([0.1, None, None, 0.1]), [False] * 4)
        self.assertEqual(english_flags([None, 0.9]), [True, True])

    def test_empty_and_all_unknown(self):
        self.assertEqual(english_flags([]), [])
        self.assertEqual(english_flags([None, None]), [True, True])


class TestBuildSpans(unittest.TestCase):
    def test_merges_runs_and_clips_to_duration(self):
        spans = build_spans([True, True, False, False, True], window_sec=15.0, total_sec=70.0)
        self.assertEqual(spans, [(0.0, 30.0, True), (30.0, 60.0, False), (60.0, 70.0, True)])

    def test_empty(self):
        self.assertEqual(build_spans([], 15.0, 0.0), [])


class TestMarker(unittest.TestCase):
    def test_minutes_and_hours(self):
        self.assertEqual(format_marker(190, 340), "[Non-English speech, 3:10-5:40]")
        self.assertEqual(format_marker(3590, 3725), "[Non-English speech, 59:50-1:02:05]")



class TestTranslationLabel(unittest.TestCase):
    def test_known_language_is_named(self):
        from language_model.language_spans import translation_label
        self.assertEqual(translation_label("es"), "[Translated from Spanish]")

    def test_unknown_code_is_shown_as_is(self):
        from language_model.language_spans import translation_label
        self.assertEqual(translation_label("xx"), "[Translated from xx]")

    def test_missing_language_is_still_labelled(self):
        from language_model.language_spans import translation_label
        self.assertEqual(translation_label(None), "[Translated from another language]")


class TestJoinWithLabels(unittest.TestCase):
    def test_label_marks_start_of_each_translated_run(self):
        from language_model.language_spans import join_with_translation_labels
        segs = [{"text": "Hello."}, {"text": "Take it twice.", "translated_from": "es"},
                {"text": "Every day.", "translated_from": "es"}, {"text": "Okay."}]
        self.assertEqual(join_with_translation_labels(segs),
                         "Hello. [Translated from Spanish] Take it twice. Every day. Okay.")

    def test_no_translation_is_a_plain_join(self):
        from language_model.language_spans import join_with_translation_labels
        self.assertEqual(join_with_translation_labels([{"text": "a"}, {"text": "b"}]), "a b")


if __name__ == "__main__":
    unittest.main()
