#!/usr/bin/env python3
"""Tests for app/services/speaker_assign.py (speaker turns -> labelled segments and transcript)."""

import unittest

from app.services.speaker_assign import assign_speakers, label_map, render_transcript


def _w(word, start, end):
    return {"word": " " + word, "start": start, "end": end, "probability": 0.9}


class TestLabelMap(unittest.TestCase):
    def test_numbered_by_first_appearance(self):
        turns = [(5.0, 6.0, "B"), (0.0, 4.0, "A"), (7.0, 8.0, "A")]
        self.assertEqual(label_map(turns), {"A": "Speaker 1", "B": "Speaker 2"})

    def test_empty(self):
        self.assertEqual(label_map([]), {})


class TestAssignSpeakers(unittest.TestCase):
    TURNS = [(0.0, 5.0, "A"), (5.0, 10.0, "B")]

    def test_segment_without_words_gets_majority_speaker(self):
        segs = [{"start": 1.0, "end": 4.0, "text": "How are you feeling today?"},
                {"start": 6.0, "end": 9.0, "text": "A bit better."}]
        out = assign_speakers(segs, self.TURNS)
        self.assertEqual([s["speaker"] for s in out], ["Speaker 1", "Speaker 2"])

    def test_segment_is_split_where_speaker_changes(self):
        seg = {"start": 3.0, "end": 8.0, "text": "Any pain? Yes, a little.",
               "words": [_w("Any", 3.0, 3.5), _w("pain?", 3.5, 4.5), _w("Yes,", 5.5, 6.0),
                         _w("a", 6.0, 6.2), _w("little.", 6.2, 7.0)]}
        out = assign_speakers([seg], self.TURNS)
        self.assertEqual([(s["speaker"], s["text"]) for s in out],
                         [("Speaker 1", "Any pain?"), ("Speaker 2", "Yes, a little.")])
        self.assertEqual(out[0]["end"], 4.5)
        self.assertEqual(out[1]["start"], 5.5)

    def test_overlapping_talk_cannot_cut_a_sentence(self):
        # the turn flips to B for the last two words, but A holds most of the sentence
        seg = {"start": 0.0, "end": 6.0, "text": "I am like where you at?",
               "words": [_w("I", 0.0, 0.5), _w("am", 0.5, 1.0), _w("like", 1.0, 2.0), _w("where", 2.0, 3.5),
                         _w("you", 3.5, 4.5), _w("at?", 4.5, 5.5)]}
        out = assign_speakers([seg], [(0.0, 3.8, "A"), (3.8, 9.0, "B")])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["speaker"], "Speaker 1")
        self.assertEqual(out[0]["text"], "I am like where you at?")

    def test_marker_segments_get_no_speaker(self):
        seg = {"start": 0.0, "end": 9.0, "text": "[Non-English speech, 0:00-0:09]", "non_english": True}
        out = assign_speakers([seg], self.TURNS)
        self.assertNotIn("speaker", out[0])

    def test_no_turns_leaves_segments_unlabelled(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        self.assertEqual(assign_speakers(segs, []), segs)

    def test_word_in_a_gap_takes_nearest_speaker(self):
        turns = [(0.0, 2.0, "A"), (4.0, 6.0, "B")]
        seg = {"start": 0.0, "end": 6.0, "text": "one. two.",
               "words": [_w("one.", 0.5, 1.0), _w("two.", 5.0, 5.5)]}
        out = assign_speakers([seg], turns)
        self.assertEqual([s["speaker"] for s in out], ["Speaker 1", "Speaker 2"])

    def test_inputs_not_mutated(self):
        segs = [{"start": 1.0, "end": 4.0, "text": "hi"}]
        before = [dict(s) for s in segs]
        assign_speakers(segs, self.TURNS)
        self.assertEqual(segs, before)


class TestRender(unittest.TestCase):
    def test_merges_consecutive_same_speaker_and_prefixes(self):
        segs = [{"text": "Any pain?", "speaker": "Speaker 1"}, {"text": "Any fever?", "speaker": "Speaker 1"},
                {"text": "No.", "speaker": "Speaker 2"}]
        self.assertEqual(render_transcript(segs), "Speaker 1: Any pain? Any fever?\nSpeaker 2: No.")

    def test_single_speaker_has_no_prefix(self):
        segs = [{"text": "Patient is 50.", "speaker": "Speaker 1"}, {"text": "Exam normal.", "speaker": "Speaker 1"}]
        self.assertEqual(render_transcript(segs), "Patient is 50. Exam normal.")

    def test_markers_and_unlabelled_stand_alone(self):
        segs = [{"text": "Hello.", "speaker": "Speaker 1"}, {"text": "[Non-English speech, 1:00-2:00]"},
                {"text": "Hi.", "speaker": "Speaker 2"}]
        self.assertEqual(render_transcript(segs),
                         "Speaker 1: Hello.\n[Non-English speech, 1:00-2:00]\nSpeaker 2: Hi.")

    def test_transform_sees_whole_turns(self):
        segs = [{"text": "Take it twice", "speaker": "Speaker 1"}, {"text": "a day.", "speaker": "Speaker 1"},
                {"text": "Okay.", "speaker": "Speaker 2"}]
        self.assertEqual(render_transcript(segs, transform=str.upper),
                         "Speaker 1: TAKE IT TWICE A DAY.\nSpeaker 2: OKAY.")

    def test_turn_that_cleans_to_nothing_is_dropped(self):
        segs = [{"text": "Hello.", "speaker": "Speaker 1"}, {"text": "Um...", "speaker": "Speaker 2"},
                {"text": "Bye.", "speaker": "Speaker 1"}]
        self.assertEqual(render_transcript(segs, transform=lambda t: t.replace("Um", "")),
                         "Speaker 1: Hello.\nSpeaker 1: Bye.")

    def test_translated_text_is_labelled_and_keeps_its_speaker(self):
        segs = [{"text": "Hello.", "speaker": "Speaker 1"},
                {"text": "It hurts here.", "speaker": "Speaker 2", "translated_from": "es"},
                {"text": "Since when?", "speaker": "Speaker 2", "translated_from": "es"},
                {"text": "Okay.", "speaker": "Speaker 1"}]
        self.assertEqual(render_transcript(segs),
                         "Speaker 1: Hello.\nSpeaker 2: [Translated from Spanish] It hurts here. Since when?\nSpeaker 1: Okay.")

    def test_empty(self):
        self.assertEqual(render_transcript([]), "")


if __name__ == "__main__":
    unittest.main()
