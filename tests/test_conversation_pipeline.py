#!/usr/bin/env python3
"""
Unit tests for Anot Medical ASR Mode B Conversation Pipeline & Reconstructor.
Tests all deterministic logic: sorting, attribution, merging, splitting, resilience,
and transcript reconstruction using mock acoustic models.
"""

import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

from audio.conversation_pipeline import (
    split_long_segments,
    attribute_speakers_to_segments,
    reconstruct_conversation_turns,
    format_transcript_text,
    ConversationInferenceBatcher,
)


class TestConversationPipeline(unittest.TestCase):
    """Unit tests for Stage 8G Mode B pipeline components."""

    def test_chronological_sorting(self):
        """1. Verify segments out of order are sorted strictly by start timestamp."""
        unordered_segments = [
            {"segment_id": "seg_002", "start": 8.5, "end": 10.0, "speaker": "SPEAKER_00", "transcript": "third"},
            {"segment_id": "seg_000", "start": 0.5, "end": 2.0, "speaker": "SPEAKER_00", "transcript": "first"},
            {"segment_id": "seg_001", "start": 3.0, "end": 5.0, "speaker": "SPEAKER_01", "transcript": "second"},
        ]
        turns = reconstruct_conversation_turns(unordered_segments, merge_same_speaker=False)
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[0]["transcript"], "first")
        self.assertEqual(turns[1]["transcript"], "second")
        self.assertEqual(turns[2]["transcript"], "third")
        self.assertEqual(turns[0]["start"], 0.5)
        self.assertEqual(turns[1]["start"], 3.0)
        self.assertEqual(turns[2]["start"], 8.5)

    def test_speaker_attribution(self):
        """2. Verify VAD segment is attributed to the speaker with dominant temporal overlap."""
        segments = [
            {"segment_id": "seg_000", "start_time": 1.0, "end_time": 4.0, "duration": 3.0},
        ]
        diarized_turns = [
            {"speaker_id": "SPEAKER_00", "start_time": 0.0, "end_time": 3.5},  # 2.5s overlap
            {"speaker_id": "SPEAKER_01", "start_time": 3.5, "end_time": 5.0},  # 0.5s overlap
        ]
        attributed = attribute_speakers_to_segments(segments, diarized_turns)
        self.assertEqual(len(attributed), 1)
        self.assertEqual(attributed[0]["speaker"], "SPEAKER_00")

    def test_timestamp_preservation(self):
        """3. Verify timestamps are strictly preserved during reconstruction."""
        segments = [
            {"segment_id": "seg_000", "start": 1.234, "end": 4.567, "duration": 3.333, "speaker": "SPEAKER_00", "transcript": "Test speech."}
        ]
        turns = reconstruct_conversation_turns(segments, merge_same_speaker=True)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["start"], 1.234)
        self.assertEqual(turns[0]["end"], 4.567)
        self.assertEqual(turns[0]["duration"], 3.333)

    def test_same_speaker_merging(self):
        """4. Verify adjacent segments from the same speaker separated by small silence are merged."""
        segments = [
            {"segment_id": "seg_000", "start": 0.0, "end": 2.5, "speaker": "SPEAKER_00", "transcript": "Patient is 58"},
            {"segment_id": "seg_001", "start": 3.0, "end": 5.0, "speaker": "SPEAKER_00", "transcript": "with diabetes."},
        ]
        # Silence gap is 0.5s (<= 1.5s threshold)
        merged = reconstruct_conversation_turns(segments, merge_same_speaker=True, max_merge_silence_seconds=1.5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["speaker"], "SPEAKER_00")
        self.assertEqual(merged[0]["start"], 0.0)
        self.assertEqual(merged[0]["end"], 5.0)
        self.assertEqual(merged[0]["transcript"], "Patient is 58 with diabetes.")
        self.assertEqual(merged[0]["segment_ids"], ["seg_000", "seg_001"])

    def test_cross_speaker_merge_prevention(self):
        """5. Verify consecutive segments from different speakers are NEVER merged."""
        segments = [
            {"segment_id": "seg_000", "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "transcript": "How long have you had the cough?"},
            {"segment_id": "seg_001", "start": 2.1, "end": 3.5, "speaker": "SPEAKER_01", "transcript": "About three weeks."},
            {"segment_id": "seg_002", "start": 3.6, "end": 5.0, "speaker": "SPEAKER_00", "transcript": "Any shortness of breath?"},
        ]
        merged = reconstruct_conversation_turns(segments, merge_same_speaker=True, max_merge_silence_seconds=1.5)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]["speaker"], "SPEAKER_00")
        self.assertEqual(merged[1]["speaker"], "SPEAKER_01")
        self.assertEqual(merged[2]["speaker"], "SPEAKER_00")

    def test_long_silence_merge_prevention(self):
        """Verify same-speaker segments separated by long silence are NOT merged."""
        segments = [
            {"segment_id": "seg_000", "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "transcript": "Sentence one."},
            {"segment_id": "seg_001", "start": 6.0, "end": 8.0, "speaker": "SPEAKER_00", "transcript": "Sentence two."},
        ]
        # Silence gap is 4.0s (> 1.5s threshold)
        merged = reconstruct_conversation_turns(segments, merge_same_speaker=True, max_merge_silence_seconds=1.5)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["transcript"], "Sentence one.")
        self.assertEqual(merged[1]["transcript"], "Sentence two.")

    def test_long_segment_splitting(self):
        """6. Verify long segment exceeding threshold is split into bounded chunks."""
        long_segments = [
            {"segment_id": "seg_000", "start_time": 10.0, "end_time": 30.0, "duration": 20.0, "start_sample": 160000, "end_sample": 480000},
            {"segment_id": "seg_001", "start_time": 35.0, "end_time": 38.0, "duration": 3.0, "start_sample": 560000, "end_sample": 608000},
        ]
        bounded = split_long_segments(long_segments, max_duration_seconds=8.0)
        # 20.0s should split into 3 chunks (each <= 8.0s) + 1 short segment = 4 total
        self.assertEqual(len(bounded), 4)
        for b in bounded:
            self.assertLessEqual(b["duration"], 8.0)
        self.assertEqual(bounded[0]["parent_segment_id"], "seg_000")
        self.assertEqual(bounded[3]["segment_id"], "seg_001")

    def test_empty_segment_handling(self):
        """7. Verify empty audio chunks are handled gracefully without breaking pipeline."""
        mock_proc = MagicMock()
        mock_model = MagicMock()
        batcher = ConversationInferenceBatcher(
            processor=mock_proc,
            model=mock_model,
        )
        empty_wave = np.zeros(100, dtype=np.float32)  # < 160 samples (< 10ms)
        segments = [{"segment_id": "seg_empty", "start_time": 0.0, "end_time": 0.005, "duration": 0.005, "start_sample": 0, "end_sample": 80}]
        results = batcher.process_waveform(
            empty_wave,
            sample_rate=16000,
            precomputed_segments=segments,
            precomputed_diarization=[{"speaker_id": "SPEAKER_00", "start_time": 0.0, "end_time": 0.005}],
        )
        self.assertEqual(results["performance"]["empty_segments"], 1)
        self.assertEqual(results["segments"][0]["status"], "empty_audio")

    def test_failed_segment_handling(self):
        """8. Verify individual segment inference failure does not crash the encounter."""
        mock_proc = MagicMock()
        mock_proc.return_value = {"input_features": torch.zeros(1, 80, 100)}
        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("CUDA/XPU out of memory simulation")

        batcher = ConversationInferenceBatcher(
            processor=mock_proc,
            model=mock_model,
        )
        wave = np.ones(16000, dtype=np.float32) * 0.1
        segments = [{"segment_id": "seg_fail", "start_time": 0.0, "end_time": 1.0, "duration": 1.0, "start_sample": 0, "end_sample": 16000}]

        # Should NOT raise exception
        results = batcher.process_waveform(
            wave,
            sample_rate=16000,
            precomputed_segments=segments,
            precomputed_diarization=[{"speaker_id": "SPEAKER_00", "start_time": 0.0, "end_time": 1.0}],
        )
        self.assertEqual(results["performance"]["failed_segments"], 1)
        self.assertEqual(results["segments"][0]["status"], "inference_error")

    def test_ambiguous_speaker_fallback(self):
        """9. Verify ambiguous speaker overlap falls back to UNKNOWN rather than guessing."""
        segments = [
            {"segment_id": "seg_000", "start_time": 1.0, "end_time": 3.0, "duration": 2.0},
        ]
        # Ambiguous: Both speakers overlap almost equally (1.0s vs 0.95s)
        diarized_turns = [
            {"speaker_id": "SPEAKER_00", "start_time": 1.0, "end_time": 2.0},
            {"speaker_id": "SPEAKER_01", "start_time": 2.0, "end_time": 3.0},
        ]
        attributed = attribute_speakers_to_segments(segments, diarized_turns)
        self.assertEqual(attributed[0]["speaker"], "UNKNOWN")

    def test_transcript_reconstruction_formatting(self):
        """10. Verify transcript text format matches clinical encounter specification."""
        turns = [
            {"start": 0.1, "end": 4.0, "speaker": "SPEAKER_00", "transcript": "Hello, how are you today?"},
            {"start": 4.2, "end": 5.5, "speaker": "SPEAKER_01", "transcript": "I am feeling short of breath."},
        ]
        timestamped_txt = format_transcript_text(turns, include_timestamps=True)
        plain_txt = format_transcript_text(turns, include_timestamps=False)

        self.assertIn("[00:00.100 - 00:04.000] SPEAKER_00:", timestamped_txt)
        self.assertIn("Hello, how are you today?", timestamped_txt)
        self.assertIn("[00:04.200 - 00:05.500] SPEAKER_01:", timestamped_txt)

        self.assertIn("SPEAKER_00: Hello, how are you today?", plain_txt)
        self.assertIn("SPEAKER_01: I am feeling short of breath.", plain_txt)


if __name__ == "__main__":
    unittest.main()
