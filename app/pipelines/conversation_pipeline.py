#!/usr/bin/env python3
"""
Anot Medical ASR — Mode B Ambient Conversation Pipeline
========================================================
Stage 11: Production wrapper for multi-speaker doctor-patient encounter pipeline.
VAD Segmentation -> Bounded Chunking -> MedASR -> Chronological Turn Reconstruction -> Normalization.
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np

from app.models.medasr_model import MedASRModel
from audio.conversation_pipeline import ConversationInferenceBatcher


class ConversationPipeline:
    """End-to-end Mode B pipeline for multi-speaker ambient doctor-patient conversation."""

    def __init__(self, model: MedASRModel):
        self.model = model

    def process(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        max_chunk_sec: float = 8.0,
        max_merge_silence_sec: float = 1.5,
        normalize: bool = True,
        segmentation_strategy: str = "fixed_duration",
        search_window_sec: float = 0.5,
        edge_context_sec: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Execute conversation pipeline on 16 kHz mono waveform.
        Returns chronological transcript, speaker-attributed turns, and performance metrics.
        """
        audio_duration = len(waveform) / sample_rate
        t_pipeline_start = time.perf_counter()

        # Instantiate batcher reusing loaded MedASR instance
        batcher = ConversationInferenceBatcher(
            model_id=self.model._model_id,
            device_str=self.model.get_device(),
            processor=self.model.processor,
            model=self.model._model,
            max_chunk_duration_seconds=max_chunk_sec,
            max_merge_silence_seconds=max_merge_silence_sec,
            segmentation_strategy=segmentation_strategy,
            search_window_seconds=search_window_sec,
            edge_context_seconds=edge_context_sec,
        )
        batcher._models_loaded = True

        # Process waveform through proven Stage 8G batcher
        res = batcher.process_waveform(waveform, sample_rate=sample_rate, normalize=normalize)

        perf = res["performance"]
        turns = res["normalized_turns"] if (normalize and res.get("normalized_turns")) else res["merged_turns"]

        # Build clean output segments with anonymous speaker IDs
        segments: List[Dict[str, Any]] = []
        for t in turns:
            segments.append({
                "start": round(t["start"], 2),
                "end": round(t["end"], 2),
                "speaker": t.get("speaker", t.get("speaker_id", "SPEAKER_UNKNOWN")),
                "text": t.get("transcript", t.get("text", ""))
            })

        # Final plain chronological transcript
        full_transcript = " ".join(s["text"] for s in segments if s["text"]).strip()

        total_time = time.perf_counter() - t_pipeline_start
        rtf = round(total_time / audio_duration, 4) if audio_duration > 0 else 0.0

        return {
            "mode": "conversation",
            "duration_seconds": round(audio_duration, 4),
            "transcript": full_transcript,
            "segments": segments,
            "turns_count": len(segments),
            "decoder": {
                "type": "greedy",
                "beam_width": None,
                "language_model": False
            },
            "normalization_enabled": normalize,
            "normalization_events_count": perf.get("normalization_events_count", 0),
            "performance": {
                "inference_seconds": perf.get("total_inference_time_seconds", 0.0),
                "decoding_seconds": 0.0,
                "normalization_seconds": perf.get("normalization_time_seconds", 0.0),
                "total_seconds": round(total_time, 4),
                "rtf": rtf
            }
        }
