#!/usr/bin/env python3
"""
Anot Medical ASR — Whisper Transcription Pipeline
=================================================
Stage 21: End-to-end pipeline for the Whisper large-v3 engine. Handles both
dictation and conversational patient-visit audio (Whisper is robust to both).

    Audio waveform -> Whisper large-v3 -> Medical Normalization (optional)

The medical normalization step is the SAME deterministic, fact-preserving layer
used by the MedASR path, so clinical representation (dosages, vitals) is
standardized consistently regardless of engine.
"""

import time
from typing import Any, Dict, Optional

import numpy as np

from app.core.config import settings
from app.models.whisper_model import WhisperASRModel
from app.services.normalization_service import NormalizationService
from audio.enhancement import enhance_waveform


class WhisperPipeline:
    """End-to-end transcription pipeline for the Whisper engine."""

    def __init__(self, model: WhisperASRModel, normalization_service: NormalizationService):
        self.model = model
        self.normalization_service = normalization_service

    def process(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        normalize: bool = True,
        mode: str = "conversation",
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        audio_duration = len(waveform) / sample_rate
        t_start = time.perf_counter()

        # Audio front-end enhancement (label-free): loudness normalization +
        # high-pass. Makes quiet/rumbly visit audio easier to recognise without
        # altering speech content.
        enhance_time = 0.0
        proc_wave = waveform
        if settings.AUDIO_ENHANCE:
            t_enh = time.perf_counter()
            proc_wave = enhance_waveform(
                waveform,
                sample_rate,
                highpass=settings.AUDIO_HIGHPASS,
                highpass_cutoff_hz=settings.AUDIO_HIGHPASS_HZ,
                loudness_norm=settings.AUDIO_LOUDNESS_NORM,
                target_dbfs=settings.AUDIO_TARGET_DBFS,
                denoise=settings.AUDIO_DENOISE,
            )
            enhance_time = time.perf_counter() - t_enh

        raw_text, raw_segments, inference_time = self.model.transcribe(
            proc_wave, sample_rate, language=language)
        detected_language = getattr(self.model, "_last_detected_language", language or "en")

        norm_time = 0.0
        final_text = raw_text
        segments = raw_segments
        if normalize and raw_text:
            final_text, _events, norm_time = self.normalization_service.normalize_text(raw_text)
            # Normalize each segment's text too, so segment views stay consistent.
            norm_segments = []
            for seg in raw_segments:
                seg_text, _ev, _t = self.normalization_service.normalize_text(seg["text"])
                norm_segments.append({**seg, "text": seg_text})
            segments = norm_segments

        if not segments and final_text:
            segments = [{"start": 0.0, "end": round(audio_duration, 2), "text": final_text}]

        total_time = time.perf_counter() - t_start
        rtf = round(total_time / audio_duration, 4) if audio_duration > 0 else 0.0

        # Confidence summary for downstream routing (review / selective fallback).
        low_conf = [s for s in segments if s.get("low_confidence")]
        confidence = {
            "segments_total": len(segments),
            "low_confidence_segments": len(low_conf),
            "low_confidence_ratio": round(len(low_conf) / len(segments), 3) if segments else 0.0,
        }

        return {
            "mode": mode,
            "language": detected_language,
            "duration_seconds": round(audio_duration, 4),
            "transcript": final_text,
            "raw_transcript": raw_text,
            "segments": segments,
            "decoder": {
                "type": "whisper",
                "engine": "whisper",
                "model": self.model.get_info().get("model_id"),
                "language": detected_language,
                "language_model": False,
            },
            "normalization_enabled": normalize,
            "confidence": confidence,
            "performance": {
                "enhancement_seconds": round(enhance_time, 4),
                "inference_seconds": round(inference_time, 4),
                "normalization_seconds": round(norm_time, 4),
                "total_seconds": round(total_time, 4),
                "rtf": rtf,
            },
        }
