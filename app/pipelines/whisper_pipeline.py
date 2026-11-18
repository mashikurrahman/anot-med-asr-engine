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

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.models.whisper_model import WhisperASRModel
from app.services.diarization_service import DiarizationError, Diarizer
from app.services.normalization_service import NormalizationService
from app.services.speaker_assign import assign_speakers, render_transcript
from audio.enhancement import enhance_waveform
from language_model.gap_recovery import keep_recovered
from language_model.language_spans import join_with_translation_labels
from language_model.term_corrector import correct_terms
from language_model.transcript_cleanup import apply_dictation_commands, strip_fillers

logger = logging.getLogger("anot.pipeline")


class WhisperPipeline:
    """End-to-end transcription pipeline for the Whisper engine."""

    def __init__(self, model: WhisperASRModel, normalization_service: NormalizationService, diarizer=None):
        self.model = model
        self.normalization_service = normalization_service
        self._diarizer = diarizer

    def _start_diarization(self, wave: np.ndarray, sample_rate: int, mode: str):
        """Run speaker diarization in a background thread so it overlaps with transcription."""
        if not (settings.DIARIZE and mode == "conversation"):
            return None
        diarizer = self._diarizer or Diarizer.get()
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(diarizer.diarize, wave, sample_rate)
        executor.shutdown(wait=False)
        return future

    @staticmethod
    def _collect_turns(future) -> List[Tuple[float, float, str]]:
        if future is None:
            return []
        try:
            return future.result()
        except DiarizationError as e:  # speaker labels are optional: keep the transcript
            logger.warning("diarization skipped: %s", e)
            return []
        except Exception:
            logger.exception("diarization crashed; continuing without speaker labels")
            return []

    @staticmethod
    def _clean(text: str) -> str:
        """Drug-name fixes, then filler removal, then spoken-punctuation commands (each opt-in)."""
        if settings.TERM_CORRECTION:
            text = correct_terms(text)
        if settings.STRIP_FILLERS:
            text = strip_fillers(text)
        if settings.DICTATION_COMMANDS:
            text = apply_dictation_commands(text)
        return text

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
                gain_riding=settings.AUDIO_GAIN_RIDE,
            )
            enhance_time = time.perf_counter() - t_enh

        diarization = self._start_diarization(proc_wave, sample_rate, mode)
        raw_text, raw_segments, inference_time = self.model.transcribe(
            proc_wave, sample_rate, language=language)
        raw_segments = assign_speakers(raw_segments, self._collect_turns(diarization))
        if any(sg.get("recovered") for sg in raw_segments):
            # Second-pass lines: keep only confident ones from the visit's main speakers (drops background voices).
            raw_segments = keep_recovered(raw_segments)
            raw_text = join_with_translation_labels(raw_segments)
        speakers_on = any(sg.get("speaker") for sg in raw_segments)
        detected_language = getattr(self.model, "_last_detected_language", language or "en")

        # Optional text fixes (English only); raw_transcript stays untouched.
        text, seg_list = raw_text, raw_segments
        if str(detected_language).lower().startswith("en"):
            text = self._clean(raw_text)
            seg_list = [{**seg, "text": self._clean(seg["text"])} for seg in raw_segments]

        norm_time = 0.0
        final_text = text
        segments = seg_list
        if normalize and text:
            final_text, _events, norm_time = self.normalization_service.normalize_text(text)
            # Normalize each segment's text too, so segment views stay consistent.
            norm_segments = []
            for seg in seg_list:
                seg_text, _ev, _t = self.normalization_service.normalize_text(seg["text"])
                norm_segments.append({**seg, "text": seg_text})
            segments = norm_segments

        if speakers_on:  # "Speaker 1: ..." lines; cleanup and normalization see each whole turn
            def tidy(turn_text: str) -> str:
                cleaned = self._clean(turn_text) if str(detected_language).lower().startswith("en") else turn_text
                return self.normalization_service.normalize_text(cleaned)[0] if normalize and cleaned else cleaned
            final_text = render_transcript(raw_segments, transform=tidy)

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
