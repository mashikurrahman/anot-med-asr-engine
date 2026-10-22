#!/usr/bin/env python3
"""
Anot Medical ASR — Mode A Physician Dictation Pipeline
======================================================
Stage 11: Production pipeline for monologue clinical dictation.
Acoustic Inference -> CTC Decoder -> Medical Normalization.
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np

from app.models.medasr_model import MedASRModel
from app.services.decoding_service import DecodingService
from app.services.normalization_service import NormalizationService


class DictationPipeline:
    """End-to-end Mode A pipeline for physician dictation."""

    def __init__(
        self,
        model: MedASRModel,
        decoding_service: DecodingService,
        normalization_service: NormalizationService
    ):
        self.model = model
        self.decoding_service = decoding_service
        self.normalization_service = normalization_service

    def process(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        decoder_type: str = "greedy",
        beam_width: Optional[int] = None,
        use_lm: bool = False,
        lm_weight: Optional[float] = None,
        medical_weight: Optional[float] = None,
        normalize: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute dictation pipeline on a 16 kHz mono waveform.
        Returns transcript, segments, decoder metadata, and timing performance.
        """
        audio_duration = len(waveform) / sample_rate
        t_pipeline_start = time.perf_counter()

        # 1. Acoustic Inference via MedASR
        logits, inference_time = self.model.infer(waveform, sample_rate=sample_rate)

        # 2. CTC Decoding (Greedy default or Beam+LM)
        decode_result = self.decoding_service.decode(
            logits=logits,
            processor=self.model.processor,
            decoder_type=decoder_type,
            beam_width=beam_width,
            use_lm=use_lm,
            lm_weight=lm_weight,
            medical_weight=medical_weight
        )
        raw_text = decode_result["text"]
        decoder_info = decode_result["decoder_info"]
        decoding_time = decode_result["decode_time_seconds"]

        # 3. Medical Normalization (representation standardization)
        norm_time = 0.0
        final_text = raw_text
        norm_events = []

        if normalize and raw_text:
            final_text, norm_events, norm_time = self.normalization_service.normalize_text(raw_text)

        total_pipeline_time = time.perf_counter() - t_pipeline_start
        rtf = round(total_pipeline_time / audio_duration, 4) if audio_duration > 0 else 0.0

        # Structured single-segment for dictation
        segments = [
            {
                "start": 0.0,
                "end": round(audio_duration, 2),
                "text": final_text
            }
        ] if final_text else []

        return {
            "mode": "dictation",
            "duration_seconds": round(audio_duration, 4),
            "transcript": final_text,
            "raw_transcript": raw_text,
            "segments": segments,
            "decoder": decoder_info,
            "normalization_enabled": normalize,
            "normalization_events_count": len(norm_events),
            "performance": {
                "inference_seconds": round(inference_time, 4),
                "decoding_seconds": round(decoding_time, 4),
                "normalization_seconds": round(norm_time, 4),
                "total_seconds": round(total_pipeline_time, 4),
                "rtf": rtf
            }
        }
