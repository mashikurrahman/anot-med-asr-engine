#!/usr/bin/env python3
"""
Anot Medical ASR — Main ASR Orchestrator Service
================================================
Stage 11: Production orchestrator coordinating audio conversion, pipeline selection,
concurrency limiting, and PHI-safe response generation.
"""

import asyncio
import pathlib
import time
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.logging import logger, log_request_event
from app.models.medasr_model import MedASRModel
from app.models.whisper_model import WhisperASRModel
from app.pipelines.conversation_pipeline import ConversationPipeline
from app.pipelines.dictation_pipeline import DictationPipeline
from app.pipelines.whisper_pipeline import WhisperPipeline
from app.security.privacy import generate_request_id
from app.services.audio_service import AudioService
from app.services.decoding_service import DecodingService
from app.services.normalization_service import NormalizationService


class ASRService:
    """Central orchestration service for Anot Medical ASR transcription requests."""

    _instance: Optional["ASRService"] = None

    def __init__(self):
        self.engine = settings.ASR_ENGINE
        self.model = MedASRModel.get_instance()
        self.whisper_model = WhisperASRModel.get_instance()
        self.audio_service = AudioService()
        self.decoding_service = DecodingService()
        self.normalization_service = NormalizationService()

        self.dictation_pipeline = DictationPipeline(
            model=self.model,
            decoding_service=self.decoding_service,
            normalization_service=self.normalization_service
        )
        self.conversation_pipeline = ConversationPipeline(
            model=self.model
        )
        self.whisper_pipeline = WhisperPipeline(
            model=self.whisper_model,
            normalization_service=self.normalization_service,
        )

        # Concurrency semaphore to prevent out-of-memory or CPU starvation
        self._concurrency_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_JOBS)

    @classmethod
    def get_instance(cls) -> "ASRService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _resolve_engine(self, engine: Optional[str]) -> str:
        """Pick the engine for a request, defaulting to the configured engine."""
        return (engine or self.engine or "whisper").lower()

    def active_model(self, engine: Optional[str] = None):
        """Return the model object for the given/active engine."""
        return self.whisper_model if self._resolve_engine(engine) == "whisper" else self.model

    def is_ready(self, engine: Optional[str] = None) -> bool:
        return self.active_model(engine).is_loaded()

    def active_device(self, engine: Optional[str] = None) -> str:
        return self.active_model(engine).get_device()

    def active_model_id(self, engine: Optional[str] = None) -> str:
        return self.active_model(engine).get_info().get("model_id", "unknown")

    def initialize(self) -> None:
        """Initialize and warm up the configured engine's model on service startup."""
        if self.engine == "whisper":
            self.whisper_model.load(device=settings.WHISPER_DEVICE)
        else:
            self.model.load(device=settings.DEVICE)

    async def transcribe_audio_file(
        self,
        audio_path: pathlib.Path,
        mode: str = "dictation",
        decoder: str = "greedy",
        beam_width: Optional[int] = None,
        language_model: bool = False,
        lm_weight: Optional[float] = None,
        medical_weight: Optional[float] = None,
        normalize: bool = True,
        request_id: Optional[str] = None,
        segmentation_strategy: Optional[str] = None,
        search_window_sec: Optional[float] = None,
        edge_context_sec: Optional[float] = None,
        engine: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute transcription on a local audio file path with concurrency management.
        Guarantees cleanup of temporary converted WAV files.
        """
        req_id = request_id or generate_request_id()
        t_req_start = time.perf_counter()
        engine_resolved = self._resolve_engine(engine)

        log_request_event("transcribe_started", req_id, {
            "engine": engine_resolved,
            "mode": mode,
            "decoder": decoder,
            "lm_enabled": language_model,
            "normalize": normalize
        })

        async with self._concurrency_semaphore:
            temp_wav: Optional[pathlib.Path] = None
            try:
                # 1. Convert source audio to 16 kHz mono WAV
                (
                    waveform,
                    sample_rate,
                    duration_sec,
                    prep_time_sec,
                    temp_wav
                ) = self.audio_service.convert_to_16k_mono(audio_path)

                # 2. Dispatch to requested engine / clinical pipeline
                mode_lower = (mode or settings.MODE_DEFAULT).lower()

                if engine_resolved == "whisper":
                    res = self.whisper_pipeline.process(
                        waveform=waveform,
                        sample_rate=sample_rate,
                        normalize=normalize,
                        mode=mode_lower,
                        language=language,
                    )
                elif mode_lower == "conversation":
                    res = self.conversation_pipeline.process(
                        waveform=waveform,
                        sample_rate=sample_rate,
                        normalize=normalize,
                        segmentation_strategy=segmentation_strategy or "fixed_duration",
                        search_window_sec=search_window_sec if search_window_sec is not None else 0.5,
                        edge_context_sec=edge_context_sec if edge_context_sec is not None else 0.0,
                    )
                else:
                    res = self.dictation_pipeline.process(
                        waveform=waveform,
                        sample_rate=sample_rate,
                        decoder_type=decoder,
                        beam_width=beam_width,
                        use_lm=language_model,
                        lm_weight=lm_weight,
                        medical_weight=medical_weight,
                        normalize=normalize
                    )

                total_req_time = time.perf_counter() - t_req_start
                perf = res.get("performance", {})
                perf["preprocessing_seconds"] = round(prep_time_sec, 4)
                perf["total_request_seconds"] = round(total_req_time, 4)

                response_payload = {
                    "request_id": req_id,
                    "status": "completed",
                    "engine": engine_resolved,
                    "mode": mode_lower,
                    "language": res.get("language", language or "en"),
                    "duration_seconds": round(duration_sec, 2),
                    "transcript": res["transcript"],
                    "segments": res["segments"],
                    "decoder": res["decoder"],
                    "confidence": res.get("confidence"),
                    "performance": perf
                }

                log_request_event("transcribe_completed", req_id, {
                    "duration_seconds": round(duration_sec, 2),
                    "total_seconds": round(total_req_time, 4),
                    "segments_count": len(res["segments"])
                })

                return response_payload

            finally:
                # Always remove converted temporary WAV file
                self.audio_service.cleanup_paths(temp_wav)

    async def transcribe_upload(
        self,
        file_bytes: bytes,
        original_filename: str,
        mode: str = "dictation",
        decoder: str = "greedy",
        beam_width: Optional[int] = None,
        language_model: bool = False,
        lm_weight: Optional[float] = None,
        medical_weight: Optional[float] = None,
        normalize: bool = True,
        request_id: Optional[str] = None,
        segmentation_strategy: Optional[str] = None,
        search_window_sec: Optional[float] = None,
        edge_context_sec: Optional[float] = None,
        engine: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle multipart upload: validate -> write temp -> transcribe -> clean up all temp files.
        """
        req_id = request_id or generate_request_id()

        # Validate file size and format before writing to disk
        self.audio_service.validate_upload(original_filename, len(file_bytes))

        temp_upload_path: Optional[pathlib.Path] = None
        try:
            temp_upload_path = self.audio_service.write_temp_upload(file_bytes, original_filename)
            return await self.transcribe_audio_file(
                audio_path=temp_upload_path,
                mode=mode,
                decoder=decoder,
                beam_width=beam_width,
                language_model=language_model,
                lm_weight=lm_weight,
                medical_weight=medical_weight,
                normalize=normalize,
                request_id=req_id,
                segmentation_strategy=segmentation_strategy,
                search_window_sec=search_window_sec,
                edge_context_sec=edge_context_sec,
                engine=engine,
                language=language,
            )
        finally:
            self.audio_service.cleanup_paths(temp_upload_path)
