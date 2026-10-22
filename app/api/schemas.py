#!/usr/bin/env python3
"""
Anot Medical ASR — API Request and Response Schemas
===================================================
Stage 11: Stable Pydantic schemas for the production ASR service.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Safe, minimal service health response."""
    status: str = Field(default="ok", example="ok")
    service: str = Field(default="anot-asr", example="anot-asr")


class ReadyResponse(BaseModel):
    """Readiness response indicating model initialization status."""
    ready: bool = Field(default=True, example=True)
    model_loaded: bool = Field(default=True, example=True)
    device: str = Field(default="cpu", example="cpu")
    model_id: str = Field(default="google/medasr", example="google/medasr")


class TranscribeSegment(BaseModel):
    """Segment within a transcription with optional speaker attribution."""
    start: float = Field(description="Start time in seconds", example=0.0)
    end: float = Field(description="End time in seconds", example=4.2)
    text: str = Field(description="Transcribed text for this segment", example="Patient is a 58-year-old male.")
    speaker: Optional[str] = Field(default=None, description="Anonymous speaker identifier (e.g. SPEAKER_00)", example="SPEAKER_00")
    avg_logprob: Optional[float] = Field(default=None, description="Whisper average token log-probability (confidence)")
    no_speech_prob: Optional[float] = Field(default=None, description="Whisper probability the segment is non-speech")
    low_confidence: Optional[bool] = Field(default=None, description="True if the segment is below the confidence threshold (review candidate)")


class DecoderInfo(BaseModel):
    """Decoder configuration metadata."""
    type: str = Field(description="Decoder type: greedy, beam, or whisper", example="greedy")
    engine: Optional[str] = Field(default=None, description="ASR engine used", example="whisper")
    model: Optional[str] = Field(default=None, description="Model identifier", example="whisper-large-v3")
    language: Optional[str] = Field(default=None, description="Transcription language (detected or requested)", example="en")
    beam_width: Optional[int] = Field(default=None, description="Beam width if beam decoder selected", example=4)
    language_model: bool = Field(default=False, description="Whether medical language model rescoring was applied", example=False)
    lm_weight: Optional[float] = Field(default=None, description="Language model prior weight", example=0.05)
    medical_weight: Optional[float] = Field(default=None, description="Medical lexicon boost weight", example=0.05)


class PerformanceMetrics(BaseModel):
    """Technical execution profiling metrics."""
    preprocessing_seconds: Optional[float] = Field(default=None, description="Audio format conversion time")
    inference_seconds: Optional[float] = Field(default=None, description="MedASR acoustic model forward pass time")
    decoding_seconds: Optional[float] = Field(default=None, description="CTC decoding time")
    normalization_seconds: Optional[float] = Field(default=None, description="Clinical representation normalization time")
    total_request_seconds: Optional[float] = Field(default=None, description="Total end-to-end request processing time")
    rtf: Optional[float] = Field(default=None, description="Real-Time Factor (total processing time / audio duration)")


class TranscribeResponse(BaseModel):
    """Standardized versioned response for all transcription requests."""
    request_id: str = Field(description="Unique neutral request ID", example="ASR-20261003-000001")
    status: str = Field(default="completed", example="completed")
    engine: Optional[str] = Field(default=None, description="ASR engine used: whisper or medasr", example="whisper")
    mode: str = Field(description="Transcription mode: dictation or conversation", example="dictation")
    language: str = Field(default="en", example="en")
    duration_seconds: float = Field(description="Total audio duration in seconds", example=62.36)
    transcript: str = Field(description="Complete normalized clinical transcript")
    segments: List[TranscribeSegment] = Field(default_factory=list, description="Chronological transcript segments")
    decoder: DecoderInfo = Field(description="Decoder configuration details")
    confidence: Optional[Dict[str, Any]] = Field(default=None, description="Confidence summary (low-confidence segment counts) for review/fallback routing")
    performance: Optional[Dict[str, Any]] = Field(default=None, description="Safe technical execution timings")


class ErrorResponse(BaseModel):
    """Safe, non-leaking HTTP error response."""
    detail: str = Field(description="Error message without internal paths or stack traces")
    request_id: Optional[str] = Field(default=None, description="Request identifier for troubleshooting")
