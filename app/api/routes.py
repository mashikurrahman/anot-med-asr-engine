#!/usr/bin/env python3
"""
Anot Medical ASR — Production FastAPI API Routes
=================================================
Stage 11: Endpoints for health, readiness, and audio transcription.
"""

import pathlib
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ReadyResponse,
    TranscribeResponse
)
from app.core.config import settings
from app.security.privacy import generate_request_id, verify_auth_boundary
from app.services.asr_service import ASRService

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Returns safe service status. Does not expose model internals or PHI."
)
async def health_check() -> HealthResponse:
    """Check service viability."""
    return HealthResponse(
        status="ok",
        service=settings.SERVICE_NAME,
        version=settings.VERSION
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Inference Readiness Check",
    description="Returns 200 ready=true only when MedASR is loaded in memory and ready for inference."
)
async def readiness_check() -> ReadyResponse:
    """Verify that the active ASR engine is loaded in memory and ready for inference."""
    asr = ASRService.get_instance()

    if not asr.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ASR service is initializing or model is not yet loaded in memory."
        )

    return ReadyResponse(
        ready=True,
        model_loaded=True,
        device=asr.active_device(),
        model_id=asr.active_model_id()
    )


@router.post(
    "/api/v1/transcribe",
    response_model=TranscribeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid audio format or excessive duration"},
        413: {"model": ErrorResponse, "description": "Audio file exceeds size limit"},
        422: {"model": ErrorResponse, "description": "Invalid parameters"},
        500: {"model": ErrorResponse, "description": "Internal transcription failure"},
        503: {"model": ErrorResponse, "description": "Service not ready"}
    },
    dependencies=[Depends(verify_auth_boundary)],
    summary="Transcribe Clinical Audio",
    description="Transcribe uploaded clinical audio via MedASR (supports Mode A dictation and Mode B conversation)."
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Uploaded clinical audio recording (.wav, .m4a, .mp3, .flac, .mp4, etc.)"),
    mode: str = Form(default="dictation", description="Transcription workflow: 'dictation' or 'conversation'"),
    decoder: str = Form(default="greedy", description="CTC Decoder: 'greedy' (default production) or 'beam'"),
    beam_width: Optional[int] = Form(default=None, description="Beam width if beam decoder is selected (e.g. 4 or 8)"),
    language_model: bool = Form(default=False, description="Enable medical language model rescoring during beam search"),
    lm_weight: Optional[float] = Form(default=None, description="LM prior weight (recommended 0.05 - 0.10)"),
    medical_weight: Optional[float] = Form(default=None, description="Medical lexicon boost weight (recommended 0.05)"),
    normalize: bool = Form(default=True, description="Enable deterministic clinical representation normalization"),
    engine: Optional[str] = Form(default=None, description="ASR engine: 'whisper' (default, best for visits) or 'medasr' (dictation)"),
    language: Optional[str] = Form(default=None, description="Language code ('en', 'bn', ...), 'auto' to detect, or blank for the configured default. Whisper engine only."),
) -> TranscribeResponse:
    # 1. Early parameter and extension validation (prior to resource allocation)
    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in settings.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format '{ext}'. Supported formats: {sorted(list(settings.SUPPORTED_EXTENSIONS))}."
        )

    mode_clean = mode.lower().strip()
    if mode_clean not in ["dictation", "conversation"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid transcription mode '{mode}'. Supported modes: 'dictation', 'conversation'."
        )

    decoder_clean = decoder.lower().strip()
    if decoder_clean not in ["greedy", "beam"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid decoder '{decoder}'. Supported decoders: 'greedy', 'beam'."
        )

    engine_clean = (engine or settings.ASR_ENGINE).lower().strip()
    if engine_clean not in ["whisper", "medasr"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid engine '{engine}'. Supported engines: 'whisper', 'medasr'."
        )

    # 2. Verify model readiness for the selected engine
    asr = ASRService.get_instance()
    if not asr.is_ready(engine_clean):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ASR service is not ready. Engine '{engine_clean}' model is still loading."
        )

    req_id = generate_request_id()

    # Read uploaded file bytes in memory
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded audio stream."
        ) from e

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    result = await asr.transcribe_upload(
        file_bytes=file_bytes,
        original_filename=file.filename or "upload.wav",
        mode=mode_clean,
        decoder=decoder_clean,
        beam_width=beam_width,
        language_model=language_model,
        lm_weight=lm_weight,
        medical_weight=medical_weight,
        normalize=normalize,
        request_id=req_id,
        engine=engine_clean,
        language=(language or None),
    )

    return TranscribeResponse(**result)


@router.post(
    "/api/v1/transcribe/conversation",
    response_model=TranscribeResponse,
    dependencies=[Depends(verify_auth_boundary)],
    summary="Transcribe Doctor–Patient Ambient Conversation",
    description="Convenience endpoint for Mode B ambient multi-speaker clinical encounters with VAD segmentation."
)
async def transcribe_conversation(
    file: UploadFile = File(..., description="Uploaded multi-speaker encounter audio"),
    normalize: bool = Form(default=True, description="Enable clinical representation normalization"),
    engine: Optional[str] = Form(default=None, description="ASR engine: 'whisper' (default) or 'medasr'"),
    language: Optional[str] = Form(default=None, description="Language code, 'auto', or blank for default"),
) -> TranscribeResponse:
    """Specialized endpoint for ambient conversation workflow."""
    return await transcribe_audio(
        file=file,
        mode="conversation",
        decoder="greedy",
        beam_width=None,
        language_model=False,
        lm_weight=None,
        medical_weight=None,
        normalize=normalize,
        engine=engine if isinstance(engine, str) else None,
        language=language if isinstance(language, str) else None,
    )


@router.post(
    "/v1/listen",
    summary="Deepgram Nova-3 Drop-In Compatibility Endpoint",
    description="Accepts raw audio or multipart forms and returns Deepgram-format JSON to migrate existing web apps with 0 client changes."
)
async def deepgram_compatible_listen(
    request: Request,
    model: Optional[str] = None,
    smart_format: Optional[bool] = True,
    language: Optional[str] = "en",
) -> Dict[str, Any]:
    """
    Drop-in compatibility route for apps built on Deepgram Nova-3 Medical SDK / REST API.
    """
    content_type = request.headers.get("content-type", "")
    filename = "recording.wav"
    file_bytes = b""

    if "multipart/form-data" in content_type:
        form = await request.form()
        for field_name in ["file", "audio", "buffer"]:
            if field_name in form and hasattr(form[field_name], "read"):
                file_bytes = await form[field_name].read()
                filename = getattr(form[field_name], "filename", "recording.wav")
                break
        if not file_bytes and form:
            first_val = list(form.values())[0]
            if hasattr(first_val, "read"):
                file_bytes = await first_val.read()
                filename = getattr(first_val, "filename", "recording.wav")
    else:
        file_bytes = await request.body()
        if "audio/m4a" in content_type or "mp4" in content_type:
            filename = "recording.m4a"
        elif "audio/mp3" in content_type or "mpeg" in content_type:
            filename = "recording.mp3"
        elif "audio/ogg" in content_type:
            filename = "recording.ogg"

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio payload."
        )

    asr = ASRService.get_instance()
    req_id = generate_request_id()

    # If model mentions dictation, route to MedASR; otherwise default to Whisper for conversational visits
    engine_choice = "medasr" if (model and "dictat" in model.lower()) else "whisper"
    mode_choice = "dictation" if engine_choice == "medasr" else "conversation"

    result = await asr.transcribe_upload(
        file_bytes=file_bytes,
        original_filename=filename,
        mode=mode_choice,
        decoder="greedy",
        normalize=bool(smart_format),
        request_id=req_id,
        engine=engine_choice,
        language=(language if language != "auto" else None),
    )

    transcript = result.get("transcript", "")
    segments = result.get("segments", [])
    words = []
    for s in segments:
        for w in (s.get("text", "")).split():
            words.append({
                "word": w,
                "start": s.get("start", 0.0),
                "end": s.get("end", 0.0),
                "confidence": 0.98
            })

    return {
        "metadata": {
            "transaction_key": req_id,
            "request_id": req_id,
            "duration": result.get("duration_seconds", 0.0),
            "channels": 1,
            "models": [f"anot-{engine_choice}"]
        },
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": transcript,
                            "confidence": 0.98,
                            "words": words,
                            "paragraphs": {
                                "transcript": transcript,
                                "paragraphs": [
                                    {"sentences": [{"text": transcript, "start": 0.0, "end": result.get("duration_seconds", 0.0)}]}
                                ]
                            }
                        }
                    ]
                }
            ]
        }
    }
