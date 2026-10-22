#!/usr/bin/env python3
"""
Anot Medical ASR — Privacy & Security Boundary
==============================================
Stage 11: Request anonymization, filename sanitization, and authentication interface.
"""

import hashlib
import os
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

# Authentication header boundary
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def generate_request_id() -> str:
    """Generate a clean, neutral, non-PHI request identifier."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:8].upper()
    return f"ASR-{date_str}-{short_uuid}"


def sanitize_filename(original_filename: Optional[str]) -> str:
    """
    Sanitize an uploaded filename into a neutral, randomized string.
    Prevents patient names, medical IDs, or clinic details in original filenames
    from leaking into filesystem paths or server logs.
    """
    if not original_filename:
        return f"upload_{uuid.uuid4().hex[:12]}.wav"

    ext = pathlib.Path(original_filename).suffix.lower()
    if not ext:
        ext = ".wav"

    # Create deterministic hash of original filename + salt for internal tracking if needed
    name_hash = hashlib.sha256(original_filename.encode("utf-8")).hexdigest()[:12]
    return f"upload_{name_hash}_{uuid.uuid4().hex[:6]}{ext}"


async def verify_auth_boundary(
    api_key: Optional[str] = Security(API_KEY_HEADER)
) -> None:
    """
    Security boundary dependency for FastAPI endpoints.
    In development / private deployment (settings.API_KEY is None), all requests are allowed.
    In authenticated environments, validates the X-API-Key header.
    """
    expected_key = settings.API_KEY
    if expected_key is not None:
        if not api_key or api_key != expected_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: invalid or missing API key."
            )
