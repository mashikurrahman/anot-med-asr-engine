#!/usr/bin/env python3
"""
Anot Medical ASR — Privacy & Security Boundary
==============================================
Stage 11: Request anonymization, filename sanitization, and authentication interface.
"""

import hashlib
import hmac
import os
import pathlib
import uuid
from datetime import datetime, timezone
from typing import Optional
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings
from app.security.client_keys import DailyUsage, SharedDailyUsage, parse_client_keys

# Authentication header boundary
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
# Deepgram-style clients send "Authorization: Token <key>" (or "Bearer <key>").
AUTHORIZATION_HEADER = APIKeyHeader(name="Authorization", auto_error=False)


def _key_from_authorization(value: Optional[str]) -> Optional[str]:
    parts = (value or "").split(None, 1)
    return parts[1].strip() if len(parts) == 2 and parts[0].lower() in ("token", "bearer") else None


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


# Request counter for client keys with a daily limit. Shared across Modal containers when ANOT_USAGE_BACKEND=modal.
_usage = SharedDailyUsage() if os.getenv("ANOT_USAGE_BACKEND", "").lower() == "modal" else DailyUsage()


def _matches(presented: str, expected: str) -> bool:
    return hmac.compare_digest(presented.encode(), expected.encode())


async def verify_auth_boundary(
    request: Request = None,
    api_key: Optional[str] = Security(API_KEY_HEADER),
    authorization: Optional[str] = Security(AUTHORIZATION_HEADER),
) -> None:
    """
    Security boundary dependency for FastAPI endpoints.
    In development / private deployment (no key configured), all requests are allowed.
    Otherwise the X-API-Key header (or a Deepgram-style "Authorization: Token <key>") must be the main key
    (settings.API_KEY, unlimited) or one of the per-client keys (settings.CLIENT_KEYS, each with an optional daily limit).
    The matched client is recorded on request.state for the usage limit.
    """
    master = settings.API_KEY
    clients = parse_client_keys(getattr(settings, "CLIENT_KEYS", "") or "")
    if master is None and not clients:
        return
    presented = api_key or _key_from_authorization(authorization)
    if presented:
        match = ("master", None) if master is not None and _matches(presented, master) else None
        for client_key, (name, limit) in clients.items():
            if match is None and _matches(presented, client_key):
                match = (name, limit)
        if match is not None:
            if request is not None:
                request.state.client_name, request.state.client_limit = match
            return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: invalid or missing API key."
    )


async def enforce_daily_limit(request: Request) -> None:
    """For endpoints that process audio: one request counts against the client's daily limit; over it is a 429."""
    name = getattr(request.state, "client_name", None)
    limit = getattr(request.state, "client_limit", None)
    if name and not _usage.allow(name, limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily request limit reached for this key. It resets at 00:00 UTC."
        )
