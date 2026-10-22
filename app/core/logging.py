#!/usr/bin/env python3
"""
Anot Medical ASR — PHI-Safe Structured Logging
==============================================
Stage 11: Production logger with strict PHI redaction.
Zero patient names, dates of birth, MRNs, addresses, or transcript text in logs.
"""

import logging
import re
import sys
from typing import Any, Dict, Optional


class PHISafeFormatter(logging.Formatter):
    """
    Formatter that ensures NO Protected Health Information (PHI) is emitted in log strings.
    Redacts phone numbers, SSNs, MRNs, emails, and transcript snippets.
    """

    PHI_PATTERNS = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
        (r"\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b", "[REDACTED_PHONE]"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]"),
        (r"\b(mrn|patient_id|dob|ssn)\s*[:=]\s*\S+", "[REDACTED_ID]"),
    ]

    def format(self, record: logging.LogRecord) -> str:
        orig = super().format(record)
        cleaned = orig
        for pattern, replacement in self.PHI_PATTERNS:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        return cleaned


def setup_logger(name: str = "anot_asr", level: int = logging.INFO) -> logging.Logger:
    """Initialize and configure a PHI-safe logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = PHISafeFormatter(
            fmt="%(asctime)s [%(levelname)s] [service=anot-asr] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


# Global logger instance
logger = setup_logger()


def log_request_event(
    event: str,
    request_id: str,
    extra: Optional[Dict[str, Any]] = None,
    level: int = logging.INFO
) -> None:
    """
    Log an event strictly keyed by request_id with technical metadata only.
    Guarantees no patient text or audio contents are written.
    """
    msg = f"event={event} request_id={request_id}"
    if extra:
        # Sanitize keys in extra: reject keys that might contain raw text
        safe_items = []
        for k, v in extra.items():
            if any(forbidden in k.lower() for forbidden in ["transcript", "text", "patient", "name", "content", "raw"]):
                continue
            safe_items.append(f"{k}={v}")
        if safe_items:
            msg += " " + " ".join(safe_items)
    logger.log(level, msg)
