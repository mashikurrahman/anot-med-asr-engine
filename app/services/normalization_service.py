#!/usr/bin/env python3
"""
Anot Medical ASR — Medical Normalization Service
================================================
Stage 11: Production wrapper for deterministic clinical normalization.
Standardizes clinical representation (e.g. 152/92, 10 mg) without altering facts.
"""

import time
from typing import Any, Dict, List, Tuple
from audio.normalization import MedicalNormalizer


class NormalizationService:
    """Wraps deterministic MedicalNormalizer for single texts and conversation turns."""

    def __init__(self):
        self._normalizer = MedicalNormalizer()

    def normalize_text(self, text: str) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Normalize a single clinical transcript string.
        Returns:
            (normalized_text, list_of_normalization_events, normalization_time_seconds)
        """
        t0 = time.perf_counter()
        normalized_text, events = self._normalizer.normalize_text(text)
        duration = time.perf_counter() - t0
        return normalized_text, events, duration

    def normalize_turns(
        self,
        turns: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        """
        Normalize structured conversation turns.
        Returns:
            (normalized_turns, list_of_normalization_events, normalization_time_seconds)
        """
        t0 = time.perf_counter()
        normalized_turns, events = self._normalizer.normalize_turns(turns)
        duration = time.perf_counter() - t0
        return normalized_turns, events, duration
