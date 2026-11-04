#!/usr/bin/env python3
"""
Plan English / non-English spans for a recording from per-window language scores.

The engine is English-only; speech in another language, if forced through an
English decoder, comes out as invented English. This module turns the English
probability of each fixed-length audio window into contiguous spans (English
spans are transcribed, others are replaced by a marker). Pure functions.
"""

from typing import List, Optional, Tuple

Span = Tuple[float, float, bool]  # (start_sec, end_sec, is_english)


def english_flags(p_en: List[Optional[float]], threshold: float = 0.4) -> List[bool]:
    """Per-window English flag. None (quiet window) inherits a neighbour; lone flips are smoothed away."""
    known = [p for p in p_en if p is not None]
    if not known:
        return [True] * len(p_en)
    flags: List[Optional[bool]] = [None if p is None else p >= threshold for p in p_en]
    last: Optional[bool] = None
    for i, f in enumerate(flags):  # forward fill
        last = f if f is not None else last
        flags[i] = last
    nxt: Optional[bool] = None
    for i in range(len(flags) - 1, -1, -1):  # back fill leading gaps
        nxt = flags[i] if flags[i] is not None else nxt
        flags[i] = nxt
    out = list(flags)
    for i in range(1, len(flags) - 1):  # a window that disagrees with both neighbours (who agree) flips
        if flags[i - 1] == flags[i + 1] and flags[i] != flags[i - 1]:
            out[i] = flags[i - 1]
    return [bool(f) for f in out]


def build_spans(flags: List[bool], window_sec: float, total_sec: float) -> List[Span]:
    """Merge consecutive equal flags into (start, end, is_english) spans, clipped to the duration."""
    spans: List[Span] = []
    for i, f in enumerate(flags):
        start, end = i * window_sec, min((i + 1) * window_sec, total_sec)
        if end <= start:
            continue
        if spans and spans[-1][2] == f:
            spans[-1] = (spans[-1][0], end, f)
        else:
            spans.append((start, end, f))
    return spans


def _clock(sec: float) -> str:
    s = int(round(sec))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_marker(start: float, end: float) -> str:
    """Placeholder text for a stretch of non-English speech."""
    return f"[Non-English speech, {_clock(start)}-{_clock(end)}]"


_LANGUAGE_NAMES = {
    "es": "Spanish", "bn": "Bengali", "hi": "Hindi", "ar": "Arabic", "zh": "Chinese", "fr": "French",
    "pt": "Portuguese", "ru": "Russian", "vi": "Vietnamese", "ur": "Urdu", "ko": "Korean", "ja": "Japanese",
    "de": "German", "it": "Italian", "tl": "Tagalog", "ht": "Haitian Creole", "pl": "Polish", "fa": "Persian",
    "ta": "Tamil", "te": "Telugu", "gu": "Gujarati", "pa": "Punjabi", "tr": "Turkish", "uk": "Ukrainian",
}


def translation_label(language_code: Optional[str]) -> str:
    """Tag placed before text that was translated into English, so a reader knows it is not a verbatim transcript."""
    if not language_code:
        return "[Translated from another language]"
    return f"[Translated from {_LANGUAGE_NAMES.get(language_code, language_code)}]"


def join_with_translation_labels(segments: List[dict]) -> str:
    """Join segment texts; a translation label is put where each run of translated text begins."""
    parts: List[str] = []
    in_translation = False
    for sg in segments:
        source = sg.get("translated_from")
        if source and not in_translation:
            parts.append(translation_label(None if source == "unknown" else source))
        in_translation = bool(source)
        parts.append(sg["text"])
    return " ".join(parts).strip()
