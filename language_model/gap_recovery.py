#!/usr/bin/env python3
"""
Find loud stretches of a recording that the first pass left untranscribed, and screen what a second pass returns.

The first pass relies on a speech detector that can miss quiet, distant or child speech. A stretch that is clearly
not silence (compared with the level of the speech that WAS transcribed) but has no text is worth a second look
with a more sensitive detector. Output of that second pass is flagged low-confidence and screened for the usual
noise inventions. Pure functions.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

Gap = Tuple[float, float]


def find_gaps(segments: Sequence[Dict[str, Any]], level: Sequence[float], duration: float,
              rel: float = 0.2, min_len: float = 5.0, close_hole: int = 3) -> List[Gap]:
    """(start, end) stretches with no transcript whose loudness is at least `rel` x the median loudness of transcribed speech.

    `level` is the RMS per second of audio. Dips of up to `close_hole` seconds inside a loud stretch do not split it;
    stretches shorter than `min_len` seconds are ignored."""
    n = min(len(level), int(duration))
    covered = np.zeros(n, dtype=bool)
    for s in segments:
        covered[max(0, int(s["start"])):min(n, int(np.ceil(s["end"])))] = True
    if not covered.any():
        return []
    lv = np.asarray(level[:n], dtype=float)
    ref = float(np.median(lv[covered]))
    loud = (~covered) & (lv >= rel * ref)
    i = 0
    while i < n:  # close short dips between loud seconds (only inside uncovered audio)
        if not loud[i] and not covered[i]:
            j = i
            while j < n and not loud[j] and not covered[j]:
                j += 1
            if j - i <= close_hole and i > 0 and j < n and loud[i - 1] and loud[j]:
                loud[i:j] = True
            i = j
        else:
            i += 1
    gaps: List[Gap] = []
    i = 0
    while i < n:
        if loud[i]:
            j = i
            while j < n and loud[j]:
                j += 1
            if j - i >= min_len:
                gaps.append((float(i), float(j)))
            i = j
        else:
            i += 1
    return gaps


def filter_recovered(segments: Sequence[Dict[str, Any]], max_no_speech: float = 0.6,
                     min_logprob: float = -1.2) -> List[Dict[str, Any]]:
    """Keep second-pass segments that look like speech; flag them recovered + low-confidence."""
    kept: List[Dict[str, Any]] = []
    seen = set()
    for s in segments:
        text = (s.get("text") or "").strip()
        key = text.lower()
        no_speech, logprob = s.get("no_speech_prob"), s.get("avg_logprob")
        if not text or key in seen:
            continue
        if no_speech is not None and no_speech > max_no_speech:
            continue
        if logprob is not None and logprob < min_logprob:
            continue
        seen.add(key)
        kept.append({**s, "recovered": True, "low_confidence": True})
    return kept
