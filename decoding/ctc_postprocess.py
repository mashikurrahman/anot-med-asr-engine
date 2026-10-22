#!/usr/bin/env python3
"""
Anot Medical ASR — Conservative CTC Glitch Corrector
====================================================
Stage 19: Deterministic, fact-preserving repair of common CTC surface glitches.

MedASR's CTC head occasionally emits surface-level glitches that are NOT genuine
word confusions — most commonly a spuriously doubled letter from imperfect blank
collapse (e.g. "examinnation" for "examination"). This module repairs ONLY those
glitches, under strict safety constraints:

SAFETY CONTRACT (never violated):
  1. A token is only ever changed when it is NOT a known word AND the repair
     produces EXACTLY ONE known word. Ambiguity => leave unchanged.
  2. Any token containing a digit is left completely untouched. Clinical numbers,
     doses, vitals, and ages can never be altered by this module.
  3. Only within-token letter edits are attempted (doubled-letter collapse).
     No word is inserted, deleted, split, merged, or semantically substituted.
  4. The "known word" vocabulary is domain vocabulary (medical lexicon + the
     n-gram LM vocabulary + number words), so repairs stay inside clinical
     language and cannot invent out-of-domain terms.

This is a SURFACE corrector, not a language model. It does not rank candidates by
probability; it acts only when the answer is unambiguous.
"""

import re
from typing import Iterable, List, Optional, Set, Tuple


class CTCGlitchCorrector:
    """Repair unambiguous CTC surface glitches against a known-word vocabulary."""

    def __init__(self, vocabulary: Iterable[str]):
        self._vocab: Set[str] = {w.lower() for w in vocabulary if w and w.isalpha()}

    # --- public API -----------------------------------------------------

    def correct_text(self, text: str) -> Tuple[str, List[dict]]:
        """
        Correct a text string. Returns (corrected_text, events).

        Tokenisation preserves all whitespace and punctuation; only the
        alphabetic cores of tokens are considered for repair.
        """
        events: List[dict] = []
        # Split into alternating word / non-word chunks, preserving separators.
        parts = re.split(r"([A-Za-z]+)", text)
        for i, part in enumerate(parts):
            if not part or not part.isalpha():
                continue
            repaired = self._correct_token(part)
            if repaired is not None and repaired != part:
                events.append({
                    "original": part,
                    "corrected": repaired,
                    "category": "ctc_doubled_letter",
                    "confidence": "deterministic_unambiguous",
                })
                parts[i] = repaired
        return "".join(parts), events

    # --- internals ------------------------------------------------------

    def _is_known(self, token_lower: str) -> bool:
        return token_lower in self._vocab

    def _correct_token(self, token: str) -> Optional[str]:
        """Return a repaired token, or None if no safe repair applies."""
        lower = token.lower()

        # Rule 2: never touch anything with a digit (enforced by caller split,
        # but re-checked defensively) and never touch already-known words.
        if any(ch.isdigit() for ch in token):
            return None
        if self._is_known(lower):
            return None

        candidates = self._doubled_letter_collapses(lower)
        known = sorted({c for c in candidates if self._is_known(c)})

        # Rule 1: act only when exactly one known candidate exists.
        if len(known) != 1:
            return None

        return self._match_case(token, known[0])

    @staticmethod
    def _doubled_letter_collapses(token_lower: str) -> Set[str]:
        """
        Generate candidates by collapsing each run of a repeated letter to a
        single letter (e.g. 'examinnation' -> 'examination', 'ssaturation' ->
        'saturation'). Only single-step collapses per run are produced.
        """
        candidates: Set[str] = set()
        n = len(token_lower)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and token_lower[j + 1] == token_lower[i]:
                j += 1
            run_len = j - i + 1
            if run_len >= 2:
                # Collapse this run to a single character.
                candidate = token_lower[:i] + token_lower[i] + token_lower[j + 1:]
                candidates.add(candidate)
            i = j + 1
        return candidates

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        """Preserve the original token's capitalisation pattern."""
        if original.isupper():
            return replacement.upper()
        if original[:1].isupper():
            return replacement.capitalize()
        return replacement


def build_default_vocabulary(
    arpa_lm=None,
    medical_lexicon=None,
    extra_words: Optional[Iterable[str]] = None,
) -> Set[str]:
    """
    Assemble a conservative known-word vocabulary from available domain assets:
    the n-gram LM vocabulary, the medical lexicon, spelled-out numbers, and any
    caller-supplied extras. All sources are domain-scoped, so repairs cannot
    drift to arbitrary English.
    """
    from audio.normalization import NUMBER_WORDS, NUM_WORDS_ORDERED

    vocab: Set[str] = set()
    vocab.update(NUMBER_WORDS.keys())
    vocab.update(w for w in NUM_WORDS_ORDERED if w.isalpha())

    if arpa_lm is not None and getattr(arpa_lm, "vocab", None):
        vocab.update(w for w in arpa_lm.vocab if isinstance(w, str))
    if medical_lexicon is not None:
        try:
            vocab.update(medical_lexicon.get_all_words())
        except Exception:
            pass
    if extra_words:
        vocab.update(extra_words)

    return {w.lower() for w in vocab if w and w.isalpha()}
