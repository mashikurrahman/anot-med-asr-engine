#!/usr/bin/env python3
"""
Anot Medical ASR — Pure Python ARPA N-Gram Language Model
==========================================================
Stage 9B: Lightweight ARPA n-gram reader for medical LM rescoring.

Implements a pure Python ARPA format parser and n-gram scorer with:
- 1-gram, 2-gram, and 3-gram support
- Backoff weight handling (Katz or Kneser-Ney style)
- Incremental state-based scoring for beam search integration
- Unknown n-gram graceful degradation (no crashes)
- Zero C++ / kenlm / pyctcdecode dependency

ARPA format reference:
    \\data\\
    ngram 1=<count>
    ngram 2=<count>
    ...
    \\1-grams:
    <log10_prob> <word> [<backoff_weight>]
    ...
    \\2-grams:
    <log10_prob> <word1> <word2> [<backoff_weight>]
    ...
    \\end\\
"""

import math
import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


class ArpaLanguageModel:
    """
    Pure Python ARPA n-gram language model.
    
    Supports 1-gram through N-gram with backoff weights.
    All probabilities are stored as natural log (ln) internally.
    ARPA files use log10; conversion happens at load time.
    """

    LOG10_TO_LN = math.log(10.0)  # Multiply log10 values by this to get ln

    def __init__(self):
        self.order: int = 0
        self.ngrams: Dict[int, Dict[Tuple[str, ...], float]] = {}  # order -> {ngram_tuple: ln_prob}
        self.backoffs: Dict[int, Dict[Tuple[str, ...], float]] = {}  # order -> {context_tuple: ln_backoff}
        self.vocab: set = set()
        self._loaded: bool = False
        self._source_path: str = ""

    @classmethod
    def from_arpa_file(cls, filepath: str) -> "ArpaLanguageModel":
        """Load an ARPA-format language model from a file."""
        lm = cls()
        lm._source_path = filepath
        with open(filepath, "r", encoding="utf-8") as f:
            lm._parse_arpa(f.read())
        return lm

    @classmethod
    def from_arpa_string(cls, arpa_text: str) -> "ArpaLanguageModel":
        """Load an ARPA-format language model from a string."""
        lm = cls()
        lm._source_path = "<string>"
        lm._parse_arpa(arpa_text)
        return lm

    def _parse_arpa(self, text: str) -> None:
        """Parse ARPA format text into internal n-gram structures."""
        lines = text.strip().split("\n")
        current_order = 0
        in_data_section = False
        expected_counts: Dict[int, int] = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Header section
            if line == "\\data\\":
                in_data_section = True
                continue

            if in_data_section and line.startswith("ngram "):
                # Parse "ngram N=COUNT"
                match = re.match(r"ngram\s+(\d+)\s*=\s*(\d+)", line)
                if match:
                    order = int(match.group(1))
                    count = int(match.group(2))
                    expected_counts[order] = count
                    self.ngrams[order] = {}
                    self.backoffs[order] = {}
                    self.order = max(self.order, order)
                continue

            # N-gram section headers
            ngram_header = re.match(r"\\(\d+)-grams?:", line)
            if ngram_header:
                current_order = int(ngram_header.group(1))
                in_data_section = False
                continue

            if line == "\\end\\":
                break

            # Parse n-gram entries
            if current_order > 0:
                parts = line.split("\t")
                if len(parts) < 2:
                    parts = line.split()
                if len(parts) < 2:
                    continue

                try:
                    log10_prob = float(parts[0])
                except ValueError:
                    continue

                words = tuple(parts[1].split()) if "\t" in line else tuple(parts[1:1 + current_order])
                if len(words) != current_order:
                    # Try splitting the remaining parts
                    all_words = []
                    for p in parts[1:]:
                        all_words.extend(p.split())
                    if len(all_words) >= current_order:
                        words = tuple(all_words[:current_order])
                    else:
                        continue

                ln_prob = log10_prob * self.LOG10_TO_LN
                self.ngrams[current_order][words] = ln_prob

                # Backoff weight (optional, only for orders < max)
                backoff_idx = 1 + current_order
                remaining = parts[1 + current_order:] if len(parts) > 1 + current_order else []
                if not remaining and len(parts) > backoff_idx:
                    # Try parsing as tab-separated
                    pass

                # Check for backoff weight
                backoff_parts = line.split("\t")
                if len(backoff_parts) >= 3 and current_order < self.order:
                    try:
                        log10_backoff = float(backoff_parts[-1])
                        self.backoffs[current_order][words] = log10_backoff * self.LOG10_TO_LN
                    except ValueError:
                        pass

                # Add to vocabulary
                for w in words:
                    if w not in ("<s>", "</s>", "<unk>"):
                        self.vocab.add(w)

        self._loaded = True

    def score_ngram(self, words: Tuple[str, ...]) -> float:
        """
        Score an n-gram with backoff.
        
        If the full n-gram exists, return its log probability.
        Otherwise, back off to shorter n-grams with backoff penalties.
        
        Returns natural log probability.
        """
        n = len(words)

        # Direct lookup
        if n in self.ngrams and words in self.ngrams[n]:
            return self.ngrams[n][words]

        # Backoff: P(w_n | w_1...w_{n-1}) = backoff(w_1...w_{n-1}) * P(w_n | w_2...w_{n-1})
        if n > 1:
            context = words[:-1]
            shorter = words[1:]

            backoff_weight = 0.0  # ln(1.0) = 0.0, no penalty by default
            context_order = len(context)
            if context_order in self.backoffs and context in self.backoffs[context_order]:
                backoff_weight = self.backoffs[context_order][context]

            return backoff_weight + self.score_ngram(shorter)

        # Unigram fallback
        if 1 in self.ngrams:
            unigram = (words[-1],)
            if unigram in self.ngrams[1]:
                return self.ngrams[1][unigram]

            # Unknown word: use <unk> if available
            unk = ("<unk>",)
            if unk in self.ngrams[1]:
                return self.ngrams[1][unk]

        # Absolute fallback for completely unknown words
        return -10.0 * self.LOG10_TO_LN  # Very low probability

    def score_sequence(self, words: List[str]) -> float:
        """
        Score a full word sequence using the highest-order n-gram available.
        
        Returns total natural log probability of the sequence.
        """
        if not words:
            return 0.0

        total = 0.0
        padded = ["<s>"] + words + ["</s>"]

        for i in range(1, len(padded)):
            # Use up to order-length context
            start = max(0, i - self.order + 1)
            ngram = tuple(padded[start:i + 1])
            total += self.score_ngram(ngram)

        return total

    def score_word_in_context(self, context: List[str], word: str) -> float:
        """
        Score a single word given preceding context.
        
        Uses the highest-order n-gram that fits the available context.
        
        Args:
            context: List of preceding words (may be empty)
            word: The word to score
            
        Returns:
            Natural log probability P(word | context)
        """
        if not context:
            return self.score_ngram((word,))

        # Build n-gram from context + word, up to model order
        ctx_len = min(len(context), self.order - 1)
        ngram = tuple(context[-ctx_len:]) + (word,)
        return self.score_ngram(ngram)

    # ------------------------------------------------------------------
    # Incremental scoring interface for beam search
    # ------------------------------------------------------------------
    def begin(self) -> List[str]:
        """Start a new scoring context with sentence-begin marker."""
        return ["<s>"]

    def score_token(self, state: List[str], token: str) -> Tuple[float, List[str]]:
        """
        Score a token given current state and return (score, new_state).
        
        Args:
            state: Current context word list
            token: Next token to score
            
        Returns:
            (ln_probability, updated_state)
        """
        score = self.score_word_in_context(state, token)
        new_state = list(state) + [token]
        # Keep only last (order - 1) words for context efficiency
        if len(new_state) > self.order:
            new_state = new_state[-(self.order - 1):]
        return score, new_state

    def finish(self, state: List[str]) -> float:
        """Score the end-of-sentence given current state."""
        return self.score_word_in_context(state, "</s>")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Return a summary of the loaded language model."""
        counts = {order: len(entries) for order, entries in self.ngrams.items()}
        return {
            "source": self._source_path,
            "order": self.order,
            "loaded": self._loaded,
            "ngram_counts": counts,
            "vocabulary_size": len(self.vocab),
            "has_backoff_weights": any(len(bo) > 0 for bo in self.backoffs.values())
        }

    def contains_word(self, word: str) -> bool:
        """Check if a word exists in the unigram vocabulary."""
        return word in self.vocab or (1 in self.ngrams and (word,) in self.ngrams[1])

    def get_vocabulary(self) -> set:
        """Return the set of known vocabulary words."""
        return set(self.vocab)
