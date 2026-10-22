#!/usr/bin/env python3
"""
Anot Medical ASR — Pure-Python N-gram LM Trainer (ARPA writer)
==============================================================
Stage 20: Trains an interpolated Witten-Bell n-gram language model from an
in-domain medical text corpus and emits a standard ARPA file.

Design goals (consistent with the project's zero-C++-dependency decoding stack):
  - Pure Python / stdlib only. No KenLM, no SRILM, no compiler.
  - Emits ARPA that `language_model.arpa_lm.ArpaLanguageModel` loads unchanged
    (tab-separated: log10_prob <TAB> words <TAB> backoff).
  - Witten-Bell smoothing with interpolation: robust on small/medium corpora and
    graceful on unseen contexts, which matters for clinical rescoring.

The resulting LM is a RESCORER prior. It never invents clinical facts; it only
reweights candidate word sequences during CTC beam search.
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
_NEG_INF_LOG10 = -99.0  # ARPA convention for "no backoff / impossible"


class NGramTrainer:
    """Train an interpolated Witten-Bell n-gram model and write ARPA."""

    def __init__(self, order: int = 3, min_count: int = 1):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.min_count = max(1, min_count)
        # counts[n] maps an n-gram tuple -> count
        self._counts: Dict[int, Dict[Tuple[str, ...], int]] = {
            n: defaultdict(int) for n in range(1, order + 1)
        }
        # distinct continuation counts for Witten-Bell: context -> #distinct next words
        self._continuations: Dict[Tuple[str, ...], set] = defaultdict(set)
        # context total counts: context tuple -> sum of counts of grams with that context
        self._ctx_total: Dict[Tuple[str, ...], int] = defaultdict(int)
        self._vocab: set = set()

    # --- training -------------------------------------------------------

    def fit(self, sentences: List[List[str]]) -> "NGramTrainer":
        """Accumulate counts from a list of tokenised sentences (lists of words)."""
        for sent in sentences:
            if not sent:
                continue
            padded = [BOS] * (self.order - 1) + list(sent) + [EOS]
            self._vocab.update(sent)
            for n in range(1, self.order + 1):
                for i in range(len(padded) - n + 1):
                    gram = tuple(padded[i:i + n])
                    self._counts[n][gram] += 1
                    if n >= 2:
                        self._continuations[gram[:-1]].add(gram[-1])
                        self._ctx_total[gram[:-1]] += 1
        self._vocab.update({BOS, EOS, UNK})
        return self

    # --- probability estimation (interpolated Witten-Bell) --------------

    def _unigram_prob(self, word: str) -> float:
        total = sum(self._counts[1].values())
        v = len(self._counts[1])
        # Reserve a small mass for <unk> via add-one on the vocabulary size.
        c = self._counts[1].get((word,), 0)
        return (c + 1.0) / (total + v + 1.0)

    def _prob(self, gram: Tuple[str, ...]) -> float:
        """Interpolated Witten-Bell probability P(w_n | w_1..w_{n-1})."""
        n = len(gram)
        if n == 1:
            return self._unigram_prob(gram[0])

        context = gram[:-1]
        ctx_total = self._ctx_total.get(context, 0)
        gram_count = self._counts[n].get(gram, 0)
        distinct = len(self._continuations.get(context, ()))

        lower = self._prob(gram[1:])  # back off to (n-1)-gram

        if ctx_total == 0:
            return lower

        # Witten-Bell lambda: weight on the higher-order estimate.
        lam = ctx_total / (ctx_total + distinct) if (ctx_total + distinct) > 0 else 0.0
        higher = gram_count / ctx_total if ctx_total > 0 else 0.0
        return lam * higher + (1.0 - lam) * lower

    # --- ARPA emission --------------------------------------------------

    def to_arpa(self) -> str:
        """Render the trained model as ARPA text (log10 probs + backoff)."""
        # Collect the n-grams we will emit (apply min_count to orders >= 2).
        emitted: Dict[int, List[Tuple[str, ...]]] = {}
        for n in range(1, self.order + 1):
            grams = []
            for gram, cnt in self._counts[n].items():
                if n >= 2 and cnt < self.min_count:
                    continue
                # Do not emit n-grams ending in <s> or starting mid-pad oddly.
                if gram[-1] == BOS:
                    continue
                grams.append(gram)
            emitted[n] = sorted(grams)

        # Backoff weights: a simple, valid scheme. For each context that has
        # emitted higher-order continuations, assign the Witten-Bell escape mass
        # distributed over backoff. We approximate with a mild constant derived
        # from the continuation diversity so unseen continuations stay scored.
        def backoff_log10(context: Tuple[str, ...]) -> float:
            ctx_total = self._ctx_total.get(context, 0)
            distinct = len(self._continuations.get(context, ()))
            if ctx_total == 0 or distinct == 0:
                return 0.0  # log10(1.0)
            escape = distinct / (ctx_total + distinct)
            escape = min(max(escape, 1e-4), 0.9999)
            return math.log10(escape)

        lines: List[str] = ["\\data\\"]
        for n in range(1, self.order + 1):
            lines.append(f"ngram {n}={len(emitted[n])}")
        lines.append("")

        for n in range(1, self.order + 1):
            lines.append(f"\\{n}-grams:")
            for gram in emitted[n]:
                p = self._prob(gram)
                log10_p = math.log10(p) if p > 0 else _NEG_INF_LOG10
                words = " ".join(gram)
                if n < self.order:
                    bo = backoff_log10(gram)
                    lines.append(f"{log10_p:.6f}\t{words}\t{bo:.6f}")
                else:
                    lines.append(f"{log10_p:.6f}\t{words}")
            lines.append("")

        lines.append("\\end\\")
        return "\n".join(lines)

    def summary(self) -> Dict[str, int]:
        return {
            "order": self.order,
            "vocab_size": len(self._vocab),
            **{f"ngrams_{n}": len(self._counts[n]) for n in range(1, self.order + 1)},
        }


def train_arpa(sentences: List[List[str]], order: int = 3, min_count: int = 1) -> str:
    """Convenience: train and return ARPA text from tokenised sentences."""
    return NGramTrainer(order=order, min_count=min_count).fit(sentences).to_arpa()
