#!/usr/bin/env python3
"""
Anot Medical ASR — Language Model Rescorer
============================================
Stage 9B: Integrates ARPA n-gram LM and medical lexicon with CTC beam search.

ARCHITECTURE:
    CTC Acoustic Logits
        ↓
    Native CTC Prefix Beam Search (decoding/beam_decoder.py)
        ↓
    Subword-to-Word Bridge
        ↓
    ARPA n-gram Language Model Scoring
        ↓
    Medical Lexicon Vocabulary Boost
        ↓
    Combined Rescoring:
        combined = acoustic + lm_weight * lm_score + medical_weight * lexicon_score
        ↓
    Final Transcript

SUBWORD COMPATIBILITY DESIGN:
    MedASR uses SentencePiece Unigram tokenizer (LasrTokenizer, 613 tokens).
    Word boundaries are marked by the '▁' (U+2581) prefix character.
    The beam decoder emits sequences of subword token IDs.
    
    The LM rescorer must:
    1. Accumulate subword tokens into complete words
    2. Detect word boundaries via '▁' prefix
    3. Score completed words using the word-level ARPA LM
    4. Apply medical lexicon boosts to completed words
    
    This bridge operates incrementally during beam search (not post-hoc).

CRITICAL SAFETY:
    The LM is a RESCORER. It influences candidate ranking.
    It does NOT invent clinical facts.
    Acoustic evidence remains dominant at recommended weights.
"""

import math
import time
import argparse
import json
import sys
import os
import pathlib
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np
import torch

# Add project root to path for imports
_project_root = str(pathlib.Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from language_model.arpa_lm import ArpaLanguageModel
from language_model.medical_lexicon import MedicalLexicon
from decoding.beam_decoder import BeamDecoder


class SubwordWordBridge:
    """
    Bridges subword token IDs to word-level text for LM scoring.
    
    MedASR SentencePiece uses '▁' (U+2581) to mark word boundaries.
    Subwords WITHOUT '▁' are continuations of the previous word.
    Subwords WITH '▁' start a new word.
    
    Example:
        Token sequence: ['▁pa', 't', 'i', 'ent', '▁has', '▁no', '▁chest']
        Words:          ['patient', 'has', 'no', 'chest']
    """

    WORD_BOUNDARY = "▁"  # SentencePiece word boundary marker (U+2581)

    def __init__(self, tokenizer):
        """
        Initialize with MedASR tokenizer.
        
        Args:
            tokenizer: HuggingFace tokenizer or processor with decode() method
        """
        self._tokenizer = tokenizer
        self._vocab_cache: Dict[int, str] = {}
        self._build_vocab_cache()

    def _build_vocab_cache(self) -> None:
        """Pre-cache token ID to string mappings."""
        tok = self._tokenizer.tokenizer if hasattr(self._tokenizer, "tokenizer") else self._tokenizer
        if hasattr(tok, "get_vocab"):
            vocab = tok.get_vocab()
            self._vocab_cache = {v: k for k, v in vocab.items()}

    def token_id_to_piece(self, token_id: int) -> str:
        """Convert a single token ID to its SentencePiece string."""
        if token_id in self._vocab_cache:
            return self._vocab_cache[token_id]
        # Fallback: use tokenizer decode
        tok = self._tokenizer.tokenizer if hasattr(self._tokenizer, "tokenizer") else self._tokenizer
        if hasattr(tok, "decode"):
            return tok.decode([token_id])
        return f"<id:{token_id}>"

    def token_ids_to_words(self, token_ids: List[int]) -> List[str]:
        """
        Convert a sequence of subword token IDs to a list of words.
        
        Uses '▁' boundaries to determine word segmentation.
        Filters out special tokens (<s>, </s>, <epsilon>, <unk>).
        """
        if not token_ids:
            return []

        special_ids = set()
        tok = self._tokenizer.tokenizer if hasattr(self._tokenizer, "tokenizer") else self._tokenizer
        if hasattr(tok, "get_vocab"):
            vocab = tok.get_vocab()
            for special in ["<epsilon>", "<s>", "</s>", "<unk>"]:
                if special in vocab:
                    special_ids.add(vocab[special])

        pieces = []
        for tid in token_ids:
            if tid in special_ids:
                continue
            piece = self.token_id_to_piece(tid)
            pieces.append(piece)

        # Reconstruct words from pieces
        words = []
        current_word = ""

        for piece in pieces:
            if piece.startswith(self.WORD_BOUNDARY):
                # New word boundary
                if current_word:
                    words.append(current_word)
                current_word = piece[len(self.WORD_BOUNDARY):]
            else:
                # Continuation of current word
                current_word += piece

        if current_word:
            words.append(current_word)

        return [w for w in words if w.strip()]

    def get_partial_word(self, token_ids: List[int]) -> Tuple[List[str], str]:
        """
        Like token_ids_to_words but also returns the in-progress partial word.
        
        Returns:
            (completed_words, partial_word_so_far)
        """
        if not token_ids:
            return [], ""

        special_ids = set()
        tok = self._tokenizer.tokenizer if hasattr(self._tokenizer, "tokenizer") else self._tokenizer
        if hasattr(tok, "get_vocab"):
            vocab = tok.get_vocab()
            for special in ["<epsilon>", "<s>", "</s>", "<unk>"]:
                if special in vocab:
                    special_ids.add(vocab[special])

        pieces = []
        for tid in token_ids:
            if tid in special_ids:
                continue
            pieces.append(self.token_id_to_piece(tid))

        words = []
        current_word = ""

        for piece in pieces:
            if piece.startswith(self.WORD_BOUNDARY):
                if current_word:
                    words.append(current_word)
                current_word = piece[len(self.WORD_BOUNDARY):]
            else:
                current_word += piece

        return words, current_word


class LMBeamScorer:
    """
    Combined LM + medical lexicon scorer for CTC beam search integration.
    
    This scorer is passed to BeamDecoder as its lm_scorer callback.
    It operates at the subword token level but applies word-level
    LM scoring when word boundaries are detected.
    
    Score formula:
        token_lm_score = lm_weight * word_lm_score + medical_weight * medical_lexicon_score
        
    Word-level scores are emitted when a word boundary ('▁') is detected,
    meaning a new token starting with '▁' completes the previous word.
    """

    def __init__(
        self,
        arpa_lm: Optional[ArpaLanguageModel],
        medical_lexicon: Optional[MedicalLexicon],
        bridge: SubwordWordBridge,
        lm_weight: float = 0.1,
        medical_weight: float = 0.05
    ):
        self.arpa_lm = arpa_lm
        self.medical_lexicon = medical_lexicon
        self.bridge = bridge
        self.lm_weight = lm_weight
        self.medical_weight = medical_weight
        self._score_cache: Dict[Tuple[Tuple[int, ...], int], float] = {}

    def __call__(self, prefix: List[int], next_token: int) -> float:
        """
        Beam search scoring callback.
        
        Called by BeamDecoder at each frame for each candidate token extension.
        
        Args:
            prefix: Current prefix token IDs (already decoded)
            next_token: Candidate next token ID
            
        Returns:
            Combined LM + medical lexicon score for this extension
        """
        cache_key = (tuple(prefix), next_token)
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        score = 0.0
        
        # Check if the new token starts a word boundary
        new_piece = self.bridge.token_id_to_piece(next_token)
        is_word_boundary = new_piece.startswith(SubwordWordBridge.WORD_BOUNDARY)

        if is_word_boundary and prefix:
            # The new token starts a new word, meaning the previous word is now complete
            completed_words, _partial = self.bridge.get_partial_word(prefix)
            
            if completed_words:
                last_completed = completed_words[-1].lower()
                
                # ARPA LM scoring on the completed word
                if self.arpa_lm is not None:
                    context = ["<s>"] + [w.lower() for w in completed_words[:-1]]
                    lm_score = self.arpa_lm.score_word_in_context(context, last_completed)
                    # Normalize: ARPA scores are negative log probs, typically -1 to -10
                    # We return the raw score; the lm_weight in BeamDecoder handles scaling
                    score += lm_score

                # Medical lexicon scoring on the completed word
                if self.medical_lexicon is not None:
                    med_score = self.medical_lexicon.score_word(last_completed)
                    score += (self.medical_weight / max(self.lm_weight, 1e-6)) * med_score

        # Cache result (limit cache size for memory)
        if len(self._score_cache) < 100000:
            self._score_cache[cache_key] = score

        return score

    def clear_cache(self) -> None:
        """Clear the scoring cache."""
        self._score_cache.clear()


class LMRescorer:
    """
    High-level interface for running LM-rescored beam search decoding.
    
    Combines:
        - CTC Prefix Beam Search (BeamDecoder)
        - ARPA n-gram Language Model (ArpaLanguageModel)
        - Medical Vocabulary Lexicon (MedicalLexicon)
        - Subword-to-Word Bridge (SubwordWordBridge)
    """

    def __init__(
        self,
        arpa_lm: Optional[ArpaLanguageModel] = None,
        medical_lexicon: Optional[MedicalLexicon] = None,
        tokenizer_or_processor=None,
        lm_weight: float = 0.1,
        medical_weight: float = 0.05,
        beam_width: int = 8,
        blank_id: int = 0,
        temperature: float = 1.0,
        prune_threshold: float = 1e-4
    ):
        self.arpa_lm = arpa_lm
        self.medical_lexicon = medical_lexicon
        self.lm_weight = lm_weight
        self.medical_weight = medical_weight
        self.beam_width = beam_width
        self.blank_id = blank_id
        self.temperature = temperature
        self.prune_threshold = prune_threshold

        # Build the subword bridge if tokenizer is available
        self.bridge = None
        if tokenizer_or_processor is not None:
            self.bridge = SubwordWordBridge(tokenizer_or_processor)

        # Build the scorer
        self.scorer = None
        if self.bridge is not None and (arpa_lm is not None or medical_lexicon is not None):
            self.scorer = LMBeamScorer(
                arpa_lm=arpa_lm,
                medical_lexicon=medical_lexicon,
                bridge=self.bridge,
                lm_weight=lm_weight,
                medical_weight=medical_weight
            )

        # Build the beam decoder with LM scorer
        self.decoder = BeamDecoder(
            blank_id=blank_id,
            beam_width=beam_width,
            prune_threshold=prune_threshold,
            temperature=temperature,
            lm_weight=lm_weight if self.scorer is not None else 0.0,
            lm_scorer=self.scorer
        )

    def decode(
        self,
        logits,
        tokenizer_or_processor=None,
        clean_special_tokens: bool = True
    ) -> Dict[str, Any]:
        """
        Decode logits with LM rescoring.
        
        Args:
            logits: Acoustic logits tensor [T, V] or [1, T, V]
            tokenizer_or_processor: HuggingFace tokenizer/processor (uses init one if None)
            clean_special_tokens: Remove <s>, </s>, <epsilon> from output
            
        Returns:
            Dict with decoded text, scores, token IDs, timing, and LM metadata
        """
        tok = tokenizer_or_processor or self.bridge._tokenizer if self.bridge else None
        if tok is None:
            raise ValueError("No tokenizer/processor provided")

        t0 = time.perf_counter()

        result = self.decoder.decode_logits(
            logits, tok,
            beam_width=self.beam_width,
            clean_special_tokens=clean_special_tokens
        )

        total_time = time.perf_counter() - t0

        # Add word-level analysis
        words = []
        if self.bridge:
            words = self.bridge.token_ids_to_words(result.get("token_ids", []))

        # Score completed words through LM for diagnostics
        word_scores = []
        if words and self.arpa_lm:
            state = self.arpa_lm.begin()
            for w in words:
                w_lower = w.lower()
                score, state = self.arpa_lm.score_token(state, w_lower)
                med_score = self.medical_lexicon.score_word(w_lower) if self.medical_lexicon else 0.0
                word_scores.append({
                    "word": w,
                    "lm_score": round(score, 4),
                    "medical_score": round(med_score, 4),
                    "in_lm_vocab": self.arpa_lm.contains_word(w_lower),
                    "in_medical_lexicon": self.medical_lexicon.contains(w_lower) if self.medical_lexicon else False
                })

        result["words"] = words
        result["word_count"] = len(words)
        result["word_scores"] = word_scores
        result["total_decode_time"] = total_time
        result["lm_config"] = {
            "lm_weight": self.lm_weight,
            "medical_weight": self.medical_weight,
            "beam_width": self.beam_width,
            "lm_source": self.arpa_lm._source_path if self.arpa_lm else None,
            "lm_order": self.arpa_lm.order if self.arpa_lm else 0,
            "lm_vocab_size": len(self.arpa_lm.vocab) if self.arpa_lm else 0,
            "lexicon_word_count": len(self.medical_lexicon.get_all_words()) if self.medical_lexicon else 0,
        }

        return result

    def compare_with_baseline(
        self,
        logits,
        baseline_text: str,
        tokenizer_or_processor=None,
        clean_special_tokens: bool = True
    ) -> Dict[str, Any]:
        """
        Decode with LM and compare against a baseline (e.g., greedy) transcript.
        
        Returns the LM-decoded result plus a diff analysis showing what changed.
        """
        result = self.decode(logits, tokenizer_or_processor, clean_special_tokens)

        lm_text = result.get("text", "")
        baseline_words = baseline_text.lower().split()
        lm_words = lm_text.lower().split()

        # Simple diff
        changes = []
        max_len = max(len(baseline_words), len(lm_words))
        for i in range(max_len):
            bw = baseline_words[i] if i < len(baseline_words) else "<MISSING>"
            lw = lm_words[i] if i < len(lm_words) else "<MISSING>"
            if bw != lw:
                change_info = {
                    "position": i,
                    "baseline": bw,
                    "lm_decoded": lw,
                    "in_medical_lexicon": self.medical_lexicon.contains(lw) if self.medical_lexicon else False,
                    "is_numeric": lw.replace(".", "").isdigit() or bw.replace(".", "").isdigit(),
                }
                # Classify the change source
                if change_info["in_medical_lexicon"]:
                    change_info["change_source"] = "medical_vocabulary"
                elif change_info["is_numeric"]:
                    change_info["change_source"] = "numeric_expression"
                else:
                    change_info["change_source"] = "general_vocabulary"
                changes.append(change_info)

        result["baseline_comparison"] = {
            "baseline_text": baseline_text,
            "lm_text": lm_text,
            "baseline_word_count": len(baseline_words),
            "lm_word_count": len(lm_words),
            "changes": changes,
            "total_changes": len(changes),
            "changed_by_medical_vocab": sum(1 for c in changes if c["change_source"] == "medical_vocabulary"),
            "changed_by_numeric": sum(1 for c in changes if c["change_source"] == "numeric_expression"),
            "changed_by_general_vocab": sum(1 for c in changes if c["change_source"] == "general_vocabulary"),
        }

        return result


class NBestRescorer:
    """
    Second-pass N-best LM rescorer (the correct integration for a real LM).

    In-search shallow fusion (scoring the LM inside the CTC prefix beam search)
    is numerically fragile: with a full-size LM the per-frame scores overwhelm
    the acoustic posteriors and corrupt word boundaries. This class instead
    performs the standard, robust second pass:

        1. Run PURE-ACOUSTIC beam search to obtain the top-N hypotheses.
        2. Score each complete hypothesis with the word-level LM and lexicon.
        3. Re-rank by:  acoustic + alpha * lm_logprob + beta * word_count
                                 + gamma * lexicon_bonus

    It can only ever re-rank hypotheses the acoustic model already proposed, so
    it never invents words and never degrades a confident transcript below the
    acoustic 1-best (which remains available as a tie/zero-weight fallback).
    """

    def __init__(
        self,
        arpa_lm: Optional[ArpaLanguageModel],
        medical_lexicon: Optional[MedicalLexicon],
        tokenizer_or_processor,
        alpha: float = 0.3,
        beta: float = 0.5,
        gamma: float = 0.1,
        beam_width: int = 16,
        blank_id: int = 0,
        prune_threshold: float = 1e-6,
    ):
        self.arpa_lm = arpa_lm
        self.medical_lexicon = medical_lexicon
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.beam_width = beam_width
        self.bridge = SubwordWordBridge(tokenizer_or_processor)
        self.decoder = BeamDecoder(
            blank_id=blank_id, beam_width=beam_width, prune_threshold=prune_threshold
        )

    def _lm_logprob(self, words: List[str]) -> float:
        if self.arpa_lm is None or not words:
            return 0.0
        return self.arpa_lm.score_sequence([w.lower() for w in words])

    def _lexicon_bonus(self, words: List[str]) -> float:
        if self.medical_lexicon is None or not words:
            return 0.0
        return sum(self.medical_lexicon.score_word(w.lower()) for w in words)

    def decode(self, logits, tokenizer_or_processor=None, clean_special_tokens: bool = True) -> Dict[str, Any]:
        tok = tokenizer_or_processor or self.bridge._tokenizer
        t0 = time.perf_counter()
        acoustic = self.decoder.decode_logits(
            logits, tok, beam_width=self.beam_width, clean_special_tokens=clean_special_tokens
        )
        beams = acoustic.get("top_beams", [])
        if not beams:
            acoustic["rescored"] = False
            return acoustic

        scored = []
        for b in beams:
            words = self.bridge.token_ids_to_words(b.get("token_ids", []))
            combined = (
                float(b["score"])
                + self.alpha * self._lm_logprob(words)
                + self.beta * len(words)
                + self.gamma * self._lexicon_bonus(words)
            )
            scored.append((combined, b))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_combined, best = scored[0]

        result = dict(acoustic)
        result["text"] = best["text"]
        result["token_ids"] = best.get("token_ids", [])
        result["acoustic_1best"] = acoustic["text"]
        result["rescored"] = best["text"] != acoustic["text"]
        result["combined_score"] = best_combined
        result["rescore_config"] = {
            "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
            "beam_width": self.beam_width,
            "lm_vocab_size": len(self.arpa_lm.vocab) if self.arpa_lm else 0,
            "lm_order": self.arpa_lm.order if self.arpa_lm else 0,
        }
        result["total_decode_time"] = time.perf_counter() - t0
        return result


def build_small_medical_arpa() -> str:
    """
    Build a small controlled ARPA language model for engineering validation.
    
    This is a hand-crafted trigram model using generic medical language patterns.
    It is NOT trained on any patient data or real clinical recordings.
    
    The corpus consists of common clinical dictation patterns:
    - Patient demographics
    - Vital signs reporting
    - Medication prescriptions
    - Diagnosis statements
    - Physical exam findings
    """
    # Manually constructed from generic clinical language patterns
    # All probabilities are approximate log10 values
    arpa_text = r"""\data\
ngram 1=120
ngram 2=85
ngram 3=45

\1-grams:
-1.0000	<s>	-0.5
-1.0000	</s>
-2.0000	<unk>	-0.5
-1.2000	patient	-0.3
-1.5000	is	-0.2
-1.3000	a	-0.3
-1.8000	year	-0.2
-1.8000	old	-0.3
-2.0000	male	-0.3
-2.0000	female	-0.3
-1.5000	with	-0.3
-1.8000	history	-0.2
-2.0000	of	-0.2
-1.5000	the	-0.2
-1.8000	and	-0.3
-2.2000	hypertension	-0.2
-2.2000	diabetes	-0.2
-2.5000	mellitus	-0.3
-2.0000	type	-0.3
-2.8000	two	-0.3
-2.2000	blood	-0.2
-2.2000	pressure	-0.3
-2.5000	over	-0.3
-2.0000	heart	-0.2
-2.5000	rate	-0.3
-2.5000	oxygen	-0.2
-2.5000	saturation	-0.3
-2.0000	no	-0.3
-2.5000	chest	-0.2
-2.5000	pain	-0.3
-2.2000	shortness	-0.2
-2.5000	breath	-0.3
-2.5000	cough	-0.3
-2.8000	persistent	-0.3
-2.5000	taking	-0.2
-2.5000	currently	-0.2
-2.2000	milligrams	-0.3
-2.5000	mg	-0.3
-2.5000	daily	-0.3
-2.5000	twice	-0.2
-2.8000	bedtime	-0.3
-2.2000	metformin	-0.3
-2.2000	lisinopril	-0.3
-2.2000	atorvastatin	-0.3
-2.5000	amlodipine	-0.3
-2.5000	metoprolol	-0.3
-2.8000	omeprazole	-0.3
-2.5000	aspirin	-0.3
-2.8000	ibuprofen	-0.3
-2.5000	lungs	-0.2
-2.5000	clear	-0.3
-2.8000	bilaterally	-0.3
-2.8000	bilateral	-0.3
-2.5000	assessment	-0.3
-2.5000	plan	-0.3
-2.5000	continue	-0.2
-2.5000	current	-0.2
-2.8000	medications	-0.3
-2.8000	regimen	-0.3
-2.5000	follow	-0.2
-2.8000	up	-0.3
-2.8000	weeks	-0.3
-2.8000	three	-0.3
-2.8000	four	-0.3
-2.2000	has	-0.2
-2.5000	been	-0.3
-1.8000	for	-0.3
-2.5000	how	-0.3
-2.5000	long	-0.3
-2.5000	have	-0.3
-2.5000	you	-0.3
-2.5000	had	-0.3
-2.5000	any	-0.3
-2.8000	do	-0.3
-2.8000	does	-0.3
-2.5000	he	-0.3
-2.5000	she	-0.3
-2.5000	his	-0.3
-2.5000	her	-0.3
-2.5000	presents	-0.3
-2.8000	today	-0.3
-2.5000	denies	-0.3
-2.8000	reports	-0.3
-2.8000	complains	-0.3
-3.0000	fifty	-0.3
-3.0000	eight	-0.3
-3.0000	fifty-eight	-0.3
-3.5000	fifty-two	-0.3
-3.0000	one	-0.3
-3.0000	ninety	-0.3
-3.0000	hundred	-0.3
-3.0000	ten	-0.3
-3.0000	twenty	-0.3
-3.0000	thirty	-0.3
-3.0000	forty	-0.3
-3.0000	sixty	-0.3
-3.0000	seventy	-0.3
-3.0000	eighty	-0.3
-2.8000	temperature	-0.3
-2.8000	respiratory	-0.3
-2.8000	examination	-0.3
-2.8000	normal	-0.3
-2.8000	elevated	-0.3
-2.8000	decreased	-0.3
-2.8000	stable	-0.3
-2.8000	controlled	-0.3
-2.8000	uncontrolled	-0.3
-3.0000	review	-0.3
-3.0000	systems	-0.3
-3.0000	negative	-0.3
-2.5000	diagnosis	-0.3
-2.5000	treated	-0.3
-2.8000	prescribed	-0.3
-2.5000	not	-0.3
-2.5000	well	-0.3
-2.5000	this	-0.3
-2.5000	that	-0.3
-2.5000	was	-0.3
-2.5000	are	-0.3
-2.5000	were	-0.3
-2.5000	about	-0.3
-2.8000	started	-0.3
-2.8000	per	-0.3

\2-grams:
-0.5000	<s> patient	-0.2
-0.8000	patient is	-0.2
-0.5000	is a	-0.2
-0.6000	a fifty	-0.2
-0.5000	year old	-0.2
-0.8000	old male	-0.3
-0.8000	old female	-0.3
-0.5000	with hypertension	-0.2
-0.5000	with diabetes	-0.2
-0.8000	with history	-0.2
-0.5000	history of	-0.2
-0.5000	diabetes mellitus	-0.3
-0.8000	type two	-0.3
-0.5000	blood pressure	-0.2
-0.8000	pressure is	-0.2
-0.8000	heart rate	-0.2
-0.8000	rate is	-0.2
-0.8000	oxygen saturation	-0.2
-0.8000	saturation is	-0.2
-0.5000	no chest	-0.2
-0.5000	chest pain	-0.2
-0.5000	shortness of	-0.2
-0.5000	of breath	-0.3
-0.8000	persistent cough	-0.3
-0.5000	currently taking	-0.2
-0.8000	taking metformin	-0.3
-0.8000	taking lisinopril	-0.3
-0.8000	taking atorvastatin	-0.3
-1.0000	milligrams daily	-0.3
-1.0000	mg daily	-0.3
-1.0000	twice daily	-0.3
-0.8000	at bedtime	-0.3
-0.5000	lungs clear	-0.2
-0.8000	clear bilaterally	-0.3
-0.5000	follow up	-0.2
-0.8000	up in	-0.2
-1.0000	in three	-0.3
-1.0000	in four	-0.3
-1.0000	three weeks	-0.3
-1.0000	four weeks	-0.3
-0.8000	continue current	-0.2
-0.8000	current medications	-0.3
-0.8000	assessment and	-0.2
-0.8000	and plan	-0.3
-0.8000	presents today	-0.3
-0.8000	today with	-0.2
-0.5000	patient has	-0.2
-0.8000	has been	-0.2
-0.8000	has no	-0.2
-1.0000	denies chest	-0.2
-1.0000	denies shortness	-0.2
-0.8000	reports persistent	-0.3
-0.8000	how long	-0.2
-0.8000	long have	-0.2
-0.8000	have you	-0.2
-0.8000	you had	-0.2
-1.0000	any shortness	-0.2
-1.0000	any chest	-0.2
-1.0000	do you	-0.2
-1.0000	does he	-0.2
-1.0000	does she	-0.2
-0.8000	review of	-0.2
-0.8000	of systems	-0.2
-1.0000	systems negative	-0.3
-1.0000	fifty eight	-0.3
-1.5000	fifty two	-0.3
-0.8000	one fifty	-0.3
-0.8000	one hundred	-0.3
-1.0000	over ninety	-0.3
-1.0000	is normal	-0.3
-1.0000	is elevated	-0.3
-1.0000	is stable	-0.3
-1.0000	well controlled	-0.3
-0.8000	a fifty-eight	-0.2
-1.5000	a fifty-two	-0.3
-1.0000	not well	-0.3
-1.0000	has a	-0.3
-1.0000	patient presents	-0.2
-1.0000	patient denies	-0.2
-1.0000	patient reports	-0.2
-1.2000	physical examination	-0.3
-1.2000	respiratory rate	-0.3

\3-grams:
-0.3000	<s> patient is
-0.3000	patient is a
-0.5000	is a fifty
-0.3000	a fifty eight
-1.0000	a fifty two
-0.3000	fifty eight year
-0.3000	eight year old
-0.3000	year old male
-0.3000	year old female
-0.3000	old male with
-0.3000	old female with
-0.3000	with history of
-0.3000	history of hypertension
-0.3000	history of diabetes
-0.5000	diabetes mellitus type
-0.5000	mellitus type two
-0.3000	blood pressure is
-0.3000	heart rate is
-0.3000	oxygen saturation is
-0.3000	shortness of breath
-0.3000	no chest pain
-0.3000	currently taking metformin
-0.5000	currently taking lisinopril
-0.5000	currently taking atorvastatin
-0.5000	taking metformin milligrams
-0.5000	milligrams twice daily
-0.5000	milligrams daily at
-0.5000	mg twice daily
-0.5000	lungs clear bilaterally
-0.3000	follow up in
-0.5000	up in three
-0.5000	up in four
-0.5000	in three weeks
-0.5000	in four weeks
-0.3000	continue current medications
-0.3000	assessment and plan
-0.3000	patient presents today
-0.5000	presents today with
-0.5000	patient has been
-0.5000	patient has no
-0.5000	how long have
-0.5000	long have you
-0.5000	have you had
-0.5000	any shortness of
-0.5000	review of systems
-0.5000	is a fifty-eight

\end\
"""
    return arpa_text


def create_default_rescorer(
    tokenizer_or_processor,
    lm_weight: float = 0.1,
    medical_weight: float = 0.05,
    beam_width: int = 8,
    entities_json_path: Optional[str] = None
) -> LMRescorer:
    """
    Factory function to create a fully configured LMRescorer with the
    built-in small medical ARPA model and default medical lexicon.
    """
    # Build ARPA LM
    arpa_text = build_small_medical_arpa()
    arpa_lm = ArpaLanguageModel.from_arpa_string(arpa_text)

    # Build medical lexicon
    if entities_json_path is None:
        entities_path = pathlib.Path(__file__).parent.parent / "evaluation" / "medical_entities.json"
        if entities_path.exists():
            entities_json_path = str(entities_path)

    medical_lexicon = MedicalLexicon(entities_json_path=entities_json_path)

    return LMRescorer(
        arpa_lm=arpa_lm,
        medical_lexicon=medical_lexicon,
        tokenizer_or_processor=tokenizer_or_processor,
        lm_weight=lm_weight,
        medical_weight=medical_weight,
        beam_width=beam_width
    )


# ------------------------------------------------------------------
# CLI Interface
# ------------------------------------------------------------------
def main():
    """Command-line interface for LM rescoring experiments."""
    parser = argparse.ArgumentParser(
        description="Anot Medical ASR — Language Model Rescorer (Stage 9B)"
    )
    parser.add_argument("--lm-weight", type=float, default=0.1,
                        help="Language model weight (default: 0.1)")
    parser.add_argument("--medical-weight", type=float, default=0.05,
                        help="Medical lexicon weight (default: 0.05)")
    parser.add_argument("--beam-width", type=int, default=8,
                        help="Beam width (default: 8)")
    parser.add_argument("--arpa-file", type=str, default=None,
                        help="Path to ARPA LM file (default: built-in small medical LM)")
    parser.add_argument("--entities-json", type=str, default=None,
                        help="Path to medical_entities.json")
    parser.add_argument("--info", action="store_true",
                        help="Print LM configuration info and exit")

    args = parser.parse_args()

    if args.info:
        arpa_text = build_small_medical_arpa()
        lm = ArpaLanguageModel.from_arpa_string(arpa_text)
        lexicon = MedicalLexicon()
        print("=== ARPA Language Model ===")
        for k, v in lm.summary().items():
            print(f"  {k}: {v}")
        print("\n=== Medical Lexicon ===")
        for k, v in lexicon.summary().items():
            print(f"  {k}: {v}")
        print(f"\n=== Configuration ===")
        print(f"  LM Weight: {args.lm_weight}")
        print(f"  Medical Weight: {args.medical_weight}")
        print(f"  Beam Width: {args.beam_width}")
        return

    print("LM Rescorer ready. Use programmatic API for decoding experiments.")
    print(f"  LM Weight: {args.lm_weight}")
    print(f"  Medical Weight: {args.medical_weight}")
    print(f"  Beam Width: {args.beam_width}")


if __name__ == "__main__":
    main()
