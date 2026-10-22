#!/usr/bin/env python3
"""
Anot Medical ASR — CTC Prefix Beam Search Decoder
=================================================
Stage 9: Native, deterministic CTC Prefix Beam Search implementation.

Implements prefix-tree search over CTC acoustic frame posteriors (Graves et al.,
Hannun et al.) with optional n-gram language model rescoring and length normalization.

Zero external C++ compiler or wheel dependencies (100% pure Python / PyTorch / NumPy).
Avoids the NumPy 1.x downgrade risks imposed by pyctcdecode on Windows.
"""

import math
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

import numpy as np
import torch


class BeamDecoder:
    """
    Deterministic CTC Prefix Beam Search Decoder for MedASR (LasrForCTC).
    
    Features:
    - Maintains top B prefix paths tracking blank (p_b) and non-blank (p_nb) probabilities.
    - Subword-aware token extensions matching SentencePiece vocabularies.
    - Configurable beam width, pruning threshold, and blank penalty.
    - Modular language model scoring interface: score = log P_CTC + alpha * log P_LM + beta * len.
    - Multi-beam hypothesis extraction for diagnostic inspectability.
    """

    def __init__(
        self,
        blank_id: int = 0,
        beam_width: int = 8,
        prune_threshold: float = 1e-4,
        temperature: float = 1.0,
        blank_penalty: float = 0.0,
        lm_weight: float = 0.0,
        length_penalty: float = 0.0,
        lm_scorer: Optional[Callable[[List[int], int], float]] = None
    ):
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.prune_threshold = prune_threshold
        self.temperature = max(1e-4, temperature)
        self.blank_penalty = blank_penalty
        self.lm_weight = lm_weight
        self.length_penalty = length_penalty
        self.lm_scorer = lm_scorer

    def decode_logits(
        self,
        logits: Union[torch.Tensor, np.ndarray],
        tokenizer,
        beam_width: Optional[int] = None,
        prune_threshold: Optional[float] = None,
        clean_special_tokens: bool = False
    ) -> Dict[str, Any]:
        """
        Execute CTC Prefix Beam Search on acoustic logits [T, V] or [1, T, V].
        
        Returns:
            Dict containing:
                - 'text': Best hypothesis decoded text
                - 'score': Total log-probability of best hypothesis
                - 'token_ids': Best hypothesis token IDs
                - 'top_beams': List of top N hypotheses with scores and texts
                - 'decode_time_seconds': Execution time
                - 'beam_width': Beam width used
        """
        t0 = time.perf_counter()
        bw = beam_width if beam_width is not None else self.beam_width
        pt = prune_threshold if prune_threshold is not None else self.prune_threshold

        if isinstance(logits, np.ndarray):
            logits_tensor = torch.from_numpy(logits)
        else:
            logits_tensor = logits

        if logits_tensor.ndim == 3:
            logits_tensor = logits_tensor.squeeze(0)

        # Scale by temperature and compute log-softmax
        if self.temperature != 1.0:
            logits_tensor = logits_tensor / self.temperature

        log_probs = torch.log_softmax(logits_tensor, dim=-1).cpu().numpy()
        T, V = log_probs.shape

        # Initial beam: empty prefix () with P_blank = 1.0 (log = 0.0), P_non_blank = 0.0 (log = -inf)
        # Beams map: prefix tuple -> (log_p_blank, log_p_non_blank)
        beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -float("inf"))}

        log_prune_thresh = math.log(max(pt, 1e-12))

        for t in range(T):
            curr_log_probs = log_probs[t]
            max_lp = float(np.max(curr_log_probs))
            cutoff = max_lp + log_prune_thresh

            # Active candidate tokens above threshold
            cand_indices = np.where(curr_log_probs >= cutoff)[0]
            if self.blank_id not in cand_indices:
                cand_indices = np.append(cand_indices, self.blank_id)

            next_beams = defaultdict(lambda: (-float("inf"), -float("inf")))
            lp_blank = float(curr_log_probs[self.blank_id]) - self.blank_penalty

            for prefix, (p_b, p_nb) in beams.items():
                p_total = np.logaddexp(p_b, p_nb)

                # 1. Blank extension: prefix remains unchanged
                curr_pb, curr_pnb = next_beams[prefix]
                next_beams[prefix] = (np.logaddexp(curr_pb, p_total + lp_blank), curr_pnb)

                # 2. Non-blank extensions
                last_token = prefix[-1] if prefix else None

                for c in cand_indices:
                    c = int(c)
                    if c == self.blank_id:
                        continue

                    lp_c = float(curr_log_probs[c])

                    # Apply LM scoring if provided
                    if self.lm_scorer is not None and self.lm_weight > 0.0:
                        lm_score = self.lm_scorer(list(prefix), c)
                        lp_c += self.lm_weight * lm_score

                    if self.length_penalty > 0.0:
                        lp_c += self.length_penalty

                    if c == last_token:
                        # Extension of repeated token:
                        # From p_nb: stays same prefix (collapsed by CTC)
                        curr_pb, curr_pnb = next_beams[prefix]
                        next_beams[prefix] = (curr_pb, np.logaddexp(curr_pnb, p_nb + lp_c))

                        # From p_b: emits new repeated token (separated by blank)
                        new_prefix = prefix + (c,)
                        np_b, np_nb = next_beams[new_prefix]
                        next_beams[new_prefix] = (np_b, np.logaddexp(np_nb, p_b + lp_c))
                    else:
                        # Extension of new token: expands prefix from either blank or non-blank state
                        new_prefix = prefix + (c,)
                        np_b, np_nb = next_beams[new_prefix]
                        next_beams[new_prefix] = (np_b, np.logaddexp(np_nb, p_total + lp_c))

            # Prune to beam width
            scored = []
            for p, (pb, pnb) in next_beams.items():
                tot = float(np.logaddexp(pb, pnb))
                scored.append((tot, p, pb, pnb))

            scored.sort(key=lambda x: x[0], reverse=True)
            beams = {p: (pb, pnb) for _, p, pb, pnb in scored[:bw]}

        # Sort final beams by total log probability
        final_candidates = []
        for prefix, (pb, pnb) in beams.items():
            tot_score = float(np.logaddexp(pb, pnb))
            final_candidates.append((tot_score, prefix))

        final_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_prefix = final_candidates[0]

        # Decode tokens to string using tokenizer or processor
        def tokens_to_text(t_ids: Tuple[int, ...]) -> str:
            tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
            if hasattr(tok, "decode"):
                t = tok.decode(list(t_ids), clean_up_tokenization_spaces=False)
            elif hasattr(tok, "batch_decode"):
                t = tok.batch_decode([list(t_ids)])[0]
            else:
                t = " ".join(str(i) for i in t_ids)
            if clean_special_tokens:
                t = t.replace("</s>", "").replace("<s>", "").replace("<epsilon>", "").strip()
            return t

        best_text = tokens_to_text(best_prefix)

        top_beams_output = []
        for score, prefix in final_candidates[:bw]:
            top_beams_output.append({
                "score": score,
                "token_ids": list(prefix),
                "text": tokens_to_text(prefix)
            })

        dec_time = time.perf_counter() - t0

        return {
            "text": best_text,
            "score": best_score,
            "token_ids": list(best_prefix),
            "top_beams": top_beams_output,
            "decode_time_seconds": dec_time,
            "beam_width": bw
        }
