#!/usr/bin/env python3
"""
Anot Medical ASR — Greedy CTC Decoder (Baseline)
=================================================
Stage 9: Reference implementation of greedy CTC decoding for MedASR.

Greedy decoding takes the argmax token at each acoustic frame, collapses
consecutive duplicate tokens, removes the CTC blank token (<epsilon>, index 0),
and decodes the resulting token sequence using the subword tokenizer.

This reproduces the exact authoritative baseline behavior of MedASR.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import numpy as np


class GreedyDecoder:
    """
    Deterministic Greedy CTC Decoder for MedASR (LasrForCTC).
    """

    def __init__(self, blank_id: int = 0):
        self.blank_id = blank_id

    def decode_token_ids(self, token_ids: List[int]) -> List[int]:
        """
        Apply CTC collapse and blank removal on a sequence of frame-level token IDs.
        1. Collapse consecutive identical tokens.
        2. Remove blank tokens (<epsilon>).
        """
        collapsed = []
        prev = None
        for tid in token_ids:
            if tid != prev:
                if tid != self.blank_id:
                    collapsed.append(tid)
                prev = tid
        return collapsed

    def decode_logits(
        self,
        logits: Union[torch.Tensor, np.ndarray],
        tokenizer_or_processor,
        clean_special_tokens: bool = False
    ) -> Dict[str, Any]:
        """
        Decode logits tensor [T, V] or [1, T, V] into raw transcript.
        
        Returns:
            Dict containing:
                - 'text': decoded text transcript
                - 'token_ids': collapsed non-blank token IDs
                - 'frame_ids': raw argmax ID per frame
        """
        if isinstance(logits, np.ndarray):
            logits_tensor = torch.from_numpy(logits)
        else:
            logits_tensor = logits

        if logits_tensor.ndim == 3:
            logits_tensor = logits_tensor.squeeze(0)

        # Frame-level argmax
        if not isinstance(logits_tensor, torch.Tensor):
            logits_tensor = torch.from_numpy(logits_tensor)
        pred_ids = torch.argmax(logits_tensor, dim=-1)

        frame_ids = pred_ids.squeeze().tolist()
        if isinstance(frame_ids, int):
            frame_ids = [frame_ids]
        collapsed_ids = self.decode_token_ids(frame_ids)

        # 1. Full Processor with CTC decoder (e.g. LasrProcessor)
        if hasattr(tokenizer_or_processor, "feature_extractor") and hasattr(tokenizer_or_processor, "batch_decode"):
            if pred_ids.ndim == 1:
                batch_tensor = pred_ids.unsqueeze(0)
            else:
                batch_tensor = pred_ids
            text = tokenizer_or_processor.batch_decode(batch_tensor)[0]
        elif hasattr(tokenizer_or_processor, "decode"):
            text = tokenizer_or_processor.decode(collapsed_ids, clean_up_tokenization_spaces=False)
        elif hasattr(tokenizer_or_processor, "batch_decode"):
            text = tokenizer_or_processor.batch_decode([collapsed_ids])[0]
        else:
            raise AttributeError("Object has neither batch_decode nor decode method.")

        if clean_special_tokens:
            text = text.replace("</s>", "").replace("<s>", "").replace("<epsilon>", "").strip()

        return {
            "text": text,
            "token_ids": collapsed_ids,
            "frame_ids": frame_ids,
            "frame_count": len(frame_ids),
            "token_count": len(collapsed_ids)
        }

    def decode_batch(
        self,
        batch_logits: Union[torch.Tensor, np.ndarray],
        tokenizer,
        clean_special_tokens: bool = False
    ) -> List[Dict[str, Any]]:
        """Decode a batch of logits tensors [B, T, V]."""
        if isinstance(batch_logits, np.ndarray):
            batch_logits = torch.from_numpy(batch_logits)

        results = []
        for i in range(batch_logits.shape[0]):
            res = self.decode_logits(batch_logits[i], tokenizer, clean_special_tokens=clean_special_tokens)
            results.append(res)
        return results
