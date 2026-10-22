#!/usr/bin/env python3
"""
Anot Medical ASR — CTC Decoding Service
=======================================
Stage 11: Modular CTC decoder service supporting Greedy (default) and Beam + Medical LM.
"""

import time
from typing import Any, Dict, Optional, Tuple
import torch
from transformers import AutoProcessor

from app.core.config import settings, PROJECT_ROOT
from decoding.greedy_decoder import GreedyDecoder
from decoding.beam_decoder import BeamDecoder
from decoding.ctc_postprocess import CTCGlitchCorrector, build_default_vocabulary
from language_model.arpa_lm import ArpaLanguageModel
from language_model.medical_lexicon import MedicalLexicon
from language_model.lm_rescorer import NBestRescorer, build_small_medical_arpa


class DecodingService:
    """Manages CTC decoding strategies: Greedy (default), Beam, and Beam + Medical LM."""

    def __init__(self):
        self._greedy_decoder = GreedyDecoder(blank_id=0)
        self._arpa_lm: Optional[ArpaLanguageModel] = None
        self._medical_lexicon: Optional[MedicalLexicon] = None
        self._glitch_corrector: Optional[CTCGlitchCorrector] = None

    def _ensure_lm_assets(self) -> Tuple[ArpaLanguageModel, MedicalLexicon]:
        """
        Lazy load the ARPA LM and medical lexicon for beam rescoring.

        Prefers a locally-trained in-domain medical LM at
        language_model/models/medical.arpa (built by build_medical_lm.py from the
        local clinical-note corpus; PHI-derived, gitignored). Falls back to the
        tiny built-in validation ARPA when no trained model is present.
        """
        if self._arpa_lm is None:
            trained_path = PROJECT_ROOT / "language_model" / "models" / "medical.arpa"
            if trained_path.exists():
                self._arpa_lm = ArpaLanguageModel.from_arpa_file(str(trained_path))
            else:
                self._arpa_lm = ArpaLanguageModel.from_arpa_string(build_small_medical_arpa())
        if self._medical_lexicon is None:
            ent_path = settings.TEMP_DIR.parent.parent / "evaluation" / "medical_entities.json"
            if ent_path.exists():
                self._medical_lexicon = MedicalLexicon(entities_json_path=str(ent_path))
            else:
                self._medical_lexicon = MedicalLexicon()
        return self._arpa_lm, self._medical_lexicon

    def _ensure_glitch_corrector(self) -> CTCGlitchCorrector:
        """
        Lazy-build the conservative CTC surface-glitch corrector from the domain
        vocabulary (medical lexicon + n-gram LM vocab + number words). It repairs
        only unambiguous non-word glitches (e.g. 'examinnation' -> 'examination')
        and never touches any token containing a digit, so clinical values are
        never altered.
        """
        if self._glitch_corrector is None:
            arpa_lm, med_lex = self._ensure_lm_assets()
            vocab = build_default_vocabulary(arpa_lm=arpa_lm, medical_lexicon=med_lex)
            self._glitch_corrector = CTCGlitchCorrector(vocab)
        return self._glitch_corrector

    def decode(
        self,
        logits: torch.Tensor,
        processor: AutoProcessor,
        decoder_type: str = "greedy",
        beam_width: Optional[int] = None,
        use_lm: bool = False,
        lm_weight: Optional[float] = None,
        medical_weight: Optional[float] = None,
        correct_glitches: bool = True,
    ) -> Dict[str, Any]:
        """
        Decode CTC acoustic logits into text.
        Default: greedy (empirically validated 11.01% baseline).
        """
        decoder_type = (decoder_type or settings.DECODER_DEFAULT).lower()
        bw = beam_width or settings.BEAM_WIDTH_DEFAULT
        lm_w = lm_weight if lm_weight is not None else settings.LM_WEIGHT_DEFAULT
        med_w = medical_weight if medical_weight is not None else settings.MEDICAL_WEIGHT_DEFAULT

        t0 = time.perf_counter()

        if decoder_type == "beam":
            if use_lm:
                # Beam search + second-pass N-best LM rescoring.
                # N-best rescoring (not in-search shallow fusion) is the robust
                # integration for a full-size LM: it re-ranks complete acoustic
                # hypotheses and can never corrupt word boundaries or invent words.
                arpa_lm, med_lex = self._ensure_lm_assets()
                # Map lm_weight -> alpha; a small default is overridden by any
                # explicit request. beta (word bonus) counteracts LM brevity bias.
                alpha = lm_w if lm_w and lm_w > 0 else 0.3
                rescorer = NBestRescorer(
                    arpa_lm=arpa_lm,
                    medical_lexicon=med_lex,
                    tokenizer_or_processor=processor,
                    alpha=alpha,
                    beta=0.5,
                    gamma=med_w,
                    beam_width=bw,
                )
                res = rescorer.decode(logits, processor, clean_special_tokens=True)
                raw_text = res["text"]
                decoder_info = {
                    "type": "beam",
                    "beam_width": bw,
                    "language_model": True,
                    "rescoring": "nbest_second_pass",
                    "lm_alpha": alpha,
                    "medical_weight": med_w,
                    "lm_vocab_size": res.get("rescore_config", {}).get("lm_vocab_size", 0),
                    "rescored": res.get("rescored", False),
                }
            else:
                # Pure acoustic Beam Search
                beam_dec = BeamDecoder(blank_id=0, beam_width=bw, prune_threshold=1e-4)
                res = beam_dec.decode_logits(logits, processor, clean_special_tokens=True)
                raw_text = res["text"]
                decoder_info = {
                    "type": "beam",
                    "beam_width": bw,
                    "language_model": False,
                    "lm_weight": None,
                    "medical_weight": None
                }
        else:
            # Default production Greedy Decoding (fastest, lowest error rate)
            res = self._greedy_decoder.decode_logits(logits, processor, clean_special_tokens=True)
            raw_text = res["text"]
            decoder_info = {
                "type": "greedy",
                "beam_width": None,
                "language_model": False,
                "lm_weight": None,
                "medical_weight": None
            }

        # Conservative, fact-preserving CTC surface-glitch repair (safe for all
        # decoder types). Only unambiguous non-word -> known-word fixes; digits
        # and clinical values are never touched.
        glitch_events: list = []
        text_out = raw_text.strip()
        if correct_glitches and text_out:
            corrector = self._ensure_glitch_corrector()
            text_out, glitch_events = corrector.correct_text(text_out)

        decoder_info["glitch_correction"] = correct_glitches
        decoder_info["glitch_corrections_count"] = len(glitch_events)
        decoder_info["glitch_corrections"] = glitch_events

        decode_time = time.perf_counter() - t0

        return {
            "text": text_out.strip(),
            "decoder_info": decoder_info,
            "decode_time_seconds": decode_time
        }
