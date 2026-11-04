#!/usr/bin/env python3
"""
Anot Medical ASR — Whisper large-v3 Model Lifecycle Manager
===========================================================
Stage 21: Production singleton for the Whisper large-v3 engine (faster-whisper /
CTranslate2). Local, batch transcription of English patient-visit recordings.

Chosen as the default engine after a head-to-head benchmark against MedASR:
on labeled conversational/vitals/medication cases Whisper large-v3 reached
~97.6% clinical accuracy vs MedASR's 92.3%, and produced clean transcripts of
real spontaneous visit audio where the dictation-only MedASR garbles.

Audio is decoded to a 16 kHz mono float32 array by the caller (AudioService),
so this wrapper never touches files and avoids faster-whisper's internal
decoder (which is sensitive to the installed PyAV version).
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.core.logging import logger
from language_model.gap_recovery import filter_recovered, find_gaps
from language_model.language_spans import build_spans, english_flags, format_marker, join_with_translation_labels
from language_model.phrase_library import build_hotwords, build_initial_prompt


def _resolve_hotwords(language: Optional[str]) -> Optional[str]:
    """
    Resolve hotwords for a language:
      explicit env override  ->  multilingual phrase library (merges the clinic
      vocabulary mined from local notes + curated terms/lay expressions).
    """
    explicit = settings.WHISPER_HOTWORDS.strip()
    if explicit:
        return explicit
    return build_hotwords(language)


def _resolve_prompt(language: Optional[str]) -> Optional[str]:
    """Explicit env prompt override, else the language-appropriate library prompt."""
    explicit = settings.WHISPER_INITIAL_PROMPT.strip()
    return explicit or build_initial_prompt(language)


class WhisperASRModel:
    """Thread-safe singleton wrapper around faster-whisper WhisperModel."""

    _instance: Optional["WhisperASRModel"] = None

    def __new__(cls) -> "WhisperASRModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._device = settings.WHISPER_DEVICE
            cls._instance._model_size = settings.WHISPER_MODEL_SIZE
            cls._instance._compute_type = settings.WHISPER_COMPUTE_TYPE
            cls._instance._load_time = 0.0
            cls._instance._is_loaded = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "WhisperASRModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_loaded(self) -> bool:
        return self._is_loaded and self._model is not None

    def is_ready(self) -> bool:
        return self.is_loaded()

    def get_device(self) -> str:
        return self._device

    def get_info(self) -> Dict[str, Any]:
        return {
            "model_id": f"whisper-{self._model_size}",
            "architecture": "Whisper (faster-whisper/CTranslate2)",
            "device": self._device,
            "compute_type": self._compute_type,
            "is_loaded": self._is_loaded,
            "load_time_seconds": round(self._load_time, 4),
        }

    def load(self, device: Optional[str] = None, target_device: Optional[str] = None) -> None:
        """Load the Whisper model. Signature mirrors MedASRModel.load for drop-in use."""
        if self._is_loaded:
            logger.info("Whisper model already loaded and ready.")
            return

        # Lazy import so the dependency is only required when this engine is used.
        from faster_whisper import WhisperModel

        chosen_device = (target_device or device or self._device).lower()
        t0 = time.perf_counter()
        logger.info(
            f"Loading Whisper '{self._model_size}' "
            f"(device={chosen_device}, compute_type={self._compute_type})..."
        )
        try:
            self._model = WhisperModel(
                self._model_size, device=chosen_device, compute_type=self._compute_type,
                cpu_threads=settings.WHISPER_CPU_THREADS,
            )
            self._device = chosen_device
        except Exception as e:
            # CTranslate2 may reject a compute_type on some hardware; fall back.
            logger.warning(f"Whisper load with compute_type={self._compute_type} failed ({e}); "
                           f"retrying with compute_type=int8 on cpu.")
            self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8",
                                       cpu_threads=settings.WHISPER_CPU_THREADS)
            self._device = "cpu"
            self._compute_type = "int8"

        self._batched = None
        if settings.WHISPER_BATCHED:
            from faster_whisper import BatchedInferencePipeline
            self._batched = BatchedInferencePipeline(model=self._model)

        # Warm-up on 1s of silence to avoid first-request latency.
        try:
            _ = list(self._model.transcribe(np.zeros(16000, dtype=np.float32), language="en")[0])
        except Exception as e:
            logger.warning(f"Whisper warm-up completed with non-fatal notice: {e}")

        self._load_time = time.perf_counter() - t0
        self._is_loaded = True
        logger.info(f"Whisper model ready for inference in {self._load_time:.2f}s.")

    def transcribe(
        self, waveform: np.ndarray, sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Transcribe a 16 kHz mono float32 waveform.

        `language`: BCP-47-ish code ("en", "bn", ...), "auto" to detect, or None
        to use the configured default. Hotwords and the context prompt are chosen
        for the resolved language from the multilingual phrase library.

        Returns:
            (full_text, segments, inference_seconds)
            where segments is a list of {"start", "end", "text"} dicts.
        """
        if not self._is_loaded or self._model is None:
            raise RuntimeError("WhisperASRModel is not loaded. Call load() before transcribe().")
        if sample_rate != 16000:
            raise ValueError("Whisper expects 16 kHz audio; resample upstream in AudioService.")

        lang = (language or settings.WHISPER_LANGUAGE or "en").lower()
        fw_language = None if lang == "auto" else lang  # None => Whisper auto-detect
        hotwords = None if settings.WHISPER_PLAIN else _resolve_hotwords(lang)
        initial_prompt = None if settings.WHISPER_PLAIN else _resolve_prompt(lang)

        t0 = time.perf_counter()
        spans = self._plan_spans(waveform) if (settings.NON_ENGLISH_MARK and lang == "en") else             [(0.0, waveform.size / 16000.0, True)]
        segments: List[Dict[str, Any]] = []
        low_conf = 0
        self._last_detected_language = fw_language or lang
        non_english_sec = 0.0
        for span_start, span_end, is_english in spans:
            if not is_english and settings.NON_ENGLISH_MODE == "translate":
                non_english_sec += span_end - span_start
                chunk = waveform[int(span_start * 16000):int(span_end * 16000)]
                segments.extend(self._translate_span(chunk, span_start))
                continue
            if not is_english:
                non_english_sec += span_end - span_start
                segments.append({
                    "start": round(span_start, 2), "end": round(span_end, 2),
                    "text": format_marker(span_start, span_end), "non_english": True,
                    "avg_logprob": None, "no_speech_prob": None, "low_confidence": True,
                })
                continue
            chunk = waveform[int(span_start * 16000):int(span_end * 16000)]
            span_segments, detected = self._decode(chunk, span_start, fw_language, initial_prompt, hotwords)
            if detected and len(spans) == 1:
                self._last_detected_language = detected
            segments.extend(span_segments)
        if settings.RECOVER_GAPS and lang == "en":
            segments = self._recover_gaps(waveform, segments, hotwords, initial_prompt)
        low_conf = sum(1 for sg in segments if sg.get("low_confidence") and not sg.get("non_english"))
        self._last_non_english_seconds = non_english_sec
        inference_time = time.perf_counter() - t0
        self._last_low_confidence_count = low_conf
        return join_with_translation_labels(segments), segments, inference_time

    def _plan_spans(self, waveform: np.ndarray) -> List[Tuple[float, float, bool]]:
        """Split the recording into English / non-English spans from per-window language detection."""
        win = settings.LANG_WINDOW_SEC
        step = int(win * 16000)
        p_en: List[Optional[float]] = []
        for k in range(0, waveform.size, step):
            seg = waveform[k:k + step]
            if seg.size < settings.LANG_MIN_SEC * 16000 or float(np.sqrt(np.mean(np.square(seg)))) < 1e-4:
                p_en.append(None)
                continue
            _lang, _prob, all_probs = self._model.detect_language(seg)
            p_en.append(float(dict(all_probs).get("en", 0.0)))
        flags = english_flags(p_en, settings.LANG_EN_THRESHOLD)
        return build_spans(flags, win, waveform.size / 16000.0)

    def _recover_gaps(self, waveform: np.ndarray, segments: List[Dict[str, Any]],
                      hotwords: Optional[str], initial_prompt: Optional[str]) -> List[Dict[str, Any]]:
        """Second pass over loud stretches with no text: a more sensitive speech detector; English is transcribed,
        other languages are translated. Results are flagged low-confidence and screened for noise inventions."""
        per_sec = waveform[: (waveform.size // 16000) * 16000].reshape(-1, 16000)
        level = np.sqrt(np.mean(np.square(per_sec), axis=1))
        duration = waveform.size / 16000.0
        found: List[Dict[str, Any]] = []
        gaps = find_gaps(segments, level, duration, rel=settings.RECOVER_REL_LEVEL)
        for g_start, g_end in gaps:
            a, b = max(0.0, g_start - 0.5), min(duration, g_end + 0.5)
            chunk = waveform[int(a * 16000):int(b * 16000)]
            try:
                _l, _p, probs = self._model.detect_language(chunk[: 30 * 16000])
                is_en = float(dict(probs).get("en", 0.0)) >= settings.LANG_EN_THRESHOLD
                if is_en:
                    segs, _d = self._decode(chunk, a, "en", initial_prompt, hotwords, relaxed=True)
                else:
                    segs = self._translate_span(chunk, a, relaxed=True)
            except Exception as exc:  # a failed second pass must never lose the first-pass transcript
                logger.warning("Gap recovery failed for %.0f-%.0fs: %s", a, b, exc)
                continue
            kept = filter_recovered(segs)
            if not kept:  # the detector still found nothing in a clearly audible stretch: let Whisper judge it, strictly screened
                try:
                    segs = (self._decode(chunk, a, "en", initial_prompt, hotwords, relaxed=True, no_vad=True)[0] if is_en
                            else self._translate_span(chunk, a, relaxed=True, no_vad=True))
                    kept = filter_recovered(segs, max_no_speech=0.4, min_logprob=-0.9)
                except Exception as exc:
                    logger.warning("Gap recovery (no detector) failed for %.0f-%.0fs: %s", a, b, exc)
            found.extend(kept)
        logger.info("Gap recovery: %d stretch(es) rechecked, %d segment(s) added", len(gaps), len(found))
        return sorted(segments + found, key=lambda sg: sg["start"])

    def _translate_span(self, chunk: np.ndarray, offset: float, relaxed: bool = False,
                        no_vad: bool = False) -> List[Dict[str, Any]]:
        """Translate a non-English stretch into English. Source language is auto-detected; each segment is labelled as translated."""
        segs, detected = self._decode(chunk, offset, None, None, None, task="translate", relaxed=relaxed, no_vad=no_vad)
        if detected and detected.lower().startswith("en"):  # the stretch turned out to be English: no translation happened, no label
            return segs
        return [{**sg, "translated_from": detected or "unknown"} for sg in segs]

    def _decode(self, chunk: np.ndarray, offset: float, fw_language: Optional[str],
                initial_prompt: Optional[str], hotwords: Optional[str],
                task: str = "transcribe", relaxed: bool = False,
                no_vad: bool = False) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Run faster-whisper on one chunk; timestamps are shifted by `offset` seconds."""
        shared = dict(
            language=fw_language,
            task=task,
            beam_size=settings.WHISPER_BEAM_SIZE,
            word_timestamps=settings.WHISPER_WORD_TIMESTAMPS,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            no_speech_threshold=settings.WHISPER_NO_SPEECH_THRESHOLD,
            log_prob_threshold=settings.WHISPER_LOGPROB_THRESHOLD,
            compression_ratio_threshold=settings.WHISPER_COMPRESSION_RATIO_THRESHOLD,
        )
        if getattr(self, "_batched", None) is not None and not relaxed:  # several windows decoded at once on the GPU: ~2x faster
            seg_iter, info = self._batched.transcribe(chunk, batch_size=settings.WHISPER_BATCH_SIZE, **shared)
        else:
            seg_iter, info = self._model.transcribe(
                chunk, vad_filter=not no_vad, condition_on_previous_text=False,  # no runaway repetition on silence
                **({"vad_parameters": {"threshold": 0.15, "min_speech_duration_ms": 100, "speech_pad_ms": 300}} if relaxed and not no_vad else {}),
                **shared)
        out: List[Dict[str, Any]] = []
        for s in seg_iter:
            text = s.text.strip()
            if not text:
                continue
            avg_lp = getattr(s, "avg_logprob", None)
            no_speech = getattr(s, "no_speech_prob", None)
            is_low = avg_lp is not None and avg_lp < settings.WHISPER_LOW_CONFIDENCE_LOGPROB
            out.append({
                "start": round(s.start + offset, 2),
                "end": round(s.end + offset, 2),
                "text": text,
                "avg_logprob": round(avg_lp, 3) if avg_lp is not None else None,
                "no_speech_prob": round(no_speech, 3) if no_speech is not None else None,
                "low_confidence": bool(is_low),
                **({"words": [{"word": w.word, "start": round(w.start + offset, 3),
                               "end": round(w.end + offset, 3), "probability": round(w.probability, 3)}
                              for w in s.words]}
                   if settings.WHISPER_WORD_TIMESTAMPS and getattr(s, "words", None) else {}),
            })
        language = getattr(info, "language", None)
        if getattr(info, "language_probability", 1.0) < 0.7:  # short or noisy audio: a guessed language name would mislead
            language = None
        return out, language

    def unload(self) -> None:
        self._model = None
        self._is_loaded = False
        logger.info("Whisper model unloaded from memory.")
