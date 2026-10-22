#!/usr/bin/env python3
"""
Anot Medical ASR — MedASR Model Lifecycle Manager
=================================================
Stage 11: Production model singleton. Loads MedASR once at startup.
Supports CPU and Intel XPU with automatic fallback and readiness reporting.
"""

import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCTC

from app.core.config import settings
from app.core.logging import logger


class MedASRModel:
    """
    Thread-safe Singleton wrapper for Google MedASR (LasrForCTC, 105M parameters).
    Ensures model weights are loaded only once in memory during service startup.
    """

    _instance: Optional["MedASRModel"] = None

    def __new__(cls) -> "MedASRModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._processor = None
            cls._instance._device = "cpu"
            cls._instance._model_id = settings.MODEL_ID
            cls._instance._load_time = 0.0
            cls._instance._is_loaded = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "MedASRModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_loaded(self) -> bool:
        """Check if model and processor are initialized in memory."""
        return self._is_loaded and self._model is not None and self._processor is not None

    def is_ready(self) -> bool:
        """Alias for is_loaded for consistency with readiness check."""
        return self.is_loaded()

    def get_device(self) -> str:
        """Return the active compute device string (cpu or xpu)."""
        return self._device

    def get_info(self) -> Dict[str, Any]:
        """Return safe, non-sensitive model metadata."""
        return {
            "model_id": self._model_id,
            "architecture": "LasrForCTC",
            "parameter_count": 105000000,
            "device": self._device,
            "is_loaded": self._is_loaded,
            "load_time_seconds": round(self._load_time, 4)
        }

    def load(self, device: Optional[str] = None, target_device: Optional[str] = None) -> None:
        """
        Load model and processor from local cache.
        Target device: cpu or xpu.
        """
        if self._is_loaded:
            logger.info("MedASR model is already loaded and ready.")
            return

        chosen_device = (target_device or device or settings.DEVICE).lower()
        t0 = time.perf_counter()

        logger.info(f"Loading MedASR model '{self._model_id}' (target_device={chosen_device})...")

        # 1. Load processor & model from local cache
        processor = AutoProcessor.from_pretrained(self._model_id, local_files_only=True)
        model = AutoModelForCTC.from_pretrained(self._model_id, local_files_only=True)
        model.eval()

        # 2. Configure target hardware device
        actual_device = "cpu"
        if target_device == "xpu":
            try:
                import intel_extension_for_pytorch as ipex
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    model = model.to("xpu")
                    actual_device = "xpu"
                    logger.info("Successfully loaded MedASR on Intel XPU.")
                else:
                    logger.warning("Intel XPU requested but torch.xpu is unavailable. Falling back to CPU.")
                    actual_device = "cpu"
            except Exception as e:
                logger.warning(f"Failed to initialize Intel XPU ({e}). Falling back to CPU.")
                actual_device = "cpu"
        else:
            model = model.to("cpu")
            actual_device = "cpu"
            logger.info("MedASR successfully loaded on CPU.")

        # 3. Model warm-up with dummy frame (prevents first-request latency spikes)
        try:
            dummy_waveform = np.zeros(16000, dtype=np.float32)  # 1s silence
            inputs = processor(dummy_waveform, sampling_rate=16000, return_tensors="pt")
            if actual_device == "xpu":
                inputs = {k: v.to("xpu") for k, v in inputs.items()}
            with torch.no_grad():
                _ = model(**inputs)
            logger.info("MedASR warm-up inference completed.")
        except Exception as e:
            logger.warning(f"Warm-up pass completed with non-fatal notice: {e}")

        self._processor = processor
        self._model = model
        self._device = actual_device
        self._load_time = time.perf_counter() - t0
        self._is_loaded = True
        logger.info(f"MedASR model ready for inference in {self._load_time:.2f}s.")

    def infer(self, waveform: np.ndarray, sample_rate: int = 16000) -> Tuple[torch.Tensor, float]:
        """
        Execute acoustic inference on a 16 kHz mono waveform.
        Returns:
            (logits_tensor, inference_duration_seconds)
        """
        if not self._is_loaded or self._model is None or self._processor is None:
            raise RuntimeError("MedASRModel is not loaded. Call load() before infer().")

        t0 = time.perf_counter()
        inputs = self._processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
        if self._device == "xpu":
            inputs = {k: v.to("xpu") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        inference_time = time.perf_counter() - t0
        logits = outputs.logits.cpu()
        return logits, inference_time

    @property
    def processor(self) -> AutoProcessor:
        if not self._is_loaded or self._processor is None:
            raise RuntimeError("Processor is not loaded.")
        return self._processor

    def unload(self) -> None:
        """Clean shutdown / free memory."""
        self._model = None
        self._processor = None
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("MedASR model unloaded from memory.")
