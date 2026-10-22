#!/usr/bin/env python3
"""
End-to-end tests for the Whisper engine path (app/models/whisper_model.py,
app/pipelines/whisper_pipeline.py). Skipped automatically when faster-whisper or
the large-v3 weights are not present locally, so offline CI stays green.
"""

import importlib.util
import pathlib
import unittest

import soundfile as sf

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_HAS_FW = importlib.util.find_spec("faster_whisper") is not None
_MODEL_CACHED = (pathlib.Path.home() / ".cache" / "huggingface" / "hub"
                 / "models--Systran--faster-whisper-large-v3").exists()
_CLIP = _ROOT / "benchmark" / "stage12" / "audio" / "B02_CONV_SHORT_RESPONSES.wav"

_SKIP = not (_HAS_FW and _MODEL_CACHED and _CLIP.exists())
_REASON = "faster-whisper / large-v3 weights / sample clip not available locally"


@unittest.skipIf(_SKIP, _REASON)
class TestWhisperEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.models.whisper_model import WhisperASRModel
        cls.model = WhisperASRModel.get_instance()
        if not cls.model.is_loaded():
            cls.model.load(device="cpu")
        cls.waveform, _sr = sf.read(str(_CLIP), dtype="float32")

    def test_model_reports_loaded(self):
        self.assertTrue(self.model.is_loaded())
        info = self.model.get_info()
        self.assertIn("whisper", info["model_id"])

    def test_transcribe_returns_nonempty_text_and_segments(self):
        text, segments, secs = self.model.transcribe(self.waveform, 16000)
        self.assertGreater(len(text), 0)
        self.assertGreater(len(segments), 0)
        self.assertGreater(secs, 0.0)
        for seg in segments:
            self.assertIn("start", seg)
            self.assertIn("end", seg)
            self.assertIn("text", seg)

    def test_pipeline_shape_and_normalization(self):
        from app.pipelines.whisper_pipeline import WhisperPipeline
        from app.services.normalization_service import NormalizationService
        pipe = WhisperPipeline(self.model, NormalizationService())
        res = pipe.process(self.waveform, 16000, normalize=True, mode="conversation")
        self.assertEqual(res["decoder"]["engine"], "whisper")
        self.assertGreater(len(res["transcript"]), 0)
        self.assertEqual(res["mode"], "conversation")
        self.assertIn("rtf", res["performance"])


if __name__ == "__main__":
    unittest.main()
