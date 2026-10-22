#!/usr/bin/env python3
"""
Unit and Integration Tests for Anot ASR Engine Service Architecture
===================================================================
Stage 11: Production Service, API Endpoints, Model Lifecycle, CLI,
Safe Logging, Temporary Storage Lifecycle, and Benchmark Regression.
"""

import io
import json
import logging
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient

from app.core.config import Settings
from app.core.logging import PHISafeFormatter
from app.security.privacy import generate_request_id, sanitize_filename
from app.models.medasr_model import MedASRModel
from app.services.audio_service import AudioService
from app.main import app
from app import cli
from accuracy.metrics import compute_clinical_accuracy


class TestServiceHealthAndReadiness(unittest.TestCase):
    """Test suite for GET /health and GET /ready endpoints."""

    def test_health_endpoint_safe_response(self):
        """GET /health must return safe service info with no internal paths or PHI."""
        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "anot-asr")

        # Must not expose internal filesystem paths, model details, or patient data
        raw_text = response.text
        self.assertNotIn("Users", raw_text)
        self.assertNotIn("Desktop", raw_text)
        self.assertNotIn("patient", raw_text.lower())

    def test_ready_endpoint_unready_before_load(self):
        """GET /ready must return 503 if model is not loaded in memory."""
        medasr_model = MedASRModel.get_instance()
        if medasr_model.is_loaded():
            medasr_model.unload()

        client = TestClient(app)
        response = client.get("/ready")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("request_id", data)

    def test_ready_endpoint_ready_after_lifespan(self):
        """GET /ready must return 200 with readiness details after lifespan startup."""
        with TestClient(app) as client:
            response = client.get("/ready")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertTrue(data["ready"])
            self.assertTrue(data["model_loaded"])
            self.assertEqual(data["model_id"], "google/medasr")
            self.assertIn(data["device"], ["cpu", "xpu"])


class TestSecurityAndPHISafety(unittest.TestCase):
    """Test suite for PHI-safe logging, request IDs, and filename sanitization."""

    def test_request_id_format(self):
        """Request IDs must follow ASR-YYYYMMDD-XXXXXXXX format without patient data."""
        req_id = generate_request_id()
        self.assertTrue(req_id.startswith("ASR-"))
        parts = req_id.split("-")
        self.assertEqual(len(parts), 3)
        self.assertEqual(len(parts[1]), 8)  # YYYYMMDD
        self.assertEqual(len(parts[2]), 8)  # Hex token

    def test_filename_sanitization(self):
        """Sanitizer must remove paths, PHI patterns, and hazardous characters."""
        unsafe_name = "C:\\patients\\John_Doe_MRN998877_ChestXray (1).wav"
        safe_name = sanitize_filename(unsafe_name)
        self.assertNotIn("\\", safe_name)
        self.assertNotIn("John_Doe", safe_name)
        self.assertTrue(safe_name.endswith(".wav"))

    def test_phi_safe_log_formatter(self):
        """Formatter must scrub MRNs, SSNs, phone numbers, and emails."""
        formatter = PHISafeFormatter("[%(levelname)s] %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Processing MRN: 12345678, SSN: 000-12-3456, phone: 555-123-4567, email: doc@clinic.org",
            args=(),
            exc_info=None
        )
        formatted = formatter.format(record)
        self.assertNotIn("12345678", formatted)
        self.assertNotIn("000-12-3456", formatted)
        self.assertNotIn("555-123-4567", formatted)
        self.assertNotIn("doc@clinic.org", formatted)
        self.assertIn("[REDACTED_ID]", formatted)
        self.assertIn("[REDACTED_PHONE]", formatted)
        self.assertIn("[REDACTED_EMAIL]", formatted)


class TestConfigAndResourceProtection(unittest.TestCase):
    """Test suite for settings and resource bounds."""

    def test_configuration_defaults(self):
        """Verify default production configuration."""
        settings = Settings()
        self.assertEqual(settings.DEVICE, "cpu")
        self.assertEqual(settings.MODE_DEFAULT, "dictation")
        self.assertEqual(settings.DECODER_DEFAULT, "greedy")
        self.assertEqual(settings.MAX_UPLOAD_MB, 100)
        self.assertEqual(settings.MAX_DURATION_SECONDS, 1800.0)
        self.assertEqual(settings.MAX_AUDIO_DURATION_SECONDS, 1800.0)
        self.assertEqual(settings.MAX_CONCURRENT_JOBS, 2)
        self.assertFalse(settings.LM_ENABLED)

    def test_environment_override(self):
        """Verify settings override from environment variables."""
        with patch.dict(os.environ, {
            "ANOT_ASR_MAX_UPLOAD_MB": "50",
            "ANOT_ASR_DECODER_DEFAULT": "beam",
            "ANOT_ASR_BEAM_WIDTH": "8"
        }):
            custom_settings = Settings()
            self.assertEqual(custom_settings.MAX_UPLOAD_MB, 50)
            self.assertEqual(custom_settings.DECODER_DEFAULT, "beam")
            self.assertEqual(custom_settings.BEAM_WIDTH, 8)


class TestAPIRequestValidation(unittest.TestCase):
    """Test suite for API input validation and error responses."""

    def test_unsupported_audio_format_rejected(self):
        """Rejecting non-audio file extensions with 400 Bad Request."""
        client = TestClient(app)
        fake_file = io.BytesIO(b"Not an audio file content")
        response = client.post(
            "/api/v1/transcribe",
            files={"file": ("report.txt", fake_file, "text/plain")},
            data={"mode": "dictation", "decoder": "greedy"}
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("Unsupported audio format", data["detail"])
        self.assertIn("request_id", data)

    def test_invalid_mode_rejected(self):
        """Invalid pipeline mode rejected with 422 Unprocessable Entity."""
        client = TestClient(app)
        fake_file = io.BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt ")
        response = client.post(
            "/api/v1/transcribe",
            files={"file": ("sample.wav", fake_file, "audio/wav")},
            data={"mode": "telepathy", "decoder": "greedy"}
        )
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("Invalid transcription mode", data["detail"])

    def test_invalid_decoder_rejected(self):
        """Invalid decoder rejected with 422 Unprocessable Entity."""
        client = TestClient(app)
        fake_file = io.BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt ")
        response = client.post(
            "/api/v1/transcribe",
            files={"file": ("sample.wav", fake_file, "audio/wav")},
            data={"mode": "dictation", "decoder": "quantum"}
        )
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("Invalid decoder", data["detail"])


class TestTemporaryStorageLifecycle(unittest.TestCase):
    """Test suite for temporary audio file processing and guaranteed cleanup."""

    def test_temp_file_cleanup_on_completion(self):
        """Uploaded temp audio and converted WAV must be deleted after request."""
        temp_dir = tempfile.mkdtemp(prefix="anot_audio_test_")
        source_wav = pathlib.Path(temp_dir) / "test.wav"

        # Write 0.1s silence WAV (16kHz mono, 16-bit PCM = 3200 bytes)
        import wave
        with wave.open(str(source_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 3200)

        audio_service = AudioService()
        with open(source_wav, "rb") as f:
            content = f.read()

        temp_src, temp_converted, duration = audio_service.process_upload(
            file_bytes=content,
            original_filename="test.wav",
            request_id="ASR-TEST-CLEANUP"
        )

        self.assertTrue(pathlib.Path(temp_converted).exists())
        self.assertAlmostEqual(duration, 0.1, places=1)

        # Cleanup
        audio_service.cleanup_temporary_files(temp_src, temp_converted, request_id="ASR-TEST-CLEANUP")
        self.assertFalse(pathlib.Path(temp_src).exists())
        self.assertFalse(pathlib.Path(temp_converted).exists())

        # Cleanup test directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestAuthoritativeBenchmarkRegression(unittest.TestCase):
    """Test suite verifying the authoritative Clinical Test 01 baseline via service.

    Baseline history:
      - 11.01% raw WER was the greedy-only baseline (Stages 1-11).
      - Stage 19 added a conservative, fact-preserving CTC glitch corrector to the
        production decode path. It repairs unambiguous surface glitches (here
        'examinNation' -> 'examination') without ever touching a token containing
        a digit, so clinical values are unchanged. This resolves one substitution,
        lowering the raw WER to 10.09% (S/D/I 6/3/2). The critical age mismatch
        (58 -> 50) is acoustic and is intentionally NOT repaired.
      - The clinically-normalized WER (unit/vital/disfluency-equivalent) is 3.88%.
    """

    @classmethod
    def setUpClass(cls):
        cls.medasr_model = MedASRModel.get_instance()
        if not cls.medasr_model.is_loaded():
            cls.medasr_model.load(target_device="cpu")

    def test_clinical_test_01_dictation_regression(self):
        """Clinical Test 01 MUST reproduce the authoritative baseline through the service API."""
        audio_path = pathlib.Path("results/audio_cache/anot_test_01_16k_mono.wav")
        ref_path = pathlib.Path("results/anot_test_01_reference.txt")

        self.assertTrue(audio_path.exists(), f"Audio file {audio_path} must exist.")
        self.assertTrue(ref_path.exists(), f"Reference file {ref_path} must exist.")

        with open(audio_path, "rb") as af:
            audio_bytes = af.read()

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/transcribe",
                files={"file": ("anot_test_01.wav", io.BytesIO(audio_bytes), "audio/wav")},
                data={
                    "mode": "dictation",
                    "decoder": "greedy",
                    "normalize": "false"
                }
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()

            # Verify response schema fields
            self.assertTrue(data["request_id"].startswith("ASR-"))
            self.assertEqual(data["status"], "completed")
            self.assertEqual(data["mode"], "dictation")
            self.assertEqual(data["language"], "en")
            self.assertGreater(data["duration_seconds"], 20.0)
            self.assertIn("transcript", data)
            self.assertGreater(len(data["segments"]), 0)
            self.assertEqual(data["decoder"]["type"], "greedy")
            self.assertFalse(data["decoder"]["language_model"])

            # Verify no PHI or paths leaked in response JSON
            response_json_str = json.dumps(data)
            self.assertNotIn("Users", response_json_str)
            self.assertNotIn("Desktop", response_json_str)
            self.assertNotIn(".cache", response_json_str)

            # Compute clinical accuracy against authoritative reference
            reference_text = ref_path.read_text(encoding="utf-8")
            hypothesis_text = data["transcript"]

            eval_res = compute_clinical_accuracy(reference_text, hypothesis_text)
            wer_pct = round(eval_res["wer_metrics"]["wer"] * 100, 2)

            # Assert authoritative raw baseline (Stage 19: glitch corrector resolves
            # 'examinNation' -> 'examination', one fewer substitution than the
            # pre-Stage-19 greedy-only baseline of 11.01% / S7 D3 I2).
            self.assertEqual(wer_pct, 10.09, f"Raw WER changed! Expected 10.09%, got {wer_pct}%")
            self.assertEqual(eval_res["wer_metrics"]["substitutions"], 6)
            self.assertEqual(eval_res["wer_metrics"]["deletions"], 3)
            self.assertEqual(eval_res["wer_metrics"]["insertions"], 2)

            # Assert authoritative clinically-normalized baseline (reported alongside
            # raw WER): unit/vital/disfluency-equivalent surface forms do not count
            # as errors, leaving only genuine acoustic errors.
            clin = eval_res["clinical_normalized_wer_metrics"]
            self.assertEqual(round(clin["wer"] * 100, 2), 3.88,
                             f"Clinical-normalized WER changed! Expected 3.88%, got {round(clin['wer']*100, 2)}%")
            self.assertEqual(clin["substitutions"], 3)
            self.assertEqual(clin["deletions"], 1)
            self.assertEqual(clin["insertions"], 0)

            # The critical age mismatch (58 -> 50) is acoustic and MUST remain
            # flagged -- it is never silently "corrected".
            self.assertEqual(eval_res["numeric_performance"]["critical_mismatches_count"], 1)
            self.assertEqual(eval_res["medication_performance"]["dosage_mismatches_count"], 0)
            self.assertEqual(eval_res["medication_performance"]["omitted_medications_count"], 0)

    def test_beam_decoder_api_option(self):
        """Beam decoder option must be selectable via API without error."""
        audio_path = pathlib.Path("results/audio_cache/anot_test_01_16k_mono.wav")
        with open(audio_path, "rb") as af:
            audio_bytes = af.read()

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/transcribe",
                files={"file": ("anot_test_01.wav", io.BytesIO(audio_bytes), "audio/wav")},
                data={
                    "mode": "dictation",
                    "decoder": "beam",
                    "beam_width": "4",
                    "language_model": "false",
                    "normalize": "true"
                }
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "completed")
            self.assertEqual(data["decoder"]["type"], "beam")
            self.assertEqual(data["decoder"]["beam_width"], 4)
            self.assertFalse(data["decoder"]["language_model"])


class TestConversationServicePipeline(unittest.TestCase):
    """Test suite for Mode B conversation transcription and anonymous speaker attribution."""

    @classmethod
    def setUpClass(cls):
        cls.medasr_model = MedASRModel.get_instance()
        if not cls.medasr_model.is_loaded():
            cls.medasr_model.load(target_device="cpu")

    def test_conversation_pipeline_execution(self):
        """30302 conversation audio must transcribe with chronological anonymous segments."""
        conv_audio_path = pathlib.Path("real benchmark/diagnostic/conversation_01/reconstructed/30302_16k_mono.wav")
        self.assertTrue(conv_audio_path.exists(), f"Audio file {conv_audio_path} must exist.")

        with open(conv_audio_path, "rb") as af:
            audio_bytes = af.read()

        with TestClient(app) as client:
            response = client.post(
                "/api/v1/transcribe/conversation",
                files={"file": ("30302.wav", io.BytesIO(audio_bytes), "audio/wav")},
                data={
                    "decoder": "greedy",
                    "normalize": "true"
                }
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "completed")
            self.assertEqual(data["mode"], "conversation")
            self.assertGreater(len(data["transcript"]), 0)
            self.assertGreater(len(data["segments"]), 0)

            # Check anonymous speaker handling:
            for seg in data["segments"]:
                self.assertIn("start", seg)
                self.assertIn("end", seg)
                self.assertIn("text", seg)
                self.assertIn("speaker", seg)
                # Must be anonymous ID like SPEAKER_00, SPEAKER_01 or UNKNOWN
                self.assertNotIn("doctor", seg["speaker"].lower())
                self.assertNotIn("patient", seg["speaker"].lower())
                self.assertTrue(seg["speaker"].startswith("SPEAKER_") or seg["speaker"] == "SPEAKER_UNKNOWN")


class TestServiceCLI(unittest.TestCase):
    """Test suite verifying CLI interface calls service layer properly."""

    def test_cli_help(self):
        """CLI --help returns 0 and displays supported commands."""
        import sys
        from unittest.mock import patch

        test_args = ["cli.py", "--help"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
            self.assertEqual(cm.exception.code, 0)

    def test_cli_transcribe_dictation(self):
        """CLI transcribe on anot_test_01 produces valid JSON output and completes."""
        audio_path = "results/audio_cache/anot_test_01_16k_mono.wav"
        output_json = tempfile.mktemp(suffix=".json", prefix="anot_cli_out_")

        try:
            cli_code = cli.cli_transcribe([
                audio_path,
                "--mode", "dictation",
                "--decoder", "greedy",
                "--normalize",
                "--out", output_json
            ])
            self.assertEqual(cli_code, 0)
            self.assertTrue(pathlib.Path(output_json).exists())

            with open(output_json, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["status"], "completed")
            self.assertEqual(data["mode"], "dictation")
            self.assertIn("metformin", data["transcript"].lower())
        finally:
            if pathlib.Path(output_json).exists():
                pathlib.Path(output_json).unlink()


if __name__ == "__main__":
    unittest.main()
