#!/usr/bin/env python3
"""
Anot Medical ASR — Audio Preprocessing & Temporary File Service
==============================================================
Stage 11: Upload validation, 16 kHz mono conversion, and guaranteed cleanup.
"""

import os
import pathlib
import shutil
import subprocess
import time
from typing import Optional, Set, Tuple
import numpy as np
import soundfile as sf
from fastapi import HTTPException, status

from app.core.config import settings
from app.core.logging import logger
from app.security.privacy import sanitize_filename


class AudioService:
    """Handles audio validation, temporary storage, FFmpeg conversion, and lifecycle cleanup."""

    def __init__(self, temp_dir: Optional[pathlib.Path] = None):
        self.temp_dir = temp_dir or settings.TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._ffmpeg_bin: Optional[str] = None

    def get_ffmpeg(self) -> str:
        """Find FFmpeg binary path."""
        if self._ffmpeg_bin:
            return self._ffmpeg_bin

        cmd = shutil.which("ffmpeg")
        if cmd:
            self._ffmpeg_bin = cmd
            return cmd

        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            winget_base = pathlib.Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            if winget_base.exists():
                for candidate in winget_base.glob("**/ffmpeg.exe"):
                    bin_dir = str(candidate.parent)
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                    self._ffmpeg_bin = str(candidate)
                    return self._ffmpeg_bin

        raise RuntimeError("FFmpeg executable not found. Ensure FFmpeg is installed.")

    def validate_upload(self, filename: str, file_size: int) -> None:
        """Validate file size and supported audio extension."""
        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        if file_size > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded audio file exceeds size limit of {settings.MAX_UPLOAD_MB} MB."
            )

        ext = pathlib.Path(filename).suffix.lower()
        if ext not in settings.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio format '{ext}'. Supported formats: {sorted(list(settings.SUPPORTED_EXTENSIONS))}."
            )

    def write_temp_upload(self, file_bytes: bytes, original_filename: str) -> pathlib.Path:
        """Save upload to a sanitized, non-PHI temporary file path."""
        safe_name = sanitize_filename(original_filename)
        temp_path = self.temp_dir / safe_name
        temp_path.write_bytes(file_bytes)
        return temp_path

    def convert_to_16k_mono(
        self,
        input_path: pathlib.Path
    ) -> Tuple[np.ndarray, int, float, float, pathlib.Path]:
        """
        Convert any input audio format to standardized 16 kHz mono WAV via FFmpeg.
        Returns:
            (waveform_np, sample_rate, duration_seconds, prep_time_seconds, output_wav_path)
        """
        t0 = time.perf_counter()
        ffmpeg_cmd = self.get_ffmpeg()

        output_wav_path = self.temp_dir / f"{input_path.stem}_16k_mono.wav"

        cmd = [
            ffmpeg_cmd, "-v", "error", "-y",
            "-i", str(input_path),
            "-f", "wav",
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            str(output_wav_path)
        ]

        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to decode audio file. The container may be corrupted or unsupported."
            ) from e

        # Read normalized waveform
        try:
            waveform, sample_rate = sf.read(str(output_wav_path), dtype="float32")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load converted audio waveform."
            ) from e

        duration_seconds = len(waveform) / sample_rate
        prep_time_seconds = time.perf_counter() - t0

        # Validate maximum duration resource limit
        if duration_seconds > settings.MAX_AUDIO_DURATION_SECONDS:
            self.cleanup_paths(input_path, output_wav_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio duration ({duration_seconds:.1f}s) exceeds maximum allowed limit of {settings.MAX_AUDIO_DURATION_SECONDS}s."
            )

        return waveform, sample_rate, duration_seconds, prep_time_seconds, output_wav_path

    def process_upload(
        self,
        file_bytes: bytes,
        original_filename: str,
        request_id: Optional[str] = None
    ) -> Tuple[pathlib.Path, pathlib.Path, float]:
        """
        Validate, save upload, and convert to 16 kHz mono WAV.
        Returns:
            (source_temp_path, converted_wav_path, duration_seconds)
        """
        self.validate_upload(original_filename, len(file_bytes))
        temp_src = self.write_temp_upload(file_bytes, original_filename)
        try:
            _, _, duration_seconds, _, converted_wav_path = self.convert_to_16k_mono(temp_src)
            return temp_src, converted_wav_path, duration_seconds
        except Exception:
            self.cleanup_paths(temp_src)
            raise

    def cleanup_temporary_files(self, *paths: Optional[pathlib.Path], request_id: Optional[str] = None) -> None:
        """Safely delete temporary files."""
        self.cleanup_paths(*paths)

    def cleanup_paths(self, *paths: Optional[pathlib.Path]) -> None:
        """Safely delete temporary files, suppressing errors."""
        for p in paths:
            if p is not None:
                p_obj = pathlib.Path(p) if not isinstance(p, pathlib.Path) else p
                if p_obj.exists():
                    try:
                        p_obj.unlink()
                    except Exception as e:
                        logger.warning(f"Could not delete temporary file {p_obj.name}: {e}")
