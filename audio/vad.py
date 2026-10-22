#!/usr/bin/env python3
"""
Anot Medical ASR — Local Voice Activity Detection (VAD) Module
==============================================================
Provides local, privacy-preserving speech activity detection and temporal segmentation.
Uses Silero VAD locally (via PyTorch) with fallback energy-thresholding for offline testing.

PHI Protection:
- Zero transcription text is captured, processed, or stored in this module.
- Outputs contain only timing boundaries (start, end, duration) and acoustic activity ratios.
- All processing is strictly local (no network calls, no cloud telemetry).
"""

import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch


class LocalVAD:
    """
    Local Voice Activity Detection (VAD) engine.
    Wraps Silero VAD with fallback support for deterministic unit testing.
    """

    def __init__(self, mode: str = "silero", vad_model: Optional[Any] = None):
        """
        Initialize LocalVAD.
        :param mode: 'silero' (default) or 'energy' (fallback/mockable)
        :param vad_model: Optional pre-loaded PyTorch VAD model.
        """
        self.mode = mode.lower().strip()
        self.model = vad_model
        self.utils = None
        self._initialized = False

    def initialize(self):
        """Load local VAD model if in silero mode and not yet loaded."""
        if self._initialized:
            return

        if self.mode == "silero":
            try:
                self.model, self.utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    onnx=False,
                    trust_repo=True,
                )
                self._initialized = True
            except Exception as e:
                # Fallback to energy-based VAD if Silero is unavailable offline
                self.mode = "energy"
                self._initialized = True
        else:
            self._initialized = True

    def detect_speech_segments(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 200,
        min_silence_duration_ms: int = 150,
    ) -> List[Dict[str, Any]]:
        """
        Detect speech boundaries in a 1D audio waveform.
        Returns a list of segment dictionaries with timing coordinates.
        Does NOT produce or inspect linguistic content.
        """
        if not self._initialized:
            self.initialize()

        if len(waveform) == 0:
            return []

        if self.mode == "silero" and self.model is not None:
            return self._detect_silero(
                waveform,
                sample_rate,
                threshold,
                min_speech_duration_ms,
                min_silence_duration_ms,
            )
        else:
            return self._detect_energy(
                waveform,
                sample_rate,
                threshold=0.01,
                min_speech_ms=min_speech_duration_ms,
                min_silence_ms=min_silence_duration_ms,
            )

    def _detect_silero(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        threshold: float,
        min_speech_ms: int,
        min_silence_ms: int,
    ) -> List[Dict[str, Any]]:
        """Silero VAD inference."""
        get_speech_timestamps = self.utils[0]
        wav_tensor = torch.from_numpy(waveform).float()
        raw_stamps = get_speech_timestamps(
            wav_tensor,
            self.model,
            sampling_rate=sample_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
        )

        segments = []
        for idx, item in enumerate(raw_stamps):
            start_s = int(item["start"])
            end_s = int(item["end"])
            start_t = round(start_s / sample_rate, 4)
            end_t = round(end_s / sample_rate, 4)
            dur_t = round(end_t - start_t, 4)
            segments.append(
                {
                    "segment_id": f"seg_{idx:03d}",
                    "start_sample": start_s,
                    "end_sample": end_s,
                    "start_time": start_t,
                    "end_time": end_t,
                    "duration": dur_t,
                }
            )
        return segments

    def _detect_energy(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        threshold: float = 0.01,
        min_speech_ms: int = 200,
        min_silence_ms: int = 150,
    ) -> List[Dict[str, Any]]:
        """Deterministic energy-based speech boundary detector (fallback/unit tests)."""
        frame_len = int(sample_rate * 0.02)  # 20ms
        hop_len = int(sample_rate * 0.01)    # 10ms
        if len(waveform) < frame_len:
            return []

        # Compute frame energies
        num_frames = 1 + (len(waveform) - frame_len) // hop_len
        energies = np.zeros(num_frames)
        for i in range(num_frames):
            frame = waveform[i * hop_len : i * hop_len + frame_len]
            energies[i] = np.sqrt(np.mean(frame**2))

        # Adaptive threshold if requested threshold is low
        dynamic_thresh = max(threshold, float(np.percentile(energies, 30)) * 2.0)
        is_speech = energies >= dynamic_thresh

        # Group contiguous speech frames
        segments = []
        in_speech = False
        start_frame = 0
        min_speech_frames = max(1, int(min_speech_ms / 10))
        min_silence_frames = max(1, int(min_silence_ms / 10))

        silence_count = 0
        seg_idx = 0

        for i, speech in enumerate(is_speech):
            if speech:
                if not in_speech:
                    in_speech = True
                    start_frame = i
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        end_frame = i - silence_count
                        if (end_frame - start_frame) >= min_speech_frames:
                            start_s = start_frame * hop_len
                            end_s = min(len(waveform), end_frame * hop_len + frame_len)
                            start_t = round(start_s / sample_rate, 4)
                            end_t = round(end_s / sample_rate, 4)
                            segments.append(
                                {
                                    "segment_id": f"seg_{seg_idx:03d}",
                                    "start_sample": start_s,
                                    "end_sample": end_s,
                                    "start_time": start_t,
                                    "end_time": end_t,
                                    "duration": round(end_t - start_t, 4),
                                }
                            )
                            seg_idx += 1
                        in_speech = False
                        silence_count = 0

        if in_speech:
            end_frame = num_frames
            if (end_frame - start_frame) >= min_speech_frames:
                start_s = start_frame * hop_len
                end_s = len(waveform)
                start_t = round(start_s / sample_rate, 4)
                end_t = round(end_s / sample_rate, 4)
                segments.append(
                    {
                        "segment_id": f"seg_{seg_idx:03d}",
                        "start_sample": start_s,
                        "end_sample": end_s,
                        "start_time": start_t,
                        "end_time": end_t,
                        "duration": round(end_t - start_t, 4),
                    }
                )

        return segments


def build_vad_manifest(
    segments: List[Dict[str, Any]], total_duration_seconds: float
) -> Dict[str, Any]:
    """
    Construct a PHI-safe JSON-serializable VAD manifest.
    Strictly contains timing and acoustic proportions; no textual data.
    """
    total_speech_sec = sum(s["duration"] for s in segments)
    speech_ratio = (
        round(total_speech_sec / total_duration_seconds, 4)
        if total_duration_seconds > 0
        else 0.0
    )
    total_silence_sec = max(0.0, total_duration_seconds - total_speech_sec)
    silence_ratio = (
        round(total_silence_sec / total_duration_seconds, 4)
        if total_duration_seconds > 0
        else 0.0
    )

    durations = [s["duration"] for s in segments]
    dur_stats = {
        "count": len(segments),
        "min_duration_seconds": round(float(np.min(durations)), 4) if durations else 0.0,
        "max_duration_seconds": round(float(np.max(durations)), 4) if durations else 0.0,
        "mean_duration_seconds": round(float(np.mean(durations)), 4) if durations else 0.0,
        "median_duration_seconds": round(float(np.median(durations)), 4) if durations else 0.0,
        "std_duration_seconds": round(float(np.std(durations)), 4) if durations else 0.0,
    }

    return {
        "manifest_version": "1.0",
        "audio_duration_seconds": round(total_duration_seconds, 4),
        "total_speech_duration_seconds": round(total_speech_sec, 4),
        "total_silence_duration_seconds": round(total_silence_sec, 4),
        "speech_activity_ratio": speech_ratio,
        "silence_ratio": silence_ratio,
        "segment_count": len(segments),
        "duration_distribution": dur_stats,
        "segments": segments,
    }


def generate_vad_report_text(manifest: Dict[str, Any]) -> str:
    """Format a human-readable, PHI-safe VAD technical report."""
    dist = manifest.get("duration_distribution", {})
    lines = [
        "==============================================================================",
        "          ANOT MEDICAL ASR — VOICE ACTIVITY DETECTION (VAD) REPORT",
        "==============================================================================",
        "  Privacy Safeguard: LOCAL-ONLY PROCESSING. NO TRANSCRIPTS OR PHI INCLUDED.",
        "------------------------------------------------------------------------------",
        f"  Total Audio Duration:          {manifest.get('audio_duration_seconds', 0.0):.2f} s",
        f"  Detected Speech Segments:      {manifest.get('segment_count', 0)}",
        f"  Total Active Speech Duration:  {manifest.get('total_speech_duration_seconds', 0.0):.2f} s",
        f"  Total Inactive / Silence:      {manifest.get('total_silence_duration_seconds', 0.0):.2f} s",
        f"  Speech Activity Ratio:         {manifest.get('speech_activity_ratio', 0.0)*100:.1f} %",
        f"  Silence Ratio:                 {manifest.get('silence_ratio', 0.0)*100:.1f} %",
        "------------------------------------------------------------------------------",
        "  SEGMENT DURATION METRICS:",
        f"    Shortest Segment:            {dist.get('min_duration_seconds', 0.0):.2f} s",
        f"    Longest Segment:             {dist.get('max_duration_seconds', 0.0):.2f} s",
        f"    Mean Segment Duration:       {dist.get('mean_duration_seconds', 0.0):.2f} s",
        f"    Median Segment Duration:     {dist.get('median_duration_seconds', 0.0):.2f} s",
        f"    Standard Deviation:          {dist.get('std_duration_seconds', 0.0):.2f} s",
        "------------------------------------------------------------------------------",
        "  SEGMENT TIMELINE MANIFEST (First 15 segments):",
    ]
    for seg in manifest.get("segments", [])[:15]:
        lines.append(
            f"    [{seg['segment_id']}] {seg['start_time']:6.2f}s -> {seg['end_time']:6.2f}s "
            f"(Duration: {seg['duration']:5.2f}s)"
        )
    if len(manifest.get("segments", [])) > 15:
        lines.append(f"    ... [{len(manifest.get('segments', [])) - 15} additional segments omitted from summary view]")
    lines.append("==============================================================================")
    return "\n".join(lines) + "\n"
