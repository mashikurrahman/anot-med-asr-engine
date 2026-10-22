#!/usr/bin/env python3
"""
Anot Medical ASR — Audio Forensics & Segmentation Utilities
===========================================================
Performs forensic acoustic measurement and stream slicing for local speech processing.

PHI Protection:
- Zero textual extraction.
- Strictly measures physical and signal acoustic properties.
- Slices audio streams into anonymous speaker and segment files.
"""

import math
import pathlib
import subprocess
import shutil
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import soundfile as sf


def compute_acoustic_forensics(
    waveform: np.ndarray,
    sample_rate: int,
    source_metadata: Optional[Dict[str, Any]] = None,
    vad_segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Perform deep technical audio forensics on an audio waveform.
    Calculates RMS, peak level, clipping indicators, and temporal distributions.
    """
    duration = len(waveform) / sample_rate if sample_rate > 0 else 0.0

    # 1. Amplitude Statistics
    abs_wave = np.abs(waveform)
    rms = float(np.sqrt(np.mean(waveform ** 2))) if len(waveform) > 0 else 0.0
    rms_dbfs = round(20 * math.log10(rms), 2) if rms > 1e-9 else -100.0

    peak = float(np.max(abs_wave)) if len(waveform) > 0 else 0.0
    peak_dbfs = round(20 * math.log10(peak), 2) if peak > 1e-9 else -100.0

    # 2. Clipping Detection
    # In 16-bit PCM or normalized float, samples near +/- 1.0 indicate potential clipping
    clipping_samples = int(np.sum(abs_wave >= 0.999))
    clipping_ratio = round(clipping_samples / len(waveform), 6) if len(waveform) > 0 else 0.0
    has_clipping = clipping_samples > 0

    # 3. Speech vs Silence Ratios
    total_speech_sec = sum(s["duration"] for s in (vad_segments or []))
    speech_ratio = round(total_speech_sec / duration, 4) if duration > 0 else 0.0
    silence_sec = max(0.0, duration - total_speech_sec)
    silence_ratio = round(silence_sec / duration, 4) if duration > 0 else 0.0

    # 4. Segment Distribution
    durations = [s["duration"] for s in (vad_segments or [])]
    dur_stats = {
        "count": len(durations),
        "min_seconds": round(float(np.min(durations)), 4) if durations else 0.0,
        "max_seconds": round(float(np.max(durations)), 4) if durations else 0.0,
        "mean_seconds": round(float(np.mean(durations)), 4) if durations else 0.0,
        "median_seconds": round(float(np.median(durations)), 4) if durations else 0.0,
        "std_seconds": round(float(np.std(durations)), 4) if durations else 0.0,
    }

    # 5. Overlap Indicators
    overlap_detected = False
    overlap_sec = 0.0
    if vad_segments and len(vad_segments) > 1:
        for i in range(len(vad_segments) - 1):
            curr_end = vad_segments[i]["end_time"]
            next_start = vad_segments[i + 1]["start_time"]
            if curr_end > next_start:
                overlap_detected = True
                overlap_sec += (curr_end - next_start)

    forensics = {
        "duration_seconds": round(duration, 4),
        "sample_rate_hz": sample_rate,
        "channels": 1,
        "rms_amplitude": round(rms, 6),
        "rms_dbfs": rms_dbfs,
        "peak_amplitude": round(peak, 6),
        "peak_dbfs": peak_dbfs,
        "clipping": {
            "has_clipping": has_clipping,
            "clipped_sample_count": clipping_samples,
            "clipping_ratio": clipping_ratio,
        },
        "speech_activity": {
            "speech_duration_seconds": round(total_speech_sec, 4),
            "silence_duration_seconds": round(silence_sec, 4),
            "speech_ratio": speech_ratio,
            "silence_ratio": silence_ratio,
        },
        "segment_distribution": dur_stats,
        "overlap": {
            "substantial_overlap_detected": overlap_detected,
            "overlap_duration_seconds": round(overlap_sec, 4),
        },
    }

    if source_metadata:
        forensics["source_properties"] = {
            "codec": source_metadata.get("codec", "unknown"),
            "container_format": source_metadata.get("container_format", "unknown"),
            "source_sample_rate_hz": source_metadata.get("sample_rate_hz", sample_rate),
            "source_channels": source_metadata.get("channels", 1),
            "file_size_bytes": source_metadata.get("file_size_bytes", 0),
        }

    return forensics


def format_audio_forensics_report(forensics: Dict[str, Any]) -> str:
    """Format a PHI-safe text report of audio forensics."""
    src = forensics.get("source_properties", {})
    clip = forensics.get("clipping", {})
    sa = forensics.get("speech_activity", {})
    sd = forensics.get("segment_distribution", {})
    ov = forensics.get("overlap", {})

    lines = [
        "==============================================================================",
        "          ANOT MEDICAL ASR — AUDIO FORENSICS & ACOUSTIC REPORT",
        "==============================================================================",
        "  Privacy Safeguard: LOCAL-ONLY PROCESSING. NO TRANSCRIPTS OR PHI INCLUDED.",
        "------------------------------------------------------------------------------",
        "  SOURCE FILE TECHNICAL ATTRIBUTES:",
        f"    Container Format:            {src.get('container_format', 'N/A')}",
        f"    Codec:                       {src.get('codec', 'N/A')}",
        f"    Source Sample Rate:          {src.get('source_sample_rate_hz', 'N/A')} Hz",
        f"    Channels:                    {src.get('source_channels', 'N/A')}",
        f"    File Size:                   {src.get('file_size_bytes', 0):,} bytes",
        "------------------------------------------------------------------------------",
        "  ACOUSTIC & LOUDNESS PROPERTIES (16 kHz PCM Mono):",
        f"    Audio Duration:              {forensics.get('duration_seconds', 0.0):.2f} s",
        f"    RMS Amplitude:               {forensics.get('rms_amplitude', 0.0):.6f}",
        f"    RMS Loudness Level:          {forensics.get('rms_dbfs', 0.0):.2f} dBFS",
        f"    Peak Amplitude:              {forensics.get('peak_amplitude', 0.0):.6f}",
        f"    Peak Level:                  {forensics.get('peak_dbfs', 0.0):.2f} dBFS",
        f"    Clipping Detected:           {'YES' if clip.get('has_clipping') else 'NO'} ({clip.get('clipped_sample_count', 0)} samples, {clip.get('clipping_ratio', 0.0)*100:.4f} %)",
        "------------------------------------------------------------------------------",
        "  TEMPORAL SPEECH & ACTIVITY PROFILE:",
        f"    Detected Speech Segments:    {sd.get('count', 0)}",
        f"    Active Speech Duration:      {sa.get('speech_duration_seconds', 0.0):.2f} s ({sa.get('speech_ratio', 0.0)*100:.1f} %)",
        f"    Silence / Inactive Duration: {sa.get('silence_duration_seconds', 0.0):.2f} s ({sa.get('silence_ratio', 0.0)*100:.1f} %)",
        f"    Substantial Overlap:         {'YES' if ov.get('substantial_overlap_detected') else 'NO'} ({ov.get('overlap_duration_seconds', 0.0):.2f} s)",
        "------------------------------------------------------------------------------",
        "  SPEECH SEGMENT DURATION DISTRIBUTION:",
        f"    Min Duration:                {sd.get('min_seconds', 0.0):.2f} s",
        f"    Max Duration:                {sd.get('max_seconds', 0.0):.2f} s",
        f"    Mean Duration:               {sd.get('mean_seconds', 0.0):.2f} s",
        f"    Median Duration:             {sd.get('median_seconds', 0.0):.2f} s",
        f"    Standard Deviation:          {sd.get('std_seconds', 0.0):.2f} s",
        "==============================================================================",
    ]
    return "\n".join(lines) + "\n"


def slice_speaker_audio_streams(
    waveform: np.ndarray,
    sample_rate: int,
    diarized_turns: List[Dict[str, Any]],
    output_dir: pathlib.Path,
    inter_segment_padding_ms: int = 150,
) -> Dict[str, pathlib.Path]:
    """
    Export isolated speaker audio files (e.g. speaker_00.wav, speaker_01.wav).
    16 kHz mono PCM WAV format.
    Does not modify original recording.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    padding_samples = int(sample_rate * (inter_segment_padding_ms / 1000.0))
    padding_wave = np.zeros(padding_samples, dtype=np.float32)

    # Group turns by speaker
    speaker_turns: Dict[str, List[Dict[str, Any]]] = {}
    for turn in diarized_turns:
        spk = turn["speaker_id"]
        speaker_turns.setdefault(spk, []).append(turn)

    created_paths = {}
    for spk, turns in speaker_turns.items():
        spk_chunks = []
        for t in turns:
            st = t.get("start_sample", int(t["start_time"] * sample_rate))
            en = t.get("end_sample", int(t["end_time"] * sample_rate))
            chunk = waveform[st:en]
            spk_chunks.append(chunk)
            if padding_samples > 0:
                spk_chunks.append(padding_wave)

        if spk_chunks:
            spk_wave = np.concatenate(spk_chunks)
        else:
            spk_wave = np.zeros(1600, dtype=np.float32)

        out_name = f"{spk.lower()}.wav"
        out_file = output_dir / out_name
        sf.write(str(out_file), spk_wave, sample_rate, subtype="PCM_16")
        created_paths[spk] = out_file

    return created_paths


def slice_individual_segments(
    waveform: np.ndarray,
    sample_rate: int,
    segments: List[Dict[str, Any]],
    output_dir: pathlib.Path,
) -> List[Tuple[str, pathlib.Path]]:
    """
    Export each speech segment as an individual 16 kHz mono WAV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_files = []
    for s in segments:
        seg_id = s["segment_id"]
        st = s.get("start_sample", int(s["start_time"] * sample_rate))
        en = s.get("end_sample", int(s["end_time"] * sample_rate))
        chunk = waveform[st:en]
        out_file = output_dir / f"{seg_id}.wav"
        sf.write(str(out_file), chunk, sample_rate, subtype="PCM_16")
        out_files.append((seg_id, out_file))
    return out_files
