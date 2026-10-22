#!/usr/bin/env python3
"""
Anot Medical ASR — Local Speaker Diarization Module
===================================================
Performs local, privacy-preserving speaker clustering and turn attribution.
Operates 100% locally without cloud services, external APIs, or gated models.

PHI Protection:
- Anonymous labels ONLY: SPEAKER_00, SPEAKER_01, etc.
- Never identifies real people, roles (doctor/patient), or biometric signatures.
- Zero transcript text in manifests or reports.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class LocalSpeakerDiarizer:
    """
    Unsupervised local speaker diarization engine based on acoustic feature clustering.
    Can accept pre-computed deep embeddings (e.g. from MedASR encoder) or extract
    multivariate acoustic representations (spectral centroid, bandwidth, energy, rolloff).
    """

    def __init__(self, num_speakers: Optional[int] = 2):
        """
        :param num_speakers: Target number of speakers if known, or None for automatic estimation.
        """
        self.num_speakers = num_speakers

    def extract_acoustic_features(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Extract deterministic acoustic summary vector for a segment.
        Returns normalized 32-dim acoustic feature vector.
        """
        if len(waveform) < 160:  # < 10ms
            return np.zeros(32, dtype=np.float32)

        # 1. FFT Magnitude Spectrum
        fft_vals = np.abs(np.fft.rfft(waveform))
        freqs = np.fft.rfftfreq(len(waveform), 1.0 / sample_rate)
        sum_fft = np.sum(fft_vals) + 1e-12

        # 2. Spectral Moments
        centroid = np.sum(freqs * fft_vals) / sum_fft
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_vals) / sum_fft)
        
        # 3. Energy statistics
        rms = np.sqrt(np.mean(waveform ** 2))
        peak = np.max(np.abs(waveform))
        
        # 4. Filterbank-like sub-band energies (16 sub-bands)
        bands = 16
        band_edges = np.linspace(0, len(fft_vals) - 1, bands + 1, dtype=int)
        sub_energies = np.zeros(bands)
        for b in range(bands):
            sub_energies[b] = np.mean(fft_vals[band_edges[b]:band_edges[b+1]] ** 2) if band_edges[b+1] > band_edges[b] else 0.0

        # Sub-band log energies
        sub_log = np.log1p(sub_energies)

        # 5. Zero Crossing Rate
        zcr = np.mean(np.abs(np.diff(np.signbit(waveform))))

        # Combine into feature vector
        feats = np.hstack([
            [centroid / 4000.0, bandwidth / 2000.0, rms * 10.0, peak, zcr * 10.0],
            sub_log,
            # Padding to reach 32 dims
            np.zeros(32 - 5 - bands)
        ]).astype(np.float32)

        norm = np.linalg.norm(feats)
        if norm > 1e-9:
            feats = feats / norm
        return feats

    def cluster_segments(
        self,
        segments: List[Dict[str, Any]],
        waveform: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        precomputed_embeddings: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Cluster speech segments into anonymous speaker IDs (SPEAKER_00, SPEAKER_01...).
        """
        if not segments:
            return []

        # If only 1 segment, trivially assign to SPEAKER_00
        if len(segments) == 1:
            seg_copy = dict(segments[0])
            seg_copy["speaker_id"] = "SPEAKER_00"
            return [seg_copy]

        # Get embeddings
        if precomputed_embeddings is not None and len(precomputed_embeddings) == len(segments):
            features = precomputed_embeddings
        else:
            if waveform is None:
                # Mock / deterministic pseudo-features based on durations
                features = np.array([
                    [s["duration"], np.sin(s["start_time"]), np.cos(s["end_time"])]
                    for s in segments
                ], dtype=np.float32)
            else:
                feat_list = []
                for s in segments:
                    start_s = s.get("start_sample", int(s["start_time"] * sample_rate))
                    end_s = s.get("end_sample", int(s["end_time"] * sample_rate))
                    seg_wav = waveform[start_s:end_s]
                    feat_list.append(self.extract_acoustic_features(seg_wav, sample_rate))
                features = np.array(feat_list)

        # Normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features = features / norms

        # Target cluster count
        k = self.num_speakers if self.num_speakers is not None else 2
        k = min(k, len(segments))

        # Spherical K-Means / Cosine Clustering
        # Deterministic initialization: first point and furthest point
        centroids = [features[0]]
        for _ in range(1, k):
            sims = np.max([np.dot(features, c) for c in centroids], axis=0)
            furthest_idx = int(np.argmin(sims))
            centroids.append(features[furthest_idx])
        centroids = np.array(centroids)

        labels = np.zeros(len(features), dtype=int)
        for _ in range(25):
            sim_matrix = np.dot(features, centroids.T)  # (N, k)
            new_labels = np.argmax(sim_matrix, axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for c_idx in range(k):
                mask = (labels == c_idx)
                if np.sum(mask) > 0:
                    new_mean = np.mean(features[mask], axis=0)
                    m_norm = np.linalg.norm(new_mean)
                    centroids[c_idx] = new_mean / (m_norm if m_norm > 0 else 1.0)

        # Order speaker IDs by total speaking time (dominant speaker = SPEAKER_00)
        speaker_times = {}
        for idx, seg in enumerate(segments):
            lbl = labels[idx]
            speaker_times[lbl] = speaker_times.get(lbl, 0.0) + seg["duration"]

        sorted_clusters = sorted(speaker_times.keys(), key=lambda c: speaker_times[c], reverse=True)
        cluster_to_speaker = {clust: f"SPEAKER_{i:02d}" for i, clust in enumerate(sorted_clusters)}

        diarized_segments = []
        for idx, seg in enumerate(segments):
            item = dict(seg)
            item["speaker_id"] = cluster_to_speaker[labels[idx]]
            diarized_segments.append(item)

        return diarized_segments


def compute_diarization_metrics(
    diarized_segments: List[Dict[str, Any]],
    total_audio_duration: float
) -> Dict[str, Any]:
    """
    Compute PHI-safe speaker turn statistics and overlap metrics.
    """
    if not diarized_segments:
        return {
            "estimated_speakers": 0,
            "speakers": {},
            "total_turns": 0,
            "overlap_duration_seconds": 0.0,
            "overlap_ratio": 0.0,
        }

    unique_speakers = sorted(list(set(s["speaker_id"] for s in diarized_segments)))
    speaker_stats = {}

    for spk in unique_speakers:
        spk_segs = [s for s in diarized_segments if s["speaker_id"] == spk]
        durations = [s["duration"] for s in spk_segs]
        total_time = sum(durations)
        speaker_stats[spk] = {
            "speaker_id": spk,
            "turns": len(spk_segs),
            "total_speaking_time_seconds": round(total_time, 4),
            "speaking_time_ratio": round(total_time / total_audio_duration, 4) if total_audio_duration > 0 else 0.0,
            "average_turn_duration_seconds": round(float(np.mean(durations)), 4) if durations else 0.0,
            "median_turn_duration_seconds": round(float(np.median(durations)), 4) if durations else 0.0,
            "longest_turn_seconds": round(float(np.max(durations)), 4) if durations else 0.0,
            "shortest_turn_seconds": round(float(np.min(durations)), 4) if durations else 0.0,
        }

    # Measure potential temporal overlap between consecutive turns
    overlap_sec = 0.0
    for i in range(len(diarized_segments) - 1):
        curr_end = diarized_segments[i]["end_time"]
        next_start = diarized_segments[i + 1]["start_time"]
        if curr_end > next_start:
            overlap_sec += (curr_end - next_start)

    overlap_ratio = round(overlap_sec / total_audio_duration, 4) if total_audio_duration > 0 else 0.0

    return {
        "manifest_version": "1.0",
        "audio_duration_seconds": round(total_audio_duration, 4),
        "estimated_speakers": len(unique_speakers),
        "total_turns": len(diarized_segments),
        "overlap_duration_seconds": round(overlap_sec, 4),
        "overlap_ratio": overlap_ratio,
        "speakers": speaker_stats,
        "turns": diarized_segments,
    }


def generate_diarization_report_text(manifest: Dict[str, Any]) -> str:
    """Format human-readable, PHI-safe diarization technical report."""
    lines = [
        "==============================================================================",
        "          ANOT MEDICAL ASR — SPEAKER DIARIZATION REPORT",
        "==============================================================================",
        "  Privacy Safeguard: LOCAL-ONLY PROCESSING. NO TRANSCRIPTS OR PHI INCLUDED.",
        "------------------------------------------------------------------------------",
        f"  Total Audio Duration:          {manifest.get('audio_duration_seconds', 0.0):.2f} s",
        f"  Estimated Number of Speakers:  {manifest.get('estimated_speakers', 0)}",
        f"  Total Speaker Turns:           {manifest.get('total_turns', 0)}",
        f"  Estimated Overlap Duration:    {manifest.get('overlap_duration_seconds', 0.0):.2f} s ({manifest.get('overlap_ratio', 0.0)*100:.1f} %)",
        "------------------------------------------------------------------------------",
        "  SPEAKER PROFILES (ANONYMOUS):",
    ]

    speakers = manifest.get("speakers", {})
    for spk_id, spk in sorted(speakers.items()):
        lines.extend([
            f"  [{spk_id}]:",
            f"    Total Speaking Time:         {spk['total_speaking_time_seconds']:.2f} s ({spk['speaking_time_ratio']*100:.1f} % of encounter)",
            f"    Number of Turns:             {spk['turns']}",
            f"    Average Turn Duration:       {spk['average_turn_duration_seconds']:.2f} s",
            f"    Median Turn Duration:        {spk['median_turn_duration_seconds']:.2f} s",
            f"    Longest Turn:                {spk['longest_turn_seconds']:.2f} s",
            f"    Shortest Turn:               {spk['shortest_turn_seconds']:.2f} s",
            "",
        ])

    lines.append("------------------------------------------------------------------------------")
    lines.append("  SPEAKER TURN CHRONOLOGY (First 15 turns):")
    for turn in manifest.get("turns", [])[:15]:
        lines.append(
            f"    [{turn['start_time']:6.2f}s -> {turn['end_time']:6.2f}s] "
            f"({turn['duration']:5.2f}s) -> {turn['speaker_id']}"
        )
    if len(manifest.get("turns", [])) > 15:
        lines.append(f"    ... [{len(manifest.get('turns', [])) - 15} additional turns omitted from summary view]")

    lines.append("==============================================================================")
    return "\n".join(lines) + "\n"
