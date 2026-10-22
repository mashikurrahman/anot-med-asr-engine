#!/usr/bin/env python3
"""
Anot Medical ASR — Acoustic-Aware Chunk Segmenter
==================================================
Stage 18H: Dynamic acoustic boundary detection and edge context padding.

Replaces blind mathematical chunk slicing with acoustic-aware energy valley
detection while preserving 100% backward compatibility with fixed-duration splitting.

Core Architecture:
1. Frame-based Short-Time Energy (STE/RMS) analysis across a configurable search window.
2. Identifies natural speech pauses and low-energy valleys near the preferred chunk duration.
3. Strict enforcement of maximum chunk duration constraint (never exceeds max_duration_seconds).
4. Balanced sub-chunking ensuring all remainder chunks satisfy minimum duration constraint.
5. Configurable edge context padding (e.g. 150 ms) with explicit core vs context sample tracking.
6. Deterministic, fully reproducible local execution without remote dependencies.
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


class AcousticAwareSegmenter:
    """
    Splits long continuous VAD speech segments into bounded chunks using
    acoustic energy valley detection and optional edge context padding.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        max_duration_seconds: float = 8.0,
        min_chunk_duration_seconds: float = 1.0,
        search_window_seconds: float = 0.5,
        edge_context_seconds: float = 0.0,
        frame_length_ms: float = 25.0,
        frame_hop_ms: float = 10.0,
        energy_dip_threshold: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.min_chunk_duration_seconds = min_chunk_duration_seconds
        self.search_window_seconds = search_window_seconds
        self.edge_context_seconds = edge_context_seconds
        self.frame_length_ms = frame_length_ms
        self.frame_hop_ms = frame_hop_ms
        self.energy_dip_threshold = energy_dip_threshold

        self.frame_len_samples = max(1, int((frame_length_ms / 1000.0) * sample_rate))
        self.frame_hop_samples = max(1, int((frame_hop_ms / 1000.0) * sample_rate))

    def compute_short_time_energy(
        self,
        waveform: np.ndarray,
        start_sample: int,
        end_sample: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute short-time RMS energy across sliding windows in the given sample range.
        Returns:
            frame_centers: 1D array of sample indices corresponding to frame centers.
            frame_rms: 1D array of RMS energy values.
        """
        start_sample = max(0, start_sample)
        end_sample = min(len(waveform), end_sample)

        if end_sample <= start_sample + self.frame_len_samples:
            slice_data = waveform[start_sample:end_sample]
            rms = float(np.sqrt(np.mean(slice_data ** 2))) if len(slice_data) > 0 else 0.0
            mid = (start_sample + end_sample) // 2
            return np.array([mid], dtype=np.int64), np.array([rms], dtype=np.float32)

        centers = []
        energies = []

        for f_start in range(start_sample, end_sample - self.frame_len_samples + 1, self.frame_hop_samples):
            f_end = f_start + self.frame_len_samples
            frame = waveform[f_start:f_end]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            centers.append(f_start + self.frame_len_samples // 2)
            energies.append(rms)

        if not energies:
            slice_data = waveform[start_sample:end_sample]
            rms = float(np.sqrt(np.mean(slice_data ** 2))) if len(slice_data) > 0 else 0.0
            mid = (start_sample + end_sample) // 2
            return np.array([mid], dtype=np.int64), np.array([rms], dtype=np.float32)

        return np.array(centers, dtype=np.int64), np.array(energies, dtype=np.float32)

    def find_acoustic_cut(
        self,
        waveform: np.ndarray,
        curr_start_sample: int,
        seg_end_sample: int,
        target_chunk_samples: int,
        search_window_samples: int,
        segment_mean_rms: float = 0.0,
    ) -> Tuple[int, float, str, float]:
        """
        Locate optimal cut sample within the search window around preferred cut point.
        Guarantees:
        - Cut sample <= curr_start_sample + max_chunk_samples.
        - Cut sample >= curr_start_sample + min_chunk_samples.
        - Remaining samples >= min_chunk_samples.

        Returns:
            cut_sample (int)
            boundary_rms (float)
            cut_method (str): 'acoustic_valley' | 'acoustic_minimum' | 'fixed_fallback'
            dip_ratio (float): boundary_rms / window_mean_rms
        """
        max_allowed_sample = curr_start_sample + int(self.max_duration_seconds * self.sample_rate)
        min_allowed_sample = curr_start_sample + int(self.min_chunk_duration_seconds * self.sample_rate)
        max_allowed_sample = min(max_allowed_sample, seg_end_sample - int(self.min_chunk_duration_seconds * self.sample_rate))

        if max_allowed_sample <= min_allowed_sample:
            # Remainder is too short for search; split exactly at midpoint
            fallback_sample = (curr_start_sample + seg_end_sample) // 2
            return fallback_sample, 0.0, "fixed_fallback", 1.0

        preferred_cut_sample = curr_start_sample + target_chunk_samples
        search_start = max(min_allowed_sample, preferred_cut_sample - search_window_samples)
        search_end = min(max_allowed_sample, preferred_cut_sample + search_window_samples)

        if search_end <= search_start:
            return preferred_cut_sample, 0.0, "fixed_fallback", 1.0

        # Compute energy across search window
        centers, energies = self.compute_short_time_energy(waveform, search_start, search_end)
        if len(energies) == 0:
            return preferred_cut_sample, 0.0, "fixed_fallback", 1.0

        window_mean_rms = float(np.mean(energies))
        window_min_rms = float(np.min(energies))

        # Check for flat signal or negligible variance
        if np.std(energies) < 1e-7:
            # Flat audio (e.g. digital silence or uniform noise): fallback to preferred cut
            return preferred_cut_sample, window_mean_rms, "fixed_fallback", 1.0

        # 1. Search for local energy valleys (dips below neighbors and below threshold)
        valley_indices = []
        for i in range(1, len(energies) - 1):
            if energies[i] <= energies[i - 1] and energies[i] <= energies[i + 1]:
                valley_indices.append(i)

        valid_valleys = []
        for idx in valley_indices:
            e = energies[idx]
            ratio = (e / window_mean_rms) if window_mean_rms > 1e-9 else 1.0
            if ratio <= self.energy_dip_threshold:
                valid_valleys.append((idx, e, ratio))

        if valid_valleys:
            # Pick valley with lowest energy; if multiple, closest to preferred_cut_sample
            best_valley = min(valid_valleys, key=lambda v: (v[1], abs(int(centers[v[0]]) - preferred_cut_sample)))
            best_sample = int(centers[best_valley[0]])
            return best_sample, float(best_valley[1]), "acoustic_valley", float(best_valley[2])

        # 2. If no distinct valley below threshold, pick the global window minimum
        min_idx = int(np.argmin(energies))
        min_sample = int(centers[min_idx])
        min_e = float(energies[min_idx])
        min_ratio = (min_e / window_mean_rms) if window_mean_rms > 1e-9 else 1.0
        return min_sample, min_e, "acoustic_minimum", min_ratio

    def split_segments(
        self,
        segments: List[Dict[str, Any]],
        waveform: Optional[np.ndarray] = None,
        strategy: str = "acoustic_aware",
        edge_context_seconds: Optional[float] = None,
        search_window_seconds: Optional[float] = None,
        max_duration_seconds: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Subdivide long speech segments into bounded chunks.
        
        Args:
            segments: List of input VAD segment dicts.
            waveform: Optional 1D audio waveform. Required if strategy == 'acoustic_aware'.
            strategy: 'acoustic_aware' (default) or 'fixed_duration' (Stage 8G baseline).
            edge_context_seconds: Audio context padding (defaults to self.edge_context_seconds).
            search_window_seconds: Search window radius (defaults to self.search_window_seconds).
            max_duration_seconds: Max chunk duration (defaults to self.max_duration_seconds).
        """
        strat = (strategy or "acoustic_aware").lower().strip()
        max_dur = max_duration_seconds or self.max_duration_seconds
        pad_sec = self.edge_context_seconds if edge_context_seconds is None else edge_context_seconds
        win_sec = self.search_window_seconds if search_window_seconds is None else search_window_seconds
        total_audio_dur = (len(waveform) / self.sample_rate) if waveform is not None else 1e9

        # If strategy is fixed_duration or waveform is missing, execute baseline fixed splitting
        if strat == "fixed_duration" or waveform is None:
            return self._split_fixed_duration(segments, max_dur, pad_sec, total_audio_dur)

        # Execute acoustic-aware splitting
        bounded_chunks: List[Dict[str, Any]] = []

        for seg in segments:
            dur = seg["duration"]
            st_sec = seg["start_time"]
            en_sec = seg["end_time"]

            if dur <= max_dur:
                # Segment fits within max duration: keep natural boundary
                chunk = dict(seg)
                chunk["is_subsplit"] = False
                chunk["boundary_type"] = "natural_vad"
                chunk["boundary_cut_method"] = "natural_vad"
                self._apply_context_padding(chunk, pad_sec, total_audio_dur)
                bounded_chunks.append(chunk)
                continue

            # Segment exceeds max duration: iteratively slice at acoustic minima
            seg_start_sample = int(round(st_sec * self.sample_rate))
            seg_end_sample = int(round(en_sec * self.sample_rate))
            seg_wave = waveform[seg_start_sample:seg_end_sample]
            seg_mean_rms = float(np.sqrt(np.mean(seg_wave ** 2))) if len(seg_wave) > 0 else 0.0

            curr_sample = seg_start_sample
            sub_index = 0

            while curr_sample < seg_end_sample:
                rem_samples = seg_end_sample - curr_sample
                rem_sec = rem_samples / self.sample_rate

                if rem_sec <= max_dur:
                    # Final remainder fits within max duration
                    c_end_sample = seg_end_sample
                    c_end_sec = en_sec
                    c_start_sec = round(curr_sample / self.sample_rate, 4)
                    c_dur = round(c_end_sec - c_start_sec, 4)

                    sub_chunk = dict(seg)
                    sub_chunk["segment_id"] = f"{seg['segment_id']}_p{sub_index:02d}"
                    sub_chunk["parent_segment_id"] = seg["segment_id"]
                    sub_chunk["is_subsplit"] = True
                    sub_chunk["start_time"] = c_start_sec
                    sub_chunk["end_time"] = c_end_sec
                    sub_chunk["duration"] = c_dur
                    sub_chunk["start_sample"] = curr_sample
                    sub_chunk["end_sample"] = c_end_sample
                    sub_chunk["boundary_type"] = "remainder_final"
                    sub_chunk["boundary_cut_method"] = "natural_vad_end"
                    sub_chunk["boundary_rms"] = 0.0
                    sub_chunk["dip_ratio"] = 1.0
                    self._apply_context_padding(sub_chunk, pad_sec, total_audio_dur)
                    bounded_chunks.append(sub_chunk)
                    break

                # Determine balanced target duration for remainder
                num_subchunks = int(np.ceil(rem_sec / max_dur))
                target_dur_sec = rem_sec / num_subchunks
                target_samples = int(round(target_dur_sec * self.sample_rate))
                search_samples = int(round(win_sec * self.sample_rate))

                # Find optimal acoustic cut point
                cut_sample, b_rms, method, dip_ratio = self.find_acoustic_cut(
                    waveform=waveform,
                    curr_start_sample=curr_sample,
                    seg_end_sample=seg_end_sample,
                    target_chunk_samples=target_samples,
                    search_window_samples=search_samples,
                    segment_mean_rms=seg_mean_rms,
                )

                c_start_sec = round(curr_sample / self.sample_rate, 4)
                c_end_sec = round(cut_sample / self.sample_rate, 4)
                c_dur = round(c_end_sec - c_start_sec, 4)

                sub_chunk = dict(seg)
                sub_chunk["segment_id"] = f"{seg['segment_id']}_p{sub_index:02d}"
                sub_chunk["parent_segment_id"] = seg["segment_id"]
                sub_chunk["is_subsplit"] = True
                sub_chunk["start_time"] = c_start_sec
                sub_chunk["end_time"] = c_end_sec
                sub_chunk["duration"] = c_dur
                sub_chunk["start_sample"] = curr_sample
                sub_chunk["end_sample"] = cut_sample
                sub_chunk["boundary_type"] = method
                sub_chunk["boundary_cut_method"] = method
                sub_chunk["boundary_rms"] = round(b_rms, 6)
                sub_chunk["dip_ratio"] = round(dip_ratio, 4)
                self._apply_context_padding(sub_chunk, pad_sec, total_audio_dur)
                bounded_chunks.append(sub_chunk)

                curr_sample = cut_sample
                sub_index += 1

        return bounded_chunks

    def _split_fixed_duration(
        self,
        segments: List[Dict[str, Any]],
        max_duration_seconds: float,
        pad_seconds: float,
        total_audio_duration: float,
    ) -> List[Dict[str, Any]]:
        """Baseline fixed-duration slicing (replicates Stage 8G split_long_segments)."""
        bounded = []
        for seg in segments:
            dur = seg["duration"]
            if dur <= max_duration_seconds:
                item = dict(seg)
                item["is_subsplit"] = False
                item["boundary_type"] = "natural_vad"
                item["boundary_cut_method"] = "natural_vad"
                self._apply_context_padding(item, pad_seconds, total_audio_duration)
                bounded.append(item)
                continue

            start_t = seg["start_time"]
            end_t = seg["end_time"]
            num_chunks = int(np.ceil(dur / max_duration_seconds))
            chunk_dur = dur / num_chunks

            for i in range(num_chunks):
                c_start = round(start_t + i * chunk_dur, 4)
                c_end = round(min(end_t, c_start + chunk_dur), 4)
                c_dur = round(c_end - c_start, 4)
                if c_dur <= 0:
                    continue

                sub_item = dict(seg)
                sub_item["segment_id"] = f"{seg['segment_id']}_p{i:02d}"
                sub_item["parent_segment_id"] = seg["segment_id"]
                sub_item["is_subsplit"] = True
                sub_item["start_time"] = c_start
                sub_item["end_time"] = c_end
                sub_item["duration"] = c_dur
                sub_item["start_sample"] = int(c_start * self.sample_rate)
                sub_item["end_sample"] = int(c_end * self.sample_rate)
                sub_item["boundary_type"] = "fixed_duration"
                sub_item["boundary_cut_method"] = "fixed_duration"
                sub_item["boundary_rms"] = 0.0
                sub_item["dip_ratio"] = 1.0
                self._apply_context_padding(sub_item, pad_seconds, total_audio_duration)
                bounded.append(sub_item)

        return bounded

    def _apply_context_padding(
        self,
        chunk: Dict[str, Any],
        pad_seconds: float,
        total_audio_duration: float,
    ) -> None:
        """
        Calculate context boundaries without shifting core start/end timestamps.
        Explicitly tracks core samples vs context padding samples.
        """
        core_st = chunk["start_time"]
        core_en = chunk["end_time"]
        core_st_s = chunk.get("start_sample", int(round(core_st * self.sample_rate)))
        core_en_s = chunk.get("end_sample", int(round(core_en * self.sample_rate)))

        if pad_seconds <= 0.0:
            chunk["context_start_time"] = core_st
            chunk["context_end_time"] = core_en
            chunk["context_duration"] = chunk["duration"]
            chunk["context_start_sample"] = core_st_s
            chunk["context_end_sample"] = core_en_s
            chunk["pad_left_seconds"] = 0.0
            chunk["pad_right_seconds"] = 0.0
            chunk["pad_left_samples"] = 0
            chunk["pad_right_samples"] = 0
            return

        ctx_st = max(0.0, round(core_st - pad_seconds, 4))
        ctx_en = min(total_audio_duration, round(core_en + pad_seconds, 4))
        ctx_dur = round(ctx_en - ctx_st, 4)

        ctx_st_s = int(round(ctx_st * self.sample_rate))
        ctx_en_s = int(round(ctx_en * self.sample_rate))

        chunk["context_start_time"] = ctx_st
        chunk["context_end_time"] = ctx_en
        chunk["context_duration"] = ctx_dur
        chunk["context_start_sample"] = ctx_st_s
        chunk["context_end_sample"] = ctx_en_s
        chunk["pad_left_seconds"] = round(core_st - ctx_st, 4)
        chunk["pad_right_seconds"] = round(ctx_en - core_en, 4)
        chunk["pad_left_samples"] = core_st_s - ctx_st_s
        chunk["pad_right_samples"] = ctx_en_s - core_en_s
