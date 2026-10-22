#!/usr/bin/env python3
"""
Anot Medical ASR — Audio Front-End Enhancement (label-free accuracy lever)
==========================================================================
Stage 22: Deterministic, dependency-light audio clean-up applied BEFORE the ASR
engine. Targets the dominant error sources on real patient-visit recordings:

  1. Quiet / distant patient speech  -> loudness (RMS) normalization ("AGC").
  2. Low-frequency room rumble / HVAC -> FFT high-pass.
  3. DC offset from cheap mics        -> mean removal.

Pure NumPy (no scipy/librosa). It changes only the acoustic signal level and
low-frequency content; it never fabricates or removes speech content, so it
cannot alter clinical facts — it only makes existing speech easier to recognise.

All operations are safe no-ops on silence and preserve sample count and dtype.
"""

from typing import Optional

import numpy as np

EPS = 1e-9


def remove_dc(wave: np.ndarray) -> np.ndarray:
    """Remove any DC offset (constant bias) from the waveform."""
    return wave - float(np.mean(wave)) if wave.size else wave


def fft_highpass(wave: np.ndarray, sample_rate: int, cutoff_hz: float = 70.0) -> np.ndarray:
    """
    Attenuate content below `cutoff_hz` using a smooth frequency-domain mask.
    Vectorised (one FFT over the whole signal) — fine for offline batch audio.
    """
    n = wave.size
    if n == 0 or cutoff_hz <= 0:
        return wave
    spectrum = np.fft.rfft(wave)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    # Smooth raised-cosine transition over [cutoff/2, cutoff] to avoid ringing.
    lo, hi = cutoff_hz * 0.5, cutoff_hz
    mask = np.ones_like(freqs)
    below = freqs < lo
    trans = (freqs >= lo) & (freqs < hi)
    mask[below] = 0.0
    mask[trans] = 0.5 * (1 - np.cos(np.pi * (freqs[trans] - lo) / (hi - lo)))
    filtered = np.fft.irfft(spectrum * mask, n=n)
    return filtered.astype(wave.dtype, copy=False)


def normalize_loudness(
    wave: np.ndarray,
    target_dbfs: float = -20.0,
    max_gain: float = 20.0,
    peak_ceiling: float = 0.97,
) -> np.ndarray:
    """
    Bring the signal to a target RMS loudness (simple AGC), capped so we don't
    over-amplify near-silence, with a peak guard to prevent clipping.
    """
    if wave.size == 0:
        return wave
    rms = float(np.sqrt(np.mean(np.square(wave))))
    if rms < EPS:
        return wave  # silence: nothing to normalize
    target_lin = 10.0 ** (target_dbfs / 20.0)
    gain = min(target_lin / rms, 10.0 ** (max_gain / 20.0))
    out = wave * gain
    peak = float(np.max(np.abs(out)))
    if peak > peak_ceiling:
        out = out * (peak_ceiling / peak)
    return out.astype(wave.dtype, copy=False)


def enhance_waveform(
    wave: np.ndarray,
    sample_rate: int = 16000,
    highpass: bool = True,
    highpass_cutoff_hz: float = 70.0,
    loudness_norm: bool = True,
    target_dbfs: float = -20.0,
    denoise: bool = False,
) -> np.ndarray:
    """
    Apply the front-end chain. Returns a new array (input is never mutated).

    Order: DC removal -> high-pass -> optional denoise -> loudness normalization.
    `denoise` uses the `noisereduce` library if installed; otherwise it is a
    graceful no-op (the chain still helps via high-pass + AGC).
    """
    if wave is None or wave.size == 0:
        return wave
    out = np.asarray(wave, dtype=np.float32).copy()
    out = remove_dc(out)
    if highpass:
        out = fft_highpass(out, sample_rate, highpass_cutoff_hz)
    if denoise:
        out = _maybe_denoise(out, sample_rate)
    if loudness_norm:
        out = normalize_loudness(out, target_dbfs=target_dbfs)
    # Final safety: clip to valid range and keep finite.
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


def _maybe_denoise(wave: np.ndarray, sample_rate: int) -> np.ndarray:
    """Spectral-gate denoise via `noisereduce` if available; else no-op."""
    try:
        import noisereduce as nr  # optional dependency
    except Exception:
        return wave
    try:
        return nr.reduce_noise(y=wave, sr=sample_rate, stationary=False).astype(np.float32)
    except Exception:
        return wave
