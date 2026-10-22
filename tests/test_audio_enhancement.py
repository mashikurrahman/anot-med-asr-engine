#!/usr/bin/env python3
"""
Tests for the label-free audio front-end (audio/enhancement.py).

Focus: the enhancement is safe (no NaNs/clipping blowups, preserves length,
no-op on silence) and actually does its job (quiet speech amplified, DC/rumble
reduced) without fabricating content.
"""

import unittest

import numpy as np

from audio.enhancement import (
    enhance_waveform,
    fft_highpass,
    normalize_loudness,
    remove_dc,
)

SR = 16000


def _tone(freq, dur=1.0, amp=0.1, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestAudioEnhancement(unittest.TestCase):
    def test_silence_is_noop_and_safe(self):
        silence = np.zeros(SR, dtype=np.float32)
        out = enhance_waveform(silence, SR)
        self.assertEqual(out.shape, silence.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertAlmostEqual(float(np.max(np.abs(out))), 0.0, places=6)

    def test_length_and_range_preserved(self):
        sig = _tone(300, amp=0.05) + 0.01 * np.random.randn(SR).astype(np.float32)
        out = enhance_waveform(sig, SR)
        self.assertEqual(out.shape, sig.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)

    def test_quiet_speech_is_amplified(self):
        quiet = _tone(300, amp=0.01)  # very quiet
        out = normalize_loudness(quiet, target_dbfs=-20.0)
        self.assertGreater(float(np.sqrt(np.mean(out**2))),
                           float(np.sqrt(np.mean(quiet**2))))

    def test_highpass_attenuates_low_frequency(self):
        low = _tone(30, amp=0.3)   # rumble below cutoff
        high = _tone(1000, amp=0.3)
        mix = low + high
        out = fft_highpass(mix, SR, cutoff_hz=70.0)
        # Low-frequency energy should drop substantially.
        self.assertLess(float(np.sqrt(np.mean(out**2))), float(np.sqrt(np.mean(mix**2))))

    def test_remove_dc(self):
        sig = _tone(300, amp=0.1) + 0.2  # add DC offset
        out = remove_dc(sig)
        self.assertAlmostEqual(float(np.mean(out)), 0.0, places=4)

    def test_does_not_amplify_pure_silence(self):
        out = normalize_loudness(np.zeros(SR, dtype=np.float32))
        self.assertAlmostEqual(float(np.max(np.abs(out))), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
