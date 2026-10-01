"""
Unit tests for metric engines with known-answer tests.
"""

import unittest
import numpy as np

from expresseval.metrics.intelligibility import compute_wer, compute_cer, levenshtein_distance
from expresseval.metrics.prosody import extract_simple_f0, simple_dtw
from expresseval.metrics.speaker_sim import cosine_similarity
from expresseval.metrics.spectral import compute_mfcc


class TestMetrics(unittest.TestCase):
    def test_levenshtein_distance(self):
        # Identical
        self.assertEqual(levenshtein_distance(["hello"], ["hello"]), 0)
        # 1 substitution
        self.assertEqual(levenshtein_distance(["cat"], ["bat"]), 1)
        # 1 insertion + 1 deletion
        self.assertEqual(levenshtein_distance(["a", "b"], ["a", "c", "d"]), 2)

    def test_wer_known_answers(self):
        ref = "the quick brown fox"
        # Exact match
        self.assertEqual(compute_wer(ref, "the quick brown fox"), 0.0)
        # 1 substitution out of 4 words = 0.25
        self.assertEqual(compute_wer(ref, "the fast brown fox"), 0.25)
        # Empty hypothesis
        self.assertEqual(compute_wer(ref, ""), 1.0)

    def test_cer_known_answers(self):
        # Chinese characters: 1 substitution out of 6 characters
        ref = "语音合成评估"
        hyp = "语音合成评测"
        cer = compute_cer(ref, hyp)
        self.assertAlmostEqual(cer, 1.0 / 6.0, places=3)

    def test_cosine_similarity(self):
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0, places=4)
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0, places=4)

    def test_f0_extraction_synthetic_tone(self):
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        # 200 Hz pure sine wave
        sine_wave = 0.5 * np.sin(2 * np.pi * 200.0 * t).astype(np.float32)

        f0, voiced = extract_simple_f0(sine_wave, sr=sr, f0_min=65.0, f0_max=500.0)
        voiced_f0 = f0[voiced]
        self.assertTrue(len(voiced_f0) > 0)
        # Median pitch should be close to 200 Hz
        self.assertAlmostEqual(float(np.median(voiced_f0)), 200.0, delta=10.0)


if __name__ == "__main__":
    unittest.main()
