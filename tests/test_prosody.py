"""
Unit tests for prosodic dynamics and fundamental frequency metrics.
"""

import unittest
import numpy as np

from voxeval.core.synthetic import generate_synthetic_prosody_profile
from voxeval.metrics.prosody_basic import compute_basic_prosody, hz_to_semitones


class TestProsodyMetrics(unittest.TestCase):
    def test_hz_to_semitones_conversion(self):
        # 440 Hz relative to 440 Hz reference should be 0 semitones
        hz = np.array([440.0])
        st = hz_to_semitones(hz, reference_hz=440.0)
        self.assertAlmostEqual(st[0], 0.0, places=4)

        # 880 Hz (one octave up) should be 12 semitones
        hz_octave = np.array([880.0])
        st_octave = hz_to_semitones(hz_octave, reference_hz=440.0)
        self.assertAlmostEqual(st_octave[0], 12.0, places=4)

    def test_compute_basic_prosody(self):
        f0, energy = generate_synthetic_prosody_profile(
            duration_s=2.0,
            base_pitch_hz=200.0,
            pitch_mod_depth=30.0,
            expressiveness_factor=1.0,
            seed=123,
        )

        metrics = compute_basic_prosody(f0, energy)

        # Verify physical sanity
        self.assertGreater(metrics.f0_mean_hz, 150.0)
        self.assertLess(metrics.f0_mean_hz, 250.0)
        self.assertGreater(metrics.f0_range_semitones, 2.0)
        self.assertGreater(metrics.voicing_ratio, 0.4)
        self.assertLessEqual(metrics.voicing_ratio, 1.0)
        self.assertGreater(metrics.pitch_velocity_mean, 0.0)


if __name__ == "__main__":
    unittest.main()
