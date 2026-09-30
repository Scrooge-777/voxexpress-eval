"""
Synthetic audio & prosody profile generators for autonomous evaluation and testing.
"""

from typing import Tuple
import numpy as np


def generate_synthetic_prosody_profile(
    duration_s: float = 3.5,
    frame_shift_s: float = 0.010,
    base_pitch_hz: float = 180.0,
    pitch_mod_depth: float = 35.0,
    expressiveness_factor: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates deterministic synthetic pitch (F0) and RMS energy trajectories
    simulating realistic human speech intonation patterns.

    Args:
        duration_s: Duration in seconds.
        frame_shift_s: Step size per frame.
        base_pitch_hz: Mean speaker fundamental frequency.
        pitch_mod_depth: Variation amplitude in Hz.
        expressiveness_factor: Multiplier for emotional dynamics.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (f0_trajectory, energy_rms) arrays.
    """
    rng = np.random.default_rng(seed)
    n_frames = int(duration_s / frame_shift_s)
    t = np.linspace(0, duration_s, n_frames)

    # Macro-intonation: Declination arc + question-rise/emphasis
    macro_contour = base_pitch_hz - 15.0 * (t / duration_s)
    
    # Syllabic stress: periodic accents
    syllabic_osc = (pitch_mod_depth * expressiveness_factor) * np.sin(2.0 * np.pi * 3.5 * t)
    
    # Emotional vibrato/inflection
    vibrato = (5.0 * expressiveness_factor) * np.sin(2.0 * np.pi * 6.0 * t)
    
    # Micro-prosodic jitter
    jitter = rng.normal(0, 1.5, n_frames)

    raw_f0 = macro_contour + syllabic_osc + vibrato + jitter

    # Simulate realistic unvoiced pauses (consonants / breathing)
    voicing_mask = np.ones(n_frames, dtype=bool)
    # Add simulated pauses every ~1 second
    for pause_start in np.arange(0.8, duration_s - 0.3, 1.1):
        idx_start = int(pause_start / frame_shift_s)
        idx_end = int((pause_start + 0.18) / frame_shift_s)
        voicing_mask[idx_start:idx_end] = False

    f0_trajectory = np.where(voicing_mask, np.clip(raw_f0, 65.0, 500.0), 0.0)

    # Correlated energy contour (higher pitch often correlates with higher energy)
    normalized_f0 = np.clip((f0_trajectory - base_pitch_hz) / (base_pitch_hz + 1e-6), -0.5, 0.5)
    base_energy = 0.05 + 0.04 * (normalized_f0 + 0.5)
    energy_rms = np.where(voicing_mask, base_energy + rng.normal(0, 0.005, n_frames), 0.001)
    energy_rms = np.clip(energy_rms, 1e-4, 1.0)

    return f0_trajectory, energy_rms
