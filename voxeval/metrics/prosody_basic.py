"""
Basic prosodic & fundamental frequency dynamics metrics.
Implements mathematical formulations for semitone pitch range, pitch velocity, and voicing dynamics.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class BasicProsodyMetrics:
    f0_mean_hz: float
    f0_std_hz: float
    f0_min_hz: float
    f0_max_hz: float
    f0_range_semitones: float
    pitch_velocity_mean: float
    pitch_velocity_max: float
    voicing_ratio: float
    energy_rms_mean: float
    energy_dynamic_range_db: float


def hz_to_semitones(f0_hz: np.ndarray, reference_hz: float = 440.0) -> np.ndarray:
    """Convert frequencies in Hertz to Semitones relative to a reference pitch (default A4 = 440Hz)."""
    valid = f0_hz > 0
    semitones = np.zeros_like(f0_hz, dtype=np.float64)
    semitones[valid] = 12.0 * np.log2(f0_hz[valid] / reference_hz)
    return semitones


def compute_basic_prosody(
    f0_trajectory: np.ndarray,
    energy_rms: np.ndarray,
    frame_shift_s: float = 0.010,
    min_f0: float = 65.0,
    max_f0: float = 500.0,
) -> BasicProsodyMetrics:
    """
    Computes baseline acoustic prosody dynamics over an aligned frame sequence.

    Args:
        f0_trajectory: 1D array of extracted fundamental frequencies per frame (0 for unvoiced).
        energy_rms: 1D array of root-mean-square energy values per frame.
        frame_shift_s: Temporal interval between consecutive frames in seconds (default 10ms).
        min_f0: Minimum valid pitch threshold in Hz.
        max_f0: Maximum valid pitch threshold in Hz.

    Returns:
        BasicProsodyMetrics dataclass containing computed statistical moments and dynamic ranges.
    """
    voiced_mask = (f0_trajectory >= min_f0) & (f0_trajectory <= max_f0)
    voiced_f0 = f0_trajectory[voiced_mask]

    if len(voiced_f0) == 0:
        return BasicProsodyMetrics(
            f0_mean_hz=0.0,
            f0_std_hz=0.0,
            f0_min_hz=0.0,
            f0_max_hz=0.0,
            f0_range_semitones=0.0,
            pitch_velocity_mean=0.0,
            pitch_velocity_max=0.0,
            voicing_ratio=0.0,
            energy_rms_mean=float(np.mean(energy_rms)) if len(energy_rms) > 0 else 0.0,
            energy_dynamic_range_db=0.0,
        )

    # 1. Pitch moments in Hertz
    f0_mean = float(np.mean(voiced_f0))
    f0_std = float(np.std(voiced_f0))
    f0_min = float(np.min(voiced_f0))
    f0_max = float(np.max(voiced_f0))

    # 2. Dynamic pitch range in perceptual Semitones
    semitones = hz_to_semitones(voiced_f0)
    f0_range_st = float(np.max(semitones) - np.min(semitones))

    # 3. Pitch velocity (first temporal derivative in semitones/sec)
    st_full = np.zeros_like(f0_trajectory, dtype=np.float64)
    st_full[voiced_mask] = hz_to_semitones(voiced_f0)
    velocity = np.abs(np.diff(st_full)) / frame_shift_s
    voiced_diff_mask = voiced_mask[:-1] & voiced_mask[1:]
    if np.any(voiced_diff_mask):
        pitch_vel_mean = float(np.mean(velocity[voiced_diff_mask]))
        pitch_vel_max = float(np.max(velocity[voiced_diff_mask]))
    else:
        pitch_vel_mean = 0.0
        pitch_vel_max = 0.0

    # 4. Voicing ratio (percentage of voiced speech frames)
    voicing_ratio = float(np.mean(voiced_mask))

    # 5. Energy statistics
    energy_mean = float(np.mean(energy_rms))
    nonzero_energy = energy_rms[energy_rms > 1e-7]
    if len(nonzero_energy) > 0:
        e_max_db = 20.0 * np.log10(np.max(nonzero_energy))
        e_min_db = 20.0 * np.log10(np.min(nonzero_energy))
        energy_dyn_range = float(e_max_db - e_min_db)
    else:
        energy_dyn_range = 0.0

    return BasicProsodyMetrics(
        f0_mean_hz=f0_mean,
        f0_std_hz=f0_std,
        f0_min_hz=f0_min,
        f0_max_hz=f0_max,
        f0_range_semitones=f0_range_st,
        pitch_velocity_mean=pitch_vel_mean,
        pitch_velocity_max=pitch_vel_max,
        voicing_ratio=voicing_ratio,
        energy_rms_mean=energy_mean,
        energy_dynamic_range_db=energy_dyn_range,
    )
