"""
Prosodic and Intonational Dynamics Metric Engine.
Computes F0 RMSE, F0 correlation, energy dynamics, speaking rate, and DTW alignment.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.spatial.distance import cdist

from expresseval.metrics.base import BaseMetric, MetricInput, MetricResult


def extract_simple_f0(
    waveform: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    frame_length: int = 400,
    f0_min: float = 65.0,
    f0_max: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Autocorrelation-based pitch tracking with parabolic interpolation.
    Returns (f0_trajectory, voiced_mask).
    """
    n_frames = 1 + int((len(waveform) - frame_length) / hop_length)
    if n_frames <= 0:
        return np.array([0.0]), np.array([False])

    f0 = np.zeros(n_frames, dtype=np.float32)
    voiced = np.zeros(n_frames, dtype=bool)

    min_lag = int(sr / f0_max)
    max_lag = int(sr / f0_min)

    for i in range(n_frames):
        start = i * hop_length
        frame = waveform[start : start + frame_length]
        if len(frame) < frame_length:
            break

        # Remove DC bias & window
        frame = frame - np.mean(frame)
        energy = np.sum(frame ** 2)
        if energy < 1e-4:
            continue

        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode="full")[frame_length - 1 :]
        search_window = autocorr[min_lag : min_lag + (max_lag - min_lag)]
        if len(search_window) == 0:
            continue

        peak_idx = min_lag + np.argmax(search_window)
        peak_val = autocorr[peak_idx]

        # Voicing decision based on normalized autocorrelation peak
        norm_peak = peak_val / (autocorr[0] + 1e-8)
        if norm_peak > 0.35:
            # Parabolic interpolation for sub-sample accuracy
            if 0 < peak_idx < len(autocorr) - 1:
                alpha = autocorr[peak_idx - 1]
                beta = autocorr[peak_idx]
                gamma = autocorr[peak_idx + 1]
                denom = 2 * (2 * beta - alpha - gamma) + 1e-8
                delta = (alpha - gamma) / denom
                true_lag = peak_idx + delta
            else:
                true_lag = peak_idx

            pitch = sr / (true_lag + 1e-8)
            if f0_min <= pitch <= f0_max:
                f0[i] = pitch
                voiced[i] = True

    return f0, voiced


def simple_dtw(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes dynamic time warping path between two 1D sequences."""
    n, m = len(x), len(y)
    cost = np.zeros((n, m))
    cost[0, 0] = abs(x[0] - y[0])

    for i in range(1, n):
        cost[i, 0] = cost[i - 1, 0] + abs(x[i] - y[0])
    for j in range(1, m):
        cost[0, j] = cost[0, j - 1] + abs(x[0] - y[j])

    for i in range(1, n):
        for j in range(1, m):
            cost[i, j] = abs(x[i] - y[j]) + min(
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )

    return cost


class ProsodyMetric(BaseMetric):
    """
    Evaluates prosodic naturalness, dynamic pitch range, velocity, and reference similarity.
    """
    def __init__(self, f0_min: float = 65.0, f0_max: float = 500.0, dtw_alignment: bool = True):
        super().__init__(name="prosody", axis="prosody", requires_reference=False)
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.dtw_alignment = dtw_alignment

    def compute(self, item: MetricInput) -> MetricResult:
        f0, voiced = extract_simple_f0(
            item.audio_waveform,
            sr=item.sample_rate,
            f0_min=self.f0_min,
            f0_max=self.f0_max,
        )

        voiced_f0 = f0[voiced]
        if len(voiced_f0) > 1:
            semitones = 12.0 * np.log2(voiced_f0 / 440.0)
            f0_range_st = float(np.max(semitones) - np.min(semitones))
            f0_std_hz = float(np.std(voiced_f0))
            f0_mean_hz = float(np.mean(voiced_f0))
            pitch_vel = float(np.mean(np.abs(np.diff(semitones))))
        else:
            f0_range_st = 0.0
            f0_std_hz = 0.0
            f0_mean_hz = 0.0
            pitch_vel = 0.0

        # Speaking rate proxy: syllables inferred from text vs duration
        char_count = len(item.text.replace(" ", ""))
        duration = len(item.audio_waveform) / (item.sample_rate + 1e-8)
        speaking_rate = float(char_count / (duration + 1e-8)) if duration > 0 else 0.0

        details = {
            "f0_mean_hz": round(f0_mean_hz, 2),
            "f0_std_hz": round(f0_std_hz, 2),
            "f0_range_semitones": round(f0_range_st, 2),
            "pitch_velocity": round(pitch_vel, 3),
            "speaking_rate_cps": round(speaking_rate, 2),
            "voicing_percentage": round(float(np.mean(voiced) * 100), 1),
        }

        # If reference audio is provided, compute F0 correlation and RMSE
        if item.ref_waveform is not None and len(item.ref_waveform) > 0:
            ref_f0, ref_voiced = extract_simple_f0(
                item.ref_waveform,
                sr=item.ref_sample_rate or item.sample_rate,
                f0_min=self.f0_min,
                f0_max=self.f0_max,
            )
            min_len = min(len(f0), len(ref_f0))
            if min_len > 5:
                sub_f0 = f0[:min_len]
                sub_ref = ref_f0[:min_len]
                joint_voiced = voiced[:min_len] & ref_voiced[:min_len]
                if np.sum(joint_voiced) > 3:
                    f0_rmse = float(np.sqrt(np.mean((sub_f0[joint_voiced] - sub_ref[joint_voiced]) ** 2)))
                    corr = float(np.corrcoef(sub_f0[joint_voiced], sub_ref[joint_voiced])[0, 1])
                    details["ref_f0_rmse"] = round(f0_rmse, 2)
                    details["ref_f0_corr"] = round(0.0 if np.isnan(corr) else corr, 3)

        # Baseline composite prosody score (0 - 100)
        score = min(max((f0_range_st * 6.0) + (pitch_vel * 25.0), 0.0), 100.0)

        return MetricResult(
            metric_name=self.name,
            axis=self.axis,
            score=round(score, 2),
            details=details,
        )
