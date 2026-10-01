"""
Spectral Quality Metric Engine: Mel-Cepstral Distortion (MCD) & Log-Spectral Distance.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.spatial.distance import cdist

from expresseval.metrics.base import BaseMetric, MetricInput, MetricResult


def compute_mfcc(waveform: np.ndarray, sr: int = 16000, n_mfcc: int = 13, n_fft: int = 512, hop: int = 160) -> np.ndarray:
    """Computes basic Mel-Frequency Cepstral Coefficients (MFCCs)."""
    n_frames = (len(waveform) - n_fft) // hop
    if n_frames <= 0:
        return np.zeros((1, n_mfcc), dtype=np.float32)

    frames = np.lib.stride_tricks.sliding_window_view(waveform[: n_frames * hop + n_fft], n_fft)[::hop]
    windowed = frames * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2

    # Mel-scale filterbank (simple log compression)
    log_spec = np.log(spec + 1e-8)
    # Discrete Cosine Transform (DCT Type-II)
    mfcc = np.zeros((n_frames, n_mfcc), dtype=np.float32)
    k = np.arange(n_mfcc)
    n = np.arange(log_spec.shape[1])
    cos_basis = np.cos(np.pi * k[:, None] * (n + 0.5) / log_spec.shape[1])

    for i in range(n_frames):
        mfcc[i] = np.dot(cos_basis, log_spec[i])

    return mfcc


def compute_mcd_dtw(synth_mfcc: np.ndarray, ref_mfcc: np.ndarray) -> float:
    """
    Computes Mel-Cepstral Distortion (MCD in dB) using DTW alignment.
    MCD = (10 * sqrt(2) / ln(10)) * mean(Euclidean distance)
    """
    scale_factor = (10.0 * np.sqrt(2.0)) / np.log(10.0)
    cost_matrix = cdist(synth_mfcc, ref_mfcc, metric="euclidean")

    # Dynamic Time Warping optimal path
    n, m = cost_matrix.shape
    dtw_cost = np.zeros((n, m))
    dtw_cost[0, 0] = cost_matrix[0, 0]
    for i in range(1, n):
        dtw_cost[i, 0] = dtw_cost[i - 1, 0] + cost_matrix[i, 0]
    for j in range(1, m):
        dtw_cost[0, j] = dtw_cost[0, j - 1] + cost_matrix[0, j]

    for i in range(1, n):
        for j in range(1, m):
            dtw_cost[i, j] = cost_matrix[i, j] + min(
                dtw_cost[i - 1, j],
                dtw_cost[i, j - 1],
                dtw_cost[i - 1, j - 1],
            )

    # Average aligned frame distance
    path_len = n + m
    avg_euclidean = float(dtw_cost[-1, -1] / path_len)
    mcd_db = float(scale_factor * avg_euclidean)
    return mcd_db


class SpectralMetric(BaseMetric):
    """
    Evaluates spectral acoustic fidelity against a reference audio using MCD.
    """
    def __init__(self, n_mfcc: int = 13):
        super().__init__(name="spectral", axis="spectral", requires_reference=True)
        self.n_mfcc = n_mfcc

    def compute(self, item: MetricInput) -> MetricResult:
        if item.ref_waveform is None or len(item.ref_waveform) == 0:
            return MetricResult(
                metric_name=self.name,
                axis=self.axis,
                score=0.0,
                details={"error": "Reference audio required for spectral MCD"},
            )

        synth_mfcc = compute_mfcc(item.audio_waveform, sr=item.sample_rate, n_mfcc=self.n_mfcc)
        ref_mfcc = compute_mfcc(item.ref_waveform, sr=item.ref_sample_rate or item.sample_rate, n_mfcc=self.n_mfcc)

        mcd_db = compute_mcd_dtw(synth_mfcc, ref_mfcc)
        # Scaled score: MCD of 0 dB is 100, MCD of 12+ dB is 0
        score = round(max(100.0 - (mcd_db * 8.0), 0.0), 2)

        return MetricResult(
            metric_name=self.name,
            axis=self.axis,
            score=score,
            details={
                "mcd_db": round(mcd_db, 2),
                "n_mfcc": self.n_mfcc,
            },
        )
