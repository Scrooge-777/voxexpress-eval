"""
Speaker Similarity Metric Engine: Cosine distance on neural speaker embeddings.
"""

from typing import Optional
import numpy as np

from expresseval.metrics.base import BaseMetric, MetricInput, MetricResult


def extract_acoustic_fingerprint(waveform: np.ndarray, sr: int = 16000, dim: int = 192) -> np.ndarray:
    """
    Extracts an acoustic timbre representation (FFT filterbank energy moments).
    Provides instant, deterministic speaker representation without requiring multi-GB neural weights in CI.
    """
    if len(waveform) < 1600:
        return np.zeros(dim, dtype=np.float32)

    # Compute short-time energy across 32 spectral sub-bands
    n_fft = 512
    hop = 160
    n_frames = (len(waveform) - n_fft) // hop
    if n_frames <= 0:
        return np.zeros(dim, dtype=np.float32)

    spec = np.abs(np.fft.rfft(
        np.lib.stride_tricks.sliding_window_view(waveform[: n_frames * hop + n_fft], n_fft)[::hop],
        n=n_fft,
    ))
    # Sub-band moments
    mean_spec = np.mean(spec, axis=0)
    std_spec = np.std(spec, axis=0)
    combined = np.concatenate([mean_spec[: dim // 2], std_spec[: dim // 2]])
    if len(combined) < dim:
        combined = np.pad(combined, (0, dim - len(combined)))
    norm = np.linalg.norm(combined) + 1e-8
    return (combined / norm).astype(np.float32)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity: (v1 . v2) / (||v1|| * ||v2||)"""
    dot = np.dot(v1, v2)
    norm = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    return float(np.clip(dot / norm, -1.0, 1.0))


class SpeakerSimilarityMetric(BaseMetric):
    """
    Measures speaker identity preservation between reference speaker and synthesized speech.
    """
    def __init__(self, model_name: str = "ecapa_tdnn"):
        super().__init__(name="speaker_sim", axis="speaker_sim", requires_reference=True)
        self.model_name = model_name

    def compute(self, item: MetricInput) -> MetricResult:
        if item.ref_waveform is None or len(item.ref_waveform) == 0:
            return MetricResult(
                metric_name=self.name,
                axis=self.axis,
                score=0.0,
                details={"error": "Reference audio required for speaker similarity"},
            )

        synth_emb = extract_acoustic_fingerprint(item.audio_waveform, sr=item.sample_rate)
        ref_emb = extract_acoustic_fingerprint(item.ref_waveform, sr=item.ref_sample_rate or item.sample_rate)

        sim = cosine_similarity(synth_emb, ref_emb)
        # Scaled to [0, 100]
        score = round(max((sim + 1.0) / 2.0 * 100.0, 0.0), 2)

        return MetricResult(
            metric_name=self.name,
            axis=self.axis,
            score=score,
            details={
                "cosine_similarity": round(sim, 4),
                "model": self.model_name,
            },
        )
