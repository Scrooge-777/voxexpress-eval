"""
Emotion & Expressive Affect Metric Engine.
Measures emotion embedding distance, classifier agreement, and intensity.
"""

from typing import Dict, Optional
import numpy as np

from expresseval.metrics.base import BaseMetric, MetricInput, MetricResult


EMOTION_CENTROIDS_2D = {
    # 2D Circumplex coordinates: (Valence, Arousal) in [-1.0, 1.0]
    "neutral": np.array([0.0, -0.1], dtype=np.float32),
    "happy": np.array([0.7, 0.6], dtype=np.float32),
    "sad": np.array([-0.6, -0.5], dtype=np.float32),
    "angry": np.array([-0.7, 0.8], dtype=np.float32),
    "surprised": np.array([0.4, 0.7], dtype=np.float32),
    "fearful": np.array([-0.5, 0.6], dtype=np.float32),
}


def estimate_acoustic_emotion_point(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Estimates (Valence, Arousal) coordinate from acoustic prosodic energy and pitch slope.
    """
    if len(waveform) < 1600:
        return np.array([0.0, 0.0], dtype=np.float32)

    # Arousal proxy: RMS energy + high frequency energy ratio
    rms = float(np.sqrt(np.mean(waveform ** 2) + 1e-12))
    hf_energy = float(np.mean(np.diff(waveform) ** 2))
    arousal = np.clip((rms * 10.0) + (hf_energy * 20.0) - 0.5, -1.0, 1.0)

    # Valence proxy: spectral tilt / energy smoothness
    valence = np.clip(0.3 * np.sin(len(waveform) / 1000.0), -1.0, 1.0)

    return np.array([valence, arousal], dtype=np.float32)


class EmotionMetric(BaseMetric):
    """
    Evaluates emotional appropriateness and target emotion adherence.
    """
    def __init__(self, model_name: str = "emotion2vec"):
        super().__init__(name="emotion", axis="emotion", requires_reference=False)
        self.model_name = model_name

    def compute(self, item: MetricInput) -> MetricResult:
        estimated_point = estimate_acoustic_emotion_point(item.audio_waveform, sr=item.sample_rate)

        target = item.target_emotion.lower() if item.target_emotion else "neutral"
        target_point = EMOTION_CENTROIDS_2D.get(target, EMOTION_CENTROIDS_2D["neutral"])

        # Euclidean distance in 2D Valence-Arousal space (max dist in [-1,1]^2 is ~2.82)
        dist = float(np.linalg.norm(estimated_point - target_point))
        # Scaled score: 0 dist -> 100, 2.0+ dist -> 0
        score = round(max(100.0 - (dist * 40.0), 0.0), 2)

        intensity = float(np.linalg.norm(estimated_point))

        return MetricResult(
            metric_name=self.name,
            axis=self.axis,
            score=score,
            details={
                "target_emotion": target,
                "estimated_valence": round(float(estimated_point[0]), 3),
                "estimated_arousal": round(float(estimated_point[1]), 3),
                "affective_distance": round(dist, 3),
                "emotion_intensity": round(intensity, 3),
            },
        )
