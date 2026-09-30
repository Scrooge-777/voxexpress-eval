"""
Core configuration and parameters for VoxExpress-Eval.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class LanguageCode(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"
    MANDARIN = "zh"
    HINDI = "hi"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"


class EmotionCategory(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    EMPATHETIC = "empathetic"


@dataclass
class AudioConfig:
    sample_rate: int = 24000
    target_lufs: float = -23.0  # EBU R128 standard loudness target
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    n_mels: int = 80
    f_min: float = 50.0
    f_max: float = 11000.0


@dataclass
class ProsodyConfig:
    f0_min_hz: float = 65.0
    f0_max_hz: float = 500.0
    voicing_threshold: float = 0.45
    cwt_octaves: int = 10
    cwt_scales_per_octave: int = 4
    min_pause_duration_ms: float = 150.0


@dataclass
class MetricWeights:
    """Weights for the composite PolyExpress Score."""
    prosody_weight: float = 0.30
    emotion_weight: float = 0.25
    speaker_weight: float = 0.25
    intelligibility_weight: float = 0.20

    def validate(self) -> None:
        total = (
            self.prosody_weight
            + self.emotion_weight
            + self.speaker_weight
            + self.intelligibility_weight
        )
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Metric weights must sum to 1.0 (currently {total:.4f})")


@dataclass
class EvalConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)
    weights: MetricWeights = field(default_factory=MetricWeights)
    target_languages: List[LanguageCode] = field(
        default_factory=lambda: [
            LanguageCode.ENGLISH,
            LanguageCode.SPANISH,
            LanguageCode.MANDARIN,
            LanguageCode.HINDI,
            LanguageCode.FRENCH,
            LanguageCode.GERMAN,
            LanguageCode.JAPANESE,
        ]
    )

    def __post_init__(self):
        self.weights.validate()
