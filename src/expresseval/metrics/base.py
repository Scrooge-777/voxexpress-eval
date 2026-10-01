"""
Abstract Metric interface for ExpressEval.
Every evaluation metric implements this contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class MetricInput:
    id: str
    audio_waveform: np.ndarray
    sample_rate: int
    text: str
    language: str
    ref_waveform: Optional[np.ndarray] = None
    ref_sample_rate: Optional[int] = None
    target_emotion: Optional[str] = None
    file_hash: Optional[str] = None


@dataclass
class MetricResult:
    metric_name: str
    axis: str
    score: float
    details: Optional[Dict[str, Any]] = None


class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics.
    """
    def __init__(self, name: str, axis: str, requires_reference: bool = False):
        self.name = name
        self.axis = axis
        self.requires_reference = requires_reference

    @abstractmethod
    def compute(self, item: MetricInput) -> MetricResult:
        """Computes metric score for a single evaluation input."""
        pass

    def compute_batch(self, batch: List[MetricInput]) -> List[MetricResult]:
        """Vectorized or sequential batch execution."""
        return [self.compute(item) for item in batch]

    def __call__(self, item_or_batch):
        if isinstance(item_or_batch, list):
            return self.compute_batch(item_or_batch)
        return self.compute(item_or_batch)
