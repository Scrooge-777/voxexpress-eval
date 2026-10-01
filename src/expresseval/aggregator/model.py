"""
Predictive models for aggregating multi-axis metrics into human MOS predictions.
"""

from typing import Optional
import numpy as np


class RidgeAggregator:
    """
    L2-regularized linear model for predicting human MOS from normalized metric vectors.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeAggregator":
        n, d = X.shape
        # Add intercept column
        X_design = np.hstack([np.ones((n, 1)), X])
        # Regularization matrix (do not regularize bias)
        reg = self.alpha * np.eye(d + 1)
        reg[0, 0] = 0.0

        # Normal equations: w = (X^T X + alpha * I)^(-1) X^T y
        params = np.linalg.solve(X_design.T @ X_design + reg, X_design.T @ y)
        self.bias = float(params[0])
        self.weights = params[1:].astype(np.float32)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Aggregator must be fitted before predict.")
        preds = (X @ self.weights) + self.bias
        # Clip to valid MOS domain [1.0, 5.0]
        return np.clip(preds, 1.0, 5.0).astype(np.float32)
