"""
Loss functions for learned aggregator: MSE + Bradley-Terry pairwise ranking loss.
"""

from typing import Tuple
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def bradley_terry_loss(y_pred: np.ndarray, y_true: np.ndarray, margin: float = 0.0) -> float:
    """
    Bradley-Terry pairwise ranking loss:
    L_BT = - (1 / |P|) * sum_{(i,j): y_i > y_j} log(sigmoid(y_hat_i - y_hat_j - margin))
    """
    diff_true = y_true[:, None] - y_true[None, :]
    pairs = diff_true > 0  # True when item i is rated better than item j

    if not np.any(pairs):
        return 0.0

    diff_pred = y_pred[:, None] - y_pred[None, :] - margin
    probs = np.clip(sigmoid(diff_pred[pairs]), 1e-7, 1.0 - 1e-7)
    return float(-np.mean(np.log(probs)))


def combined_aggregator_loss(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    lambda_ranking: float = 0.5,
    margin: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Computes L = MSE(y, y_hat) + lambda * L_BT(y, y_hat).
    Returns: (total_loss, mse_loss, ranking_loss)
    """
    mse = float(np.mean((y_pred - y_true) ** 2))
    ranking = bradley_terry_loss(y_pred, y_true, margin=margin)
    total = mse + (lambda_ranking * ranking)
    return total, mse, ranking
