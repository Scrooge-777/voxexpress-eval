"""
Correlation statistics: Pearson r, Spearman rho, and Kendall tau.
"""

from typing import Dict, Tuple
import numpy as np
import scipy.stats as stats


def compute_correlations(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Computes Pearson r, Spearman rho, and Kendall tau correlation coefficients.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have identical lengths.")
    if len(predictions) < 3:
        return {"pearson": 0.0, "spearman": 0.0, "kendall_tau": 0.0}

    # Pearson r
    pearson_r, _ = stats.pearsonr(predictions, ground_truth)
    # Spearman rho
    spearman_rho, _ = stats.spearmanr(predictions, ground_truth)
    # Kendall tau
    kendall_tau, _ = stats.kendalltau(predictions, ground_truth)

    return {
        "pearson": round(float(0.0 if np.isnan(pearson_r) else pearson_r), 4),
        "spearman": round(float(0.0 if np.isnan(spearman_rho) else spearman_rho), 4),
        "kendall_tau": round(float(0.0 if np.isnan(kendall_tau) else kendall_tau), 4),
    }
