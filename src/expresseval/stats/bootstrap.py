"""
Bootstrap confidence interval estimation for evaluation metrics.
"""

from typing import Callable, Tuple
import numpy as np


def bootstrap_ci(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes empirical bootstrap estimate and (1 - alpha) Confidence Intervals.
    Returns: (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(y_pred)
    if n < 5:
        point = float(metric_fn(y_pred, y_true))
        return point, point, point

    point_estimate = float(metric_fn(y_pred, y_true))
    boot_estimates = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, n, size=n)
        sample_pred = y_pred[indices]
        sample_true = y_true[indices]
        try:
            val = metric_fn(sample_pred, sample_true)
            if not np.isnan(val):
                boot_estimates.append(val)
        except Exception:
            continue

    if len(boot_estimates) < 10:
        return point_estimate, point_estimate, point_estimate

    ci_lower = float(np.percentile(boot_estimates, (alpha / 2.0) * 100))
    ci_upper = float(np.percentile(boot_estimates, (1.0 - alpha / 2.0) * 100))

    return round(point_estimate, 4), round(ci_lower, 4), round(ci_upper, 4)
