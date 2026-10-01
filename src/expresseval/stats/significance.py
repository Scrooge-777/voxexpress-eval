"""
Statistical significance testing: Fisher z-transform and Steiger's test
for comparing two dependent correlations sharing one common variable (human MOS).
"""

from typing import Tuple
import numpy as np
import scipy.stats as stats


def fisher_z_transform(r: float) -> float:
    """Fisher r-to-z transformation: z = 0.5 * ln((1 + r) / (1 - r))"""
    r_clipped = np.clip(r, -0.9999, 0.9999)
    return float(0.5 * np.log((1.0 + r_clipped) / (1.0 - r_clipped)))


def steiger_test(r12: float, r13: float, r23: float, n: int) -> Tuple[float, float]:
    """
    Steiger's Z-test for comparing two dependent correlations r12 and r13
    sharing a common variable 1 (e.g. human MOS).

    r12: Correlation between Model 1 and Human MOS
    r13: Correlation between Model 2 and Human MOS
    r23: Correlation between Model 1 and Model 2
    n: Number of evaluated utterances

    Returns:
        (z_score, p_value)
    """
    if n <= 3:
        return 0.0, 1.0

    r12 = float(np.clip(r12, -0.999, 0.999))
    r13 = float(np.clip(r13, -0.999, 0.999))
    r23 = float(np.clip(r23, -0.999, 0.999))

    # Average correlation
    r_bar = (r12 + r13) / 2.0
    f = (1.0 - r23) / (2.0 * (1.0 - r_bar ** 2) + 1e-8)
    h = (1.0 - f * (r_bar ** 2)) / (1.0 - r_bar ** 2 + 1e-8)

    z12 = fisher_z_transform(r12)
    z13 = fisher_z_transform(r13)

    denom = np.sqrt((2.0 * (1.0 - r23) * h) / (n - 3.0) + 1e-8)
    z = (z12 - z13) / denom

    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return round(float(z), 4), round(p_val, 4)
