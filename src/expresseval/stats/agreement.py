"""
Inter-annotator agreement: Krippendorff's Alpha metric.
Calculates the theoretical ceiling for automatic evaluation systems based on human agreement.
"""

import numpy as np


def krippendorff_alpha(ratings_matrix: np.ndarray) -> float:
    """
    Computes Krippendorff's alpha for interval/ordinal data.
    ratings_matrix: 2D array of shape (n_units, n_raters), NaN for missing ratings.
    """
    matrix = np.asarray(ratings_matrix, dtype=np.float64)
    n_units, n_raters = matrix.shape

    # Find valid pairs
    valid_mask = ~np.isnan(matrix)
    m_u = np.sum(valid_mask, axis=1)
    # Units with at least 2 ratings
    pairable = m_u >= 2
    if not np.any(pairable):
        return 0.0

    valid_matrix = matrix[pairable]
    m_u = m_u[pairable]

    # Observed disagreement D_o
    d_o = 0.0
    total_pairs = 0
    for u in range(len(valid_matrix)):
        row = valid_matrix[u][~np.isnan(valid_matrix[u])]
        k = len(row)
        for i in range(k):
            for j in range(i + 1, k):
                d_o += (row[i] - row[j]) ** 2
                total_pairs += 1

    if total_pairs == 0:
        return 1.0
    d_o = d_o / total_pairs

    # Expected disagreement D_e
    all_values = matrix[valid_mask]
    n_total = len(all_values)
    if n_total < 2:
        return 0.0

    d_e = 0.0
    for i in range(n_total):
        for j in range(i + 1, n_total):
            d_e += (all_values[i] - all_values[j]) ** 2
    d_e = d_e / (n_total * (n_total - 1) / 2.0)

    if d_e == 0:
        return 1.0

    alpha = 1.0 - (d_o / d_e)
    return round(float(np.clip(alpha, -1.0, 1.0)), 4)
