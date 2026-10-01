"""
Unit tests for statistical correlation, bootstrap CIs, and significance testing.
"""

import unittest
import numpy as np

from expresseval.stats.correlation import compute_correlations
from expresseval.stats.bootstrap import bootstrap_ci
from expresseval.stats.significance import steiger_test, fisher_z_transform
from expresseval.stats.agreement import krippendorff_alpha


class TestStats(unittest.TestCase):
    def test_correlations(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 1.9, 3.2, 3.9, 5.1])
        res = compute_correlations(x, y)
        self.assertGreater(res["pearson"], 0.95)
        self.assertGreater(res["spearman"], 0.95)
        self.assertGreater(res["kendall_tau"], 0.90)

    def test_bootstrap_ci(self):
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y_true = np.array([1.2, 1.8, 3.1, 4.2, 4.9, 6.1, 6.9, 8.2])

        def pearson_fn(p, t):
            return float(np.corrcoef(p, t)[0, 1])

        pt, lower, upper = bootstrap_ci(y_pred, y_true, pearson_fn, n_bootstraps=100)
        self.assertLessEqual(lower, pt)
        self.assertGreaterEqual(upper, pt)

    def test_steiger_test(self):
        # Model 1 has r=0.90 with human MOS, Model 2 has r=0.50, N=100
        z, p_val = steiger_test(r12=0.90, r13=0.50, r23=0.60, n=100)
        self.assertGreater(z, 2.0)
        self.assertLess(p_val, 0.05)  # Statistically significant difference

    def test_krippendorff_alpha(self):
        # Perfect agreement
        ratings = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ])
        alpha = krippendorff_alpha(ratings)
        self.assertAlmostEqual(alpha, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
