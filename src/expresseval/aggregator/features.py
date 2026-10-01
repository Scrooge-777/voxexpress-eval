"""
Feature engineering and language-fair z-score normalization.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def z_score_by_language(
    df: pd.DataFrame,
    feature_cols: List[str],
    language_col: str = "language",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Tuple[float, float]]]]:
    """
    Applies per-language z-score normalization: z = (x - mu_lang) / sigma_lang.
    Prevents tonal languages (e.g. Mandarin) from distorting cross-lingual scales.

    Returns:
        Tuple of (normalized_dataframe, normalization_stats_dict)
    """
    df_norm = df.copy()
    stats = {}

    for lang, group in df.groupby(language_col):
        stats[lang] = {}
        for col in feature_cols:
            vals = group[col].values
            mean = float(np.mean(vals))
            std = float(np.std(vals)) + 1e-6
            stats[lang][col] = (mean, std)
            df_norm.loc[group.index, col] = (vals - mean) / std

    return df_norm, stats
