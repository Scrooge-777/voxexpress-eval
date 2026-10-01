"""
Table formatting utilities for system, language, and emotion breakdowns.
"""

from typing import List
import pandas as pd


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates metrics by system_id."""
    metric_cols = [c for c in df.columns if c.endswith("_score")]
    summary = df.groupby("system_id")[metric_cols].mean().round(2)
    return summary.sort_values(by="overall_score", ascending=False) if "overall_score" in summary.columns else summary


def generate_language_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates metrics by language."""
    metric_cols = [c for c in df.columns if c.endswith("_score")]
    return df.groupby("language")[metric_cols].mean().round(2)
