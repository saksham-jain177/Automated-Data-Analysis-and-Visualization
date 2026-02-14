"""Shared UI helpers used across all sections."""

import pandas as pd
import streamlit as st


def safe_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy with object columns cast to string for Streamlit display."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str)
    return out


def sample_for_plot(df: pd.DataFrame, max_samples: int = 10_000) -> pd.DataFrame:
    if len(df) > max_samples:
        return df.sample(n=max_samples, random_state=42)
    return df
