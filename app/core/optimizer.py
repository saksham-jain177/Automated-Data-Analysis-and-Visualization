"""Performance optimization utilities."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import streamlit as st
from .data_quality import enhanced_detect_and_handle_outliers


def optimize_dtypes(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Optimize DataFrame dtypes to reduce memory usage."""
    initial_mem = df.memory_usage(deep=True).sum() / 1024**2
    out = df.copy()

    for col in out.select_dtypes(include=["int64", "float64"]).columns:
        c = out[col]
        if c.dtype == "float64":
            if c.isnull().any():
                out[col] = pd.to_numeric(c, downcast="float")
            elif (c == c.astype(int)).all():
                out[col] = pd.to_numeric(c, downcast="integer")
            else:
                out[col] = pd.to_numeric(c, downcast="float")
        else:
            out[col] = pd.to_numeric(c, downcast="integer")

    for col in out.select_dtypes(include=["object"]).columns:
        if out[col].nunique() / len(out[col]) < 0.5:
            out[col] = out[col].astype("category")

    final_mem = out.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        pct = (1 - final_mem / initial_mem) * 100
        st.info(f"Memory optimized: {initial_mem:.2f}MB → {final_mem:.2f}MB ({pct:.1f}% reduction)")
    return out


def smart_sampling(df: pd.DataFrame, max_rows: int = 10_000) -> pd.DataFrame:
    """Intelligently sample data for visualization while preserving distribution."""
    if len(df) <= max_rows:
        return df

    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].nunique() < 20:
            try:
                return (
                    df.groupby(col, group_keys=False)
                    .apply(lambda x: x.sample(min(len(x), max(1, int(max_rows * len(x) / len(df)))), random_state=42))
                    .reset_index(drop=True)
                )
            except Exception:
                pass

    return df.sample(n=max_rows, random_state=42)


def detect_and_handle_outliers(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> Tuple[pd.DataFrame, dict]:
    """Detect and cap outliers using enhanced robust methodology."""
    return enhanced_detect_and_handle_outliers(df, columns, method, threshold)


def auto_feature_selection(df: pd.DataFrame, target_col: str, max_features: int = 20) -> list:
    """Automatically select top features based on importance."""
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression
    from sklearn.preprocessing import LabelEncoder

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_enc = X.copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X_enc[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        score_func = f_regression
    else:
        score_func = f_classif
        if not pd.api.types.is_numeric_dtype(y):
            y = LabelEncoder().fit_transform(y)

    sel = SelectKBest(score_func, k=min(max_features, X_enc.shape[1]))
    sel.fit(X_enc, y)
    return X.columns[sel.get_support()].tolist()
