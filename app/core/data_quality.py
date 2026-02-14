"""Data quality assessment utilities with standardized methodology."""

import pandas as pd
import numpy as np
from typing import Dict, Any


def check_type_consistency_correctly(df: pd.DataFrame) -> float:
    """Check type consistency — returns score as percentage (0-100)."""
    type_issues = 0
    total = len(df.columns)
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                pd.to_numeric(df[col].dropna().head(100), errors="raise")
            except (ValueError, TypeError):
                type_issues += 1
    return (1 - type_issues / total) * 100 if total > 0 else 0


def enhanced_outlier_detection(
    df_col: pd.Series, method: str = "iqr", threshold: float = 1.5
) -> pd.Series:
    """Robust outlier detection with edge-case handling."""
    if method == "iqr":
        q1, q3 = df_col.quantile(0.25), df_col.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return pd.Series([False] * len(df_col), index=df_col.index)
        return (df_col < q1 - threshold * iqr) | (df_col > q3 + threshold * iqr)

    if method == "zscore":
        mean, std = df_col.mean(), df_col.std()
        if std == 0:
            return pd.Series([False] * len(df_col), index=df_col.index)
        return np.abs((df_col - mean) / std) > threshold

    raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")


def calculate_unified_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """Unified data quality scoring (completeness 0.4, uniqueness 0.3, consistency 0.3)."""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())
    completeness = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
    duplicate_rows = int(df.duplicated().sum())
    uniqueness = (1 - duplicate_rows / df.shape[0]) * 100 if df.shape[0] > 0 else 0
    consistency = check_type_consistency_correctly(df)
    quality_score = completeness * 0.4 + uniqueness * 0.3 + consistency * 0.3

    return {
        "quality_score": round(quality_score, 2),
        "completeness": round(completeness, 2),
        "uniqueness": round(uniqueness, 2),
        "consistency": round(consistency, 2),
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "total_cells": int(total_cells),
    }


def enhanced_detect_and_handle_outliers(
    df: pd.DataFrame,
    columns: list = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> tuple:
    """Detect and cap outliers in numeric columns."""
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()

    outlier_info: Dict[str, Any] = {}
    df_clean = df.copy()

    for col in columns:
        mask = enhanced_outlier_detection(df[col], method, threshold)
        count = int(mask.sum())
        if count == 0:
            continue

        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1

        if method == "iqr":
            lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
            outlier_info[col] = {
                "count": count,
                "percentage": float(count / len(df) * 100),
                "lower_bound": float(lo),
                "upper_bound": float(hi),
            }
            df_clean.loc[df_clean[col] < lo, col] = lo
            df_clean.loc[df_clean[col] > hi, col] = hi
        elif method == "zscore":
            outlier_info[col] = {
                "count": count,
                "percentage": float(count / len(df) * 100),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            }

    return df_clean, outlier_info
