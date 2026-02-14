"""Enhanced Exploratory Data Analysis with automated insights."""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from ..core.data_quality import calculate_unified_quality_score


def detect_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Intelligently detect and categorize column types."""
    result: Dict[str, List[str]] = {"numeric": [], "categorical": [], "datetime": [], "text": [], "binary": []}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            result["datetime"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() == 2:
                result["binary"].append(col)
            else:
                result["numeric"].append(col)
        elif pd.api.types.is_categorical_dtype(df[col]) or (df[col].dtype == "object" and df[col].nunique() < 20):
            result["categorical"].append(col)
        else:
            result["text"].append(col)
    return result


def generate_insights(df: pd.DataFrame) -> List[str]:
    """Generate automated insights from the data."""
    insights: List[str] = []
    insights.append(f"Dataset has {df.shape[0]:,} rows and {df.shape[1]} columns.")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        worst = missing.idxmax()
        insights.append(f"Total missing values: {total_missing:,}. Worst column: '{worst}' ({missing[worst]:,} missing, {missing[worst]/len(df)*100:.1f}%).")
    else:
        insights.append("No missing values found — data is complete.")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for col in num_cols[:5]:
                skew = df[col].skew()
                if abs(skew) > 1:
                    direction = "right" if skew > 0 else "left"
                    insights.append(f"'{col}' is highly skewed to the {direction} (skewness={skew:.2f}).")

    if cat_cols:
        for col in cat_cols[:3]:
            nunique = df[col].nunique()
            insights.append(f"'{col}' has {nunique} unique categories.")

    dups = df.duplicated().sum()
    if dups > 0:
        insights.append(f"Found {dups:,} duplicate rows ({dups/len(df)*100:.1f}%).")

    if len(num_cols) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = df[num_cols].corr().abs()
            np.fill_diagonal(corr.values, 0)
            max_corr = corr.max().max()
            if max_corr > 0.8:
                idx = np.unravel_index(corr.values.argmax(), corr.shape)
                c1, c2 = corr.columns[idx[0]], corr.columns[idx[1]]
                insights.append(f"High correlation ({max_corr:.2f}) between '{c1}' and '{c2}'.")

    return insights


def detect_problem_type(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    """Detect ML problem type (classification vs regression)."""
    if target_col is None:
        return "classification"
    y = df[target_col]
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= 20 or y.nunique() / len(y) < 0.05:
            return "classification"
        return "regression"
    return "classification"


def recommend_models(
    df: pd.DataFrame, target_col: str = None, problem_type: str = None,
) -> List[Dict[str, str]]:
    """Enhanced model recommendations based on data analysis."""
    if problem_type is None:
        problem_type = detect_problem_type(df, target_col)

    n_samples, n_features = df.shape
    recs: List[Dict[str, str]] = []

    if problem_type == "classification":
        if n_samples < 1000:
            recs.append({"model": "Logistic Regression", "reason": "Good baseline for small datasets"})
        recs.append({"model": "Random Forest", "reason": "Robust, handles mixed types well"})
        if n_samples > 500:
            recs.append({"model": "XGBoost", "reason": "Strong gradient boosting, great for tabular data"})
            recs.append({"model": "LightGBM", "reason": "Fast gradient boosting, efficient with large datasets"})
        recs.append({"model": "Extra Trees", "reason": "Less overfitting than Random Forest"})
    else:
        if n_samples < 1000:
            recs.append({"model": "Ridge Regression", "reason": "Regularized linear model for small datasets"})
        recs.append({"model": "Random Forest", "reason": "Non-linear, handles complex relationships"})
        recs.append({"model": "XGBoost", "reason": "High accuracy for tabular regression"})
        recs.append({"model": "LightGBM", "reason": "Fast training on large datasets"})

    return recs


def calculate_class_imbalance(y: pd.Series) -> float:
    """Calculate class imbalance ratio (0=balanced, 1=completely imbalanced)."""
    counts = y.value_counts(normalize=True)
    return 1 - counts.min() / counts.max() if len(counts) > 1 else 0.0


def create_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Create a comprehensive data quality report."""
    return calculate_unified_quality_score(df)


def summary_cards(df: pd.DataFrame) -> Dict[str, Any]:
    """Return compact metrics for minimal UI cards."""
    num = len(df.select_dtypes(include=["number"]).columns)
    cat = len(df.columns) - num
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric_cols": num,
        "categorical_cols": cat,
        "missing_cells": int(df.isnull().sum().sum()),
    }
