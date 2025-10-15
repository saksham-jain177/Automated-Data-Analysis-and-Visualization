"""Enhanced Exploratory Data Analysis module with automated insights."""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats


def detect_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Intelligently detect and categorize column types."""
    
    types = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "text": [],
        "binary": [],
        "id": [],
    }
    
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types["datetime"].append(col)
        # Check for numeric
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() == 2:
                types["binary"].append(col)
            elif unique_ratio > 0.95:  # Likely ID column
                types["id"].append(col)
            else:
                types["numeric"].append(col)
        # Check for text vs categorical
        elif pd.api.types.is_object_dtype(df[col]):
            avg_length = df[col].astype(str).str.len().mean()
            if unique_ratio > 0.5 or avg_length > 50:
                types["text"].append(col)
            elif df[col].nunique() == 2:
                types["binary"].append(col)
            else:
                types["categorical"].append(col)
    
    return types


def generate_insights(df: pd.DataFrame) -> List[str]:
    """Generate automated insights from the data."""
    
    insights = []
    types = detect_data_types(df)
    
    # Basic stats
    insights.append(f"📊 Dataset contains {len(df):,} rows and {len(df.columns)} columns")
    
    # Missing data insights
    missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
    high_missing = missing_pct[missing_pct > 20]
    if not high_missing.empty:
        insights.append(f"⚠️ High missing data in: {', '.join(high_missing.index[:3])}")
    
    # Numeric insights
    if types["numeric"]:
        for col in types["numeric"][:3]:  # Top 3 numeric columns
            skew = df[col].skew()
            if abs(skew) > 1:
                direction = "right" if skew > 0 else "left"
                insights.append(f"📈 '{col}' is highly skewed to the {direction} (skew={skew:.2f})")
            
            # Outlier detection using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                outlier_pct = (outliers / len(df) * 100)
                insights.append(f"🔍 {outliers} outliers detected in '{col}' ({outlier_pct:.1f}% of data)")
    
    # Correlation insights
    if len(types["numeric"]) >= 2:
        corr_matrix = df[types["numeric"]].corr()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr:
            top_corr = sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[0]
            insights.append(f"🔗 Strong correlation: '{top_corr[0]}' ↔ '{top_corr[1]}' (r={top_corr[2]:.2f})")
    
    # Categorical insights
    if types["categorical"]:
        for col in types["categorical"][:2]:
            n_unique = df[col].nunique()
            mode_val = df[col].mode()[0] if not df[col].mode().empty else None
            if mode_val:
                mode_pct = (df[col] == mode_val).sum() / len(df) * 100
                insights.append(f"📝 '{col}' has {n_unique} categories, most common: '{mode_val}' ({mode_pct:.1f}%)")
    
    # Time series detection
    if types["datetime"]:
        insights.append(f"📅 Time series data detected in: {', '.join(types['datetime'])}")
    
    return insights


def detect_problem_type(df: pd.DataFrame, target_col: Optional[str] = None) -> str:
    """Automatically detect the ML problem type."""
    
    if target_col is None:
        return "unsupervised"
    
    if target_col not in df.columns:
        return "unsupervised"
    
    target = df[target_col]
    n_unique = target.nunique()
    
    # Binary classification
    if n_unique == 2:
        return "binary_classification"
    
    # Multi-class classification (typically < 20 classes and not numeric)
    elif n_unique < 20 and not pd.api.types.is_numeric_dtype(target):
        return "multiclass_classification"
    
    # Regression
    elif pd.api.types.is_numeric_dtype(target):
        # Check if it might be classification despite being numeric
        if n_unique < 20 and all(target.dropna() == target.dropna().astype(int)):
            return "multiclass_classification"
        return "regression"
    
    # Time series
    elif pd.api.types.is_datetime64_any_dtype(target):
        return "time_series"
    
    return "unsupervised"


def recommend_models(problem_type: str, n_features: int, n_samples: int) -> List[str]:
    """Recommend models based on problem type and data characteristics."""
    
    recommendations = []
    
    if problem_type == "binary_classification":
        if n_samples < 1000:
            recommendations = ["LogisticRegression", "RandomForest", "XGBoost"]
        else:
            recommendations = ["XGBoost", "LightGBM", "RandomForest"]
    
    elif problem_type == "multiclass_classification":
        recommendations = ["RandomForest", "XGBoost", "ExtraTrees"]
    
    elif problem_type == "regression":
        if n_features > 100:
            recommendations = ["XGBoost", "LightGBM", "ElasticNet"]
        else:
            recommendations = ["XGBoost", "RandomForest", "GradientBoosting"]
    
    elif problem_type == "time_series":
        recommendations = ["ARIMA", "Prophet", "XGBoost_TS"]
    
    else:  # unsupervised
        recommendations = ["KMeans", "DBSCAN", "IsolationForest"]
    
    return recommendations


def create_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Create a comprehensive data quality report."""
    
    report = {
        "shape": df.shape,
        "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
        "duplicates": df.duplicated().sum(),
        "missing_by_column": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "unique_counts": df.nunique().to_dict(),
    }
    
    # Add quality score (0-100)
    completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    uniqueness = (1 - df.duplicated().sum() / df.shape[0]) * 100
    report["quality_score"] = round((completeness + uniqueness) / 2, 1)
    
    return report


def summary_cards(df: pd.DataFrame) -> Dict[str, Any]:
    """Return compact metrics for minimal UI cards."""

    types = detect_data_types(df)
    missing_total = int(df.isnull().sum().sum())
    out = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_cols": int(len(types["numeric"])),
        "categorical_cols": int(len(types["categorical"])),
        "datetime_cols": int(len(types["datetime"])),
        "missing_cells": missing_total,
    }
    return out
