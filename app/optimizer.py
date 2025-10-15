"""Performance optimization utilities for the application."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import streamlit as st


def optimize_dtypes(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Optimize DataFrame dtypes to reduce memory usage."""
    
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
    df_optimized = df.copy()
    
    # Optimize numeric columns
    for col in df_optimized.select_dtypes(include=['int64', 'float64']).columns:
        col_data = df_optimized[col]
        
        # Skip if has nulls and we're converting to int
        if col_data.dtype == 'float64':
            if col_data.isnull().any():
                # Keep as float but try to downcast
                df_optimized[col] = pd.to_numeric(col_data, downcast='float')
            else:
                # Check if it's actually integers
                if (col_data == col_data.astype(int)).all():
                    df_optimized[col] = pd.to_numeric(col_data, downcast='integer')
                else:
                    df_optimized[col] = pd.to_numeric(col_data, downcast='float')
        else:
            # Integer column
            df_optimized[col] = pd.to_numeric(col_data, downcast='integer')
    
    # Convert string columns with low cardinality to category
    for col in df_optimized.select_dtypes(include=['object']).columns:
        num_unique = df_optimized[col].nunique()
        num_total = len(df_optimized[col])
        
        # Convert to category if cardinality is low
        if num_unique / num_total < 0.5:  # Less than 50% unique values
            df_optimized[col] = df_optimized[col].astype('category')
    
    final_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2  # MB
    reduction_pct = (1 - final_memory/initial_memory) * 100
    
    if verbose:
        st.info(f"Memory optimized: {initial_memory:.2f}MB → {final_memory:.2f}MB ({reduction_pct:.1f}% reduction)")
    
    return df_optimized


def smart_sampling(df: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
    """Intelligently sample data for visualization while preserving distribution."""
    
    if len(df) <= max_rows:
        return df
    
    # Try stratified sampling if there's a categorical column with reasonable cardinality
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if df[col].nunique() < 20:  # Reasonable number of strata
            try:
                # Stratified sampling
                sample = df.groupby(col, group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max(1, int(max_rows * len(x) / len(df)))), random_state=42)
                )
                return sample.reset_index(drop=True)
            except Exception:
                pass
    
    # Fallback to random sampling
    return df.sample(n=max_rows, random_state=42)


def detect_and_handle_outliers(df: pd.DataFrame, columns: Optional[list] = None, method: str = 'iqr') -> Tuple[pd.DataFrame, dict]:
    """Detect and optionally handle outliers in numeric columns."""
    
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    outlier_info = {}
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                outlier_info[col] = {
                    'count': int(outlier_count),
                    'percentage': float(outlier_count / len(df) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
                
                # Cap outliers instead of removing
                df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
        
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers_mask = z_scores > 3
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                outlier_info[col] = {
                    'count': int(outlier_count),
                    'percentage': float(outlier_count / len(df) * 100)
                }
    
    return df_clean, outlier_info


def auto_feature_selection(df: pd.DataFrame, target_col: str, max_features: int = 20) -> list:
    """Automatically select top features based on importance."""
    
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression
    from sklearn.preprocessing import LabelEncoder
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle categorical features
    X_encoded = X.copy()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
    
    # Determine if regression or classification
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        score_func = f_regression
    else:
        score_func = f_classif
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
    
    # Select top features
    selector = SelectKBest(score_func, k=min(max_features, X_encoded.shape[1]))
    selector.fit(X_encoded, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    return selected_features

