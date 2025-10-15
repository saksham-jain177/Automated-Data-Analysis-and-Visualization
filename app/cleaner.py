"""
Agentic Data Preprocessing Agent
Automated, intelligent data cleaning with multiple imputation and outlier handling strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Literal
import streamlit as st
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

ImputationMethod = Literal["mean", "median", "knn", "mode"]
OutlierMethod = Literal["iqr", "zscore", "none"]


class AgenticDataPreprocessor:
    """
    Intelligent data preprocessing agent that automatically cleans and corrects datasets.
    Supports multiple imputation strategies and outlier detection methods.
    """
    
    def __init__(
        self,
        imputation_method: ImputationMethod = "median",
        outlier_method: OutlierMethod = "iqr",
        outlier_threshold: float = 1.5,
        aggressive: bool = False
    ):
        self.imputation_method = imputation_method
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.aggressive = aggressive
        self.quality_report = {}
        self.preprocessing_log = []
        
    def log_action(self, action: str, details: str):
        """Log preprocessing actions for transparency"""
        self.preprocessing_log.append({"action": action, "details": details})
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment
        Returns quality score (0-100) and detailed report
        """
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        # Calculate quality metrics
        completeness = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
        uniqueness = (1 - duplicate_rows / len(df)) * 100 if len(df) > 0 else 0
        
        # Check type consistency
        type_issues = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric data stored as string
                try:
                    pd.to_numeric(df[col].dropna().head(100), errors='raise')
                    type_issues += 1
                except (ValueError, TypeError):
                    pass
        
        consistency = (1 - type_issues / len(df.columns)) * 100 if len(df.columns) > 0 else 0
        
        # Overall quality score
        quality_score = (completeness * 0.5 + uniqueness * 0.3 + consistency * 0.2)
        
        report = {
            "quality_score": round(quality_score, 2),
            "completeness": round(completeness, 2),
            "uniqueness": round(uniqueness, 2),
            "consistency": round(consistency, 2),
            "missing_cells": int(missing_cells),
            "missing_percentage": round(missing_cells / total_cells * 100, 2) if total_cells > 0 else 0,
            "duplicate_rows": int(duplicate_rows),
            "type_issues": type_issues,
            "total_rows": df.shape[0],
            "total_columns": df.shape[1],
        }
        
        self.quality_report = report
        self.log_action("quality_check", f"Quality score: {quality_score:.2f}")
        return report
    
    def handle_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Remove duplicate rows"""
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        
        if removed > 0:
            self.log_action("remove_duplicates", f"Removed {removed} duplicate rows")
        
        return df, removed
    
    def handle_missing_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Intelligent missing data imputation using configured method
        Supports: mean, median, knn, mode
        """
        imputed_counts = {}
        df_clean = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle numeric columns
        if numeric_cols:
            numeric_missing = df_clean[numeric_cols].isnull().sum()
            cols_with_missing = numeric_missing[numeric_missing > 0].index.tolist()
            
            if cols_with_missing:
                if self.imputation_method == "mean":
                    for col in cols_with_missing:
                        mean_val = df_clean[col].mean()
                        df_clean[col].fillna(mean_val, inplace=True)
                        imputed_counts[col] = int(numeric_missing[col])
                    self.log_action("impute_numeric", f"Mean imputation on {len(cols_with_missing)} columns")
                    
                elif self.imputation_method == "median":
                    for col in cols_with_missing:
                        median_val = df_clean[col].median()
                        df_clean[col].fillna(median_val, inplace=True)
                        imputed_counts[col] = int(numeric_missing[col])
                    self.log_action("impute_numeric", f"Median imputation on {len(cols_with_missing)} columns")
                    
                elif self.imputation_method == "knn":
                    try:
                        # KNN imputation requires at least some non-null values
                        if df_clean[cols_with_missing].notna().sum().sum() > 0:
                            imputer = KNNImputer(n_neighbors=5, weights='uniform')
                            df_clean[cols_with_missing] = imputer.fit_transform(df_clean[cols_with_missing])
                            for col in cols_with_missing:
                                imputed_counts[col] = int(numeric_missing[col])
                            self.log_action("impute_numeric", f"KNN imputation on {len(cols_with_missing)} columns")
                        else:
                            # Fallback to median if KNN can't work
                            for col in cols_with_missing:
                                median_val = df_clean[col].median()
                                df_clean[col].fillna(median_val, inplace=True)
                                imputed_counts[col] = int(numeric_missing[col])
                            self.log_action("impute_numeric", f"Fallback to median (KNN failed) on {len(cols_with_missing)} columns")
                    except Exception as e:
                        # Fallback to median if KNN fails
                        for col in cols_with_missing:
                            median_val = df_clean[col].median()
                            df_clean[col].fillna(median_val, inplace=True)
                            imputed_counts[col] = int(numeric_missing[col])
                        self.log_action("impute_numeric", f"Fallback to median (KNN error: {e})")
        
        # Handle categorical columns with mode
        if categorical_cols:
            cat_missing = df_clean[categorical_cols].isnull().sum()
            cols_with_missing = cat_missing[cat_missing > 0].index.tolist()
            
            for col in cols_with_missing:
                if len(df_clean[col].dropna()) > 0:
                    mode_val = df_clean[col].mode()[0]
                    df_clean[col].fillna(mode_val, inplace=True)
                    imputed_counts[col] = int(cat_missing[col])
            
            if cols_with_missing:
                self.log_action("impute_categorical", f"Mode imputation on {len(cols_with_missing)} columns")
        
        return df_clean, imputed_counts
    
    def handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Detect and handle outliers using IQR or Z-score method
        Options: iqr, zscore, none
        """
        if self.outlier_method == "none":
            return df, {}
        
        outlier_counts = {}
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if df_clean[col].nunique() < 2:  # Skip constant columns
                continue
            
            initial_outliers = 0
            
            if self.outlier_method == "iqr":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
                
                outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                initial_outliers = outliers_mask.sum()
                
                if initial_outliers > 0:
                    # Cap outliers instead of removing (preserve data)
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    outlier_counts[col] = int(initial_outliers)
            
            elif self.outlier_method == "zscore":
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                
                if std == 0:
                    continue
                
                z_scores = np.abs((df_clean[col] - mean) / std)
                outliers_mask = z_scores > self.outlier_threshold
                initial_outliers = outliers_mask.sum()
                
                if initial_outliers > 0:
                    # Cap at threshold standard deviations
                    lower_bound = mean - self.outlier_threshold * std
                    upper_bound = mean + self.outlier_threshold * std
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                    outlier_counts[col] = int(initial_outliers)
        
        if outlier_counts:
            self.log_action("handle_outliers", f"{self.outlier_method.upper()} method: capped outliers in {len(outlier_counts)} columns")
        
        return df_clean, outlier_counts
    
    def correct_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Automatically detect and correct column data types
        """
        type_changes = {}
        df_clean = df.copy()
        
        for col in df_clean.columns:
            original_dtype = str(df_clean[col].dtype)
            
            # Try numeric conversion
            if df_clean[col].dtype == 'object':
                try:
                    # Check if most values can be converted to numeric
                    non_null = df_clean[col].dropna()
                    if len(non_null) > 0:
                        converted = pd.to_numeric(non_null, errors='coerce')
                        conversion_success_rate = converted.notna().sum() / len(non_null)
                        
                        if conversion_success_rate > 0.8:  # 80% threshold
                            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                            type_changes[col] = f"{original_dtype} → numeric ({conversion_success_rate:.1%} success)"
                            continue
                except Exception:
                    pass
                
                # Try datetime conversion
                try:
                    non_null = df_clean[col].dropna()
                    if len(non_null) > 0:
                        converted = pd.to_datetime(non_null, errors='coerce', format='mixed')
                        conversion_success_rate = converted.notna().sum() / len(non_null)
                        
                        if conversion_success_rate > 0.8:
                            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce', format='mixed')
                            type_changes[col] = f"{original_dtype} → datetime ({conversion_success_rate:.1%} success)"
                            continue
                except Exception:
                    pass
                
                # Convert to category if low cardinality
                if df_clean[col].nunique() / len(df_clean[col]) < 0.5:
                    df_clean[col] = df_clean[col].astype('category')
                    type_changes[col] = f"{original_dtype} → category"
        
        if type_changes:
            self.log_action("type_correction", f"Corrected types for {len(type_changes)} columns")
        
        return df_clean, type_changes
    
    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Main preprocessing pipeline
        Returns: (cleaned_dataframe, comprehensive_report)
        """
        initial_shape = df.shape
        
        # Step 1: Quality check
        quality_report = self.check_data_quality(df)
        
        # Step 2: Remove duplicates
        df, duplicates_removed = self.handle_duplicates(df)
        
        # Step 3: Type correction (before imputation for better results)
        df, type_changes = self.correct_types(df)
        
        # Step 4: Handle missing data
        df, missing_imputed = self.handle_missing_data(df)
        
        # Step 5: Handle outliers (if enabled)
        outliers_handled = {}
        if self.aggressive or self.outlier_method != "none":
            df, outliers_handled = self.handle_outliers(df)
        
        # Step 6: Final quality check
        final_quality = self.check_data_quality(df)
        
        final_shape = df.shape
        
        # Comprehensive report
        report = {
            "initial_shape": initial_shape,
            "final_shape": final_shape,
            "initial_quality": quality_report,
            "final_quality": final_quality,
            "quality_improvement": round(final_quality["quality_score"] - quality_report["quality_score"], 2),
            "duplicates_removed": duplicates_removed,
            "type_changes": type_changes,
            "missing_imputed": missing_imputed,
            "outliers_handled": outliers_handled,
            "preprocessing_log": self.preprocessing_log,
            "config": {
                "imputation_method": self.imputation_method,
                "outlier_method": self.outlier_method,
                "outlier_threshold": self.outlier_threshold,
                "aggressive": self.aggressive
            }
        }
        
        return df, report


def display_preprocessing_report(report: Dict[str, Any]):
    """Display comprehensive preprocessing report in Streamlit"""
    st.subheader("🧹 Preprocessing Report")
    
    # Quality improvement
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Initial Quality Score",
            f"{report['initial_quality']['quality_score']:.1f}/100",
        )
    with col2:
        st.metric(
            "Final Quality Score",
            f"{report['final_quality']['quality_score']:.1f}/100",
            delta=f"+{report['quality_improvement']:.1f}"
        )
    with col3:
        st.metric(
            "Rows",
            f"{report['final_shape'][0]:,}",
            delta=f"{report['final_shape'][0] - report['initial_shape'][0]:,}" if report['final_shape'][0] != report['initial_shape'][0] else None
        )
    
    # Detailed actions
    with st.expander("📋 Detailed Actions", expanded=True):
        if report['duplicates_removed'] > 0:
            st.info(f"✓ Removed **{report['duplicates_removed']:,}** duplicate rows")
        
        if report['type_changes']:
            st.info(f"✓ Corrected **{len(report['type_changes'])}** column types:")
            for col, change in list(report['type_changes'].items())[:5]:  # Show first 5
                st.text(f"  • {col}: {change}")
            if len(report['type_changes']) > 5:
                st.text(f"  ... and {len(report['type_changes']) - 5} more")
        
        if report['missing_imputed']:
            st.info(f"✓ Imputed missing values in **{len(report['missing_imputed'])}** columns using **{report['config']['imputation_method']}** method:")
            for col, count in list(report['missing_imputed'].items())[:5]:
                st.text(f"  • {col}: {count:,} values")
            if len(report['missing_imputed']) > 5:
                st.text(f"  ... and {len(report['missing_imputed']) - 5} more")
        
        if report['outliers_handled']:
            st.info(f"✓ Handled outliers in **{len(report['outliers_handled'])}** columns using **{report['config']['outlier_method'].upper()}** method:")
            for col, count in list(report['outliers_handled'].items())[:5]:
                st.text(f"  • {col}: {count:,} outliers capped")
            if len(report['outliers_handled']) > 5:
                st.text(f"  ... and {len(report['outliers_handled']) - 5} more")
    
    # Configuration used
    with st.expander("⚙️ Configuration Used"):
        st.json(report['config'])
