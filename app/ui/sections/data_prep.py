"""Data preparation section — cleaning + feature engineering config."""

import pandas as pd
import streamlit as st
from ...core.preprocessing import (
    UnifiedDataPreprocessor, CleaningConfig, FeatureConfig,
    display_unified_preprocessing_report, detect_column_types, build_preprocessor,
)


def render_data_prep(df: pd.DataFrame, settings, guided: bool):
    """Render the data preparation section. Returns (cleaned_df, preprocessor, report, config_dict)."""

    st.subheader("🧹 Data Preparation")
    st.caption("Intelligent data cleaning and feature engineering in a single pipeline")

    # defaults
    cleaning_imputation = settings.imputation_method
    outlier_method = settings.outlier_method
    outlier_threshold = settings.outlier_threshold
    handle_duplicates = True
    correct_types = True
    impute_num = "median"
    impute_cat = "most_frequent"
    scaling = "standard"
    add_poly = False
    poly_degree = 2

    if not guided:
        with st.expander("⚙️ Data Preparation Configuration", expanded=True):
            t1, t2 = st.tabs(["🧽 Cleaning", "⚙️ Features"])
            with t1:
                c1, c2 = st.columns(2)
                with c1:
                    cleaning_imputation = st.selectbox(
                        "How to handle missing values", ["median", "mean", "knn", "mode"],
                        index=["median", "mean", "knn", "mode"].index(settings.imputation_method),
                        help="Choose how to fill in empty cells."
                    )
                    handle_duplicates = st.checkbox("Remove Duplicate Rows", True)
                with c2:
                    outlier_method = st.selectbox(
                        "How to handle outliers (weird values)", ["iqr", "zscore", "none"],
                        index=["iqr", "zscore", "none"].index(settings.outlier_method),
                        help="Detect and remove values that are statistically unusual."
                    )
                    correct_types = st.checkbox("Auto-correct Data Types", True)
                if outlier_method != "none":
                    outlier_threshold = st.slider("Strictness (Threshold)", 1.0, 3.0, float(settings.outlier_threshold), 0.1, help="Lower values remove more data.")
            with t2:
                c1, c2 = st.columns(2)
                with c1:
                    impute_num = st.selectbox("Fill missing numbers with:", ["mean", "median", "most_frequent"], index=1)
                    scaling = st.selectbox("Scale numbers range", ["none", "standard", "minmax"], index=1, help="Standard: centered around 0. MinMax: between 0 and 1.")
                with c2:
                    impute_cat = st.selectbox("Fill missing text with:", ["most_frequent", "constant"], index=0)
                    add_poly = st.checkbox("Create complex features (Polynomial)", False, help="Generate interactions like A*B or A^2 for better model performance.")
                if add_poly:
                    poly_degree = st.slider("Polynomial Degree", 2, 3, 2)

    enable = st.checkbox("🚀 Enable Data Preparation", True)
    if not enable:
        return df.copy(), df.copy(), None, None, _make_config_dict(impute_num, impute_cat, scaling, add_poly, poly_degree)

    with st.spinner("Running unified data preparation pipeline..."):
        # Create config dictionaries (hashable)
        cc_dict = {
            "imputation_method": cleaning_imputation, "outlier_method": outlier_method,
            "outlier_threshold": outlier_threshold, "aggressive": settings.aggressive_cleaning,
            "handle_duplicates": handle_duplicates, "correct_types": correct_types,
        }
        fc_dict = {
            "impute_strategy_num": impute_num, "impute_strategy_cat": impute_cat,
            "scaling": scaling, "one_hot_drop": "if_binary",
            "add_polynomial": add_poly, "poly_degree": poly_degree,
        }
        
        cleaned_df, feat_df, preprocessor, report = _run_preprocessing_cached(df, cc_dict, fc_dict)
        display_unified_preprocessing_report(report)

    cfg = _make_config_dict(impute_num, impute_cat, scaling, add_poly, poly_degree)
    return cleaned_df, feat_df, preprocessor, report, cfg


@st.cache_data(show_spinner=False)
def _run_preprocessing_cached(df: pd.DataFrame, cc_dict: dict, fc_dict: dict):
    """Cached execution of the unified pipeline."""
    cc = CleaningConfig(**cc_dict)
    fc = FeatureConfig(**fc_dict)
    proc = UnifiedDataPreprocessor(cc, fc)
    
    cleaned_df, c_report = proc.clean_data(df)
    feat_df, preprocessor, f_report = proc.engineer_features(cleaned_df)
    
    report = {
        "cleaning": c_report, "feature_engineering": f_report,
        "configuration": {"cleaning": cc_dict, "feature_engineering": fc_dict},
        "processing_log": proc.preprocessing_log,
    }
    return cleaned_df, feat_df, preprocessor, report




def _make_config_dict(impute_num, impute_cat, scaling, add_poly, poly_degree):
    return dict(impute_num=impute_num, impute_cat=impute_cat, scaling=scaling, add_poly=add_poly, poly_degree=poly_degree)
