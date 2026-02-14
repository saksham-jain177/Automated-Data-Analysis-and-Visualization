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
                        "Missing Data Strategy", ["median", "mean", "knn", "mode"],
                        index=["median", "mean", "knn", "mode"].index(settings.imputation_method),
                    )
                    handle_duplicates = st.checkbox("Remove Duplicates", True)
                with c2:
                    outlier_method = st.selectbox(
                        "Outlier Detection", ["iqr", "zscore", "none"],
                        index=["iqr", "zscore", "none"].index(settings.outlier_method),
                    )
                    correct_types = st.checkbox("Auto-correct Types", True)
                if outlier_method != "none":
                    outlier_threshold = st.slider("Outlier Threshold", 1.0, 3.0, float(settings.outlier_threshold), 0.1)
            with t2:
                c1, c2 = st.columns(2)
                with c1:
                    impute_num = st.selectbox("Numeric Imputation", ["mean", "median", "most_frequent"], index=1)
                    scaling = st.selectbox("Feature Scaling", ["none", "standard", "minmax"], index=1)
                with c2:
                    impute_cat = st.selectbox("Categorical Imputation", ["most_frequent", "constant"], index=0)
                    add_poly = st.checkbox("Add Polynomial Features", False)
                if add_poly:
                    poly_degree = st.slider("Polynomial Degree", 2, 3, 2)

    enable = st.checkbox("🚀 Enable Data Preparation", True)
    if not enable:
        return df.copy(), None, None, _make_config_dict(impute_num, impute_cat, scaling, add_poly, poly_degree)

    with st.spinner("Running unified data preparation pipeline..."):
        cc = CleaningConfig(
            imputation_method=cleaning_imputation, outlier_method=outlier_method,
            outlier_threshold=outlier_threshold, aggressive=settings.aggressive_cleaning,
            handle_duplicates=handle_duplicates, correct_types=correct_types,
        )
        fc = FeatureConfig(
            impute_strategy_num=impute_num, impute_strategy_cat=impute_cat,
            scaling=scaling, one_hot_drop="if_binary", add_polynomial=add_poly, poly_degree=poly_degree,
        )
        proc = UnifiedDataPreprocessor(cc, fc)
        cleaned_df, c_report = proc.clean_data(df)
        feat_df, preprocessor, f_report = proc.engineer_features(cleaned_df)
        report = {
            "cleaning": c_report, "feature_engineering": f_report,
            "configuration": {"cleaning": vars(cc), "feature_engineering": vars(fc)},
            "processing_log": proc.preprocessing_log,
        }
        display_unified_preprocessing_report(report)

    cfg = _make_config_dict(impute_num, impute_cat, scaling, add_poly, poly_degree)
    return feat_df, preprocessor, report, cfg


def _make_config_dict(impute_num, impute_cat, scaling, add_poly, poly_degree):
    return dict(impute_num=impute_num, impute_cat=impute_cat, scaling=scaling, add_poly=add_poly, poly_degree=poly_degree)
