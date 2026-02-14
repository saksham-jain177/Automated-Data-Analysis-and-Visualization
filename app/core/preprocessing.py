"""
Unified Data Preprocessing — single module for cleaning + feature engineering.
Consolidates the former cleaner.py, preprocessing.py, and unified_preprocessing.py.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Literal, List, Optional
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder,
)

from .data_quality import calculate_unified_quality_score

# ── type aliases ─────────────────────────────────────────────────────
ImputationMethod = Literal["mean", "median", "knn", "mode", "most_frequent"]
OutlierMethod = Literal["iqr", "zscore", "none"]
ScalingMethod = Literal["none", "standard", "minmax"]


# ── config dataclasses ───────────────────────────────────────────────
class CleaningConfig:
    """Configuration for data cleaning operations."""

    def __init__(
        self,
        imputation_method: ImputationMethod = "median",
        outlier_method: OutlierMethod = "iqr",
        outlier_threshold: float = 1.5,
        aggressive: bool = False,
        handle_duplicates: bool = True,
        correct_types: bool = True,
    ):
        self.imputation_method = imputation_method
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.aggressive = aggressive
        self.handle_duplicates = handle_duplicates
        self.correct_types = correct_types


class FeatureConfig:
    """Configuration for feature engineering operations."""

    def __init__(
        self,
        impute_strategy_num: ImputationMethod = "median",
        impute_strategy_cat: ImputationMethod = "most_frequent",
        scaling: ScalingMethod = "standard",
        one_hot_drop: str = "if_binary",
        add_polynomial: bool = False,
        poly_degree: int = 2,
    ):
        self.impute_strategy_num = impute_strategy_num
        self.impute_strategy_cat = impute_strategy_cat
        self.scaling = scaling
        self.one_hot_drop = one_hot_drop
        self.add_polynomial = add_polynomial
        self.poly_degree = poly_degree


# ── standalone helpers (formerly in preprocessing.py) ────────────────
def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return lists of (numeric_cols, categorical_cols)."""
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    impute_strategy_num: str = "median",
    impute_strategy_cat: str = "most_frequent",
    scaling: Optional[str] = "standard",
    one_hot_drop: Optional[str] = "if_binary",
    add_polynomial: bool = False,
    poly_degree: int = 2,
) -> ColumnTransformer:
    """Create a ColumnTransformer for numeric + categorical preprocessing."""
    num_steps = [("imputer", SimpleImputer(strategy=impute_strategy_num))]
    if scaling == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scaling == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))
    if add_polynomial:
        num_steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=impute_strategy_cat)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop=one_hot_drop)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_pipeline(estimator, preprocessor: ColumnTransformer) -> Pipeline:
    """Create a modeling pipeline combining preprocessing and estimator."""
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


# ── main preprocessor class ─────────────────────────────────────────
class UnifiedDataPreprocessor:
    """Single intelligent pipeline for both data cleaning and feature preparation."""

    def __init__(self, cleaning_config: CleaningConfig, feature_config: FeatureConfig):
        self.cleaning_config = cleaning_config
        self.feature_config = feature_config
        self.quality_report: Dict[str, Any] = {}
        self.preprocessing_log: List[Dict[str, str]] = []
        self.cleaning_report: Dict[str, Any] = {}
        self.feature_report: Dict[str, Any] = {}

    def log_action(self, step: str, action: str, details: str):
        self.preprocessing_log.append({"step": step, "action": action, "details": details})

    # ── quality check ────────────────────────────────────────────────
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        report = calculate_unified_quality_score(df)
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        report.update({
            "missing_percentage": round(missing_cells / total_cells * 100, 2) if total_cells > 0 else 0,
            "total_rows": df.shape[0],
            "total_columns": df.shape[1],
        })
        type_issues = 0
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    pd.to_numeric(df[col].dropna().head(100), errors="raise")
                except (ValueError, TypeError):
                    type_issues += 1
        report["type_issues"] = type_issues
        return report

    # ── cleaning steps ───────────────────────────────────────────────
    def handle_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        if not self.cleaning_config.handle_duplicates:
            return df, 0
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed > 0:
            self.log_action("cleaning", "remove_duplicates", f"Removed {removed} duplicate rows")
        return df, removed

    def handle_missing_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        imputed: Dict[str, int] = {}
        out = df.copy()
        num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = out.select_dtypes(include=["object", "category"]).columns.tolist()

        # numeric
        if num_cols:
            num_miss = out[num_cols].isnull().sum()
            missing = num_miss[num_miss > 0].index.tolist()
            if missing:
                method = self.cleaning_config.imputation_method
                if method == "knn":
                    try:
                        if out[missing].notna().sum().sum() > 0:
                            out[missing] = KNNImputer(n_neighbors=5).fit_transform(out[missing])
                        else:
                            for c in missing:
                                out[c] = out[c].fillna(out[c].median())
                    except Exception:
                        for c in missing:
                            out[c] = out[c].fillna(out[c].median())
                elif method == "mean":
                    for c in missing:
                        out[c] = out[c].fillna(out[c].mean())
                else:  # median (default)
                    for c in missing:
                        out[c] = out[c].fillna(out[c].median())
                for c in missing:
                    imputed[c] = int(num_miss[c])
                self.log_action("cleaning", "impute_numeric", f"{method} on {len(missing)} cols")

        # categorical
        if cat_cols:
            cat_miss = out[cat_cols].isnull().sum()
            missing = cat_miss[cat_miss > 0].index.tolist()
            for c in missing:
                if len(out[c].dropna()) > 0:
                    out[c] = out[c].fillna(out[c].mode()[0])
                    imputed[c] = int(cat_miss[c])
            if missing:
                self.log_action("cleaning", "impute_categorical", f"mode on {len(missing)} cols")

        return out, imputed

    def handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        if self.cleaning_config.outlier_method == "none":
            return df, {}
        counts: Dict[str, int] = {}
        out = df.copy()
        for col in out.select_dtypes(include=[np.number]).columns:
            if out[col].nunique() < 2:
                continue
            method = self.cleaning_config.outlier_method
            threshold = self.cleaning_config.outlier_threshold
            if method == "iqr":
                q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
                mask = (out[col] < lo) | (out[col] > hi)
            else:  # zscore
                mean, std = out[col].mean(), out[col].std()
                if std == 0:
                    continue
                mask = np.abs((out[col] - mean) / std) > threshold
                lo, hi = mean - threshold * std, mean + threshold * std
            n = int(mask.sum())
            if n > 0:
                out[col] = out[col].clip(lower=lo, upper=hi)
                counts[col] = n
        if counts:
            self.log_action(
                "cleaning", "handle_outliers",
                f"{self.cleaning_config.outlier_method.upper()}: capped in {len(counts)} cols",
            )
        return out, counts

    def correct_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        if not self.cleaning_config.correct_types:
            return df, {}
        changes: Dict[str, str] = {}
        out = df.copy()
        for col in out.columns:
            orig = str(out[col].dtype)
            if out[col].dtype != "object":
                continue
            non_null = out[col].dropna()
            if len(non_null) == 0:
                continue
            # try numeric
            conv = pd.to_numeric(non_null, errors="coerce")
            rate = conv.notna().sum() / len(non_null)
            if rate > 0.8:
                out[col] = pd.to_numeric(out[col], errors="coerce")
                changes[col] = f"{orig} → numeric ({rate:.0%})"
                continue
            # try datetime
            try:
                conv = pd.to_datetime(non_null, errors="coerce", format="mixed")
                rate = conv.notna().sum() / len(non_null)
                if rate > 0.8:
                    out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")
                    changes[col] = f"{orig} → datetime ({rate:.0%})"
                    continue
            except Exception:
                pass
            # category if low cardinality
            if out[col].nunique() / len(out[col]) < 0.5:
                out[col] = out[col].astype("category")
                changes[col] = f"{orig} → category"
        if changes:
            self.log_action("cleaning", "type_correction", f"Corrected {len(changes)} cols")
        return out, changes

    # ── pipelines ────────────────────────────────────────────────────
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        initial_shape = df.shape
        initial_q = self.check_data_quality(df)
        df, dup_n = self.handle_duplicates(df)
        df, type_ch = self.correct_types(df)
        df, miss_imp = self.handle_missing_data(df)
        outlier_h: Dict[str, int] = {}
        if self.cleaning_config.aggressive or self.cleaning_config.outlier_method != "none":
            df, outlier_h = self.handle_outliers(df)
        final_q = self.check_data_quality(df)
        report = {
            "initial_shape": initial_shape,
            "final_shape": df.shape,
            "initial_quality": initial_q,
            "final_quality": final_q,
            "quality_improvement": round(final_q["quality_score"] - initial_q["quality_score"], 2),
            "duplicates_removed": dup_n,
            "type_changes": type_ch,
            "missing_imputed": miss_imp,
            "outliers_handled": outlier_h,
        }
        self.cleaning_report = report
        return df, report

    def engineer_features(
        self, df: pd.DataFrame, target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, ColumnTransformer, Dict[str, Any]]:
        feat_df = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df.copy()
        num_cols, cat_cols = detect_column_types(feat_df)
        preprocessor = build_preprocessor(
            num_cols, cat_cols,
            impute_strategy_num=self.feature_config.impute_strategy_num,
            impute_strategy_cat=self.feature_config.impute_strategy_cat,
            scaling=None if self.feature_config.scaling == "none" else self.feature_config.scaling,
            one_hot_drop=self.feature_config.one_hot_drop,
            add_polynomial=self.feature_config.add_polynomial,
            poly_degree=self.feature_config.poly_degree,
        )
        self.log_action("feature_engineering", "column_detection", f"{len(num_cols)} numeric, {len(cat_cols)} categorical")
        if self.feature_config.scaling != "none":
            self.log_action("feature_engineering", "scaling", f"{self.feature_config.scaling} scaling")
        if self.feature_config.add_polynomial:
            self.log_action("feature_engineering", "polynomial", f"degree {self.feature_config.poly_degree}")
        report = {
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "total_columns": len(num_cols) + len(cat_cols),
            "scaling_method": self.feature_config.scaling,
            "polynomial_features": self.feature_config.add_polynomial,
            "polynomial_degree": self.feature_config.poly_degree if self.feature_config.add_polynomial else None,
        }
        self.feature_report = report
        return feat_df, preprocessor, report

    def preprocess_data(
        self, df: pd.DataFrame, target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, ColumnTransformer, Dict[str, Any]]:
        cleaned, c_report = self.clean_data(df)
        feat_df, preprocessor, f_report = self.engineer_features(cleaned, target_col)
        return cleaned, preprocessor, {
            "cleaning": c_report,
            "feature_engineering": f_report,
            "configuration": {
                "cleaning": vars(self.cleaning_config),
                "feature_engineering": vars(self.feature_config),
            },
            "processing_log": self.preprocessing_log,
        }


# ── Streamlit report display ────────────────────────────────────────
def display_unified_preprocessing_report(report: Dict[str, Any]):
    """Display comprehensive unified preprocessing report in Streamlit."""
    st.subheader("🧹 Data Preparation Report")

    c1, c2, c3, c4 = st.columns(4)
    init_q = report["cleaning"]["initial_quality"]["quality_score"]
    final_q = report["cleaning"]["final_quality"]["quality_score"]
    imp = report["cleaning"]["quality_improvement"]
    c1.metric("Initial Quality", f"{init_q:.1f}/100")
    c2.metric("Final Quality", f"{final_q:.1f}/100", delta=f"+{imp:.1f}")
    c3.metric("Features", f"{report['feature_engineering']['total_columns']}")
    row_d = report["cleaning"]["final_shape"][0] - report["cleaning"]["initial_shape"][0]
    c4.metric("Rows", f"{report['cleaning']['final_shape'][0]:,}", delta=f"{row_d:,}" if row_d else None)

    t1, t2 = st.tabs(["🧽 Cleaning", "⚙️ Features"])
    with t1:
        cl = report["cleaning"]
        if cl["duplicates_removed"] > 0:
            st.success(f"✓ Removed **{cl['duplicates_removed']:,}** duplicate rows")
        if cl["type_changes"]:
            st.success(f"✓ Corrected **{len(cl['type_changes'])}** column types")
            for col, ch in list(cl["type_changes"].items())[:5]:
                st.text(f"  • {col}: {ch}")
        if cl["missing_imputed"]:
            m = report["configuration"]["cleaning"]["imputation_method"]
            st.success(f"✓ Imputed **{len(cl['missing_imputed'])}** columns ({m})")
        if cl["outliers_handled"]:
            m = report["configuration"]["cleaning"]["outlier_method"]
            st.success(f"✓ Capped outliers in **{len(cl['outliers_handled'])}** columns ({m.upper()})")
    with t2:
        fe = report["feature_engineering"]
        st.info(f"📊 {len(fe['numeric_columns'])} numeric, {len(fe['categorical_columns'])} categorical")
        s = fe["scaling_method"]
        if s != "none":
            st.success(f"✓ **{s}** scaling applied")
        if fe["polynomial_features"]:
            st.success(f"✓ Polynomial features (degree {fe['polynomial_degree']})")

    with st.expander("⚙️ Configuration Details"):
        c1, c2 = st.columns(2)
        c1.json(report["configuration"]["cleaning"])
        c2.json(report["configuration"]["feature_engineering"])
