from typing import List, Tuple, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures


def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return lists of numeric and categorical column names."""

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


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
    """Create a ColumnTransformer for numeric and categorical preprocessing."""

    num_steps = [("imputer", SimpleImputer(strategy=impute_strategy_num))]
    if scaling == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scaling == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))
    if add_polynomial:
        num_steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
    num_pipeline = Pipeline(steps=num_steps)

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=impute_strategy_cat)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop=one_hot_drop)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor


def build_pipeline(estimator, preprocessor: ColumnTransformer) -> Pipeline:
    """Create a modeling pipeline combining preprocessing and estimator."""

    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


