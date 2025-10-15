from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


DEFAULT_MODELS: Dict[str, ClassifierMixin] = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=None),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# XGBoost added dynamically based on task type (see get_default_model)

# Add LightGBM if available  
if LIGHTGBM_AVAILABLE:
    DEFAULT_MODELS["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )


def evaluate_with_cv(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_state: int,
) -> Dict[str, Any]:
    """Evaluate a pipeline with cross-validation and return metrics."""
    
    from sklearn.model_selection import KFold
    
    # Detect if classification or regression
    is_classification = False
    unique_vals = y.nunique()
    if unique_vals < 20 and (y.dtype == 'object' or not pd.api.types.is_float_dtype(y)):
        is_classification = True
    
    # Use appropriate CV strategy
    if is_classification:
        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scoring = {
                "accuracy": "accuracy",
                "f1": "f1_weighted",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
            }
        except Exception:
            # Fallback if stratification fails
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scoring = {
                "accuracy": "accuracy",
                "f1": "f1_weighted",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
            }
    else:
        # Regression
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }
    
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=True)
    summary = {f"mean_{k}": float(np.mean(v)) for k, v in scores.items() if k.startswith("test_")}
    summary.update({f"std_{k}": float(np.std(v)) for k, v in scores.items() if k.startswith("test_")})
    return {"scores": scores, "summary": summary}


def compute_permutation_importance(
    fitted_pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance on a fitted pipeline."""

    result = permutation_importance(
        fitted_pipeline, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )

    try:
        preprocess = fitted_pipeline.named_steps["preprocess"]
        feature_names = []
        for name, transformer, cols in preprocess.transformers_:
            if name == "remainder":
                continue
            try:
                feature_names.extend(transformer.get_feature_names_out(cols))
            except Exception:
                feature_names.extend(cols)
    except Exception:
        feature_names = list(X.columns)

    importances = pd.DataFrame(
        {"feature": feature_names[: len(result.importances_mean)], "importance": result.importances_mean}
    ).sort_values(by="importance", ascending=False)
    return importances


def get_default_model(model_name: str, task_type: str = "classification") -> ClassifierMixin:
    """Return a default model by name, appropriate for the task type."""

    if model_name == "XGBoost" and XGBOOST_AVAILABLE:
        if task_type == "regression":
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        else:
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
    elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
        if task_type == "regression":
            return lgb.LGBMRegressor(
                n_estimators=200,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
        else:
            return DEFAULT_MODELS.get(model_name)
    
    if model_name not in DEFAULT_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return DEFAULT_MODELS[model_name]


def create_ensemble_model(base_models: list, method: str = "voting") -> ClassifierMixin:
    """Create an ensemble model from base models.
    
    Args:
        base_models: List of (name, model) tuples
        method: 'voting', 'bagging', or 'stacking'
    """
    
    if method == "voting":
        return VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)
    else:
        # Default to voting for now
        return VotingClassifier(estimators=base_models, voting='soft', n_jobs=-1)


def auto_select_model(X: pd.DataFrame, y: pd.Series, problem_type: str) -> ClassifierMixin:
    """Automatically select the best model based on data characteristics."""
    
    n_samples, n_features = X.shape
    
    # For small datasets, use simpler models
    if n_samples < 1000:
        if problem_type == "binary_classification":
            return get_default_model("Logistic Regression")
        else:
            return get_default_model("Random Forest")
    
    # For larger datasets, use more powerful models
    if XGBOOST_AVAILABLE:
        return get_default_model("XGBoost")
    elif LIGHTGBM_AVAILABLE:
        return get_default_model("LightGBM")
    else:
        # Create ensemble of available models
        models = [
            ("rf", get_default_model("Random Forest")),
            ("et", get_default_model("Extra Trees")),
            ("gb", get_default_model("Gradient Boosting"))
        ]
        return create_ensemble_model(models, "voting")


def try_flaml_automl(X: pd.DataFrame, y: pd.Series, time_budget: int, task: str = "classification", random_state: int = 42):
    """Optionally run FLAML AutoML and return the fitted estimator, or None if unavailable.
    
    Args:
        X: Feature dataframe
        y: Target series
        time_budget: Time budget in seconds
        task: "classification" or "regression"
        random_state: Random seed
    """

    try:
        from flaml import AutoML
    except Exception:
        return None

    # Set metric based on task
    if task == "regression":
        metric = "r2"
    else:
        metric = "accuracy"
    
    automl = AutoML()
    automl_settings = {
        "time_budget": time_budget,
        "metric": metric,
        "task": task,
        "log_file_name": "flaml.log",
        "seed": random_state,
    }
    automl.fit(X_train=X, y_train=y, **automl_settings)
    return automl.model


