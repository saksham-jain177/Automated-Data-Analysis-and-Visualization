"""ML modeling: CV evaluation, permutation importance, AutoML, ensembles."""

from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
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

# ── default model registry ──────────────────────────────────────────
DEFAULT_MODELS: Dict[str, ClassifierMixin] = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=None),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}
if LIGHTGBM_AVAILABLE:
    DEFAULT_MODELS["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=200, num_leaves=31, random_state=42, n_jobs=-1, verbosity=-1,
    )


# ── evaluation ───────────────────────────────────────────────────────
def evaluate_with_cv(pipeline, X: pd.DataFrame, y: pd.Series, cv_folds: int, random_state: int) -> Dict[str, Any]:
    """Evaluate a pipeline with cross-validation."""
    is_clf = y.nunique() < 20 and (y.dtype == "object" or not pd.api.types.is_float_dtype(y))
    if is_clf:
        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        except Exception:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = {"accuracy": "accuracy", "f1": "f1_weighted", "precision": "precision_weighted", "recall": "recall_weighted"}
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}

    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=True)
    summary = {f"mean_{k}": float(np.mean(v)) for k, v in scores.items() if k.startswith("test_")}
    summary.update({f"std_{k}": float(np.std(v)) for k, v in scores.items() if k.startswith("test_")})
    return {"scores": scores, "summary": summary}


def compute_permutation_importance(fitted_pipeline, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Compute permutation importance on a fitted pipeline."""
    result = permutation_importance(fitted_pipeline, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    try:
        pre = fitted_pipeline.named_steps["preprocess"]
        names = []
        for name, transformer, cols in pre.transformers_:
            if name == "remainder":
                continue
            try:
                names.extend(transformer.get_feature_names_out(cols))
            except Exception:
                names.extend(cols)
    except Exception:
        names = list(X.columns)
    return pd.DataFrame({"feature": names[: len(result.importances_mean)], "importance": result.importances_mean}).sort_values("importance", ascending=False)


# ── model selection ──────────────────────────────────────────────────
def get_default_model(model_name: str, task_type: str = "classification"):
    """Return a default model by name, appropriate for the task type."""
    if model_name == "XGBoost" and XGBOOST_AVAILABLE:
        return (xgb.XGBRegressor if task_type == "regression" else xgb.XGBClassifier)(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1,
        )
    if model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
        if task_type == "regression":
            return lgb.LGBMRegressor(n_estimators=200, num_leaves=31, random_state=42, n_jobs=-1, verbosity=-1)
        return DEFAULT_MODELS.get(model_name)
    if model_name not in DEFAULT_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return DEFAULT_MODELS[model_name]


def create_ensemble_model(base_models: list, method: str = "voting"):
    """Create an ensemble model from (name, model) tuples."""
    return VotingClassifier(estimators=base_models, voting="soft", n_jobs=-1)


def auto_select_model(X: pd.DataFrame, y: pd.Series, problem_type: str):
    """Automatically select the best model via quick CV comparison."""
    if len(X) < 1000:
        m = try_flaml_automl(X, y, time_budget=30, task=problem_type)
        if m is not None:
            return m
    return _select_best_model_cv(X, y, problem_type)


def _select_best_model_cv(X, y, problem_type):
    candidates = _get_candidate_models(problem_type, X.shape)
    best, best_score = None, float("-inf")
    for name in candidates:
        try:
            model = get_default_model(name, problem_type)
            res = evaluate_with_cv(model, X, y, cv_folds=5, random_state=42)
            score = res["summary"].get("mean_r2" if problem_type == "regression" else "mean_accuracy", float("-inf"))
            if score > best_score:
                best, best_score = model, score
        except Exception:
            continue
    return best if best else _auto_select_model_fallback(X, y, problem_type)


def _get_candidate_models(problem_type: str, shape: tuple) -> list:
    n, p = shape
    if problem_type == "regression":
        return (["ElasticNet"] if p > 100 else []) + ["XGBoost", "Random Forest", "Gradient Boosting", "Extra Trees"]
    if n < 500:
        return ["Logistic Regression", "Random Forest", "Extra Trees"]
    if n < 5000:
        return ["Random Forest", "XGBoost", "Extra Trees", "Gradient Boosting"]
    return ["XGBoost", "Random Forest", "LightGBM", "Extra Trees"]


def _auto_select_model_fallback(X, y, problem_type):
    if len(X) < 1000:
        return get_default_model("Logistic Regression" if problem_type == "binary_classification" else "Random Forest")
    if XGBOOST_AVAILABLE:
        return get_default_model("XGBoost")
    if LIGHTGBM_AVAILABLE:
        return get_default_model("LightGBM")
    return create_ensemble_model([("rf", get_default_model("Random Forest")), ("et", get_default_model("Extra Trees")), ("gb", get_default_model("Gradient Boosting"))])


def try_flaml_automl(X, y, time_budget: int, task: str = "classification", random_state: int = 42):
    """Run FLAML AutoML if available; returns fitted model or None."""
    try:
        from flaml import AutoML
    except Exception:
        return None
    automl = AutoML()
    automl.fit(X_train=X, y_train=y, time_budget=time_budget, metric="r2" if task == "regression" else "accuracy", task=task, log_file_name="flaml.log", seed=random_state)
    return automl.model
