"""ML sub-package: modeling, evaluation, AutoML."""

from .modeling import (
    evaluate_with_cv,
    compute_permutation_importance,
    get_default_model,
    auto_select_model,
    try_flaml_automl,
    create_ensemble_model,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
)
