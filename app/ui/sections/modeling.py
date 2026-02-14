"""ML modeling section — model selection, evaluation, AutoML."""

import pandas as pd
import streamlit as st
from io import BytesIO
from sklearn.pipeline import Pipeline
from ...ml.modeling import (
    get_default_model, evaluate_with_cv, compute_permutation_importance,
    try_flaml_automl, auto_select_model, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE,
)
from ...core.preprocessing import detect_column_types, build_preprocessor, build_pipeline
from ...analysis.eda import detect_problem_type, recommend_models


def render_modeling(df: pd.DataFrame, preprocessor, settings, guided: bool, prep_cfg: dict):
    """Render the ML modeling section."""
    st.subheader("🤖 Machine Learning Modeling")

    with st.expander("ℹ️ What is Machine Learning?", expanded=False):
        st.markdown(
            "**ML** finds patterns in data to make predictions.\n"
            "1. Select a target column  2. App trains a model  3. View accuracy & feature importance"
        )

    target_col = st.selectbox("🎯 Target column", df.columns)
    problem_type = detect_problem_type(df, target_col)
    recommended = recommend_models(df, target_col, problem_type)
    task_type = "regression" if problem_type == "regression" else "classification"

    if problem_type == "classification":
        st.info(f"📊 **Classification** ({df[target_col].nunique()} categories)")
    else:
        st.info("📈 **Regression** (continuous values)")

    with st.expander("🔍 Recommended models"):
        for rec in recommended[:3]:
            st.write(f"• **{rec['model']}**: {rec['reason']}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Model selection
    if guided:
        with st.spinner("🤖 Selecting best model..."):
            selected_model = auto_select_model(X, y, task_type)
        st.success(f"✓ **{type(selected_model).__name__}** selected via cross-validation")
    else:
        avail = ["Logistic Regression", "Random Forest", "Extra Trees", "AdaBoost", "Gradient Boosting"]
        if XGBOOST_AVAILABLE:
            avail.append("XGBoost")
        if LIGHTGBM_AVAILABLE:
            avail.append("LightGBM")
        model_name = st.selectbox("🧠 Algorithm", avail)
        selected_model = None

    # Build pipeline
    modeling_pre = _build_modeling_preprocessor(X, preprocessor, prep_cfg)
    if guided and selected_model:
        pipeline = build_pipeline(selected_model, modeling_pre)
    else:
        pipeline = build_pipeline(get_default_model(model_name, task_type), modeling_pre)

    # Evaluate & feature importance
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None
    if "feature_importance" not in st.session_state:
        st.session_state.feature_importance = None

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📊 Evaluate Model", use_container_width=True):
            with st.spinner("Testing model accuracy..."):
                st.session_state.eval_results = evaluate_with_cv(pipeline, X, y, settings.cv_folds, settings.random_state)
    with c2:
        if st.button("🔍 Feature Importance", use_container_width=True):
            with st.spinner("Analyzing features..."):
                fitted = pipeline.fit(X, y)
                st.session_state.feature_importance = compute_permutation_importance(fitted, X, y, random_state=settings.random_state)

    if st.session_state.eval_results:
        st.success("✓ Evaluation complete!")
        st.json(st.session_state.eval_results["summary"])
    if st.session_state.feature_importance is not None:
        st.success("✓ Top features:")
        st.dataframe(st.session_state.feature_importance.head(15))

    # AutoML
    _render_automl(X, y, task_type, modeling_pre, settings, guided)


def _build_modeling_preprocessor(X, preprocessor, cfg):
    """Build a preprocessor matched to feature columns."""
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    return build_preprocessor(
        num, cat,
        impute_strategy_num=cfg.get("impute_num", "median"),
        impute_strategy_cat=cfg.get("impute_cat", "most_frequent"),
        scaling=None if cfg.get("scaling") == "none" else cfg.get("scaling", "standard"),
        add_polynomial=cfg.get("add_poly", False),
        poly_degree=cfg.get("poly_degree", 2),
    )


def _render_automl(X, y, task_type, preprocessor, settings, guided):
    st.subheader("🤖 AutoML")
    if guided:
        enable = st.checkbox("🚀 Enable AutoML", False)
        budget = int(settings.automl_time_budget)
    else:
        enable = st.checkbox("🚀 Enable AutoML", settings.automl_enabled)
        budget = st.slider("⏱️ Time budget (s)", 5, 600, int(settings.automl_time_budget))

    if enable and st.button("🔬 Start AutoML Search", type="primary", use_container_width=True):
        with st.spinner(f"🔍 Searching for {budget}s..."):
            model = try_flaml_automl(X, y, budget, task_type, settings.random_state)
        if model is None:
            st.error("⚠️ FLAML not installed. `pip install 'flaml[automl]'`")
        else:
            st.session_state.automl_model = model
            st.success(f"🏆 Best model: **{type(model).__name__}**")
            st.code(str(model), language="python")
            st.balloons()

    if "automl_model" in st.session_state and st.session_state.automl_model:
        st.subheader("🎯 AutoML Results")
        tabs = st.tabs(["📊 Evaluate", "💾 Save", "🔮 Predict"])
        with tabs[0]:
            if st.button("Evaluate AutoML", use_container_width=True):
                pipe = Pipeline([("preprocess", preprocessor), ("model", st.session_state.automl_model)])
                res = evaluate_with_cv(pipe, X, y, settings.cv_folds, settings.random_state)
                st.json(res["summary"])
        with tabs[1]:
            import joblib
            buf = BytesIO()
            joblib.dump(st.session_state.automl_model, buf)
            buf.seek(0)
            st.download_button("⬇️ Download Model (.pkl)", buf, "automl_model.pkl", "application/octet-stream")
        with tabs[2]:
            pred_file = st.file_uploader("Upload prediction CSV", type=["csv"], key="pred_upload")
            if pred_file:
                pred_df = pd.read_csv(pred_file)
                pipe = Pipeline([("preprocess", preprocessor), ("model", st.session_state.automl_model)])
                pipe.fit(X, y)
                pred_df["Prediction"] = pipe.predict(pred_df)
                st.dataframe(pred_df.head(20))
                st.download_button("⬇️ Download", pred_df.to_csv(index=False), "predictions.csv", "text/csv")
