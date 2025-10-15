import os
from typing import Optional

import pandas as pd
import streamlit as st
import json

from sklearn.pipeline import Pipeline

from .config import get_settings
from .preprocessing import detect_column_types, build_preprocessor, build_pipeline
from .modeling import get_default_model, evaluate_with_cv, compute_permutation_importance, try_flaml_automl, auto_select_model
from .chat import chat_with_openrouter
from .timeseries import TimeSeriesSpec, make_univariate_ts, arima_forecast, pmdarima_status
from .charts import suggest_charts, render_chart, export_dashboard_html, parse_nl_chart
from .eda import detect_data_types, generate_insights, detect_problem_type, recommend_models, create_data_quality_report, summary_cards
from .data_loader import DataLoader, validate_dataframe
from .optimizer import optimize_dtypes, smart_sampling, detect_and_handle_outliers, auto_feature_selection
from .cleaner import AgenticDataPreprocessor, display_preprocessing_report
from .tutorial import show_tutorial, show_sample_data_selector, show_help_section, show_pro_tips


def load_csv(file) -> pd.DataFrame:
    """Load a CSV into a DataFrame with caching."""

    @st.cache_data(show_spinner=False)
    def _read(file_bytes: bytes) -> pd.DataFrame:
        from io import BytesIO

        return pd.read_csv(BytesIO(file_bytes))

    return _read(file.getvalue())


def load_any(file) -> pd.DataFrame:
    """Load CSV, Excel, JSON, or Parquet based on extension."""
    
    @st.cache_data(show_spinner=False)
    def _load(file_bytes: bytes, filename: str) -> pd.DataFrame:
        from io import BytesIO
        
        name = filename.lower()
        bio = BytesIO(file_bytes)
        
        try:
            if name.endswith(('.csv', '.tsv')):
                return pd.read_csv(bio)
            elif name.endswith(('.xlsx', '.xls')):
                return pd.read_excel(bio)
            elif name.endswith('.jsonl'):
                return pd.read_json(bio, lines=True)
            elif name.endswith('.json'):
                return pd.read_json(bio)
            elif name.endswith('.parquet'):
                return pd.read_parquet(bio)
            else:
                # Fallback to CSV
                return pd.read_csv(bio)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            # Try CSV as last resort
            bio.seek(0)
            return pd.read_csv(bio)
    
    return _load(file.getvalue(), file.name)


def sample_for_plot(df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    """Downsample dataset for plotting if necessary."""

    if len(df) > max_samples:
        return df.sample(n=max_samples, random_state=42)
    return df


def _safe_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with object columns cast to string for safer Streamlit display."""

    safe = df.copy()
    for c in safe.columns:
        if safe[c].dtype == "object":
            safe[c] = safe[c].astype(str)
    return safe


def render_app():
    """Render the Streamlit application UI."""

    settings = get_settings()
    st.title("📊 Automated Data Analysis and Visualization")
    
    # Help section with user guide link
    with st.expander("📚 Need Help? User Guide & Resources", expanded=False):
        st.markdown("""
        ### 🎯 Quick Links
        - 📖 **[Complete User Guide](USER_GUIDE.md)** - Everything you need to know
        - 🎓 **Tutorial Mode** - Click "Show Tutorial" below for step-by-step walkthrough
        - 💡 **Pro Tips** - Check sidebar for helpful hints
        - 🐛 **Having issues?** - See troubleshooting section in User Guide
        
        ### 🚀 Quick Start
        1. Upload your file (CSV, Excel, JSON, Parquet)
        2. Enable "Guided Mode" for simplified interface
        3. Click "Quick Analyze" or explore each section
        4. Use chat assistant to ask questions about your data
        
        ### 🆘 Common Questions
        - **What file types work?** CSV, Excel, JSON, Parquet (max 200MB)
        - **What is Guided Mode?** Simplified interface with smart defaults (recommended for beginners)
        - **How do I forecast?** Use Time Series section with a time/date column
        - **What is AutoML?** Automatically finds the best machine learning model for you
        """)
    
    # Show tutorial for first-time users
    show_tutorial()
    
    # Sidebar with tips and sample data
    show_pro_tips()
    sample_df, sample_name = show_sample_data_selector()
    
    guided = st.toggle(
        "🎨 Guided Mode", 
        value=settings.guided_mode_default, 
        help="✨ Simplified interface for non-technical users. Hides advanced options and shows smart recommendations."
    )

    uploaded_file = st.file_uploader("Upload your input dataset", type=["csv", "tsv", "xlsx", "xls", "json", "jsonl", "parquet"])
    
    # Use sample data if selected
    if sample_df is not None:
        df = sample_df
        st.success(f"Loaded sample dataset: {sample_name}")
    elif uploaded_file is None:
        st.info("Upload a file to get started. Supported: CSV, Excel, JSON, Parquet")
        return
    else:
        # Use optimized data loader
        with st.spinner("Loading and optimizing data..."):
            df = DataLoader.load(uploaded_file.getvalue(), uploaded_file.name)
            
            # Validate loaded data
            is_valid, message = validate_dataframe(df)
            if not is_valid:
                st.error(message)
                return
            
            # Optimize memory usage
            df = optimize_dtypes(df, verbose=False)
        
        st.success(message)
    
    # Agentic data preprocessing
    st.subheader("🤖 Agentic Data Preprocessing")
    
    with st.expander("⚙️ Preprocessing Configuration", expanded=not guided):
        col1, col2 = st.columns(2)
        with col1:
            imputation_method = st.selectbox(
                "Missing Data Strategy",
                options=["median", "mean", "knn", "mode"],
                index=["median", "mean", "knn", "mode"].index(settings.imputation_method),
                help="Method for filling missing values. KNN uses neighboring values, median/mean for numeric, mode for categorical"
            )
        with col2:
            outlier_method = st.selectbox(
                "Outlier Detection",
                options=["iqr", "zscore", "none"],
                index=["iqr", "zscore", "none"].index(settings.outlier_method),
                help="IQR: Interquartile Range (robust), Z-score: Standard deviations, None: skip outlier handling"
            )
        
        if outlier_method != "none":
            outlier_threshold = st.slider(
                "Outlier Threshold",
                min_value=1.0,
                max_value=3.0,
                value=float(settings.outlier_threshold),
                step=0.1,
                help="IQR: multiplier (1.5=standard), Z-score: number of std devs (3.0=standard)"
            )
        else:
            outlier_threshold = settings.outlier_threshold
    
    if guided:
        # In guided mode, use defaults with simple toggle
        imputation_method = settings.imputation_method
        outlier_method = settings.outlier_method
        outlier_threshold = settings.outlier_threshold
    
    clean_data = st.checkbox(
        "🧹 Enable Intelligent Preprocessing", 
        value=True, 
        help="Automatically clean data: remove duplicates, impute missing values, correct types, handle outliers"
    )
    
    if clean_data:
        with st.spinner("Running agentic preprocessing pipeline..."):
            preprocessor = AgenticDataPreprocessor(
                imputation_method=imputation_method,
                outlier_method=outlier_method,
                outlier_threshold=outlier_threshold,
                aggressive=settings.aggressive_cleaning
            )
            df, preprocessing_report = preprocessor.preprocess(df)
            display_preprocessing_report(preprocessing_report)
    st.write("Data Preview:")
    try:
        st.dataframe(_safe_df_for_display(df.head()))
    except Exception:
        st.table(_safe_df_for_display(df.head()))

    # Minimal Quick Analyze (Guided)
    st.subheader("🤖 Automated Insights")
    with st.spinner("Analyzing your data..."):
        insights = generate_insights(df)
        quality_report = create_data_quality_report(df)
        cards = summary_cards(df)

    cols = st.columns(5)
    cols[0].metric("Rows", f"{cards['rows']:,}")
    cols[1].metric("Columns", f"{cards['columns']}")
    cols[2].metric("Numeric", f"{cards['numeric_cols']}")
    cols[3].metric("Categorical", f"{cards['categorical_cols']}")
    cols[4].metric("Missing cells", f"{cards['missing_cells']:,}")

    if not guided:
        st.info(f"**Data Quality Score: {quality_report['quality_score']}%**")
        for insight in insights:
            st.write(insight)
    
    # Show data types breakdown
    data_types = detect_data_types(df)
    if data_types["numeric"]:
        st.write(f"**Numeric columns ({len(data_types['numeric'])}):** {', '.join(data_types['numeric'][:5])}")
    if data_types["categorical"]:
        st.write(f"**Categorical columns ({len(data_types['categorical'])}):** {', '.join(data_types['categorical'][:5])}")
    if data_types["datetime"]:
        st.write(f"**DateTime columns ({len(data_types['datetime'])}):** {', '.join(data_types['datetime'])}")
    
    st.subheader("Basic Data Information")
    with st.expander("View detailed statistics", expanded=not guided):
        st.write({"shape": df.shape, "columns": df.columns.tolist()})
        st.write("Data types:")
        dtypes_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
        st.dataframe(_safe_df_for_display(dtypes_df))
        st.write("Missing values:")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["column", "missing"]
        st.dataframe(_safe_df_for_display(missing_df))
        st.write("Summary Statistics:")
        summary_df = df.describe(include="all").transpose().reset_index().rename(columns={"index": "column"})
        st.dataframe(_safe_df_for_display(summary_df))

    numeric_cols, categorical_cols = detect_column_types(df)

    st.subheader("Preprocessing")
    st.caption("We'll prepare your data by filling missing values, converting text to numbers, and optionally creating extra features.")
    with st.form("preprocess_form"):
        if guided:
            impute_num = "median"
            impute_cat = "most_frequent"
            scaling = "standard"
            add_poly = False
            poly_degree = 2
            st.write("Using defaults: median imputation, one-hot encoding, standard scaling")
        else:
            impute_num = st.selectbox("Numeric imputation", ["mean", "median", "most_frequent"], index=1)
            impute_cat = st.selectbox("Categorical imputation", ["most_frequent", "constant"], index=0)
            scaling = st.selectbox("Scaling", ["none", "standard", "minmax"], index=1)
            add_poly = st.checkbox("Add polynomial features", value=False)
            poly_degree = st.slider("Polynomial degree", 2, 3, 2, disabled=not add_poly)
        submitted = st.form_submit_button("Apply")

    scaling_opt = None if scaling == "none" else scaling
    # Build preprocessor with feature-only columns to avoid mismatches
    # (constructed after target selection below)

    st.subheader("Visualization")
    # Use smart sampling for better visualization performance
    df_plot = smart_sampling(df, max_rows=settings.max_plot_samples)
    if numeric_cols:
        if guided:
            # Auto Dashboard in guided mode (no duplicate histograms)
            specs = suggest_charts(df_plot)
            figs = []
            for spec in specs:
                fig = render_chart(spec, df_plot)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                    figs.append(fig)
            if figs:
                html = export_dashboard_html(figs)
                st.download_button("Download auto dashboard (HTML)", data=html, file_name="dashboard.html", mime="text/html")
        else:
            st.write("Histograms")
            import plotly.express as px
            for col in numeric_cols:
                st.plotly_chart(px.histogram(df_plot, x=col, nbins=30), use_container_width=True)
    if len(numeric_cols) > 1:
        st.write("Correlation heatmap")
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(method=settings.corr_method), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.subheader("🤖 Machine Learning Modeling")
    
    with st.expander("ℹ️ What is Machine Learning?", expanded=False):
        st.write("""
        **Machine Learning** teaches computers to find patterns and make predictions from data.
        
        **Use it when:**
        - You want to predict future outcomes (e.g., customer churn, sales)
        - Classify things into categories (spam/not spam, disease/healthy)
        - Find patterns in complex data
        
        **How it works:**
        1. Select what you want to predict (target column)
        2. The app analyzes patterns automatically
        3. Trains a model and tests its accuracy
        4. Shows you which factors matter most
        
        **Example:** Predict wine quality based on acidity, sugar, alcohol content.
        """)
    
    target_col = st.selectbox(
        "🎯 Target column (what to predict)", 
        df.columns, 
        help="Choose the column you want to predict. The model uses all other columns as inputs."
    )
    
    # Detect problem type and recommend models
    problem_type = detect_problem_type(df, target_col)
    recommended = recommend_models(problem_type, len(df.columns)-1, len(df))
    
    if problem_type == "classification":
        unique_classes = df[target_col].nunique()
        st.info(f"📊 **Problem Type:** Classification (predicting {unique_classes} categories)")
    else:
        st.info(f"📈 **Problem Type:** Regression (predicting continuous numbers)")
    
    with st.expander("🔍 View recommended models"):
        st.write(f"**Top picks for your data:** {', '.join(recommended[:3])}")
        st.caption("These models work well for datasets of your size and problem type.")
    
    # Determine task type for model selection
    is_regression = problem_type == "regression"
    task_type = "regression" if is_regression else "classification"
    
    if guided:
        # Auto-select best model
        model_name = recommended[0] if recommended[0] in ["XGBoost", "LightGBM", "Random Forest"] else "Random Forest"
        st.success(f"✓ Auto-selected: **{model_name}** (best for your data)")
    else:
        available_models = ["Logistic Regression", "Random Forest", "Extra Trees", "AdaBoost", "Gradient Boosting"]
        # Add XGBoost/LightGBM if available
        from .modeling import XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE
        if XGBOOST_AVAILABLE:
            available_models.append("XGBoost")
        if LIGHTGBM_AVAILABLE:
            available_models.append("LightGBM")
        
        model_name = st.selectbox(
            "🧠 Choose algorithm", 
            available_models, 
            help="🌟 XGBoost and Random Forest are great all-around choices. Logistic Regression is fast and simple."
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Build preprocessor with feature-only columns to avoid mismatches
    num_X, cat_X = detect_column_types(X)
    preprocessor = build_preprocessor(
        num_X,
        cat_X,
        impute_strategy_num=impute_num,
        impute_strategy_cat=impute_cat,
        scaling=scaling_opt,
        add_polynomial=add_poly,
        poly_degree=poly_degree,
    )
    pipeline = build_pipeline(get_default_model(model_name, task_type=task_type), preprocessor)

    # Use session state to persist results
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = None
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Evaluate Model Accuracy", use_container_width=True, help="Tests model accuracy using cross-validation (splits data into parts and tests multiple times)"):
            with st.spinner("Testing model accuracy..."):
                res = evaluate_with_cv(pipeline, X, y, cv_folds=settings.cv_folds, random_state=settings.random_state)
                st.session_state.eval_results = res
    
    # Display evaluation results if they exist
    if st.session_state.eval_results is not None:
        st.success("✓ Evaluation complete!")
        st.write("**Accuracy Scores:**")
        st.json(st.session_state.eval_results["summary"])
        
        with st.expander("What do these scores mean?"):
            if task_type == "classification":
                st.write("""
                **Mean Scores** (average across cross-validation folds):
                - **Accuracy**: % of correct predictions (higher is better)
                - **F1 Score**: Balance between precision and recall (0-1, higher is better)
                - **Precision**: % of positive predictions that were correct
                - **Recall**: % of actual positives that were found
                
                **Std (Standard Deviation)** - Consistency of the model:
                - **Low std (< 0.05)**: 🎯 Model is consistent and reliable (GOOD!)
                - **High std (> 0.10)**: ⚠️ Model is unstable, predictions vary a lot
                
                **Your goal:** High mean scores + Low std = Excellent model!
                """)
            else:
                st.write("""
                **Mean Scores** (average across cross-validation folds):
                - **R²**: How well the model explains variance (1.0 is perfect, 0 is no better than guessing)
                    - > 0.90 = Excellent
                    - 0.70-0.90 = Good
                    - < 0.70 = Needs improvement
                - **MAE**: Average error in predictions (lower is better)
                - **RMSE**: Root mean squared error (lower is better, penalizes large errors more)
                
                **Std (Standard Deviation)** - Consistency of the model:
                - **Low std (< 0.05)**: 🎯 Model is consistent and reliable (GOOD!)
                - **High std (> 0.10)**: ⚠️ Model is unstable, predictions vary a lot
                """)

    with col2:
        if st.button("🔍 Show Feature Importance", use_container_width=True, help="Reveals which columns matter most for predictions"):
            with st.spinner("Analyzing feature importance..."):
                fitted = pipeline.fit(X, y)
                importances = compute_permutation_importance(fitted, X, y, random_state=settings.random_state)
                st.session_state.feature_importance = importances
    
    # Display feature importance if it exists
    if st.session_state.feature_importance is not None:
        st.success("✓ Feature importance analysis complete!")
        st.write("**Top important features:**")
        st.dataframe(st.session_state.feature_importance.head(15))
        
        with st.expander("What is feature importance?"):
            st.write("""
            Feature importance shows **which columns have the biggest impact** on predictions.
            
            Higher values = more important for the model's decisions.
            
            Use this to understand what drives your predictions!
            """)

    st.subheader("🤖 AutoML (Let AI Find the Best Model)")
    
    with st.expander("ℹ️ What is AutoML?", expanded=False):
        st.write("""
        **AutoML** (Automated Machine Learning) automatically tests many different models and finds the best one for you.
        
        **Use it when:**
        - You're not sure which model to use
        - You want to maximize accuracy
        - You have time to let the system search
        
        **How it works:**
        - Tests dozens of models automatically
        - Tunes their settings for best performance
        - Returns the champion model
        
        **Time budget:** How long to search (more time = better chance of finding optimal model)
        """)
    
    if guided:
        enable_automl = st.checkbox("🚀 Enable AutoML (searches for best model)", value=False)
        time_budget = int(settings.automl_time_budget)
        if enable_automl:
            st.info(f"⏱️ Will search for {time_budget} seconds")
    else:
        enable_automl = st.checkbox("🚀 Enable AutoML", value=settings.automl_enabled)
        time_budget = st.slider(
            "⏱️ Search time budget (seconds)", 
            min_value=5, 
            max_value=600, 
            value=int(settings.automl_time_budget),
            help="Longer time = more models tested = potentially better accuracy"
        )
    
    if enable_automl and st.button("🔬 Start AutoML Search", type="primary", use_container_width=True):
        with st.spinner(f"🔍 Testing multiple models for {time_budget} seconds..."):
            model = try_flaml_automl(X, y, time_budget=time_budget, task=task_type, random_state=settings.random_state)
        if model is None:
            st.error("⚠️ FLAML not installed. Run: `pip install 'flaml[automl]'` to enable AutoML.")
        else:
            # Store in session state
            st.session_state.automl_model = model
            st.session_state.automl_task_type = task_type  # Store task type too
            
            model_type = "Regressor" if task_type == "regression" else "Classifier"
            st.success(f"🏆 AutoML complete! Best {model_type} found:")
            st.code(str(model), language="python")
            st.balloons()
    
    # Display AutoML model results if available
    if 'automl_model' in st.session_state and st.session_state.automl_model is not None:
        st.subheader("🎯 Use AutoML Model")
        
        tabs = st.tabs(["📊 Evaluate", "🔍 Feature Importance", "💾 Save Model", "🔮 Make Predictions"])
        
        with tabs[0]:
            st.write("**Evaluate the AutoML model:**")
            if st.button("📊 Evaluate AutoML Model", use_container_width=True):
                with st.spinner("Evaluating best model..."):
                    automl_pipeline = Pipeline([("preprocess", preprocessor), ("model", st.session_state.automl_model)])
                    res = evaluate_with_cv(automl_pipeline, X, y, cv_folds=settings.cv_folds, random_state=settings.random_state)
                    st.session_state.automl_eval = res
            
            if 'automl_eval' in st.session_state and st.session_state.automl_eval is not None:
                st.success("✓ AutoML evaluation complete!")
                st.json(st.session_state.automl_eval["summary"])
        
        with tabs[1]:
            st.write("**See what matters most:**")
            if st.button("🔍 Analyze AutoML Importance", use_container_width=True):
                with st.spinner("Computing feature importance..."):
                    automl_pipeline = Pipeline([("preprocess", preprocessor), ("model", st.session_state.automl_model)])
                    automl_pipeline.fit(X, y)
                    importances = compute_permutation_importance(automl_pipeline, X, y, random_state=settings.random_state)
                    st.session_state.automl_importance = importances
            
            if 'automl_importance' in st.session_state and st.session_state.automl_importance is not None:
                st.success("✓ Feature importance computed!")
                st.dataframe(st.session_state.automl_importance.head(15))
        
        with tabs[2]:
            st.write("**Save for later use:**")
            st.code("""
import joblib

# Save the model
joblib.dump(automl_model, 'best_model.pkl')

# Later, load it:
loaded_model = joblib.load('best_model.pkl')
predictions = loaded_model.predict(new_data)
            """, language="python")
            
            # Provide download button
            import joblib
            from io import BytesIO
            buffer = BytesIO()
            joblib.dump(st.session_state.automl_model, buffer)
            buffer.seek(0)
            
            st.download_button(
                label="⬇️ Download Model (.pkl)",
                data=buffer,
                file_name="automl_best_model.pkl",
                mime="application/octet-stream",
                help="Download the trained model to use later"
            )
        
        with tabs[3]:
            st.write("**Make predictions on new data:**")
            st.write("Upload a CSV with the same columns (except target) to get predictions:")
            
            pred_file = st.file_uploader("Upload prediction data", type=["csv"], key="pred_upload")
            
            if pred_file is not None:
                try:
                    pred_df = pd.read_csv(pred_file)
                    
                    # Check columns match (excluding target)
                    expected_cols = set(X.columns)
                    pred_cols = set(pred_df.columns)
                    
                    if expected_cols == pred_cols:
                        # Fit full pipeline and predict
                        full_pipeline = Pipeline([("preprocess", preprocessor), ("model", st.session_state.automl_model)])
                        full_pipeline.fit(X, y)
                        predictions = full_pipeline.predict(pred_df)
                        
                        # Show results
                        result_df = pred_df.copy()
                        result_df[f'Predicted_{target_col}'] = predictions
                        
                        st.success(f"✓ Predictions generated for {len(predictions)} rows!")
                        st.dataframe(result_df.head(20))
                        
                        # Download predictions
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Predictions CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        missing = expected_cols - pred_cols
                        extra = pred_cols - expected_cols
                        st.error(f"Column mismatch!")
                        if missing:
                            st.write(f"Missing columns: {missing}")
                        if extra:
                            st.write(f"Extra columns: {extra}")
                except Exception as e:
                    st.error(f"Error making predictions: {e}")
            else:
                st.info("💡 Or use the trained model in your own code - see 'Save Model' tab")

    st.subheader("📈 Time Series Forecasting")
    
    with st.expander("ℹ️ What is Time Series Forecasting?", expanded=False):
        st.write("""
        **Time Series Forecasting** predicts future values based on historical data over time.
        
        **Use it when:**
        - You have data collected over time (daily, monthly, yearly)
        - You want to predict future trends
        - Examples: sales forecasts, stock prices, temperature predictions
        
        **You need:**
        1. A **time column** (dates, periods, timestamps)
        2. A **numeric column** to forecast (values that change over time)
        
        **The app will:**
        - Automatically detect patterns and trends
        - Predict future values with confidence intervals
        - Show you a visual forecast chart
        """)
    
    # Check if data is suitable for time series
    has_datetime_col = any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)
    time_like_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'period' in c.lower() or 'year' in c.lower() or 'month' in c.lower()]
    numeric_cols_ts = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    has_time_component = has_datetime_col or len(time_like_cols) > 0
    
    if not has_time_component:
        st.warning("""
        ⚠️ **No time component detected in this dataset.**
        
        Time series forecasting requires temporal data (dates, timestamps, or time periods).
        
        **Your dataset appears to be cross-sectional** (snapshot at one point in time), not time-based.
        
        **What you can do instead:**
        - Use **Machine Learning Modeling** above for predictions
        - Upload a dataset with time/date columns if you need forecasting
        
        **Examples of time columns:** `date`, `timestamp`, `period`, `year_month`, `2024-01-15`
        """)
        return
    
    if not numeric_cols_ts:
        st.warning("⚠️ No numeric columns found to forecast. Time series needs numeric values to predict.")
        return
    
    available, pmd_version = pmdarima_status()
    
    if not available:
        st.error("⚠️ Time series requires pmdarima library. Install it with: `pip install pmdarima` then restart Streamlit.")
        return
    
    st.success(f"✓ Ready to forecast (pmdarima v{pmd_version})")
    
    if not has_datetime_col and time_like_cols:
        st.info(f"💡 Detected potential time column(s): {', '.join(time_like_cols[:3])}. The app will attempt to parse them as dates.")
    
    # Auto-detect time column
    time_candidates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or 
                      'date' in c.lower() or 'time' in c.lower() or 'period' in c.lower()]
    default_time = time_candidates[0] if time_candidates else df.columns[0]
    
    # Auto-detect value column
    value_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    default_value = value_candidates[0] if value_candidates else None
    
    col1, col2 = st.columns(2)
    with col1:
        time_col = st.selectbox(
            "📅 Time column (when data was recorded)", 
            df.columns, 
            index=list(df.columns).index(default_time),
            help="Select the column with dates, timestamps, or time periods"
        )
    
    with col2:
        if not value_candidates:
            st.error("No numeric columns found to forecast!")
            return
        value_col = st.selectbox(
            "📊 Value column (what to predict)", 
            value_candidates,
            index=value_candidates.index(default_value) if default_value else 0,
            help="Select the numeric column you want to forecast"
        )
    
    horizon = st.slider(
        "🔮 How many periods ahead to forecast?", 
        min_value=1, 
        max_value=60, 
        value=12,
        help="Number of future time periods to predict (e.g., 12 months ahead)"
    )
    
    if st.button("🚀 Run Forecast", type="primary", use_container_width=True):
        with st.spinner("Analyzing patterns and forecasting..."):
            try:
                ts = make_univariate_ts(df[[time_col, value_col]].dropna(), TimeSeriesSpec(time_col=time_col, value_col=value_col))
                fc, conf = arima_forecast(ts, horizon=int(horizon))
                
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts.index, y=ts.values, 
                    mode="lines+markers", 
                    name="Historical Data",
                    line=dict(color='royalblue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=fc.index, y=fc.values, 
                    mode="lines+markers", 
                    name="Forecast",
                    line=dict(color='red', width=2, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=conf.index, y=conf["upper"], 
                    mode="lines", 
                    line=dict(width=0), 
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=conf.index, y=conf["lower"], 
                    mode="lines", 
                    fill="tonexty", 
                    line=dict(width=0), 
                    name="95% Confidence Interval",
                    fillcolor='rgba(255,0,0,0.2)'
                ))
                fig.update_layout(
                    title=f"Forecast: {value_col} over time",
                    xaxis_title="Time",
                    yaxis_title=value_col,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✓ Forecast complete! Predicted next {horizon} periods.")
                
                # Show forecast table
                with st.expander("View forecast data"):
                    forecast_df = pd.DataFrame({
                        'Time': fc.index,
                        'Predicted Value': fc.values,
                        'Lower Bound (95%)': conf['lower'].values,
                        'Upper Bound (95%)': conf['upper'].values
                    })
                    st.dataframe(forecast_df)
                    st.download_button(
                        "Download forecast CSV",
                        data=forecast_df.to_csv(index=False),
                        file_name=f"forecast_{value_col}.csv",
                        mime="text/csv"
                    )
                
            except ImportError as e:
                st.error("⚠️ pmdarima not properly installed. Run: `pip install pmdarima` and restart.")
            except Exception as exc:
                st.error(f"Forecasting error: {exc}")
                st.info("💡 Tip: Make sure your time column is in date format and values are numeric. Try a different column if the error persists.")

    st.subheader("Automated Report")
    numeric_cols, _ = detect_column_types(df)
    report_data = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": df.isnull().sum().astype(int).to_dict(),
    }
    try:
        report_data["summary_stats"] = df.describe(include="all").fillna(0).to_dict()
    except Exception:
        report_data["summary_stats"] = {}
    if numeric_cols:
        try:
            report_data["correlation"] = df[numeric_cols].corr(method=settings.corr_method).round(4).to_dict()
        except Exception:
            report_data["correlation"] = {}

    md = []
    md.append("### Dataset Overview")
    md.append(f"- Rows: {report_data['shape']['rows']}")
    md.append(f"- Columns: {report_data['shape']['columns']}")
    md.append("")
    md.append("### Column Types")
    md.append("| Column | Type |\n|---|---|")
    for c, t in report_data["dtypes"].items():
        md.append(f"| {c} | {t} |")
    md.append("")
    md.append("### Missing Values")
    md.append("| Column | Missing |\n|---|---|")
    for c, m in report_data["missing"].items():
        md.append(f"| {c} | {m} |")
    if report_data.get("correlation"):
        md.append("")
        md.append("### Correlation (numeric)")
        num_cols_preview = list(report_data["correlation"].keys())[:8]
        if num_cols_preview:
            header = "| | " + " | ".join(num_cols_preview) + " |\n|---|" + "---|" * len(num_cols_preview)
            md.append(header)
            for r in num_cols_preview:
                row = [str(report_data["correlation"].get(r, {}).get(c, "")) for c in num_cols_preview]
                md.append(f"| {r} | " + " | ".join(row) + " |")

    markdown_report = "\n".join(md)
    st.text_area("Report (Markdown)", value=markdown_report, height=260)
    # Include insights JSON for minimal export (convert numpy types to native Python)
    def convert_to_json_serializable(obj):
        """Convert numpy/pandas types to JSON-serializable types."""
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        return obj
    
    insights_export = {
        "quality": convert_to_json_serializable(quality_report),
        "cards": convert_to_json_serializable(cards),
        "insights": insights if not guided else insights[:10],
        "report": convert_to_json_serializable(report_data),
    }
    st.download_button(label="Download insights.json", data=json.dumps(insights_export, indent=2), file_name="insights.json", mime="application/json")
    # CSV exports
    try:
        st.download_button(label="Download summary_stats.csv", data=df.describe(include="all").to_csv(), file_name="summary_stats.csv", mime="text/csv")
    except Exception:
        pass
    st.download_button(label="Download missing_values.csv", data=df.isnull().sum().reset_index().to_csv(index=False), file_name="missing_values.csv", mime="text/csv")
    if numeric_cols:
        st.download_button(label="Download correlation.csv", data=df[numeric_cols].corr(method=settings.corr_method).to_csv(), file_name="correlation.csv", mime="text/csv")

    st.subheader("Chat with your data (OpenRouter)")
    st.caption(f"Model: {settings.openrouter_model} — set via ADV_OPENROUTER_MODEL. Requires ADV_OPENROUTER_API_KEY.")
    user_msg = st.text_input("Your question")
    # NL chart commands
    nl_cmd = st.text_input("Quick chart command (e.g., 'hist Data_value', 'scatter Period vs Data_value')")
    if st.button("Render chart from command") and nl_cmd:
        spec = parse_nl_chart(nl_cmd, df)
        if spec is None:
            st.warning("Could not parse command.")
        else:
            fig = render_chart(spec, df)
            if fig is None:
                st.warning("Unsupported chart type.")
            else:
                st.plotly_chart(fig, use_container_width=True)
    if st.button("Ask") and user_msg:
        with st.spinner("Thinking..."):
            reply = chat_with_openrouter(
                api_key=settings.openrouter_api_key,
                model=settings.openrouter_model,
                user_message=user_msg,
                df_context=df,
                history=st.session_state.get("chat_history"),
            )
        st.session_state.setdefault("chat_history", [])
        st.session_state["chat_history"].append({"role": "user", "content": user_msg})
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
    hist = st.session_state.get("chat_history", [])
    if hist:
        for m in hist[-8:]:
            st.markdown(("**You:** " if m["role"] == "user" else "**Assistant:** ") + m["content"]) 

    # Verify Facts Panel
    with st.expander("Verify chat claims against data", expanded=False):
        st.write("Use this section to sanity-check the chat output. Compare statements with facts below.")
        st.write({"rows": int(df.shape[0]), "columns": int(df.shape[1])})
        dtypes_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
        st.dataframe(_safe_df_for_display(dtypes_df))
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["column", "missing"]
        st.dataframe(_safe_df_for_display(missing_df))


