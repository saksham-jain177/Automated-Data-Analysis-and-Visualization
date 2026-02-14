"""Main Streamlit application orchestrator — delegates to section modules."""

import warnings
import streamlit as st

# Suppress numpy warnings (e.g. mean of empty slice)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from ..config import get_settings
from ..core.data_loader import DataLoader, validate_dataframe
from ..core.optimizer import optimize_dtypes
from .helpers import safe_df_for_display
from .sections.data_prep import render_data_prep
from .sections.insights import render_insights
from .sections.visualization import render_visualization
from .sections.modeling import render_modeling
from .sections.timeseries import render_timeseries
from .sections.report import render_report
from .sections.chat import render_chat


def render_app():
    """Main entry point for the Streamlit UI."""
    settings = get_settings()
    st.title("📊 Automated Data Analysis & Visualization")

    with st.expander("📚 Quick Start", expanded=False):
        st.markdown(
            "1. Upload a file (CSV, Excel, JSON, Parquet)\n"
            "2. Enable **Guided Mode** for smart defaults\n"
            "3. Explore insights, models, and forecasts\n"
            "4. Chat with your data using the AI assistant"
        )

    guided = st.toggle(
        "🎨 Guided Mode", value=settings.guided_mode_default,
        help="Simplified interface with smart defaults",
    )

    uploaded = st.file_uploader("Upload your dataset", type=["csv", "tsv", "xlsx", "xls", "json", "jsonl", "parquet"])

    if uploaded is None:
        st.info("Upload a file to get started.")
        return

    # Load and optimize
    # Load and optimize
    with st.spinner("Loading and optimizing data..."):
        file_bytes = uploaded.getvalue()
        sheet_name = None

        if uploaded.name.endswith(tuple(DataLoader.SUPPORTED_FORMATS["excel"])):
            sheets = DataLoader.get_sheet_names(file_bytes)
            if len(sheets) > 1:
                sheet_name = st.selectbox("📚 Select Excel sheet", sheets, index=0)

        df = DataLoader.load(file_bytes, uploaded.name, sheet_name=sheet_name)
        is_valid, msg = validate_dataframe(df)
        if not is_valid:
            st.error(msg)
            return
        df = optimize_dtypes(df, verbose=False)
    st.success(msg)

    # ── Section: Data Preparation ──
    cleaned_df, preprocessor, prep_report, prep_cfg = render_data_prep(df, settings, guided)

    # ── Data Preview ──
    st.write("Data Preview:")
    try:
        st.dataframe(safe_df_for_display(cleaned_df.head()))
    except Exception:
        st.table(safe_df_for_display(cleaned_df.head()))

    # ── Section: Automated Insights ──
    insights, quality, cards = render_insights(cleaned_df, guided)

    # ── Section: Visualization ──
    render_visualization(cleaned_df, settings, guided)

    # ── Section: ML Modeling ──
    render_modeling(cleaned_df, preprocessor, settings, guided, prep_cfg)

    # ── Section: Time Series ──
    render_timeseries(cleaned_df)

    # ── Section: Report ──
    render_report(cleaned_df, insights, quality, cards, settings, guided)

    # ── Section: Chat ──
    render_chat(cleaned_df, settings)
