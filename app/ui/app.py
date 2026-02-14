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
    
    # ── Sidebar ──
    with st.sidebar:
        st.title("📊 AutoData")
        
        uploaded = st.file_uploader(
            "📂 Upload Dataset", 
            type=["csv", "tsv", "xlsx", "xls", "json", "jsonl", "parquet"]
        )
        
        st.markdown("---")
        page = st.radio(
            "📍 Navigation", 
            ["1. Data Setup", "2. Exploratory Analysis", "3. Machine Learning", "4. Reports & Chat"]
        )
        
        st.markdown("---")
        guided = st.toggle(
            "🎨 Guided Mode", value=settings.guided_mode_default,
            help="Simplified interface with smart defaults",
        )

    if uploaded is None:
        st.info("👈 Upload a file in the sidebar to get started.")
        with st.expander("📚 Quick Start", expanded=True):
            st.markdown(
                "1. **Upload** a file (CSV, Excel, JSON)\n"
                "2. **Data Setup**: Clean and prepare your data\n"
                "3. **Explore**: Automated insights and visualization\n"
                "4. **Analysis**: Machine learning and AI chat"
            )
        return

    # ── Load Data ──
    with st.spinner("Loading..."):
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

    # ── Session State Management ──
    # Reset state if file changes
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded.name:
        st.session_state.current_file = uploaded.name
        # Default: raw df is "cleaned" until prep runs
        st.session_state.processed_data = (df, df, None, {}, {}) 

    # ── Page: Data Setup ──
    if page == "1. Data Setup":
        st.header("🛠️ Data Setup")
        # Run prep (UI controls enabled)
        results = render_data_prep(df, settings, guided)
        st.session_state.processed_data = results
        
        cleaned_df, _, _, _, _ = results
        
        st.subheader("👀 Data Preview")
        try:
            st.dataframe(safe_df_for_display(cleaned_df.head()))
        except Exception:
            st.table(safe_df_for_display(cleaned_df.head()))

    # Unpack current state for other pages
    cleaned_df, feat_df, preprocessor, prep_report, prep_cfg = st.session_state.processed_data

    # ── Page: Exploratory Analysis ──
    if page == "2. Exploratory Analysis":
        render_insights(cleaned_df, guided)
        st.markdown("---")
        render_visualization(cleaned_df, settings, guided)

    # ── Page: Machine Learning ──
    if page == "3. Machine Learning":
        # Note: Modeling uses cleaned_df and builds its own pipeline
        render_modeling(cleaned_df, preprocessor, settings, guided, prep_cfg)
        st.markdown("---")
        render_timeseries(cleaned_df)

    # ── Page: Reports & Chat ──
    if page == "4. Reports & Chat":
        # Generate report data on the fly since we are not on the insights page
        from ...analysis.eda import generate_insights as gen_insights_core
        from ...analysis.eda import create_data_quality_report, summary_cards
        
        with st.spinner("Generating report data..."):
            cw_insights = gen_insights_core(cleaned_df)
            cw_quality = create_data_quality_report(cleaned_df)
            cw_cards = summary_cards(cleaned_df)
            
        render_report(cleaned_df, cw_insights, cw_quality, cw_cards, settings, guided)
        st.markdown("---")
        render_chat(cleaned_df, settings)
