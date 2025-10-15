"""Interactive tutorial and onboarding system."""

import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer


def show_tutorial():
    """Display interactive tutorial for first-time users."""
    
    if 'tutorial_completed' not in st.session_state:
        st.session_state['tutorial_completed'] = False
    
    if 'tutorial_step' not in st.session_state:
        st.session_state['tutorial_step'] = 0
    
    if not st.session_state['tutorial_completed']:
        with st.expander("📚 Quick Start Tutorial", expanded=True):
            st.write("### Welcome to Automated Data Analysis! 👋")
            st.write("This tool helps you analyze data and build ML models automatically.")
            
            tutorial_steps = [
                {
                    "title": "Step 1: Upload Data",
                    "content": "Click 'Browse files' above to upload your CSV, Excel, JSON, or Parquet file.\n\nOr try a sample dataset below!",
                    "tip": "💡 Tip: The app supports files up to 200MB"
                },
                {
                    "title": "Step 2: Data Cleaning",
                    "content": "The app automatically:\n- Removes duplicates\n- Fills missing values\n- Fixes data types\n- Handles outliers",
                    "tip": "💡 Tip: Review the cleaning report to see what changed"
                },
                {
                    "title": "Step 3: Explore Insights",
                    "content": "View automated insights:\n- Data quality score\n- Column types\n- Statistical patterns\n- Correlations",
                    "tip": "💡 Tip: Use 'Guided mode' for simpler interface"
                },
                {
                    "title": "Step 4: Build Models",
                    "content": "Select your target column and the app will:\n- Recommend best models\n- Train and validate\n- Show performance metrics\n- Explain predictions",
                    "tip": "💡 Tip: XGBoost usually performs best for tabular data"
                },
                {
                    "title": "Step 5: Download Results",
                    "content": "Export:\n- Insights report (JSON)\n- Auto dashboard (HTML)\n- Statistical summaries (CSV)\n- Model predictions",
                    "tip": "💡 Tip: Share the HTML dashboard with your team"
                }
            ]
            
            step = st.session_state['tutorial_step']
            
            if step < len(tutorial_steps):
                current = tutorial_steps[step]
                st.write(f"### {current['title']}")
                st.info(current['content'])
                st.success(current['tip'])
                
                col1, col2, col3 = st.columns([1, 1, 1])
                if step > 0:
                    if col1.button("← Previous"):
                        st.session_state['tutorial_step'] = step - 1
                        st.rerun()
                
                if step < len(tutorial_steps) - 1:
                    if col3.button("Next →"):
                        st.session_state['tutorial_step'] = step + 1
                        st.rerun()
                else:
                    if col3.button("Finish Tutorial"):
                        st.session_state['tutorial_completed'] = True
                        st.rerun()
            
            # Progress indicator
            progress = (step + 1) / len(tutorial_steps)
            st.progress(progress)
            st.caption(f"Step {step + 1} of {len(tutorial_steps)}")


def load_sample_dataset(name: str) -> pd.DataFrame:
    """Load REAL sample datasets from sklearn (NOT fabricated).
    
    These are well-known, peer-reviewed datasets:
    - Iris: Fisher's iris flower dataset (1936)
    - Wine: Wine recognition dataset (UCI ML Repository)
    - Breast Cancer: Wisconsin Diagnostic Breast Cancer (WDBC)
    """
    
    if name == "Iris":
        # Fisher's Iris dataset - 150 samples, 4 features, 3 classes
        # Source: R.A. Fisher (1936) "The use of multiple measurements in taxonomic problems"
        data = load_iris(as_frame=True)
        df = data.frame
        df['target'] = data.target_names[data.target]
        return df
    
    elif name == "Wine":
        # Wine recognition dataset - 178 samples, 13 features, 3 classes
        # Source: UCI Machine Learning Repository
        data = load_wine(as_frame=True)
        df = data.frame
        df['target'] = data.target
        return df
    
    elif name == "Breast Cancer":
        # Wisconsin Diagnostic Breast Cancer - 569 samples, 30 features, 2 classes
        # Source: William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (1995)
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        df['target'] = data.target_names[data.target]
        return df
    
    return None


def show_sample_data_selector():
    """Display sample dataset selector."""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Try Sample Data")
    sample_choice = st.sidebar.selectbox(
        "Load a sample dataset:",
        ["None", "Iris", "Wine", "Breast Cancer"],
        help="Try the app with pre-loaded datasets"
    )
    
    if sample_choice != "None":
        if st.sidebar.button(f"Load {sample_choice} dataset"):
            st.session_state['sample_data'] = load_sample_dataset(sample_choice)
            st.session_state['sample_name'] = sample_choice
            st.rerun()
    
    return st.session_state.get('sample_data', None), st.session_state.get('sample_name', None)


def show_help_section(section: str):
    """Display contextual help for different sections."""
    
    help_content = {
        "preprocessing": {
            "title": "What is Preprocessing?",
            "content": """
            Preprocessing prepares your data for modeling:
            
            - **Imputation**: Fills missing values
            - **Scaling**: Normalizes numeric ranges
            - **Encoding**: Converts text to numbers
            - **Feature Engineering**: Creates new features
            
            The app chooses sensible defaults automatically!
            """,
            "example": "Example: Income ($20K-$200K) → scaled to (0-1)"
        },
        "modeling": {
            "title": "How Does Modeling Work?",
            "content": """
            The app automatically:
            
            1. Splits data (80% train, 20% test)
            2. Trains multiple models
            3. Validates with cross-validation
            4. Shows performance metrics
            5. Explains predictions
            
            You just pick the target column!
            """,
            "example": "Example: Predict house prices based on size, location, etc."
        },
        "evaluation": {
            "title": "Understanding Metrics",
            "content": """
            **For Classification:**
            - Accuracy: % correctly predicted
            - Precision: % of positive predictions that were correct
            - Recall: % of actual positives found
            - F1-score: Balance between precision and recall
            
            **For Regression:**
            - MAE: Average error size
            - RMSE: Penalizes large errors more
            - R²: How well model explains variance (higher is better)
            """,
            "example": "Example: 95% accuracy means 95 out of 100 predictions are correct"
        }
    }
    
    if section in help_content:
        with st.expander(f"ℹ️ {help_content[section]['title']}"):
            st.write(help_content[section]['content'])
            st.info(f"**{help_content[section]['example']}**")


def show_pro_tips():
    """Display pro tips in sidebar."""
    
    with st.sidebar.expander("💡 Pro Tips"):
        st.write("""
        **For Best Results:**
        - Clean your data first
        - Use at least 100 samples
        - Check data quality score > 70%
        - Try multiple models
        - Validate with cross-validation
        
        **Keyboard Shortcuts:**
        - R: Rerun app
        - C: Clear cache
        - S: Open settings
        """)


def show_video_tutorial():
    """Embed video tutorial (placeholder)."""
    
    with st.expander("🎥 Video Tutorial"):
        st.write("Coming soon: Step-by-step video walkthrough")
        # Future: st.video("tutorial.mp4")
