"""Automated insights section."""

import pandas as pd
import streamlit as st
from ..helpers import safe_df_for_display
from ...analysis.eda import (
    generate_insights, create_data_quality_report, summary_cards,
    detect_data_types, detect_problem_type, recommend_models,
)
from ...core.preprocessing import detect_column_types


def render_insights(df: pd.DataFrame, guided: bool):
    """Render the automated insights section."""
    st.subheader("🤖 Automated Insights")
    with st.spinner("Analyzing your data..."):
        insights = generate_insights(df)
        quality = create_data_quality_report(df)
        cards = summary_cards(df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{cards['rows']:,}")
    c2.metric("Columns", f"{cards['columns']}")
    c3.metric("Numeric", f"{cards['numeric_cols']}")
    c4.metric("Categorical", f"{cards['categorical_cols']}")
    c5.metric("Missing", f"{cards['missing_cells']:,}")

    if not guided:
        st.info(f"**Data Quality Score: {quality['quality_score']}%**")
        for i in insights:
            st.write(i)

    dtypes = detect_data_types(df)
    if dtypes["numeric"]:
        st.write(f"**Numeric ({len(dtypes['numeric'])}):** {', '.join(dtypes['numeric'][:5])}")
    if dtypes["categorical"]:
        st.write(f"**Categorical ({len(dtypes['categorical'])}):** {', '.join(dtypes['categorical'][:5])}")
    if dtypes["datetime"]:
        st.write(f"**DateTime ({len(dtypes['datetime'])}):** {', '.join(dtypes['datetime'])}")

    st.subheader("Basic Data Information")
    with st.expander("View detailed statistics", expanded=not guided):
        st.write({"shape": df.shape, "columns": df.columns.tolist()})
        dtypes_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values})
        st.dataframe(safe_df_for_display(dtypes_df))
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["column", "missing"]
        st.dataframe(safe_df_for_display(missing_df))
        st.dataframe(safe_df_for_display(df.describe(include="all").T.reset_index().rename(columns={"index": "column"})))

    return insights, quality, cards
