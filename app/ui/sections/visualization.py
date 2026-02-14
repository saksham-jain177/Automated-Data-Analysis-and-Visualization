"""Visualization section."""

import pandas as pd
import plotly.express as px
import streamlit as st
from ...viz.charts import suggest_charts, create_chart, parse_nl_chart
from ...core.optimizer import smart_sampling
from ...core.preprocessing import detect_column_types


def render_visualization(df: pd.DataFrame, settings, guided: bool):
    """Render the visualization section."""
    st.subheader("Visualization")
    num_cols, _ = detect_column_types(df)
    df_plot = smart_sampling(df, max_rows=settings.max_plot_samples)

    if num_cols:
        if guided:
            specs = suggest_charts(df_plot)
            for i, spec in enumerate(specs):
                fig = create_chart(spec, df_plot)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key=f"viz_chart_{i}")
        else:
            st.write("Histograms")
            for col in num_cols:
                st.plotly_chart(px.histogram(df_plot, x=col, nbins=30), use_container_width=True, key=f"hist_{col}")

    if len(num_cols) > 1:
        st.write("Correlation heatmap")
        corr = df[num_cols].corr(method=settings.corr_method)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True, key="viz_heatmap")
