"""Time-series forecasting section."""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from ...analysis.timeseries import TimeSeriesSpec, make_univariate_ts, arima_forecast, pmdarima_status


def render_timeseries(df: pd.DataFrame):
    """Render the time-series forecasting section."""
    st.subheader("📈 Time Series Forecasting")

    with st.expander("ℹ️ What is Time Series?", expanded=False):
        st.markdown(
            "Predicts future values from historical data. "
            "You need: a **time column** (dates/periods) and a **numeric column** to forecast."
        )

    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    time_like = [c for c in df.columns if any(k in c.lower() for k in ("date", "time", "period", "year", "month"))]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if not dt_cols and not time_like:
        st.warning("⚠️ No time component detected. Upload data with date/time columns for forecasting.")
        return
    if not num_cols:
        st.warning("⚠️ No numeric columns to forecast.")
        return

    available, ver = pmdarima_status()
    if not available:
        st.error("⚠️ `pip install pmdarima` required.")
        return
    st.success(f"✓ Ready (pmdarima v{ver})")

    time_candidates = dt_cols + [c for c in time_like if c not in dt_cols]
    val_default = num_cols[0] if num_cols else None

    c1, c2 = st.columns(2)
    with c1:
        time_col = st.selectbox("📅 Time column", df.columns, index=list(df.columns).index(time_candidates[0]) if time_candidates else 0)
    with c2:
        value_col = st.selectbox("📊 Value column", num_cols, index=num_cols.index(val_default) if val_default else 0)

    horizon = st.slider("🔮 Forecast Horizon (Steps)", 1, 60, 12, help="How many future time periods to predict.")

    if st.button("🚀 Run Forecast", type="primary", use_container_width=True, help="Train a model on past data to predict the future"):
        with st.spinner("Forecasting..."):
            try:
                ts = make_univariate_ts(df[[time_col, value_col]].dropna(), TimeSeriesSpec(time_col, value_col))
                fc, conf = arima_forecast(ts, int(horizon))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines+markers", name="Historical", line=dict(color="royalblue", width=2)))
                fig.add_trace(go.Scatter(x=fc.index, y=fc.values, mode="lines+markers", name="Forecast", line=dict(color="red", width=2, dash="dash")))
                fig.add_trace(go.Scatter(x=conf.index, y=conf["upper"], mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=conf.index, y=conf["lower"], mode="lines", fill="tonexty", line=dict(width=0), name="Likely Range (95%)", fillcolor="rgba(255,0,0,0.2)"))
                fig.update_layout(title=f"Forecast: {value_col}", xaxis_title="Time", yaxis_title=value_col, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                st.success(f"✓ Predicted {horizon} steps ahead.")

                with st.expander("Forecast data"):
                    fdf = pd.DataFrame({"Time": fc.index, "Predicted": fc.values, "Lower": conf["lower"].values, "Upper": conf["upper"].values})
                    st.dataframe(fdf)
                    st.download_button("Download CSV", fdf.to_csv(index=False), f"forecast_{value_col}.csv", "text/csv")
            except ImportError:
                st.error("pmdarima not properly installed.")
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("💡 Try a different column or check your data format.")
