"""Automated report section — markdown report + downloads."""

import json
import numpy as np
import pandas as pd
import streamlit as st
from ...core.preprocessing import detect_column_types


def render_report(df: pd.DataFrame, insights: list, quality: dict, cards: dict, settings, guided: bool):
    """Render the automated report section."""
    st.subheader("Automated Report")

    num_cols, _ = detect_column_types(df)
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
    if num_cols:
        try:
            report_data["correlation"] = df[num_cols].corr(method=settings.corr_method).round(4).to_dict()
        except Exception:
            report_data["correlation"] = {}

    md = _build_markdown(report_data)
    st.text_area("Report (Markdown)", value=md, height=260)

    export = _json_safe({
        "quality": quality, "cards": cards,
        "insights": insights if not guided else insights[:10],
        "report": report_data,
    })
    st.download_button("Download insights.json", json.dumps(export, indent=2), "insights.json", "application/json")
    try:
        st.download_button("Download overall stats.csv", df.describe(include="all").to_csv(), "summary_stats.csv", "text/csv")
    except Exception:
        pass
    st.download_button("Download missing_values_report.csv", df.isnull().sum().reset_index().to_csv(index=False), "missing_values.csv", "text/csv")
    if num_cols:
        st.download_button("Download correlations.csv", df[num_cols].corr(method=settings.corr_method).to_csv(), "correlation.csv", "text/csv")


def _build_markdown(data: dict) -> str:
    lines = [
        "### Dataset Overview",
        f"- Rows: {data['shape']['rows']}", f"- Columns: {data['shape']['columns']}", "",
        "### Column Types", "| Column | Type |\n|---|---|",
    ]
    for c, t in data["dtypes"].items():
        lines.append(f"| {c} | {t} |")
    lines += ["", "### Missing Values", "| Column | Missing |\n|---|---|"]
    for c, m in data["missing"].items():
        lines.append(f"| {c} | {m} |")
    if data.get("correlation"):
        cols = list(data["correlation"].keys())[:8]
        if cols:
            lines += ["", "### Correlation", "| | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols)]
            for r in cols:
                row = [str(data["correlation"].get(r, {}).get(c, "")) for c in cols]
                lines.append(f"| {r} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(i) for i in obj]
    return obj
