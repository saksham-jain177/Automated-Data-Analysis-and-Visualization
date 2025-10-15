from __future__ import annotations

from typing import List, Dict, Any
import io

import pandas as pd


def suggest_charts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return a small set of recommended chart specs based on column types."""

    specs: List[Dict[str, Any]] = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    if numeric_cols:
        specs.append({"type": "hist", "x": numeric_cols[0], "title": f"Distribution of {numeric_cols[0]}"})
    if len(numeric_cols) >= 2:
        specs.append({"type": "scatter", "x": numeric_cols[0], "y": numeric_cols[1], "title": f"{numeric_cols[0]} vs {numeric_cols[1]}"})
    if categorical_cols and numeric_cols:
        specs.append({"type": "bar", "x": categorical_cols[0], "y": numeric_cols[0], "agg": "mean", "title": f"Average {numeric_cols[0]} by {categorical_cols[0]}"})

    return specs[:3]


def render_chart(spec: Dict[str, Any], df: pd.DataFrame):
    import plotly.express as px
    t = spec.get("type")
    if t == "hist":
        return px.histogram(df, x=spec["x"], nbins=30, title=spec.get("title"))
    if t == "scatter":
        return px.scatter(df, x=spec["x"], y=spec["y"], title=spec.get("title"))
    if t == "bar":
        x, y = spec["x"], spec["y"]
        agg = spec.get("agg", "mean")
        grouped = df.groupby(x)[y].agg(agg).reset_index()
        return px.bar(grouped, x=x, y=y, title=spec.get("title"))
    return None


def export_dashboard_html(figs: List[Any]) -> bytes:
    """Export a simple HTML report with embedded Plotly figures."""

    html_parts = ["<html><head><meta charset='utf-8'><title>Auto Dashboard</title></head><body>"]
    for i, fig in enumerate(figs):
        if fig is None:
            continue
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    html_parts.append("</body></html>")
    return "\n".join(html_parts).encode("utf-8")


def parse_nl_chart(command: str, df: pd.DataFrame) -> Dict[str, Any] | None:
    """Very simple NL parser for demo: 'hist col', 'scatter x vs y', 'bar avg y by x'."""

    cmd = command.lower().strip()
    tokens = cmd.split()
    if not tokens:
        return None
    if tokens[0] in {"hist", "histogram"} and len(tokens) >= 2 and tokens[1] in df.columns:
        return {"type": "hist", "x": tokens[1]}
    if tokens[0] in {"scatter", "plot"} and "vs" in tokens:
        try:
            x = tokens[1]
            y = tokens[tokens.index("vs") + 1]
            if x in df.columns and y in df.columns:
                return {"type": "scatter", "x": x, "y": y}
        except Exception:
            return None
    if tokens[0] == "bar" and "by" in tokens and len(tokens) >= 5:
        # 'bar avg y by x'
        agg = tokens[1] if tokens[1] in {"avg", "mean", "sum", "count", "median"} else "mean"
        y = tokens[2]
        x = tokens[-1]
        if x in df.columns and y in df.columns:
            return {"type": "bar", "x": x, "y": y, "agg": "mean" if agg == "avg" else agg}
    return None


