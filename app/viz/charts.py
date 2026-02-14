"""
Enhanced chart generation with semantic NL parsing.
Replaces the rigid regex-based parser with fuzzy intent detection.
"""

from typing import Dict, Any, Optional, List
import difflib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ── intent keywords ──────────────────────────────────────────────────
_CHART_INTENTS: Dict[str, List[str]] = {
    "hist":     ["hist", "histogram", "distribution", "dist", "frequency"],
    "scatter":  ["scatter", "plot", "relationship", "relation", "correlate", "compare", "xy"],
    "bar":      ["bar", "barchart", "count", "countplot", "barplot"],
    "box":      ["box", "boxplot", "spread", "range", "quartile"],
    "violin":   ["violin", "violinplot", "density"],
    "line":     ["line", "lineplot", "trend", "timeseries", "time"],
    "heatmap":  ["heatmap", "correlation", "corr", "corrmatrix"],
    "pie":      ["pie", "donut", "proportion", "share", "piechart"],
}


def _match_column(token: str, columns: list[str]) -> Optional[str]:
    """Fuzzy-match a token to the closest column name."""
    token_lower = token.lower().strip()
    # exact match first
    for c in columns:
        if c.lower() == token_lower:
            return c
    # close matches
    matches = difflib.get_close_matches(token_lower, [c.lower() for c in columns], n=1, cutoff=0.6)
    if matches:
        idx = [c.lower() for c in columns].index(matches[0])
        return columns[idx]
    return None


def _detect_intent(tokens: list[str]) -> Optional[str]:
    """Map the first recognized keyword to a chart type."""
    for t in tokens:
        for chart_type, keywords in _CHART_INTENTS.items():
            if t.lower() in keywords:
                return chart_type
    return None


def _extract_columns(tokens: list[str], df: pd.DataFrame) -> list[str]:
    """Extract column references from tokens using fuzzy matching."""
    cols = []
    skip = {"vs", "by", "and", "of", "for", "the", "a", "an", "in", "on", "with", "avg", "sum", "mean", "median", "max", "min"}
    for t in tokens:
        if t.lower() in skip:
            continue
        if _detect_intent([t]):
            continue
        m = _match_column(t, df.columns.tolist())
        if m and m not in cols:
            cols.append(m)
    return cols


# ── public API ───────────────────────────────────────────────────────
def parse_nl_chart(command: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Parse a natural language chart command into a chart spec.

    Returns dict with keys: type, x, y (optional), color (optional)
    or None if the command cannot be parsed.
    """
    tokens = command.strip().split()
    if not tokens:
        return None

    chart_type = _detect_intent(tokens)
    cols = _extract_columns(tokens, df)

    if chart_type is None:
        # fallback: if only column names given, guess chart type
        if len(cols) == 1:
            if pd.api.types.is_numeric_dtype(df[cols[0]]):
                chart_type = "hist"
            else:
                chart_type = "bar"
        elif len(cols) == 2:
            chart_type = "scatter"
        else:
            return None

    # build spec
    spec: Dict[str, Any] = {"type": chart_type}

    if chart_type == "heatmap":
        spec["columns"] = cols if cols else df.select_dtypes(include="number").columns.tolist()
        return spec

    if chart_type == "pie":
        if cols:
            spec["x"] = cols[0]
        else:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            spec["x"] = cat_cols[0] if cat_cols else df.columns[0]
        return spec

    if cols:
        spec["x"] = cols[0]
    if len(cols) >= 2:
        spec["y"] = cols[1]
    if len(cols) >= 3:
        spec["color"] = cols[2]

    return spec


def create_chart(spec: Dict[str, Any], df: pd.DataFrame) -> Optional[go.Figure]:
    """Render a Plotly figure from a chart spec."""
    t = spec.get("type")
    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")

    try:
        if t == "hist":
            return px.histogram(df, x=x, color=color, title=f"Distribution of {x}")
        if t == "scatter":
            return px.scatter(df, x=x, y=y, color=color, title=f"{x} vs {y}")
        if t == "bar":
            if y:
                return px.bar(df, x=x, y=y, color=color, title=f"{y} by {x}")
            return px.histogram(df, x=x, color=color, title=f"Count of {x}")
        if t == "box":
            return px.box(df, x=x if not pd.api.types.is_numeric_dtype(df[x]) else None,
                          y=x if pd.api.types.is_numeric_dtype(df[x]) else (y or x),
                          color=color, title=f"Box plot — {x}")
        if t == "violin":
            return px.violin(df, x=x if not pd.api.types.is_numeric_dtype(df[x]) else None,
                             y=y or x, color=color, title=f"Violin — {y or x}")
        if t == "line":
            return px.line(df, x=x, y=y, color=color, title=f"{y} over {x}")
        if t == "heatmap":
            cols = spec.get("columns", df.select_dtypes(include="number").columns.tolist())
            corr = df[cols].corr()
            return px.imshow(corr, text_auto=".2f", title="Correlation Heatmap", color_continuous_scale="RdBu_r")
        if t == "pie":
            counts = df[x].value_counts().nlargest(15)
            return px.pie(values=counts.values, names=counts.index, title=f"Proportions of {x}")
    except Exception as e:
        st.warning(f"Could not create {t} chart: {e}")
    return None


def suggest_charts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Suggest meaningful chart specs based on the data schema."""
    suggestions: List[Dict[str, Any]] = []
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    # histograms for first 3 numeric columns
    for c in num_cols[:3]:
        suggestions.append({"type": "hist", "x": c, "label": f"Distribution of {c}"})

    # scatter for first numeric pair
    if len(num_cols) >= 2:
        suggestions.append({"type": "scatter", "x": num_cols[0], "y": num_cols[1], "label": f"{num_cols[0]} vs {num_cols[1]}"})

    # bar for first categorical
    if cat_cols:
        suggestions.append({"type": "bar", "x": cat_cols[0], "label": f"Counts by {cat_cols[0]}"})

    # box for categorical × numeric
    if cat_cols and num_cols:
        suggestions.append({"type": "box", "x": cat_cols[0], "y": num_cols[0], "label": f"{num_cols[0]} by {cat_cols[0]}"})

    # line chart for datetime columns
    if dt_cols and num_cols:
        suggestions.append({"type": "line", "x": dt_cols[0], "y": num_cols[0], "label": f"{num_cols[0]} over time"})

    # correlation heatmap
    if len(num_cols) >= 3:
        suggestions.append({"type": "heatmap", "columns": num_cols[:10], "label": "Correlation heatmap"})

    # pie for low-cardinality categorical
    for c in cat_cols:
        if df[c].nunique() <= 10:
            suggestions.append({"type": "pie", "x": c, "label": f"Proportions of {c}"})
            break

    return suggestions
