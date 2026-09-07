"""
components/metrics_card.py
Reusable styled metric card widgets.
"""

import streamlit as st
import plotly.graph_objects as go


def render_metric_row(metrics: dict, best_key: str = None, higher_is_better: bool = True):
    """
    Render a row of metric cards.
    
    Args:
        metrics: {label: value} dict
        best_key: The metric key that determines best (highlighted green).
        higher_is_better: If True, highest value is best; if False, lowest is best.
    """
    cols = st.columns(len(metrics))
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i]:
            if isinstance(value, float):
                fmt_value = f"{value:.4f}"
            else:
                fmt_value = str(value)
            st.metric(label=label, value=fmt_value)


def render_leaderboard(results: list[dict], task_type: str = "classification") -> None:
    """
    Render an interactive leaderboard table with the best model highlighted.
    
    Args:
        results: list of metric dicts (each dict has 'Model' key + metric keys).
        task_type: 'classification' or 'regression'.
    """
    import pandas as pd

    if not results:
        st.info("No models trained yet.")
        return

    df = pd.DataFrame(results)

    if task_type == "classification":
        sort_col = "F1-Score"
        ascending = False
        primary_metric = "F1-Score"
    else:
        sort_col = "RMSE"
        ascending = True
        primary_metric = "RMSE"

    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

    # Style the best model row
    def highlight_best(row):
        styles = [""] * len(row)
        if row["Rank"] == 1:
            styles = ["background-color: rgba(16,185,129,0.15); font-weight: bold; "
                      "border-left: 3px solid #10B981;"] * len(row)
        return styles

    styled = df.style.apply(highlight_best, axis=1)

    # Format numeric columns
    float_cols = df.select_dtypes(include="float").columns.tolist()
    for col in float_cols:
        styled = styled.format({col: "{:.4f}"})

    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Best model banner
    best = df.iloc[0]
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(5,150,105,0.05));
             border: 2px solid #10B981; border-radius: 12px; padding: 16px 20px; margin-top: 12px;">
            <span style="font-size: 1.3rem;">🏆</span>
            <strong style="color: #059669; font-size: 1.1rem;"> Best Model: {best['Model']}</strong>
            <br>
            <span style="color: #374151; font-size: 0.9rem;">
                {primary_metric}: <strong>{best.get(sort_col, 'N/A'):.4f}</strong>
                &nbsp;|&nbsp; Rank #1 out of {len(df)} models
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gauge(value: float, title: str, min_val: float = 0.0, max_val: float = 1.0) -> go.Figure:
    """Render a gauge chart for a single metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": "#10B981"},
            "steps": [
                {"range": [min_val, (max_val - min_val) * 0.5 + min_val], "color": "rgba(239,68,68,0.1)"},
                {"range": [(max_val - min_val) * 0.5 + min_val, (max_val - min_val) * 0.75 + min_val], "color": "rgba(251,191,36,0.1)"},
                {"range": [(max_val - min_val) * 0.75 + min_val, max_val], "color": "rgba(16,185,129,0.1)"},
            ],
            "threshold": {
                "line": {"color": "#0F172A", "width": 3},
                "thickness": 0.75,
                "value": value,
            },
        },
        number={"font": {"size": 28, "color": "#0F172A"}, "valueformat": ".3f"},
    ))
    fig.update_layout(height=200, margin=dict(t=30, b=10, l=20, r=20),
                       template="plotly_white")
    return fig
