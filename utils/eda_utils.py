"""
utils/eda_utils.py
EDA helper functions: statistics, outlier detection, and Plotly visualizations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats as scipy_stats


# ─── Outlier Detection ────────────────────────────────────────────────────────

def detect_outliers_iqr(series: pd.Series) -> dict:
    """Detect outliers using IQR, safely handling empty/all-missing series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "Q1": np.nan, "Q3": np.nan, "IQR": np.nan,
            "lower_bound": np.nan, "upper_bound": np.nan,
            "outlier_count": 0, "outlier_pct": 0.0,
        }

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = clean[(clean < lower) | (clean > upper)]
    return {
        "Q1": round(float(q1), 4),
        "Q3": round(float(q3), 4),
        "IQR": round(float(iqr), 4),
        "lower_bound": round(float(lower), 4),
        "upper_bound": round(float(upper), 4),
        "outlier_count": int(len(outliers)),
        "outlier_pct": round(len(outliers) / len(clean) * 100, 2),
    }

def detect_all_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Return an outlier summary for numeric columns, including an empty safe frame."""
    columns = [
        "Column", "Q1", "Q3", "IQR", "lower_bound", "upper_bound",
        "outlier_count", "outlier_pct"
    ]
    rows = []
    for col in df.select_dtypes(include="number").columns:
        info = detect_outliers_iqr(df[col])
        info["Column"] = col
        rows.append(info)
    return pd.DataFrame(rows, columns=columns)

# ─── Univariate Plots ─────────────────────────────────────────────────────────

def plot_histogram(df: pd.DataFrame, col: str, color: str = "#10B981") -> go.Figure:
    fig = px.histogram(
        df, x=col, marginal="box", nbins=40,
        color_discrete_sequence=[color],
        title=f"Distribution of {col}",
        template="plotly_white",
    )
    fig.update_layout(bargap=0.05)
    return fig


def plot_box(df: pd.DataFrame, col: str, color: str = "#10B981") -> go.Figure:
    fig = px.box(
        df, y=col, points="outliers",
        color_discrete_sequence=[color],
        title=f"Box Plot — {col}",
        template="plotly_white",
    )
    return fig


def plot_bar_counts(df: pd.DataFrame, col: str, top_n: int = 20) -> go.Figure:
    counts = df[col].value_counts().head(top_n).reset_index()
    counts.columns = [col, "Count"]
    fig = px.bar(
        counts, x=col, y="Count",
        color="Count", color_continuous_scale="Teal",
        title=f"Value Counts — {col} (Top {top_n})",
        template="plotly_white",
    )
    return fig


def plot_pie(df: pd.DataFrame, col: str, top_n: int = 10) -> go.Figure:
    counts = df[col].value_counts().head(top_n)
    fig = px.pie(
        values=counts.values, names=counts.index,
        title=f"Proportion — {col}",
        color_discrete_sequence=px.colors.sequential.Teal,
        template="plotly_white",
    )
    return fig


# ─── Bivariate Plots ──────────────────────────────────────────────────────────

def plot_scatter(df: pd.DataFrame, x: str, y: str, color_col: str = None) -> go.Figure:
    """Scatter plot. Adds an OLS trendline when statsmodels is available."""
    kwargs = dict(
        data_frame=df,
        x=x,
        y=y,
        color=color_col,
        color_discrete_sequence=["#10B981"],
        color_continuous_scale="Teal",
        title=f"{x} vs {y}",
        template="plotly_white",
        opacity=0.7,
    )
    try:
        fig = px.scatter(trendline="ols", **kwargs)
    except Exception:
        fig = px.scatter(**kwargs)
    return fig

def plot_box_grouped(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    fig = px.box(
        df, x=x, y=y, color=x,
        title=f"{y} by {x}",
        template="plotly_white",
    )
    return fig


def plot_violin(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    fig = px.violin(
        df, x=x, y=y, color=x, box=True, points="outliers",
        title=f"{y} distribution by {x}",
        template="plotly_white",
    )
    return fig


# ─── Multivariate Plots ───────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] == 0:
        fig = go.Figure()
        fig.add_annotation(text="No numeric columns available", showarrow=False)
        return fig
    corr = num_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Correlation Heatmap",
        template="plotly_white",
        height=max(400, len(corr) * 35),
    )
    return fig

def plot_pairplot(df: pd.DataFrame, cols: list, color_col: str = None) -> go.Figure:
    subset = df[cols + ([color_col] if color_col else [])].dropna()
    fig = px.scatter_matrix(
        subset,
        dimensions=cols,
        color=color_col,
        color_discrete_sequence=px.colors.qualitative.Prism,
        title="Pair Plot",
        template="plotly_white",
        opacity=0.6,
    )
    fig.update_traces(diagonal_visible=False)
    return fig


def plot_missing_heatmap(df: pd.DataFrame) -> go.Figure:
    """Visual representation of missing values as a heatmap."""
    display_df = df if len(df) <= 5000 else df.sample(5000, random_state=42).sort_index()
    missing_matrix = display_df.isnull().astype(int)
    fig = go.Figure(data=go.Heatmap(
        z=missing_matrix.values.T,
        x=list(range(len(df))),
        y=missing_matrix.columns.tolist(),
        colorscale=[[0, "#10B981"], [1, "#EF4444"]],
        showscale=True,
        colorbar=dict(
            tickvals=[0, 1],
            ticktext=["Present", "Missing"],
        ),
    ))
    fig.update_layout(
        title="Missing Value Map (Red = Missing)",
        xaxis_title="Row Index",
        yaxis_title="Column",
        template="plotly_white",
        height=max(300, len(df.columns) * 20),
    )
    return fig


# ─── Statistical Helpers ──────────────────────────────────────────────────────

def compute_extended_stats(series: pd.Series) -> dict:
    """Return extended numeric statistics, safely handling empty input."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0, "mean": np.nan, "median": np.nan, "std": np.nan,
            "min": np.nan, "max": np.nan, "skewness": np.nan,
            "kurtosis": np.nan, "q1": np.nan, "q3": np.nan,
        }
    return {
        "count": int(len(clean)),
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "std": round(float(clean.std()), 4) if len(clean) > 1 else 0.0,
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "skewness": round(float(clean.skew()), 4) if len(clean) > 2 else np.nan,
        "kurtosis": round(float(clean.kurtosis()), 4) if len(clean) > 3 else np.nan,
        "q1": round(float(clean.quantile(0.25)), 4),
        "q3": round(float(clean.quantile(0.75)), 4),
    }

def get_feature_target_correlation(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Compute correlation of all numerical features with the target column."""
    num_df = df.select_dtypes(include="number")
    if target not in num_df.columns:
        return pd.DataFrame()
    corr = num_df.corr()[target].drop(target).sort_values(key=abs, ascending=False)
    return corr.reset_index().rename(columns={"index": "Feature", target: "Correlation"})
