"""
utils/feature_engineering_utils.py
Scaling, transformation, and feature creation utilities.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA


# ─── Scaling ──────────────────────────────────────────────────────────────────

SCALER_MAP = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
    "RobustScaler": RobustScaler,
}


def apply_scaler(df: pd.DataFrame, cols: list, scaler_name: str) -> tuple[pd.DataFrame, object]:
    """Fit and apply a scaler to selected numeric columns in the interactive workbench."""
    if not cols:
        raise ValueError("Select at least one column to scale.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    bad = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"Scaling requires numeric columns. Non-numeric: {bad}")

    df = df.copy()
    scaler_cls = SCALER_MAP.get(scaler_name)
    if scaler_cls is None:
        raise ValueError(f"Unknown scaler: {scaler_name}")

    scaler = scaler_cls()
    df.loc[:, cols] = scaler.fit_transform(df[cols])

    if not isinstance(st.session_state.get("fitted_scalers"), dict):
        st.session_state["fitted_scalers"] = {}
    st.session_state["fitted_scalers"][scaler_name] = {
        "scaler": scaler,
        "cols": list(cols),
    }
    return df, scaler

def apply_log_transform(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Apply log1p transform to specified columns."""
    df = df.copy()
    for col in cols:
        min_val = df[col].min()
        if min_val <= 0:
            df[col] = np.log1p(df[col] - min_val + 1)
        else:
            df[col] = np.log1p(df[col])
    return df


# ─── Feature Creation ─────────────────────────────────────────────────────────

def create_interaction_feature(df: pd.DataFrame, col1: str, col2: str,
                                operation: str = "multiply") -> pd.DataFrame:
    """
    Create an interaction feature between two numerical columns.
    operation: 'multiply' | 'divide' | 'add' | 'subtract'
    """
    df = df.copy()
    name = f"{col1}__{operation}__{col2}"
    if operation == "multiply":
        df[name] = df[col1] * df[col2]
    elif operation == "divide":
        df[name] = df[col1] / (df[col2].replace(0, np.nan))
    elif operation == "add":
        df[name] = df[col1] + df[col2]
    elif operation == "subtract":
        df[name] = df[col1] - df[col2]
    return df


def create_polynomial_features(df: pd.DataFrame, col: str, degree: int = 2) -> pd.DataFrame:
    """Create polynomial features for a numerical column (degree 2 or 3)."""
    df = df.copy()
    for d in range(2, degree + 1):
        df[f"{col}^{d}"] = df[col] ** d
    return df


def bin_column(df: pd.DataFrame, col: str, n_bins: int = 5,
                labels: list = None) -> pd.DataFrame:
    """Bin a continuous column into n_bins equal-width categories."""
    df = df.copy()
    new_col = f"{col}_binned"
    df[new_col] = pd.cut(df[col], bins=n_bins, labels=labels)
    df[new_col] = df[new_col].astype(str)
    return df


def extract_datetime_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Extract year, month, day, weekday, is_weekend from a datetime column."""
    df = df.copy()
    try:
        dt = pd.to_datetime(df[col])
        df[f"{col}_year"] = dt.dt.year
        df[f"{col}_month"] = dt.dt.month
        df[f"{col}_day"] = dt.dt.day
        df[f"{col}_weekday"] = dt.dt.weekday
        df[f"{col}_is_weekend"] = dt.dt.weekday.isin([5, 6]).astype(int)
        df.drop(columns=[col], inplace=True)
    except Exception:
        pass
    return df


# ─── PCA Visualization ────────────────────────────────────────────────────────

def plot_pca_variance(df: pd.DataFrame, num_cols: list) -> go.Figure:
    """Plot PCA explained variance ratio with safe median filling for visualization."""
    valid_cols = [c for c in num_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_cols) < 2:
        raise ValueError("PCA visualization requires at least two numeric columns.")

    clean = df[valid_cols].replace([np.inf, -np.inf], np.nan).copy()
    # Drop columns that are entirely missing or constant.
    clean = clean.dropna(axis=1, how="all")
    clean = clean.loc[:, clean.nunique(dropna=True) > 1]
    if clean.shape[1] < 2:
        raise ValueError("PCA needs at least two non-constant numeric columns.")
    if len(clean) < 2:
        raise ValueError("PCA needs at least two rows.")

    clean = clean.fillna(clean.median(numeric_only=True))
    from sklearn.preprocessing import StandardScaler as SS
    scaled = SS().fit_transform(clean)
    pca = PCA()
    pca.fit(scaled)

    cumulative = np.cumsum(pca.explained_variance_ratio_) * 100
    individual = pca.explained_variance_ratio_ * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"PC{i+1}" for i in range(len(individual))],
        y=individual,
        name="Individual",
        marker_color="#10B981",
    ))
    fig.add_trace(go.Scatter(
        x=[f"PC{i+1}" for i in range(len(cumulative))],
        y=cumulative,
        name="Cumulative",
        line=dict(color="#0F172A", width=2),
        mode="lines+markers",
    ))
    fig.update_layout(
        title="PCA Explained Variance",
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained (%)",
        template="plotly_white",
        legend=dict(orientation="h"),
    )
    return fig

def get_scaler_recommendation(skewness: float, has_outliers: bool) -> str:
    """Rule-based scaler recommendation."""
    if has_outliers:
        return "RobustScaler"
    if abs(skewness) > 1.0:
        return "Log Transform then StandardScaler"
    return "StandardScaler"
