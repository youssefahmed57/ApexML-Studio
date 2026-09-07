"""
utils/data_loader.py
Handles dataset loading from CSV and Excel files with encoding detection.
"""

import pandas as pd
import chardet
import io
import streamlit as st


@st.cache_data(show_spinner=False)
def load_dataset(file_content: bytes, file_name: str) -> pd.DataFrame:
    """
    Load CSV/Excel bytes robustly.

    CSV loading first detects encoding, then lets pandas sniff the delimiter.
    Clear errors are raised for empty/unsupported files.
    """
    if not file_content:
        raise ValueError("The uploaded file is empty.")

    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if ext in ("xls", "xlsx"):
        df = pd.read_excel(io.BytesIO(file_content))
        if df.empty and len(df.columns) == 0:
            raise ValueError("The Excel sheet contains no tabular data.")
        return df

    if ext not in {"csv", "txt"}:
        raise ValueError("Supported formats are CSV, XLS, and XLSX.")

    detected = chardet.detect(file_content)
    candidates = [
        detected.get("encoding"),
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]
    seen = set()
    last_error = None
    for encoding in candidates:
        if not encoding or encoding.lower() in seen:
            continue
        seen.add(encoding.lower())
        try:
            # sep=None + python engine detects comma/semicolon/tab in common files.
            df = pd.read_csv(
                io.BytesIO(file_content),
                encoding=encoding,
                sep=None,
                engine="python",
            )
            if df.empty and len(df.columns) == 0:
                raise ValueError("The CSV contains no tabular data.")
            return df
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Could not parse CSV file: {last_error}")

def get_dataset_metadata(df: pd.DataFrame) -> dict:
    """Return key dataset metadata, including safe handling for empty frames."""
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    missing_cells = int(df.isnull().sum().sum()) if n_cols else 0
    missing_pct = round(missing_cells / total_cells * 100, 2) if total_cells else 0.0
    duplicates = int(df.duplicated().sum()) if n_rows else 0
    memory_mb = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    return {
        "rows": n_rows,
        "columns": n_cols,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicates": duplicates,
        "memory_mb": memory_mb,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }

def get_column_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a per-column statistics table suitable for display.
    Returns a DataFrame with columns:
    Column, Dtype, Missing, Missing%, Unique, Mean, Std, Min, Max, Skewness
    """
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = int(df[col].isnull().sum())
        missing_pct = round(missing / len(df) * 100, 1) if len(df) else 0.0
        unique = int(df[col].nunique())
        if pd.api.types.is_numeric_dtype(df[col]):
            mean = round(df[col].mean(), 4)
            std = round(df[col].std(), 4)
            mn = round(df[col].min(), 4)
            mx = round(df[col].max(), 4)
            skew = round(df[col].skew(), 4)
        else:
            mean = std = mn = mx = skew = "—"
        rows.append({
            "Column": col,
            "Dtype": dtype,
            "Missing": missing,
            "Missing %": missing_pct,
            "Unique": unique,
            "Mean": mean,
            "Std Dev": std,
            "Min": mn,
            "Max": mx,
            "Skewness": skew,
        })
    return pd.DataFrame(rows)
