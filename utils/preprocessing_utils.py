"""
utils/preprocessing_utils.py
All data cleaning, imputation, encoding, and outlier treatment functions.
Operations update st.session_state["df"] and log each step.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


# ─── Session State Helpers ────────────────────────────────────────────────────

def _ensure_list_state(key: str):
    """Return a list-valued session-state entry, repairing legacy None/wrong types."""
    value = st.session_state.get(key)
    if not isinstance(value, list):
        value = []
        st.session_state[key] = value
    return value


def _ensure_dict_state(key: str):
    """Return a dict-valued session-state entry, repairing legacy None/wrong types."""
    value = st.session_state.get(key)
    if not isinstance(value, dict):
        value = {}
        st.session_state[key] = value
    return value


def log_step(message: str):
    """Append a preprocessing step to the session state log safely."""
    _ensure_list_state("preprocessing_log").append(str(message))


# ─── Missing Value Handling ───────────────────────────────────────────────────

def impute_column(df: pd.DataFrame, col: str, strategy: str,
                  fill_value=None, knn_neighbors: int = 5) -> pd.DataFrame:
    """
    Impute missing values in a single column.

    Strategies:
        mean | median | mode | constant | knn | drop_rows | drop_col

    This function is used by the interactive cleaning workbench. Final model
    evaluation should still use the leakage-safe training pipeline.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if col not in df.columns:
        raise KeyError(f"Column '{col}' does not exist.")

    df = df.copy()
    n_missing = int(df[col].isna().sum())
    if n_missing == 0:
        log_step(f"No missing values found in '{col}'; nothing changed.")
        return df

    strategy = str(strategy).lower().strip()

    if strategy == "drop_col":
        df = df.drop(columns=[col])
        log_step(f"Dropped column '{col}' ({n_missing} missing values).")
        return df

    if strategy == "drop_rows":
        before = len(df)
        df = df.dropna(subset=[col]).reset_index(drop=True)
        log_step(f"Dropped {before - len(df)} rows with missing '{col}'.")
        return df

    if strategy == "mode":
        mode_val = df[col].mode(dropna=True)
        if mode_val.empty:
            raise ValueError(
                f"Column '{col}' contains no non-missing values, so mode imputation is impossible."
            )
        fill = mode_val.iloc[0]
        df[col] = df[col].fillna(fill)
        log_step(f"Imputed '{col}' with mode ({fill}). {n_missing} values filled.")
        return df

    if strategy == "constant":
        if fill_value is None:
            raise ValueError("A constant fill value is required.")
        # Preserve numeric dtype when possible.
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                fill_value = float(fill_value)
            except Exception as exc:
                raise ValueError(
                    f"Numeric column '{col}' requires a numeric constant."
                ) from exc
        df[col] = df[col].fillna(fill_value)
        log_step(f"Imputed '{col}' with constant ({fill_value}). {n_missing} values filled.")
        return df

    if strategy == "knn":
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError("KNN imputation is available only for numeric columns.")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        usable = [c for c in numeric_cols if df[c].notna().any()]
        if col not in usable:
            raise ValueError(
                f"Column '{col}' is completely missing; KNN has no observed target values to learn from."
            )
        if len(df) < 2:
            raise ValueError("KNN imputation needs at least two rows.")

        n_neighbors = max(1, min(int(knn_neighbors), max(1, len(df) - 1)))
        imputer = KNNImputer(n_neighbors=n_neighbors)
        transformed = imputer.fit_transform(df[usable])

        # IMPORTANT: this workbench is intentionally column-by-column.
        # KNN can use the other numeric columns to find neighbours, but only the
        # selected column is written back. Missing values in every other column
        # remain untouched until the user chooses that column explicitly.
        transformed_df = pd.DataFrame(
            transformed, index=df.index, columns=usable
        )
        selected_missing_mask = df[col].isna()
        df.loc[selected_missing_mask, col] = transformed_df.loc[selected_missing_mask, col]

        filled = n_missing - int(df[col].isna().sum())
        log_step(
            f"Applied column-only KNN imputation (k={n_neighbors}) to '{col}' "
            f"using {len(usable)} numeric feature(s) as neighbour context; "
            f"filled {filled} selected-column value(s)."
        )
        return df

    if strategy in {"mean", "median"}:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{strategy.title()} imputation requires a numeric column.")
        series = pd.to_numeric(df[col], errors="coerce")
        observed = series.dropna()
        if observed.empty:
            raise ValueError(
                f"Column '{col}' contains no observed numeric values, so {strategy} is undefined."
            )
        fill = float(observed.mean() if strategy == "mean" else observed.median())
        df[col] = series.fillna(fill)
        log_step(
            f"Imputed '{col}' with {strategy} ({fill:.4g}). {n_missing} values filled."
        )
        return df

    raise ValueError(f"Unsupported imputation strategy: {strategy}")

# ─── Duplicate Removal ────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    log_step(f"Removed {removed} duplicate rows.")
    return df


# ─── Outlier Treatment ────────────────────────────────────────────────────────

def treat_outliers(df: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
    """
    Treat outliers in a numerical column.
    
    Args:
        method: 'cap' (IQR winsorization) | 'remove' | 'log'
    """
    df = df.copy()
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    if method == "cap":
        before = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        log_step(f"Capped {before} outliers in '{col}' to [{round(lower,2)}, {round(upper,2)}].")

    elif method == "remove":
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)].reset_index(drop=True)
        log_step(f"Removed {before - len(df)} outlier rows in '{col}'.")

    elif method == "log":
        if df[col].min() <= 0:
            df[col] = np.log1p(df[col] - df[col].min() + 1)
        else:
            df[col] = np.log1p(df[col])
        log_step(f"Applied log1p transform to '{col}' to reduce outlier impact.")

    return df


# ─── Encoding ─────────────────────────────────────────────────────────────────

def encode_column(df: pd.DataFrame, col: str, method: str,
                   ordinal_order: list = None) -> pd.DataFrame:
    """
    Encode a categorical column.
    
    Args:
        method: 'onehot' | 'label' | 'ordinal'
        ordinal_order: list of category values in ascending order (required for 'ordinal').
    """
    df = df.copy()
    n_unique = df[col].nunique()

    if method == "onehot":
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        log_step(f"One-Hot Encoded '{col}' → {n_unique} new binary columns.")

    elif method == "label":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        # Store encoder in session state for inference
        _ensure_dict_state("label_encoders")[col] = le
        log_step(f"Label Encoded '{col}' ({n_unique} unique values).")

    elif method == "ordinal":
        # Normalize categories to strings so text input and dataframe values match.
        working = df[col].where(df[col].isna(), df[col].astype(str))
        if ordinal_order is None:
            ordinal_order = sorted(working.dropna().unique().tolist())
        ordinal_order = [str(v) for v in ordinal_order]
        enc = OrdinalEncoder(
            categories=[ordinal_order],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        df[col] = enc.fit_transform(working.to_frame())
        _ensure_dict_state("ordinal_encoders")[col] = enc
        log_step(f"Ordinal Encoded '{col}' with order: {ordinal_order}.")

    return df


def get_dtype_recommendation(series: pd.Series) -> str:
    """
    Heuristic to suggest encoding method based on series properties.
    Returns: 'onehot' | 'label' | 'ordinal' | 'skip'
    """
    if pd.api.types.is_numeric_dtype(series):
        return "skip"
    n_unique = series.nunique()
    if n_unique == 2:
        return "label"
    if n_unique <= 15:
        return "onehot"
    return "label"  # High cardinality → label encoding
