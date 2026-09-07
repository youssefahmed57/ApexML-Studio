from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any
import io
import math
import pickle
import copy
import time
import hashlib
import json
import html

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, KFold, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef, log_loss,
    average_precision_score, precision_recall_curve, roc_curve, brier_score_loss,
    confusion_matrix, classification_report,
    mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    IMBALANCED_AVAILABLE = True
except Exception:
    ImbPipeline = None
    RandomOverSampler = None
    SMOTE = None
    IMBALANCED_AVAILABLE = False


RANDOM_STATE = 42


class IQRClipper(BaseEstimator, TransformerMixin):
    """Learn IQR clipping bounds on training data only, then clip future data."""
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.q1_ = np.nanpercentile(arr, 25, axis=0)
        self.q3_ = np.nanpercentile(arr, 75, axis=0)
        iqr = self.q3_ - self.q1_
        self.lower_ = self.q1_ - self.factor * iqr
        self.upper_ = self.q3_ + self.factor * iqr
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_, self.upper_)


class SafeVarianceThreshold(BaseEstimator, TransformerMixin):
    """Variance filter that never collapses the entire feature matrix."""
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("Variance filtering expects a 2D feature matrix.")
        variances = np.nanvar(arr, axis=0)
        support = variances > float(self.threshold)
        # If every feature is constant, keep all features so downstream models can
        # either fit an intercept-only relation or fail with a clear model-specific reason.
        if not np.any(support):
            support = np.ones(arr.shape[1], dtype=bool)
        self.support_ = np.asarray(support, dtype=bool)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        return X[:, self.support_] if hasattr(X, "__getitem__") else np.asarray(X)[:, self.support_]

    def get_support(self, indices=False):
        if indices:
            return np.flatnonzero(self.support_)
        return self.support_


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    """SelectKBest wrapper that automatically caps k to the available feature count."""
    def __init__(self, score_func=f_classif, k: int = 10):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        n_features = int(X.shape[1])
        if n_features < 1:
            raise ValueError("Feature selection received an empty feature matrix.")
        requested = max(1, int(self.k))
        self.effective_k_ = min(requested, n_features)
        self.selector_ = SelectKBest(score_func=self.score_func, k=self.effective_k_)
        self.selector_.fit(X, y)
        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def get_support(self, indices=False):
        return self.selector_.get_support(indices=indices)

class FeatureRecipeTransformer(BaseEstimator, TransformerMixin):
    """Leakage-safe feature creation driven by a list of user recipes.

    Supported operations:
    - interaction: multiply/divide/add/subtract two numeric columns
    - polynomial: create powers 2..degree for one numeric column
    - log: log1p with a shift learned on training data
    - bin: equal-width bins whose edges are learned on training data
    - datetime: extract year/month/day/dayofweek/dayofyear/is_weekend
    """
    def __init__(self, recipes=None):
        # Keep constructor parameters unchanged so sklearn.clone can reproduce the estimator.
        self.recipes = recipes

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        self.log_shifts_ = {}
        self.bin_edges_ = {}
        for i, recipe in enumerate(self.recipes or []):
            op = recipe.get("type")
            col = recipe.get("column")
            if op == "log" and col in X.columns:
                vals = pd.to_numeric(X[col], errors="coerce")
                min_val = vals.min(skipna=True)
                shift = 0.0 if pd.isna(min_val) or min_val > 0 else float(-min_val + 1.0)
                self.log_shifts_[i] = shift
            elif op == "bin" and col in X.columns:
                vals = pd.to_numeric(X[col], errors="coerce").dropna()
                n_bins = max(2, int(recipe.get("n_bins", 5)))
                if len(vals) and vals.nunique() > 1:
                    edges = np.linspace(float(vals.min()), float(vals.max()), n_bins + 1)
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                    self.bin_edges_[i] = edges
        return self

    def transform(self, X):
        out = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        for i, recipe in enumerate(self.recipes or []):
            op = recipe.get("type")
            if op == "interaction":
                a, b = recipe.get("col1"), recipe.get("col2")
                if a not in out.columns or b not in out.columns:
                    continue
                operation = recipe.get("operation", "multiply")
                av = pd.to_numeric(out[a], errors="coerce")
                bv = pd.to_numeric(out[b], errors="coerce")
                name = recipe.get("name") or f"{a}__{operation}__{b}"
                if operation == "multiply": out[name] = av * bv
                elif operation == "divide": out[name] = av / bv.replace(0, np.nan)
                elif operation == "add": out[name] = av + bv
                elif operation == "subtract": out[name] = av - bv
            elif op == "polynomial":
                col = recipe.get("column")
                if col not in out.columns:
                    continue
                vals = pd.to_numeric(out[col], errors="coerce")
                degree = max(2, min(5, int(recipe.get("degree", 2))))
                for d in range(2, degree + 1):
                    out[f"{col}^{d}"] = vals ** d
            elif op == "log":
                col = recipe.get("column")
                if col in out.columns:
                    vals = pd.to_numeric(out[col], errors="coerce")
                    shift = self.log_shifts_.get(i, 0.0)
                    out[recipe.get("name") or f"{col}__log1p"] = np.log1p(vals + shift)
            elif op == "bin":
                col = recipe.get("column")
                edges = self.bin_edges_.get(i)
                if col in out.columns and edges is not None:
                    vals = pd.to_numeric(out[col], errors="coerce")
                    out[recipe.get("name") or f"{col}__binned"] = pd.cut(
                        vals, bins=edges, labels=False, include_lowest=True
                    ).astype("Int64").astype(str)
            elif op == "datetime":
                col = recipe.get("column")
                if col not in out.columns:
                    continue
                dt = pd.to_datetime(out[col], errors="coerce")
                prefix = recipe.get("prefix") or col
                out[f"{prefix}__year"] = dt.dt.year
                out[f"{prefix}__month"] = dt.dt.month
                out[f"{prefix}__day"] = dt.dt.day
                out[f"{prefix}__dayofweek"] = dt.dt.dayofweek
                out[f"{prefix}__dayofyear"] = dt.dt.dayofyear
                out[f"{prefix}__is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(float)
                if recipe.get("drop_original", True):
                    out = out.drop(columns=[col])
        return out

    def get_feature_names_out(self, input_features=None):
        # ColumnTransformer downstream owns the final feature names; this is informational.
        return np.asarray(input_features if input_features is not None else [], dtype=object)


@dataclass
class DatasetConfig:
    target: str
    task: str
    excluded_columns: list[str]
    date_columns: list[str]
    force_numeric: list[str]
    force_categorical: list[str]
    test_size: float
    random_state: int
    numeric_imputer: str
    categorical_imputer: str
    scaler: str
    outlier_strategy: str
    iqr_factor: float
    max_categories: int | None
    min_category_frequency: int | float | None = None
    remove_zero_variance: bool = False
    feature_selection: str = "None"
    k_best: int | None = None
    categorical_encoder: str = "OneHot"
    imbalance_strategy: str = "None"
    feature_recipe: list[dict] = field(default_factory=list)


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Supported files: CSV, XLSX, XLS.")


def infer_task(y: pd.Series) -> str:
    non_null = y.dropna()
    if len(non_null) == 0:
        return "classification"
    if (
        pd.api.types.is_object_dtype(non_null)
        or isinstance(non_null.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(non_null)
        or pd.api.types.is_string_dtype(non_null)
    ):
        return "classification"
    unique = non_null.nunique()
    ratio = unique / max(len(non_null), 1)
    threshold = max(15, int(math.sqrt(max(len(non_null), 1))))
    if unique <= threshold and ratio <= 0.10:
        return "classification"
    return "regression"


def detect_id_like_columns(df: pd.DataFrame, target: str | None = None) -> list[str]:
    result = []
    n = max(len(df), 1)
    for col in df.columns:
        if col == target:
            continue
        name = col.lower()
        nunique = df[col].nunique(dropna=True)
        unique_ratio = nunique / n
        if (
            name in {"id", "index", "uuid"}
            or name.endswith("_id")
            or name.startswith("id_")
            or (unique_ratio > 0.985 and nunique > 20)
        ):
            result.append(col)
    return result


def detect_date_candidates(df: pd.DataFrame, target: str | None = None) -> list[str]:
    candidates = []
    for col in df.columns:
        if col == target:
            continue
        name = col.lower()
        if any(k in name for k in ["date", "time", "timestamp", "created", "updated"]):
            candidates.append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
    return list(dict.fromkeys(candidates))


def prepare_dataframe(
    df: pd.DataFrame,
    target: str,
    excluded_columns: list[str],
    date_columns: list[str],
    force_numeric: list[str],
    force_categorical: list[str],
):
    work = df.copy()
    work = work.drop_duplicates().copy()
    work = work.dropna(subset=[target]).copy()

    date_columns = [c for c in date_columns if c in work.columns and c != target]
    for col in date_columns:
        parsed = pd.to_datetime(work[col], errors="coerce")
        work[f"{col}__year"] = parsed.dt.year
        work[f"{col}__month"] = parsed.dt.month
        work[f"{col}__day"] = parsed.dt.day
        work[f"{col}__dayofweek"] = parsed.dt.dayofweek
        work[f"{col}__dayofyear"] = parsed.dt.dayofyear
        work.drop(columns=[col], inplace=True)

    excluded = [c for c in excluded_columns if c in work.columns and c != target]
    X = work.drop(columns=[target] + excluded)
    y = work[target].copy()

    for col in force_numeric:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in force_categorical:
        if col in X.columns:
            X[col] = X[col].where(X[col].isna(), X[col].astype(str))

    # Normalize object/category/string columns so imputing + OHE is reliable.
    for col in X.columns:
        if (
            pd.api.types.is_object_dtype(X[col])
            or isinstance(X[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(X[col])
            or pd.api.types.is_bool_dtype(X[col])
        ):
            X[col] = X[col].where(X[col].isna(), X[col].astype(str))

    return X, y, work



def prepare_inference_features(
    df: pd.DataFrame,
    excluded_columns: list[str],
    date_columns: list[str],
    force_numeric: list[str],
    force_categorical: list[str],
) -> pd.DataFrame:
    """
    Apply deterministic column-role transformations for prediction.

    Unlike prepare_dataframe(), this function never removes duplicates or rows,
    because batch prediction must preserve one output row for every input row.
    """
    work = df.copy()

    date_columns = [c for c in date_columns if c in work.columns]
    for col in date_columns:
        parsed = pd.to_datetime(work[col], errors="coerce")
        work[f"{col}__year"] = parsed.dt.year
        work[f"{col}__month"] = parsed.dt.month
        work[f"{col}__day"] = parsed.dt.day
        work[f"{col}__dayofweek"] = parsed.dt.dayofweek
        work[f"{col}__dayofyear"] = parsed.dt.dayofyear
        work.drop(columns=[col], inplace=True)

    excluded = [c for c in excluded_columns if c in work.columns]
    if excluded:
        work = work.drop(columns=excluded)

    for col in force_numeric:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    for col in force_categorical:
        if col in work.columns:
            work[col] = work[col].where(work[col].isna(), work[col].astype(str))

    for col in work.columns:
        if (
            pd.api.types.is_object_dtype(work[col])
            or isinstance(work[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(work[col])
            or pd.api.types.is_bool_dtype(work[col])
        ):
            work[col] = work[col].where(work[col].isna(), work[col].astype(str))

    return work
def get_feature_groups(X: pd.DataFrame):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_scaler(name: str):
    if name == "StandardScaler":
        return StandardScaler()
    if name == "MinMaxScaler":
        return MinMaxScaler()
    if name == "RobustScaler":
        return RobustScaler()
    return "passthrough"


def build_preprocessor(
    X: pd.DataFrame,
    numeric_imputer: str = "median",
    categorical_imputer: str = "most_frequent",
    scaler: str = "StandardScaler",
    outlier_strategy: str = "None",
    iqr_factor: float = 1.5,
    max_categories: int | None = 50,
    min_category_frequency: int | float | None = None,
    categorical_encoder: str = "OneHot",
    dense: bool = False,
):
    numeric_cols, categorical_cols = get_feature_groups(X)

    if numeric_imputer == "knn":
        numeric_steps = [("imputer", KNNImputer(n_neighbors=5))]
    else:
        numeric_steps = [("imputer", SimpleImputer(strategy=numeric_imputer))]
    if outlier_strategy == "IQR clip":
        numeric_steps.append(("iqr_clip", IQRClipper(factor=iqr_factor)))
    selected_scaler = build_scaler(scaler)
    if selected_scaler != "passthrough":
        numeric_steps.append(("scaler", selected_scaler))

    numeric_pipe = Pipeline(numeric_steps)

    ohe_kwargs = {
        "handle_unknown": "ignore",
        "sparse_output": not dense,
    }
    if max_categories and max_categories > 1:
        ohe_kwargs["max_categories"] = int(max_categories)
    if min_category_frequency not in (None, 0, 0.0):
        ohe_kwargs["min_frequency"] = min_category_frequency

    if categorical_encoder in {"Ordinal", "Label"}:
        categorical_encoder_step = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
            encoded_missing_value=-1,
        )
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=categorical_imputer)),
            ("ordinal", categorical_encoder_step),
        ])
    else:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=categorical_imputer)),
            ("onehot", OneHotEncoder(**ohe_kwargs)),
        ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers, remainder="drop"), numeric_cols, categorical_cols


def build_postprocess_steps(task: str, remove_zero_variance: bool = False, feature_selection: str = "None", k_best: int | None = None):
    steps = []
    if remove_zero_variance:
        steps.append(("variance", SafeVarianceThreshold(threshold=0.0)))
    if feature_selection == "SelectKBest" and k_best is not None and int(k_best) > 0:
        score_func = f_classif if task == "classification" else f_regression
        steps.append(("select", SafeSelectKBest(score_func=score_func, k=int(k_best))))
    return steps


def auto_nn_architecture(n_input_features: int, n_samples: int | None = None) -> tuple[int, ...]:
    """
    Conservative automatic MLP architecture for tabular data.

    It considers both the transformed input width and (when supplied) sample count
    so a small dataset does not receive an unnecessarily large network.
    """
    n_samples = n_samples or 1000

    if n_input_features <= 12:
        base = (32, 16)
    elif n_input_features <= 50:
        base = (64, 32)
    elif n_input_features <= 200:
        base = (128, 64, 32)
    else:
        base = (256, 128, 64)

    # Small data -> shrink the network to reduce variance and training cost.
    if n_samples < 500:
        return tuple(max(8, x // 2) for x in base[:2])
    if n_samples < 1500 and len(base) > 2:
        return base[:-1]
    return base


def recommend_primary_metric(task: str, y: pd.Series | np.ndarray) -> tuple[str, str]:
    """Return a sensible default leaderboard metric and a human-readable reason."""
    if task == "regression":
        return "Test_RMSE", "RMSE penalizes large prediction errors and is a strong general-purpose regression metric."

    values, counts = np.unique(np.asarray(y), return_counts=True)
    if len(counts) <= 1:
        return "Test_F1_Weighted", "Weighted F1 is used as the general classification metric."

    imbalance_ratio = counts.max() / max(counts.min(), 1)
    if imbalance_ratio >= 2.0:
        return (
            "Test_F1_Macro",
            "The target is imbalanced, so Macro F1 gives every class equal importance."
        )
    return (
        "Test_F1_Weighted",
        "The classes are reasonably balanced, so Weighted F1 is a stable overall comparison metric."
    )



def safe_cv_folds(task: str, y, requested: int) -> int:
    requested = max(2, int(requested))
    n = len(y)
    if n < 4:
        raise ValueError("At least 4 training rows are required for cross-validation.")

    if task == "classification":
        _, counts = np.unique(np.asarray(y), return_counts=True)
        min_class = int(counts.min())
        if min_class < 2:
            raise ValueError(
                "Every class needs at least two training examples for stratified cross-validation."
            )
        return max(2, min(requested, min_class, n))

    return max(2, min(requested, n))
def data_health_report(df: pd.DataFrame, target: str | None = None) -> dict:
    """Create a compact data-quality report and 0-100 health score."""
    rows, cols = df.shape
    missing_pct = float(df.isna().mean().mean() * 100) if rows and cols else 0.0
    duplicate_pct = float(df.duplicated().mean() * 100) if rows else 0.0

    constant_cols = [
        c for c in df.columns
        if df[c].nunique(dropna=True) <= 1
    ]

    high_cardinality = []
    for c in df.columns:
        if c == target:
            continue
        unique = df[c].nunique(dropna=True)
        if rows and unique >= 50 and unique / rows > 0.50:
            high_cardinality.append(c)

    id_like = detect_id_like_columns(df, target)
    date_like = detect_date_candidates(df, target)

    score = 100.0
    score -= min(30.0, missing_pct * 0.8)
    score -= min(15.0, duplicate_pct * 1.2)
    score -= min(15.0, len(constant_cols) * 4.0)
    score -= min(15.0, len(high_cardinality) * 2.5)
    score -= min(10.0, len(id_like) * 1.5)
    score = max(0.0, round(score, 1))

    warnings = []
    if missing_pct > 0:
        warnings.append(f"{missing_pct:.1f}% of cells are missing.")
    if duplicate_pct > 0:
        warnings.append(f"{duplicate_pct:.1f}% of rows are duplicates.")
    if constant_cols:
        warnings.append(f"{len(constant_cols)} constant column(s) add no predictive information.")
    if high_cardinality:
        warnings.append(f"{len(high_cardinality)} high-cardinality column(s) may expand one-hot encoding.")
    if id_like:
        warnings.append(f"{len(id_like)} ID-like column(s) should usually be excluded.")
    if target is not None and target in df:
        target_missing = float(df[target].isna().mean() * 100)
        if target_missing > 0:
            warnings.append(f"{target_missing:.1f}% of target values are missing and will be dropped.")

    return {
        "score": score,
        "rows": rows,
        "columns": cols,
        "missing_pct": missing_pct,
        "duplicate_pct": duplicate_pct,
        "constant_columns": constant_cols,
        "high_cardinality_columns": high_cardinality,
        "id_like_columns": id_like,
        "date_candidates": date_like,
        "warnings": warnings,
    }


def build_baseline(task: str):
    if task == "classification":
        return DummyClassifier(strategy="prior", random_state=RANDOM_STATE)
    return DummyRegressor(strategy="mean")

def build_model(task: str, name: str, params: dict[str, Any], n_classes: int | None = None):
    if task == "classification":
        if name == "Logistic Regression":
            return LogisticRegression(
                C=float(params.get("C", 1.0)),
                max_iter=int(params.get("max_iter", 2000)),
                class_weight=params.get("class_weight"),
                random_state=RANDOM_STATE,
            )
        if name == "Naive Bayes":
            return GaussianNB(var_smoothing=float(params.get("var_smoothing", 1e-9)))
        if name == "Decision Tree":
            return DecisionTreeClassifier(
                max_depth=params.get("max_depth"),
                min_samples_split=int(params.get("min_samples_split", 2)),
                min_samples_leaf=int(params.get("min_samples_leaf", 1)),
                class_weight=params.get("class_weight"),
                random_state=RANDOM_STATE,
            )
        if name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=params.get("max_depth"),
                min_samples_split=int(params.get("min_samples_split", 2)),
                min_samples_leaf=int(params.get("min_samples_leaf", 1)),
                class_weight=params.get("class_weight"),
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        if name == "Extra Trees":
            return ExtraTreesClassifier(n_estimators=int(params.get("n_estimators", 300)), max_depth=params.get("max_depth"), min_samples_split=int(params.get("min_samples_split", 2)), min_samples_leaf=int(params.get("min_samples_leaf", 1)), class_weight=params.get("class_weight"), n_jobs=-1, random_state=RANDOM_STATE)
        if name == "SVM":
            return SVC(C=float(params.get("C", 1.0)), kernel=params.get("kernel", "rbf"), gamma=params.get("gamma", "scale"), class_weight=params.get("class_weight"), probability=True, random_state=RANDOM_STATE)
        if name == "AdaBoost":
            return AdaBoostClassifier(n_estimators=int(params.get("n_estimators", 150)), learning_rate=float(params.get("learning_rate", 0.05)), random_state=RANDOM_STATE)
        if name == "Hist Gradient Boosting":
            return HistGradientBoostingClassifier(learning_rate=float(params.get("learning_rate", 0.08)), max_iter=int(params.get("max_iter", 200)), max_leaf_nodes=int(params.get("max_leaf_nodes", 31)), l2_regularization=float(params.get("l2_regularization", 0.0)), random_state=RANDOM_STATE)
        if name == "Gradient Boosting":
            return GradientBoostingClassifier(
                n_estimators=int(params.get("n_estimators", 100)),
                learning_rate=float(params.get("learning_rate", 0.1)),
                max_depth=int(params.get("max_depth", 3)),
                random_state=RANDOM_STATE,
            )
        if name == "KNN":
            return KNeighborsClassifier(
                n_neighbors=int(params.get("n_neighbors", 7)),
                weights=params.get("weights", "uniform"),
                p=int(params.get("p", 2)),
            )
        if name == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise RuntimeError("XGBoost is not installed.")
            objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
            eval_metric = "logloss" if n_classes == 2 else "mlogloss"
            return XGBClassifier(
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=int(params.get("max_depth", 6)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                subsample=float(params.get("subsample", 0.9)),
                colsample_bytree=float(params.get("colsample_bytree", 0.9)),
                reg_alpha=float(params.get("reg_alpha", 0.0)),
                reg_lambda=float(params.get("reg_lambda", 1.0)),
                objective=objective,
                eval_metric=eval_metric,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        if name == "LightGBM":
            if not LIGHTGBM_AVAILABLE:
                raise RuntimeError("LightGBM is not installed.")
            return LGBMClassifier(
                n_estimators=int(params.get("n_estimators", 250)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                num_leaves=int(params.get("num_leaves", 31)),
                max_depth=int(params.get("max_depth", -1)),
                subsample=float(params.get("subsample", 0.9)),
                colsample_bytree=float(params.get("colsample_bytree", 0.9)),
                reg_alpha=float(params.get("reg_alpha", 0.0)),
                reg_lambda=float(params.get("reg_lambda", 0.0)),
                class_weight=params.get("class_weight"),
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            )
        if name == "Neural Network":
            batch = params.get("batch_size", "auto")
            return MLPClassifier(
                hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (64, 32))),
                activation=params.get("activation", "relu"),
                solver=params.get("solver", "adam"),
                alpha=float(params.get("alpha", 0.0001)),
                batch_size=batch,
                learning_rate=params.get("learning_rate", "constant"),
                learning_rate_init=float(params.get("learning_rate_init", 0.001)),
                max_iter=int(params.get("max_iter", 500)),
                early_stopping=bool(params.get("early_stopping", True)),
                validation_fraction=float(params.get("validation_fraction", 0.1)),
                n_iter_no_change=int(params.get("n_iter_no_change", 15)),
                random_state=RANDOM_STATE,
            )

    if task == "regression":
        if name == "Linear Regression":
            return LinearRegression()
        if name == "Polynomial Regression":
            degree = int(params.get("degree", 2))
            return Pipeline([
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("scale", StandardScaler(with_mean=False)),
                ("linear", LinearRegression()),
            ])
        if name == "Ridge":
            return Ridge(alpha=float(params.get("alpha", 1.0)))
        if name == "Lasso":
            return Lasso(alpha=float(params.get("alpha", 0.1)), max_iter=int(params.get("max_iter", 5000)), random_state=RANDOM_STATE)
        if name == "ElasticNet":
            return ElasticNet(alpha=float(params.get("alpha", 0.1)), l1_ratio=float(params.get("l1_ratio", 0.5)), max_iter=int(params.get("max_iter", 5000)), random_state=RANDOM_STATE)
        if name == "SVM":
            return SVR(C=float(params.get("C", 1.0)), kernel=params.get("kernel", "rbf"), gamma=params.get("gamma", "scale"), epsilon=float(params.get("epsilon", 0.1)))
        if name == "Decision Tree":
            return DecisionTreeRegressor(
                max_depth=params.get("max_depth"),
                min_samples_split=int(params.get("min_samples_split", 2)),
                min_samples_leaf=int(params.get("min_samples_leaf", 1)),
                random_state=RANDOM_STATE,
            )
        if name == "Random Forest":
            return RandomForestRegressor(
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=params.get("max_depth"),
                min_samples_split=int(params.get("min_samples_split", 2)),
                min_samples_leaf=int(params.get("min_samples_leaf", 1)),
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        if name == "Extra Trees":
            return ExtraTreesRegressor(n_estimators=int(params.get("n_estimators", 300)), max_depth=params.get("max_depth"), min_samples_split=int(params.get("min_samples_split", 2)), min_samples_leaf=int(params.get("min_samples_leaf", 1)), n_jobs=-1, random_state=RANDOM_STATE)
        if name == "AdaBoost":
            return AdaBoostRegressor(n_estimators=int(params.get("n_estimators", 150)), learning_rate=float(params.get("learning_rate", 0.05)), random_state=RANDOM_STATE)
        if name == "Hist Gradient Boosting":
            return HistGradientBoostingRegressor(learning_rate=float(params.get("learning_rate", 0.08)), max_iter=int(params.get("max_iter", 200)), max_leaf_nodes=int(params.get("max_leaf_nodes", 31)), l2_regularization=float(params.get("l2_regularization", 0.0)), random_state=RANDOM_STATE)
        if name == "Gradient Boosting":
            return GradientBoostingRegressor(
                n_estimators=int(params.get("n_estimators", 100)),
                learning_rate=float(params.get("learning_rate", 0.1)),
                max_depth=int(params.get("max_depth", 3)),
                random_state=RANDOM_STATE,
            )
        if name == "KNN":
            return KNeighborsRegressor(
                n_neighbors=int(params.get("n_neighbors", 7)),
                weights=params.get("weights", "uniform"),
                p=int(params.get("p", 2)),
            )
        if name == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise RuntimeError("XGBoost is not installed.")
            return XGBRegressor(
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=int(params.get("max_depth", 6)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                subsample=float(params.get("subsample", 0.9)),
                colsample_bytree=float(params.get("colsample_bytree", 0.9)),
                reg_alpha=float(params.get("reg_alpha", 0.0)),
                reg_lambda=float(params.get("reg_lambda", 1.0)),
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        if name == "LightGBM":
            if not LIGHTGBM_AVAILABLE:
                raise RuntimeError("LightGBM is not installed.")
            return LGBMRegressor(
                n_estimators=int(params.get("n_estimators", 250)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                num_leaves=int(params.get("num_leaves", 31)),
                max_depth=int(params.get("max_depth", -1)),
                subsample=float(params.get("subsample", 0.9)),
                colsample_bytree=float(params.get("colsample_bytree", 0.9)),
                reg_alpha=float(params.get("reg_alpha", 0.0)),
                reg_lambda=float(params.get("reg_lambda", 0.0)),
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            )
        if name == "Neural Network":
            batch = params.get("batch_size", "auto")
            return MLPRegressor(
                hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (64, 32))),
                activation=params.get("activation", "relu"),
                solver=params.get("solver", "adam"),
                alpha=float(params.get("alpha", 0.0001)),
                batch_size=batch,
                learning_rate=params.get("learning_rate", "constant"),
                learning_rate_init=float(params.get("learning_rate_init", 0.001)),
                max_iter=int(params.get("max_iter", 500)),
                early_stopping=bool(params.get("early_stopping", True)),
                validation_fraction=float(params.get("validation_fraction", 0.1)),
                n_iter_no_change=int(params.get("n_iter_no_change", 15)),
                random_state=RANDOM_STATE,
            )

    raise ValueError(f"Unsupported model: {name} for {task}")


def classification_metrics(y_true, y_pred, y_prob=None):
    n_classes = len(np.unique(y_true))
    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision_Weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall_Weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1_Weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1_Macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    result["ROC_AUC"] = np.nan
    result["PR_AUC"] = np.nan
    result["Log_Loss"] = np.nan
    result["Brier"] = np.nan

    if y_prob is not None:
        try:
            if n_classes == 2:
                result["ROC_AUC"] = roc_auc_score(y_true, y_prob[:, 1])
                result["PR_AUC"] = average_precision_score(y_true, y_prob[:, 1])
                result["Brier"] = brier_score_loss(y_true, y_prob[:, 1])
            else:
                result["ROC_AUC"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted"
                )
        except Exception:
            pass
        try:
            result["Log_Loss"] = log_loss(y_true, y_prob)
        except Exception:
            pass

    return result
def regression_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "Median_AE": median_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred),
    }
def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    selected_models: list[str],
    params_by_model: dict[str, dict[str, Any]],
    config: DatasetConfig,
    cv_folds: int = 5,
    run_cv: bool = True,
):
    """Train multiple model families safely.

    One failing estimator does not abort the whole AutoML run. Successful models are
    returned in the leaderboard and failures are stored under details['__failures__'].
    """
    target_encoder = None
    y_model = y.copy()

    if task == "classification":
        target_encoder = LabelEncoder()
        y_model = target_encoder.fit_transform(y.astype(str))
        n_classes = len(target_encoder.classes_)
        if n_classes < 2:
            raise ValueError("Classification requires at least two target classes.")
        counts = pd.Series(y_model).value_counts()
        n_total = len(y_model)
        n_test = max(1, int(round(n_total * config.test_size)))
        n_train = n_total - n_test
        can_stratify = counts.min() >= 2 and n_test >= n_classes and n_train >= n_classes
        stratify = y_model if can_stratify else None
    else:
        y_numeric = pd.to_numeric(y_model, errors="coerce")
        valid = ~pd.isna(y_numeric)
        X = X.loc[valid].reset_index(drop=True)
        y_model = np.asarray(y_numeric.loc[valid], dtype=float)
        if len(y_model) < 5:
            raise ValueError("Regression requires at least five valid numeric target rows.")
        stratify = None
        n_classes = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_model, test_size=config.test_size,
        random_state=config.random_state, stratify=stratify,
    )

    recipe_preview = FeatureRecipeTransformer(config.feature_recipe)
    X_train_design = recipe_preview.fit_transform(X_train)

    results, bundles, details, failures = [], {}, {}, []

    for name in selected_models:
        try:
            dense = name in {"Naive Bayes", "Hist Gradient Boosting"}
            preprocessor, numeric_cols, categorical_cols = build_preprocessor(
                X_train_design,
                numeric_imputer=config.numeric_imputer,
                categorical_imputer=config.categorical_imputer,
                scaler=config.scaler,
                outlier_strategy=config.outlier_strategy,
                iqr_factor=config.iqr_factor,
                max_categories=config.max_categories,
                min_category_frequency=config.min_category_frequency,
                categorical_encoder=config.categorical_encoder,
                dense=dense,
            )

            model_params = dict(params_by_model.get(name, {}))
            auto_architecture = bool(model_params.pop("_auto_architecture", False))
            nn_architecture_info = None

            if name == "Neural Network" and auto_architecture:
                transformed_train = preprocessor.fit_transform(X_train_design)
                n_transformed = int(transformed_train.shape[1])
                if config.remove_zero_variance:
                    try:
                        tmp_var = VarianceThreshold(0.0).fit_transform(transformed_train)
                        n_transformed = int(tmp_var.shape[1])
                    except Exception:
                        pass
                if config.feature_selection == "SelectKBest" and config.k_best:
                    n_transformed = min(n_transformed, int(config.k_best))
                architecture = auto_nn_architecture(n_transformed, n_samples=len(X_train))
                model_params["hidden_layer_sizes"] = architecture
                nn_architecture_info = {
                    "mode": "Auto",
                    "transformed_input_features": n_transformed,
                    "hidden_layer_sizes": architecture,
                }
                # Clean unfitted preprocessor for the final pipeline.
                preprocessor, numeric_cols, categorical_cols = build_preprocessor(
                    X_train_design,
                    numeric_imputer=config.numeric_imputer,
                    categorical_imputer=config.categorical_imputer,
                    scaler=config.scaler,
                    outlier_strategy=config.outlier_strategy,
                    iqr_factor=config.iqr_factor,
                    max_categories=config.max_categories,
                    min_category_frequency=config.min_category_frequency,
                    categorical_encoder=config.categorical_encoder,
                    dense=dense,
                )
            elif name == "Neural Network":
                nn_architecture_info = {
                    "mode": "Custom",
                    "transformed_input_features": None,
                    "hidden_layer_sizes": tuple(model_params.get("hidden_layer_sizes", (64, 32))),
                }

            model = build_model(task, name, model_params, n_classes=n_classes)
            post_steps = build_postprocess_steps(
                task, config.remove_zero_variance,
                config.feature_selection, config.k_best,
            )
            pipeline_cls = ImbPipeline if (task == "classification" and config.imbalance_strategy != "None" and IMBALANCED_AVAILABLE) else Pipeline
            pipeline_steps = [("feature_recipe", FeatureRecipeTransformer(config.feature_recipe)), ("preprocess", preprocessor), *post_steps]
            if task == "classification" and config.imbalance_strategy != "None":
                if not IMBALANCED_AVAILABLE:
                    raise RuntimeError("imbalanced-learn is required for the selected imbalance strategy.")
                if config.imbalance_strategy == "RandomOverSampler":
                    pipeline_steps.append(("sampler", RandomOverSampler(random_state=config.random_state)))
                elif config.imbalance_strategy == "SMOTE":
                    if categorical_cols:
                        raise ValueError("SMOTE is disabled for mixed categorical feature spaces in ApexML Ultimate; use RandomOverSampler or class weights instead.")
                    pipeline_steps.append(("sampler", SMOTE(random_state=config.random_state)))
            pipeline_steps.append(("model", model))
            pipe = pipeline_cls(pipeline_steps)

            fit_t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - fit_t0

            train_pred = pipe.predict(X_train)
            pred_t0 = time.perf_counter()
            test_pred = pipe.predict(X_test)
            predict_seconds = time.perf_counter() - pred_t0
            predict_ms_per_1000 = (predict_seconds / max(len(X_test), 1)) * 1_000_000
            try:
                model_size_kb = len(pickle.dumps(pipe)) / 1024.0
            except Exception:
                model_size_kb = np.nan

            if task == "classification":
                train_prob = pipe.predict_proba(X_train) if hasattr(pipe, "predict_proba") else None
                test_prob = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
                train_m = classification_metrics(y_train, train_pred, train_prob)
                test_m = classification_metrics(y_test, test_pred, test_prob)
                score_name, cv_scoring = "F1_Weighted", "f1_weighted"
                row = {
                    "Model": name, "Status": "OK",
                    **{f"Train_{k}": v for k, v in train_m.items()},
                    **{f"Test_{k}": v for k, v in test_m.items()},
                    "Generalization_Gap": train_m["F1_Weighted"] - test_m["F1_Weighted"],
                    "Fit_Seconds": fit_seconds,
                    "Predict_ms_per_1000": predict_ms_per_1000,
                    "Model_Size_KB": model_size_kb,
                }
                report = classification_report(
                    y_test, test_pred,
                    labels=np.arange(len(target_encoder.classes_)),
                    target_names=[str(x) for x in target_encoder.classes_],
                    output_dict=True, zero_division=0,
                )
                cm = confusion_matrix(y_test, test_pred)
            else:
                train_m = regression_metrics(y_train, train_pred)
                test_m = regression_metrics(y_test, test_pred)
                score_name, cv_scoring = "RMSE", "neg_root_mean_squared_error"
                row = {
                    "Model": name, "Status": "OK",
                    **{f"Train_{k}": v for k, v in train_m.items()},
                    **{f"Test_{k}": v for k, v in test_m.items()},
                    "Generalization_Gap": test_m["RMSE"] - train_m["RMSE"],
                    "Fit_Seconds": fit_seconds,
                    "Predict_ms_per_1000": predict_ms_per_1000,
                    "Model_Size_KB": model_size_kb,
                }
                report, cm = None, None

            if run_cv and cv_folds >= 2:
                try:
                    actual_cv_folds = safe_cv_folds(task, y_train, cv_folds)
                    cv_splitter = (
                        StratifiedKFold(actual_cv_folds, shuffle=True, random_state=config.random_state)
                        if task == "classification"
                        else KFold(actual_cv_folds, shuffle=True, random_state=config.random_state)
                    )
                    scores = cross_val_score(
                        pipe, X_train, y_train, cv=cv_splitter,
                        scoring=cv_scoring, n_jobs=1,
                    )
                    if task == "regression":
                        scores = -scores
                    row[f"CV_{score_name}_Mean"] = float(np.mean(scores))
                    row[f"CV_{score_name}_Std"] = float(np.std(scores))
                    row["CV_Folds"] = int(actual_cv_folds)
                except Exception as cv_exc:
                    row[f"CV_{score_name}_Mean"] = np.nan
                    row[f"CV_{score_name}_Std"] = np.nan
                    row["CV_Error"] = str(cv_exc)[:220]

            results.append(row)
            bundles[name] = {
                "pipeline": pipe,
                "task": task,
                "target_encoder": target_encoder,
                "target_name": config.target,
                "config": asdict(config),
                "feature_columns": X.columns.tolist(),
                "original_dtypes": {c: str(X[c].dtype) for c in X.columns},
            }

            model_obj = pipe.named_steps["model"]
            nn_parameter_count = nn_layer_shapes = None
            if name == "Neural Network" and hasattr(model_obj, "coefs_"):
                nn_parameter_count = int(sum(w.size for w in model_obj.coefs_) + sum(b.size for b in model_obj.intercepts_))
                input_width = int(model_obj.coefs_[0].shape[0])
                output_width = int(model_obj.coefs_[-1].shape[1])
                hidden = tuple(int(x) for x in model_obj.hidden_layer_sizes)
                nn_layer_shapes = [input_width, *hidden, output_width]
                if nn_architecture_info is not None:
                    nn_architecture_info.update({
                        "transformed_input_features": input_width,
                        "output_neurons": output_width,
                        "all_layers": nn_layer_shapes,
                        "trainable_parameters": nn_parameter_count,
                    })

            details[name] = {
                "y_test": y_test, "test_pred": test_pred, "X_test": X_test,
                "classification_report": report, "confusion_matrix": cm,
                "loss_curve": getattr(model_obj, "loss_curve_", None),
                "validation_scores": getattr(model_obj, "validation_scores_", None),
                "n_iter": getattr(model_obj, "n_iter_", None),
                "nn_architecture": nn_architecture_info,
                "nn_parameter_count": nn_parameter_count,
                "nn_layer_shapes": nn_layer_shapes,
                "numeric_cols": numeric_cols, "categorical_cols": categorical_cols,
            }

        except Exception as exc:
            failures.append({"Model": name, "Error": f"{type(exc).__name__}: {str(exc)[:400]}"})

    if not results:
        message = failures[0]["Error"] if failures else "Unknown training error."
        raise RuntimeError(f"All selected models failed. First error: {message}")

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("Test_F1_Weighted", ascending=False) if task == "classification" else result_df.sort_values("Test_RMSE")
    details["__failures__"] = failures
    return result_df.reset_index(drop=True), bundles, details, (X_train, X_test, y_train, y_test)

def best_model_name(result_df: pd.DataFrame, task: str, metric: str | None = None) -> str:
    if task == "classification":
        metric = metric or "Test_F1_Weighted"
        ascending = metric in {"Test_Log_Loss"}
    else:
        metric = metric or "Test_RMSE"
        ascending = metric not in {"Test_R2", "R2"}
    return result_df.sort_values(metric, ascending=ascending).iloc[0]["Model"]

def bundle_to_bytes(bundle: dict) -> bytes:
    buffer = io.BytesIO()
    pickle.dump(bundle, buffer)
    return buffer.getvalue()


def processed_feature_names(pipe: Pipeline):
    prep = pipe.named_steps.get("preprocess")
    if prep is None:
        return None
    try:
        names = np.asarray(prep.get_feature_names_out(), dtype=object)
    except Exception:
        return None
    for step_name in ["variance", "select"]:
        step = pipe.named_steps.get(step_name)
        if step is not None and hasattr(step, "get_support"):
            try:
                names = names[step.get_support()]
            except Exception:
                return None
    return names


def feature_importance_table(bundle: dict):
    pipe = bundle["pipeline"]
    model = pipe.named_steps["model"]
    names = processed_feature_names(pipe)
    if names is None:
        return None

    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        vals = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None

    if len(vals) != len(names):
        return None
    return (
        pd.DataFrame({"Feature": names, "Importance": vals})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
def default_tuning_grid(model_name: str, task: str, size: str = "Quick"):
    quick = size == "Quick"

    if model_name == "Logistic Regression":
        return {"model__C": [0.1, 1.0, 10.0] if quick else [0.01, 0.1, 1.0, 10.0, 100.0]}
    if model_name == "Naive Bayes":
        return {"model__var_smoothing": np.logspace(-11, -7, 5 if quick else 9).tolist()}
    if model_name == "Decision Tree":
        return {
            "model__max_depth": [None, 5, 10] if quick else [None, 3, 5, 8, 12, 20],
            "model__min_samples_split": [2, 5] if quick else [2, 5, 10],
            "model__min_samples_leaf": [1, 2] if quick else [1, 2, 4],
        }
    if model_name == "Random Forest":
        return {
            "model__n_estimators": [150, 350] if quick else [100, 250, 500],
            "model__max_depth": [None, 8] if quick else [None, 6, 10, 16],
            "model__min_samples_leaf": [1, 2],
        }
    if model_name == "Gradient Boosting":
        return {
            "model__n_estimators": [100, 250],
            "model__learning_rate": [0.03, 0.1] if quick else [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3] if quick else [2, 3, 4],
        }
    if model_name == "Extra Trees":
        return {"model__n_estimators": [150, 350] if quick else [100, 250, 500], "model__max_depth": [None, 10] if quick else [None, 6, 10, 18], "model__min_samples_leaf": [1, 2]}
    if model_name == "SVM":
        return {"model__C": [0.5, 2.0] if quick else [0.1, 0.5, 1.0, 2.0, 10.0], "model__kernel": ["rbf", "linear"] if quick else ["rbf", "linear", "poly"]}
    if model_name == "AdaBoost":
        return {"model__n_estimators": [75, 200] if quick else [50, 100, 200, 400], "model__learning_rate": [0.03, 0.1] if quick else [0.01, 0.03, 0.1, 0.3]}
    if model_name == "Hist Gradient Boosting":
        return {"model__max_iter": [100, 250] if quick else [100, 200, 400], "model__learning_rate": [0.03, 0.1] if quick else [0.01, 0.03, 0.05, 0.1], "model__max_leaf_nodes": [15, 31] if quick else [15, 31, 63]}
    if model_name == "KNN":
        return {
            "model__n_neighbors": [3, 7, 15] if quick else [3, 5, 7, 11, 15, 21],
            "model__weights": ["uniform", "distance"],
        }
    if model_name == "XGBoost":
        return {
            "model__n_estimators": [150, 300] if quick else [100, 250, 500],
            "model__max_depth": [3, 6] if quick else [3, 5, 7, 10],
            "model__learning_rate": [0.03, 0.1] if quick else [0.01, 0.03, 0.05, 0.1],
        }
    if model_name == "LightGBM":
        return {
            "model__n_estimators": [150, 350] if quick else [100, 250, 500],
            "model__learning_rate": [0.03, 0.1] if quick else [0.01, 0.03, 0.05, 0.1],
            "model__num_leaves": [15, 31] if quick else [15, 31, 63],
        }
    if model_name == "Neural Network":
        return {
            "model__hidden_layer_sizes": [(32,), (64, 32)] if quick else [(32,), (64, 32), (128, 64), (128, 64, 32)],
            "model__alpha": [0.0001, 0.001] if quick else [0.00001, 0.0001, 0.001, 0.01],
            "model__learning_rate_init": [0.001, 0.01] if quick else [0.0003, 0.001, 0.003, 0.01],
        }

    if task == "regression" and model_name == "Polynomial Regression":
        return {"model__poly__degree": [2] if quick else [2, 3]}
    if task == "regression" and model_name in {"Ridge", "Lasso"}:
        return {"model__alpha": [0.01, 0.1, 1.0, 10.0] if quick else [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    if task == "regression" and model_name == "ElasticNet":
        return {"model__alpha": [0.01, 0.1, 1.0] if quick else [0.001, 0.01, 0.1, 1.0, 10.0], "model__l1_ratio": [0.2, 0.8] if quick else [0.1, 0.25, 0.5, 0.75, 0.9]}

    return {}


def tune_trained_bundle(
    bundle: dict,
    model_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    cv_folds: int = 3,
    size: str = "Quick",
    scoring: str | None = None,
):
    task = bundle["task"]
    estimator = copy.deepcopy(bundle["pipeline"])
    grid = default_tuning_grid(model_name, task, size=size)
    if not grid:
        raise ValueError(f"No tuning grid is defined for {model_name}.")

    if scoring is None:
        scoring = "f1_weighted" if task == "classification" else "neg_root_mean_squared_error"

    actual_cv_folds = safe_cv_folds(task, y_train, cv_folds)
    cv_splitter = (
        StratifiedKFold(n_splits=actual_cv_folds, shuffle=True, random_state=RANDOM_STATE)
        if task == "classification"
        else KFold(n_splits=actual_cv_folds, shuffle=True, random_state=RANDOM_STATE)
    )

    search = GridSearchCV(
        estimator,
        param_grid=grid,
        scoring=scoring,
        cv=cv_splitter,
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    pred = search.best_estimator_.predict(X_test)

    if task == "classification":
        prob = search.best_estimator_.predict_proba(X_test) if hasattr(search.best_estimator_, "predict_proba") else None
        metrics = classification_metrics(y_test, pred, prob)
    else:
        metrics = regression_metrics(y_test, pred)

    tuned_bundle = copy.deepcopy(bundle)
    tuned_bundle["pipeline"] = search.best_estimator_
    tuned_bundle["tuning"] = {
        "best_params": search.best_params_,
        "best_cv_score": float(-search.best_score_ if str(scoring).startswith("neg_") else search.best_score_),
        "cv_folds": actual_cv_folds,
        "grid_size": size,
    }
    return {
        "search": search,
        "best_params": search.best_params_,
        "best_cv_score": float(-search.best_score_ if str(scoring).startswith("neg_") else search.best_score_),
        "test_metrics": metrics,
        "tuned_bundle": tuned_bundle,
        "cv_results": pd.DataFrame(search.cv_results_),
    }


def evaluate_baseline(X_train, X_test, y_train, y_test, task: str) -> dict:
    baseline = build_baseline(task)
    baseline.fit(X_train, y_train)
    pred = baseline.predict(X_test)
    if task == "classification":
        prob = baseline.predict_proba(X_test) if hasattr(baseline, "predict_proba") else None
        return classification_metrics(y_test, pred, prob)
    return regression_metrics(y_test, pred)


def model_card_markdown(
    model_name: str,
    bundle: dict,
    metrics: dict,
    primary_metric: str,
    dataset_name: str = "Uploaded dataset",
) -> str:
    task = bundle["task"]
    target = bundle["target_name"]
    config = bundle.get("config", {})
    lines = [
        f"# Model Card — {model_name}",
        "",
        f"- **Dataset:** {dataset_name}",
        f"- **Task:** {task.title()}",
        f"- **Target:** `{target}`",
        f"- **Primary metric:** `{primary_metric}`",
        "",
        "## Test Metrics",
        "",
    ]
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)) and not pd.isna(value):
            lines.append(f"- **{key}:** {float(value):.6f}")
    lines += [
        "",
        "## Pipeline",
        "",
        f"- Numeric imputer: `{config.get('numeric_imputer')}`",
        f"- Categorical imputer: `{config.get('categorical_imputer')}`",
        f"- Scaler: `{config.get('scaler')}`",
        f"- Outlier strategy: `{config.get('outlier_strategy')}`",
        f"- Rare-category minimum frequency: `{config.get('min_category_frequency')}`",
        f"- Remove zero variance: `{config.get('remove_zero_variance')}`",
        f"- Feature selection: `{config.get('feature_selection')}`",
        f"- K best: `{config.get('k_best')}`",
        f"- Test size: `{config.get('test_size')}`",
        f"- Random state: `{config.get('random_state')}`",
        "",
        "## Intended Use",
        "",
        "This model is intended for supervised tabular prediction on data that follows the same feature schema as the training dataset.",
        "",
        "## Limitations",
        "",
        "- Performance can degrade when production data differs from the training distribution.",
        "- Correlation and feature importance do not prove causation.",
        "- Predictions should be reviewed in the context of the real business or scientific objective.",
    ]
    return "\n".join(lines)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def cleaning_report(
    df: pd.DataFrame,
    target: str,
    excluded_columns: list[str],
    date_columns: list[str],
    force_numeric: list[str],
) -> dict:
    report = {
        "input_rows": int(len(df)),
        "input_columns": int(df.shape[1]),
        "duplicate_rows_removed": int(df.duplicated().sum()),
        "missing_target_rows_removed": int(df[target].isna().sum()) if target in df else 0,
        "excluded_columns": list(excluded_columns),
        "date_columns": list(date_columns),
        "numeric_coercion": {},
    }
    for col in force_numeric:
        if col in df.columns:
            before = int(df[col].notna().sum())
            after = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
            report["numeric_coercion"][col] = max(0, before - after)
    report["estimated_output_rows"] = max(
        0, report["input_rows"] - report["duplicate_rows_removed"] - report["missing_target_rows_removed"]
    )
    return report


def re_normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def leakage_scan(df: pd.DataFrame, target: str, task: str) -> pd.DataFrame:
    """Heuristic leakage scan. It flags suspicious features; human review is still required."""
    if target not in df.columns:
        return pd.DataFrame(columns=["Feature", "Risk", "Reason", "Evidence"])

    rows = []
    y = df[target]
    target_norm = re_normalize_name(target)

    for col in df.columns:
        if col == target:
            continue
        x = df[col]
        col_norm = re_normalize_name(col)

        try:
            same = (x.fillna("__NA__").astype(str).values == y.fillna("__NA__").astype(str).values)
            if len(same) and float(np.mean(same)) >= 0.999:
                rows.append({"Feature": col, "Risk": "Critical", "Reason": "Feature is effectively identical to the target.", "Evidence": f"match={np.mean(same):.3f}"})
                continue
        except Exception:
            pass

        suspicious_tokens = ["target", "label", "outcome", "prediction", "predicted", "result"]
        if (target_norm and (target_norm in col_norm or col_norm in target_norm) and len(col_norm) > 3) or any(t in col_norm for t in suspicious_tokens):
            rows.append({"Feature": col, "Risk": "High", "Reason": "Column name suggests it may contain target-derived information.", "Evidence": col})

        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            pair = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
            if len(pair) >= 10 and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1:
                corr = float(pair.corr().iloc[0, 1])
                if abs(corr) >= 0.995:
                    rows.append({"Feature": col, "Risk": "Critical", "Reason": "Near-perfect numeric correlation with target may indicate leakage.", "Evidence": f"corr={corr:.4f}"})
                elif abs(corr) >= 0.95:
                    rows.append({"Feature": col, "Risk": "Medium", "Reason": "Extremely strong target correlation. Verify prediction-time availability.", "Evidence": f"corr={corr:.4f}"})

        if task == "classification":
            temp = pd.DataFrame({"x": x, "y": y}).dropna()
            nunique = temp["x"].nunique()
            if 2 <= nunique <= min(500, max(2, int(len(temp) * 0.5))) and len(temp) >= 20:
                mapping = temp.groupby("x", dropna=False)["y"].nunique()
                if len(mapping) and float((mapping <= 1).mean()) >= 0.995:
                    rows.append({"Feature": col, "Risk": "High", "Reason": "Feature almost deterministically maps to target classes.", "Evidence": f"deterministic_groups={(mapping <= 1).mean():.3f}"})

    if not rows:
        return pd.DataFrame(columns=["Feature", "Risk", "Reason", "Evidence"])
    out = pd.DataFrame(rows).drop_duplicates(subset=["Feature", "Reason"])
    order = pd.Categorical(out["Risk"], categories=["Critical", "High", "Medium", "Low"], ordered=True)
    return out.assign(_order=order).sort_values(["_order", "Feature"]).drop(columns="_order").reset_index(drop=True)


def duplicate_column_report(df: pd.DataFrame) -> list[tuple[str, str]]:
    cols = list(df.columns)
    duplicates = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            try:
                if df[a].equals(df[b]):
                    duplicates.append((a, b))
            except Exception:
                pass
    return duplicates


def class_balance_report(y: pd.Series) -> pd.DataFrame:
    counts = y.astype(str).value_counts(dropna=False)
    out = counts.rename("Count").to_frame()
    out["Percent"] = out["Count"] / max(out["Count"].sum(), 1) * 100
    out.index.name = "Class"
    return out.reset_index()


def recommended_models(task: str, n_rows: int, n_features: int, full: bool = False) -> list[str]:
    if task == "classification":
        models = ["Logistic Regression", "Random Forest", "Extra Trees", "Gradient Boosting"]
        if XGBOOST_AVAILABLE:
            models.append("XGBoost")
        if LIGHTGBM_AVAILABLE:
            models.append("LightGBM")
        if n_rows <= 15000:
            models.append("Neural Network")
        if full:
            models += ["Naive Bayes", "Decision Tree", "KNN", "AdaBoost", "Hist Gradient Boosting"]
            if n_rows <= 10000:
                models.append("SVM")
    else:
        models = ["Linear Regression", "Ridge", "Random Forest", "Extra Trees", "Gradient Boosting"]
        if XGBOOST_AVAILABLE:
            models.append("XGBoost")
        if LIGHTGBM_AVAILABLE:
            models.append("LightGBM")
        if n_rows <= 15000:
            models.append("Neural Network")
        if full:
            models += ["Lasso", "ElasticNet", "Polynomial Regression", "Decision Tree", "KNN", "AdaBoost", "Hist Gradient Boosting"]
            if n_rows <= 10000:
                models.append("SVM")
    return list(dict.fromkeys(models))


def default_params_for_models(models: list[str], task: str) -> dict[str, dict[str, Any]]:
    out = {}
    for m in models:
        if m == "Neural Network":
            out[m] = {"_auto_architecture": True, "early_stopping": True, "max_iter": 700}
        elif m in {"Random Forest", "Extra Trees"}:
            out[m] = {"n_estimators": 250, "max_depth": None}
        elif m in {"Gradient Boosting", "AdaBoost"}:
            out[m] = {"n_estimators": 150, "learning_rate": 0.05}
        elif m == "XGBoost":
            out[m] = {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.05}
        elif m == "LightGBM":
            out[m] = {"n_estimators": 250, "num_leaves": 31, "learning_rate": 0.05}
        elif m == "Polynomial Regression":
            out[m] = {"degree": 2}
        else:
            out[m] = {}
    return out


def compute_permutation_importance(bundle: dict, X_test, y_test, scoring: str | None = None, n_repeats: int = 5, max_rows: int = 2000) -> pd.DataFrame:
    pipe = bundle["pipeline"]
    if len(X_test) > max_rows:
        rng = np.random.default_rng(RANDOM_STATE)
        positions = np.sort(rng.choice(len(X_test), size=max_rows, replace=False))
        sample = X_test.iloc[positions]
        y_sample = np.asarray(y_test)[positions]
    else:
        sample = X_test
        y_sample = y_test

    if scoring is None:
        scoring = "f1_weighted" if bundle["task"] == "classification" else "neg_root_mean_squared_error"

    result = permutation_importance(
        pipe, sample, y_sample, scoring=scoring, n_repeats=n_repeats,
        random_state=RANDOM_STATE, n_jobs=1
    )
    return pd.DataFrame({
        "Feature": sample.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std,
    }).sort_values("Importance_Mean", ascending=False).reset_index(drop=True)


def compute_learning_curve_data(bundle: dict, X_train, y_train, cv_folds: int = 5, points: int = 5) -> pd.DataFrame:
    task = bundle["task"]
    scoring = "f1_weighted" if task == "classification" else "neg_root_mean_squared_error"
    folds = safe_cv_folds(task, y_train, cv_folds)
    cv = StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE) if task == "classification" else KFold(folds, shuffle=True, random_state=RANDOM_STATE)
    sizes = np.linspace(0.25, 1.0, points)
    train_sizes, train_scores, valid_scores = learning_curve(
        copy.deepcopy(bundle["pipeline"]), X_train, y_train,
        cv=cv, scoring=scoring, train_sizes=sizes, n_jobs=1, shuffle=True, random_state=RANDOM_STATE
    )
    if task == "regression":
        train_scores, valid_scores = -train_scores, -valid_scores
    return pd.DataFrame({
        "Train_Size": train_sizes,
        "Train_Mean": train_scores.mean(axis=1),
        "Train_Std": train_scores.std(axis=1),
        "Validation_Mean": valid_scores.mean(axis=1),
        "Validation_Std": valid_scores.std(axis=1),
    })


def binary_threshold_table(bundle: dict, X_test, y_test, steps: int = 99) -> pd.DataFrame:
    pipe = bundle["pipeline"]
    if bundle["task"] != "classification" or not hasattr(pipe, "predict_proba"):
        raise ValueError("Threshold analysis requires a probabilistic classification model.")
    probs = pipe.predict_proba(X_test)
    if probs.shape[1] != 2:
        raise ValueError("Threshold tuning is available for binary classification only.")
    p = probs[:, 1]
    rows = []
    for threshold in np.linspace(0.01, 0.99, steps):
        pred = (p >= threshold).astype(int)
        rows.append({
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced_Accuracy": balanced_accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def binary_curve_data(bundle: dict, X_test, y_test) -> dict[str, pd.DataFrame]:
    pipe = bundle["pipeline"]
    probs = pipe.predict_proba(X_test)
    if probs.shape[1] != 2:
        raise ValueError("ROC/PR curve helper is for binary classification.")
    p = probs[:, 1]
    fpr, tpr, roc_thr = roc_curve(y_test, p)
    precision, recall, _ = precision_recall_curve(y_test, p)
    return {
        "roc": pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": roc_thr}),
        "pr": pd.DataFrame({"Recall": recall, "Precision": precision}),
    }


def winner_explanation(results: pd.DataFrame, best_model: str, task: str, metric: str, baseline_metrics: dict | None = None) -> list[str]:
    row = results.loc[results["Model"] == best_model].iloc[0]
    notes = [f"{best_model} ranked first on the selected held-out metric {metric}."]
    gap = row.get("Generalization_Gap", np.nan)
    if pd.notna(gap):
        if task == "classification":
            notes.append("Its train/test F1 gap is small, suggesting reasonable generalization." if abs(gap) < 0.05 else "Its train/test gap deserves attention for possible overfitting.")
        else:
            notes.append("Its train/test RMSE gap is part of the generalization check; compare it with competing models rather than test RMSE alone.")
    cv_col = "CV_F1_Weighted_Mean" if task == "classification" else "CV_RMSE_Mean"
    std_col = "CV_F1_Weighted_Std" if task == "classification" else "CV_RMSE_Std"
    if cv_col in row and pd.notna(row[cv_col]):
        notes.append(f"Cross-validation mean is {row[cv_col]:.4f} with std {row.get(std_col, np.nan):.4f}, showing stability across folds.")
    if "Fit_Seconds" in row and pd.notna(row["Fit_Seconds"]):
        notes.append(f"Training took about {row['Fit_Seconds']:.2f}s, so predictive quality can be weighed against compute cost.")
    if baseline_metrics:
        if task == "classification" and "F1_Weighted" in baseline_metrics:
            gain = row.get("Test_F1_Weighted", np.nan) - baseline_metrics["F1_Weighted"]
            if pd.notna(gain):
                notes.append(f"It improves Weighted F1 over the no-skill baseline by {gain:.4f}.")
        elif task == "regression" and "RMSE" in baseline_metrics:
            gain = baseline_metrics["RMSE"] - row.get("Test_RMSE", np.nan)
            if pd.notna(gain):
                notes.append(f"It reduces RMSE versus the mean baseline by {gain:.4f}.")
    return notes


def generate_html_report(
    dataset_name: str,
    df: pd.DataFrame,
    setup: dict,
    preprocessing: dict,
    results: pd.DataFrame,
    best_model: str,
    primary_metric: str,
    baseline_metrics: dict | None = None,
    leakage: pd.DataFrame | None = None,
) -> str:
    health = data_health_report(df, setup.get("target"))
    result_html = results.round(6).to_html(index=False, classes="table")
    leakage_html = "<p>No high-confidence leakage flags.</p>" if leakage is None or leakage.empty else leakage.to_html(index=False, classes="table")
    baseline_html = "<p>Not available.</p>" if not baseline_metrics else pd.DataFrame([baseline_metrics]).round(6).to_html(index=False, classes="table")
    prep_json = html.escape(json.dumps(preprocessing, indent=2, default=str))
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>AutoML Studio Final Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#172033}}
h1,h2{{color:#3f39c7}} .hero{{background:#f4f2ff;padding:24px;border-radius:18px}}
.kpis{{display:flex;gap:12px;flex-wrap:wrap}} .kpi{{padding:12px 18px;border:1px solid #ddd;border-radius:12px}}
.table{{border-collapse:collapse;width:100%;font-size:13px}} .table th,.table td{{border:1px solid #ddd;padding:7px;text-align:left}}
code{{background:#f1f3f7;padding:2px 5px;border-radius:4px}} pre{{white-space:pre-wrap;background:#f7f7fb;padding:16px;border-radius:10px}}
</style></head><body>
<div class='hero'><h1>AutoML Studio — Final Experiment Report</h1>
<p><b>Dataset:</b> {html.escape(dataset_name)} &nbsp; | &nbsp; <b>Task:</b> {html.escape(setup.get('task',''))} &nbsp; | &nbsp; <b>Target:</b> <code>{html.escape(setup.get('target',''))}</code></p>
<p><b>Best model:</b> {html.escape(best_model)} &nbsp; | &nbsp; <b>Primary metric:</b> {html.escape(primary_metric)}</p></div>
<h2>Dataset Health</h2><div class='kpis'>
<div class='kpi'><b>Rows</b><br>{len(df):,}</div><div class='kpi'><b>Columns</b><br>{df.shape[1]}</div>
<div class='kpi'><b>Health Score</b><br>{health['score']}/100</div><div class='kpi'><b>Missing</b><br>{health['missing_pct']:.2f}%</div>
<div class='kpi'><b>Duplicates</b><br>{health['duplicate_pct']:.2f}%</div></div>
<h2>Data Leakage Scan</h2>{leakage_html}
<h2>Preprocessing Configuration</h2><pre>{prep_json}</pre>
<h2>No-Skill Baseline</h2>{baseline_html}
<h2>Model Leaderboard</h2>{result_html}
<h2>Methodology</h2><p>All candidate models used the same held-out split and preprocessing policy. Learned preprocessing was fitted on training data. Cross-validation was performed on training data only. Final metrics are reported on the held-out test set.</p>
<h2>Limitations</h2><p>Automated checks cannot guarantee the absence of target leakage or distribution shift. Domain knowledge is required to verify prediction-time feature availability and the meaning of the selected target.</p>
</body></html>"""
