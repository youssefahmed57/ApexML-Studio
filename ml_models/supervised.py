"""
ml_models/supervised.py
Training and evaluation of all supervised classification and regression models.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except Exception:
    HAS_LGB = False


# ─── Classifier Registry ──────────────────────────────────────────────────────

def get_classifier_registry() -> dict:
    registry = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVC": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "ANN (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500,
            early_stopping=True, validation_fraction=0.1,
            random_state=42,
        ),
    }
    if HAS_XGB:
        registry["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    if HAS_LGB:
        registry["LightGBM"] = LGBMClassifier(
            n_estimators=200, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1,
        )
    return registry


def get_regressor_registry() -> dict:
    registry = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "Polynomial (deg 2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ]),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "SVR": SVR(kernel="rbf"),
        "KNN": KNeighborsRegressor(n_neighbors=7),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "ANN (MLP)": MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500,
            early_stopping=True, validation_fraction=0.1,
            random_state=42,
        ),
    }
    if HAS_XGB:
        registry["XGBoost"] = XGBRegressor(
            n_estimators=200, learning_rate=0.1,
            random_state=42, n_jobs=-1,
        )
    if HAS_LGB:
        registry["LightGBM"] = LGBMRegressor(
            n_estimators=200, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1,
        )
    return registry


# ─── Training ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_classifiers(
    X_train_key: str,  # Used as a cache discriminator
    y_train_key: str,
    selected_names: tuple,
    _X_train: np.ndarray,
    _y_train: np.ndarray,
) -> dict:
    """
    Train selected classifiers and return {name: fitted_model}.
    _X_train / _y_train are prefixed with _ to avoid hashing by st.cache_resource.
    Use selected_names (tuple) as the cache key discriminator.
    """
    registry = get_classifier_registry()
    trained = {}
    for name in selected_names:
        if name in registry:
            model = registry[name]
            model.fit(_X_train, _y_train)
            trained[name] = model
    return trained


@st.cache_resource(show_spinner=False)
def train_regressors(
    X_train_key: str,
    y_train_key: str,
    selected_names: tuple,
    _X_train: np.ndarray,
    _y_train: np.ndarray,
) -> dict:
    registry = get_regressor_registry()
    trained = {}
    for name in selected_names:
        if name in registry:
            model = registry[name]
            model.fit(_X_train, _y_train)
            trained[name] = model
    return trained


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_classifier(
    model, X_test: np.ndarray, y_test: np.ndarray, model_name: str
) -> dict:
    """Compute full classification metrics for a single model."""
    y_pred = model.predict(X_test)
    is_binary = len(np.unique(y_test)) == 2
    avg = "binary" if is_binary else "weighted"

    metrics = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
    }

    # ROC-AUC
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if is_binary:
                metrics["ROC-AUC"] = round(roc_auc_score(y_test, y_proba[:, 1]), 4)
            else:
                metrics["ROC-AUC"] = round(
                    roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"), 4
                )
        else:
            metrics["ROC-AUC"] = "N/A"
    except Exception:
        metrics["ROC-AUC"] = "N/A"

    return metrics


def evaluate_regressor(model, X_test: np.ndarray, y_test: np.ndarray,
                        model_name: str) -> dict:
    """Compute full regression metrics for a single model."""
    y_pred = model.predict(X_test)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    r2 = round(r2_score(y_test, y_pred), 4)
    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
    }


def get_confusion_matrix_data(model, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def get_roc_curve_data(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return FPR, TPR, and AUC for binary classifiers."""
    from sklearn.metrics import roc_curve
    if not hasattr(model, "predict_proba"):
        return None
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        return {"fpr": fpr, "tpr": tpr, "auc": round(auc, 4)}
    except Exception:
        return None
