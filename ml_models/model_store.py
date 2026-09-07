"""
ml_models/model_store.py
Manages saving, loading, and tracking of trained models in st.session_state.
Also provides joblib-based persistence.
"""

import os
import joblib
import streamlit as st
from pathlib import Path

_MODELS_DIR = Path("saved_models")


def save_model_to_state(name: str, model, task_type: str, metrics: dict):
    """
    Store a trained model and its metrics in st.session_state.
    """
    if not isinstance(st.session_state.get("trained_models"), dict):
        st.session_state["trained_models"] = {}
    st.session_state["trained_models"][name] = {
        "model": model,
        "task_type": task_type,
        "metrics": metrics,
    }


def get_all_models() -> dict:
    """Return all trained models from session state."""
    return st.session_state.get("trained_models", {})


def get_best_model(primary_metric: str = None) -> tuple:
    """
    Return (name, model_dict) of the best-performing model.
    
    For classification: sorts by F1-Score DESC.
    For regression: sorts by RMSE ASC.
    
    Returns (None, None) if no models are trained.
    """
    models = get_all_models()
    if not models:
        return None, None

    # Determine task type from first model
    first = next(iter(models.values()))
    task = first.get("task_type", "classification")

    if task == "regression":
        metric = primary_metric or "RMSE"
        best_name = min(
            models,
            key=lambda n: models[n]["metrics"].get(metric, float("inf"))
        )
    else:
        metric = primary_metric or "F1-Score"
        best_name = max(
            models,
            key=lambda n: models[n]["metrics"].get(metric, 0.0)
        )

    return best_name, models[best_name]


def export_model(name: str) -> str:
    """Save a model to disk using joblib. Returns the saved file path."""
    models = get_all_models()
    if name not in models:
        return ""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = _MODELS_DIR / f"{name.replace(' ', '_')}.joblib"
    joblib.dump(models[name]["model"], path)
    return str(path)


def load_model_from_disk(path: str):
    """Load a joblib model from disk."""
    return joblib.load(path)


def clear_models():
    """Clear all trained models from session state."""
    if "trained_models" in st.session_state:
        del st.session_state["trained_models"]
