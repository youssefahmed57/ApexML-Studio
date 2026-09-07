"""Utilities that make ApexML unsupervised workflows evaluable and reusable.

This module intentionally has no Streamlit dependency so clustering metrics and
out-of-sample assignment can be regression-tested independently of the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def evaluate_clustering(X_scaled: np.ndarray, labels: np.ndarray, inertia: float | None = None) -> dict[str, float | int]:
    """Evaluate a clustering result with internal metrics.

    DBSCAN noise points (label -1) are excluded from silhouette/DB/CH metrics,
    while their count/rate are reported separately.
    """
    X = np.asarray(X_scaled, dtype=float)
    lab = np.asarray(labels)
    if X.ndim != 2 or len(X) != len(lab):
        raise ValueError("X and labels must contain the same number of rows.")

    noise_mask = lab == -1
    valid_mask = ~noise_mask
    valid_labels = lab[valid_mask]
    valid_X = X[valid_mask]
    clusters = sorted(set(valid_labels.tolist()))
    n_clusters = len(clusters)

    out: dict[str, float | int] = {
        "Rows": int(len(lab)),
        "Clusters": int(n_clusters),
        "Noise_Points": int(noise_mask.sum()),
        "Noise_Rate": float(noise_mask.mean()) if len(lab) else 0.0,
        "Silhouette": np.nan,
        "Davies_Bouldin": np.nan,
        "Calinski_Harabasz": np.nan,
    }
    if inertia is not None:
        out["Inertia"] = float(inertia)

    # These metrics need at least 2 clusters and more samples than clusters.
    if n_clusters >= 2 and len(valid_X) > n_clusters:
        try:
            out["Silhouette"] = float(silhouette_score(valid_X, valid_labels))
        except Exception:
            pass
        try:
            out["Davies_Bouldin"] = float(davies_bouldin_score(valid_X, valid_labels))
        except Exception:
            pass
        try:
            out["Calinski_Harabasz"] = float(calinski_harabasz_score(valid_X, valid_labels))
        except Exception:
            pass
    return out


def cluster_size_table(labels: np.ndarray) -> pd.DataFrame:
    lab = pd.Series(np.asarray(labels), name="Cluster")
    counts = lab.value_counts(dropna=False).sort_index()
    total = max(1, int(counts.sum()))
    rows = []
    for cluster, count in counts.items():
        name = "Noise (-1)" if cluster == -1 else str(cluster)
        rows.append({"Cluster": name, "Rows": int(count), "Share": float(count / total)})
    return pd.DataFrame(rows)


def cluster_profile_table(train_frame: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Return mean feature profile for every discovered cluster/noise group."""
    frame = train_frame.reset_index(drop=True).copy()
    labels = np.asarray(labels)
    if len(frame) != len(labels):
        raise ValueError("Training frame and labels have different row counts.")
    frame.insert(0, "Cluster", labels)
    numeric = [c for c in frame.columns if c != "Cluster" and pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric:
        return pd.DataFrame()
    return frame.groupby("Cluster", dropna=False)[numeric].mean().reset_index()


def _required_frame(raw: pd.DataFrame, bundle: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    features = list(bundle.get("feature_columns") or [])
    if not features:
        raise ValueError("The unsupervised bundle does not contain a feature schema.")
    missing = [c for c in features if c not in raw.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))

    frame = raw[features].copy()
    for col in features:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    fill_values = bundle.get("fill_values") or {}
    for col in features:
        fill = fill_values.get(col)
        if fill is None or not np.isfinite(float(fill)):
            fill = 0.0
        frame[col] = frame[col].fillna(float(fill))
    return frame, features


def transform_new_rows(raw: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    frame, _ = _required_frame(raw, bundle)
    scaler = bundle.get("scaler")
    if scaler is None:
        raise ValueError("The unsupervised bundle does not contain its fitted scaler.")
    return np.asarray(scaler.transform(frame), dtype=float)


def _nearest_centroid_assignment(bundle: dict[str, Any], X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(bundle.get("labels"))
    X_train = np.asarray(bundle.get("X_train_scaled"), dtype=float)
    if len(labels) != len(X_train):
        raise ValueError("Hierarchical bundle is missing consistent training labels/features.")
    cluster_ids = np.array(sorted(c for c in set(labels.tolist()) if c != -1))
    if len(cluster_ids) == 0:
        raise ValueError("No non-noise clusters are available for assignment.")
    centroids = np.vstack([X_train[labels == c].mean(axis=0) for c in cluster_ids])
    dist = np.linalg.norm(X_new[:, None, :] - centroids[None, :, :], axis=2)
    best = dist.argmin(axis=1)
    return cluster_ids[best], dist[np.arange(len(X_new)), best]


def _dbscan_assignment(bundle: dict[str, Any], X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = bundle.get("model")
    if model is None:
        raise ValueError("DBSCAN bundle is missing the fitted model.")
    core_idx = np.asarray(getattr(model, "core_sample_indices_", []), dtype=int)
    components = np.asarray(getattr(model, "components_", []), dtype=float)
    if len(core_idx) == 0 or components.size == 0:
        return np.full(len(X_new), -1, dtype=int), np.full(len(X_new), np.nan)

    train_labels = np.asarray(model.labels_)
    core_labels = train_labels[core_idx]
    dist = np.linalg.norm(X_new[:, None, :] - components[None, :, :], axis=2)
    best = dist.argmin(axis=1)
    best_dist = dist[np.arange(len(X_new)), best]
    pred = np.where(best_dist <= float(model.eps), core_labels[best], -1)
    return pred.astype(int), best_dist


def predict_or_transform(raw: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Apply a fitted unsupervised object to new rows where the algorithm permits it.

    K-Means: native cluster prediction.
    Hierarchical: nearest-centroid assignment (approximation; sklearn agglomerative has no predict).
    DBSCAN: nearest fitted core-point assignment inside eps; otherwise noise (-1).
    PCA: native transform into principal components.
    t-SNE: intentionally unsupported for out-of-sample rows.
    """
    algorithm = str(bundle.get("algorithm", ""))
    X_new = transform_new_rows(raw, bundle)

    if algorithm == "K-Means Clustering":
        model = bundle.get("model")
        labels = np.asarray(model.predict(X_new), dtype=int)
        distances = np.asarray(model.transform(X_new), dtype=float)
        best_dist = distances[np.arange(len(X_new)), labels]
        return pd.DataFrame({"Cluster": labels, "Distance_to_Center": best_dist})

    if algorithm == "Hierarchical Clustering":
        labels, distances = _nearest_centroid_assignment(bundle, X_new)
        return pd.DataFrame({"Cluster": labels, "Distance_to_Centroid": distances})

    if algorithm == "DBSCAN":
        labels, distances = _dbscan_assignment(bundle, X_new)
        return pd.DataFrame({"Cluster": labels, "Distance_to_Nearest_Core": distances})

    if algorithm == "PCA":
        model = bundle.get("model")
        if model is None:
            raise ValueError("PCA bundle is missing the fitted PCA model.")
        projected = np.asarray(model.transform(X_new), dtype=float)
        return pd.DataFrame(projected, columns=[f"PC{i+1}" for i in range(projected.shape[1])])

    if algorithm == "t-SNE":
        raise ValueError(
            "scikit-learn t-SNE does not provide an out-of-sample transform for new rows. "
            "Use PCA when you need reusable projection of future cases."
        )

    raise ValueError(f"Inference is not defined for unsupervised mode: {algorithm}")
