from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.unsupervised_engine import (
    evaluate_clustering,
    cluster_size_table,
    cluster_profile_table,
    predict_or_transform,
)


def base_bundle(algorithm, features, scaler, train, X):
    return {
        "algorithm": algorithm,
        "feature_columns": features,
        "scaler": scaler,
        "fill_values": {c: float(train[c].median()) for c in features},
        "train_frame": train.reset_index(drop=True),
        "X_train_scaled": X,
    }


def main():
    df = pd.read_csv(ROOT / "datasets_tests" / "Mall_Customers.csv")
    features = [c for c in df.select_dtypes(include="number").columns if "CustomerID" not in c][:3]
    if len(features) < 2:
        features = df.select_dtypes(include="number").columns.tolist()[-2:]
    train = df[features].dropna().copy()
    scaler = StandardScaler().fit(train)
    X = scaler.transform(train)
    raw10 = train.head(10).copy()

    # K-Means: metrics + native prediction.
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
    km_metrics = evaluate_clustering(X, km.labels_, inertia=km.inertia_)
    assert km_metrics["Clusters"] == 3
    assert np.isfinite(km_metrics["Silhouette"])
    km_bundle = base_bundle("K-Means Clustering", features, scaler, train, X)
    km_bundle.update({"model": km, "labels": km.labels_})
    out = predict_or_transform(raw10, km_bundle)
    assert len(out) == 10 and "Cluster" in out and "Distance_to_Center" in out

    # Hierarchical: internal metrics + nearest-centroid assignment for new rows.
    hier = AgglomerativeClustering(n_clusters=3, linkage="ward").fit(X)
    hier_metrics = evaluate_clustering(X, hier.labels_)
    assert hier_metrics["Clusters"] == 3
    hier_bundle = base_bundle("Hierarchical Clustering", features, scaler, train, X)
    hier_bundle.update({"model": hier, "labels": hier.labels_})
    out = predict_or_transform(raw10, hier_bundle)
    assert len(out) == 10 and "Distance_to_Centroid" in out

    # DBSCAN: reusable core-point assignment always returns one label per new row.
    dbs = DBSCAN(eps=0.8, min_samples=4).fit(X)
    db_metrics = evaluate_clustering(X, dbs.labels_)
    assert db_metrics["Rows"] == len(train)
    db_bundle = base_bundle("DBSCAN", features, scaler, train, X)
    db_bundle.update({"model": dbs, "labels": dbs.labels_})
    out = predict_or_transform(raw10, db_bundle)
    assert len(out) == 10 and "Distance_to_Nearest_Core" in out

    # PCA: native out-of-sample transform.
    pca = PCA(n_components=2, random_state=42).fit(X)
    pca_bundle = base_bundle("PCA", features, scaler, train, X)
    pca_bundle.update({"model": pca})
    out = predict_or_transform(raw10, pca_bundle)
    assert out.shape == (10, 2) and list(out.columns) == ["PC1", "PC2"]

    # Cluster profile exports are aligned.
    sizes = cluster_size_table(km.labels_)
    profile = cluster_profile_table(train, km.labels_)
    assert sizes["Rows"].sum() == len(train)
    assert len(profile) == 3

    # UI integration: existing Evaluation and Inference pages must expose unsupervised modes.
    p5 = (ROOT / "pages" / "p5_evaluation.py").read_text(encoding="utf-8")
    p6 = (ROOT / "pages" / "p6_inference.py").read_text(encoding="utf-8")
    assert "Unsupervised Evaluation" in p5
    assert "Silhouette" in p5 and "Davies-Bouldin" in p5
    assert "Unsupervised Inference / Assignment" in p6
    assert "Exactly 10 Rows" in p6

    print("UNSUPERVISED EVALUATION + INFERENCE TEST PASSED")


if __name__ == "__main__":
    main()
