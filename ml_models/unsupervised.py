"""
ml_models/unsupervised.py
Unsupervised learning: Clustering, Dimensionality Reduction, and Association Rules.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")


# ─── K-Means ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_kmeans_elbow(
    _X: np.ndarray, k_range: tuple = (2, 11)
) -> tuple[list, list, list]:
    """Run K-Means across a safe K range and return inertia + silhouette scores."""
    X = np.asarray(_X)
    n_samples = len(X)
    if n_samples < 3:
        raise ValueError("K-Means comparison needs at least 3 complete rows.")

    k_start = max(2, int(k_range[0]))
    # Silhouette is undefined when n_clusters == n_samples.
    k_stop_exclusive = min(int(k_range[1]), n_samples)
    if k_start >= k_stop_exclusive:
        raise ValueError("Not enough rows for the requested K range.")

    k_values, inertias, sil_scores = [], [], []
    for k in range(k_start, k_stop_exclusive):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        k_values.append(k)
        inertias.append(float(km.inertia_))
        try:
            sil_scores.append(float(silhouette_score(X, labels)))
        except Exception:
            sil_scores.append(np.nan)
    return k_values, inertias, sil_scores

def plot_elbow(k_values: list, inertias: list, sil_scores: list) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=("Elbow Curve (Inertia)", "Silhouette Score"))
    fig.add_trace(go.Scatter(x=k_values, y=inertias, mode="lines+markers",
                              marker_color="#10B981", name="Inertia"), row=1, col=1)
    fig.add_trace(go.Scatter(x=k_values, y=sil_scores, mode="lines+markers",
                              marker_color="#0F172A", name="Silhouette"), row=1, col=2)
    fig.update_layout(template="plotly_white", height=350, showlegend=False)
    return fig


@st.cache_resource(show_spinner=False)
def fit_kmeans(_X: np.ndarray, n_clusters: int) -> KMeans:
    X = np.asarray(_X)
    if len(X) < 2:
        raise ValueError("K-Means needs at least two rows.")
    if not 2 <= int(n_clusters) <= len(X):
        raise ValueError(f"n_clusters must be between 2 and {len(X)}.")
    km = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
    km.fit(X)
    return km


# ─── Hierarchical Clustering ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def fit_hierarchical(_X: np.ndarray, n_clusters: int, linkage_method: str = "ward") -> AgglomerativeClustering:
    X = np.asarray(_X)
    if len(X) < 2:
        raise ValueError("Hierarchical clustering needs at least two rows.")
    if not 2 <= int(n_clusters) <= len(X):
        raise ValueError(f"n_clusters must be between 2 and {len(X)}.")
    model = AgglomerativeClustering(n_clusters=int(n_clusters), linkage=linkage_method)
    model.fit(X)
    return model


def plot_dendrogram(_X: np.ndarray, method: str = "ward") -> go.Figure:
    """Generate a dendrogram for hierarchical clustering (sampled to 200 pts)."""
    import scipy.cluster.hierarchy as sch
    sample = _X[:200] if len(_X) > 200 else _X
    Z = sch.linkage(sample, method=method)
    dn = sch.dendrogram(Z, no_plot=True)
    
    fig = go.Figure()
    for i, d in zip(dn["icoord"], dn["dcoord"]):
        fig.add_trace(go.Scatter(x=i, y=d, mode="lines",
                                  line=dict(color="#10B981", width=1),
                                  showlegend=False))
    fig.update_layout(
        title="Hierarchical Clustering Dendrogram",
        xaxis_title="Sample Index",
        yaxis_title="Distance",
        template="plotly_white",
        height=400,
    )
    return fig


# ─── DBSCAN ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def fit_dbscan(_X: np.ndarray, eps: float, min_samples: int) -> DBSCAN:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(_X)
    return model


# ─── PCA ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_pca(_X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    X = np.asarray(_X)
    if X.ndim != 2 or len(X) < 2 or X.shape[1] < 1:
        raise ValueError("PCA requires at least two rows and one feature.")
    max_components = min(X.shape[0], X.shape[1])
    n_components = max(1, min(int(n_components), max_components))
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


# ─── t-SNE ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_tsne(_X: np.ndarray, n_components: int = 2, perplexity: float = 30.0) -> np.ndarray:
    X = np.asarray(_X)
    if len(X) < 3:
        raise ValueError("t-SNE needs at least three complete rows.")
    safe_perplexity = min(float(perplexity), max(1.0, float(len(X) - 1)))
    if safe_perplexity >= len(X):
        safe_perplexity = max(1.0, len(X) - 1.0)
    tsne = TSNE(
        n_components=int(n_components),
        perplexity=safe_perplexity,
        random_state=42,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X)

# ─── Cluster Visualization ────────────────────────────────────────────────────

def plot_clusters_2d(X_2d: np.ndarray, labels: np.ndarray,
                      title: str = "Cluster Visualization") -> go.Figure:
    df_plot = pd.DataFrame({
        "x": X_2d[:, 0],
        "y": X_2d[:, 1],
        "Cluster": labels.astype(str),
    })
    fig = px.scatter(
        df_plot, x="x", y="y", color="Cluster",
        title=title,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.75,
        size_max=8,
    )
    fig.update_traces(marker=dict(size=6))
    return fig


# ─── Association Rules (Apriori) ─────────────────────────────────────────────

def run_apriori(df: pd.DataFrame, min_support: float = 0.05,
                 min_confidence: float = 0.5) -> pd.DataFrame:
    """
    Run the Apriori algorithm using mlxtend.
    Expects a boolean/binary DataFrame (transaction matrix).
    Returns a DataFrame of association rules.
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty:
            return pd.DataFrame()
        rules = association_rules(frequent_itemsets, metric="confidence",
                                   min_threshold=min_confidence)
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        return rules.sort_values("lift", ascending=False).reset_index(drop=True)
    except ImportError:
        return pd.DataFrame({"Error": ["mlxtend not installed. Run: pip install mlxtend"]})
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})
