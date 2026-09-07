"""
app.py — ApexML Main Entry Point
Renders the sidebar and routes to the selected page.
All pages share state via st.session_state.
"""

import streamlit as st

# ── Page Config (must be the FIRST Streamlit command) ─────────────────────────
st.set_page_config(
    page_title="ApexML — Elevating Data. Maximizing Metrics.",
    page_icon="🔺",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "**ApexML** v1.0 | Enterprise-Grade ML Dashboard | Powered by Gemini AI",
    },
)

# ── Imports ───────────────────────────────────────────────────────────────────
from components.sidebar import render_sidebar
import pages.p1_dataset_overview as p1
import pages.p2_preprocessing as p2
import pages.p3_feature_engineering as p3
import pages.p4_ml_studio as p4
import pages.p5_evaluation as p5
import pages.p6_inference as p6


# ── Session State Initialization ──────────────────────────────────────────────
def init_session_state():
    defaults = {
        "df": None,
        "df_original": None,
        "file_name": None,
        "file_signature": None,
        "preprocessing_log": [],
        "label_encoders": {},
        "ordinal_encoders": {},
        "fitted_scalers": {},
        "trained_models": {},
        "train_test_data": None,
        "eval_results": [],
        "eval_task_type": "classification",
        "selected_models": [],
        "task_type": "Classification",
        "target_col": None,
        "sidebar_open": True,
        # Final-project ML state
        "dataset_setup": None,
        "pipeline_config": None,
        "feature_recipe": [],
        "model_params": {},
        "selected_models": [],
        "training_mode": "Standard Benchmark",
        "primary_metric": None,
        "model_results": None,
        "model_bundles": {},
        "model_details": {},
        "split_data_safe": None,
        "final_bundle": None,
        "competition_run": None,
        "predictions_10": None,
        "last_doctor_score": None,
        # Reusable unsupervised evaluation/inference state
        "unsup_bundles": {},
        "unsup_results": [],
        "unsup_active_run": None,
    }
    import copy
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(val)


def render_home():
    """Home / landing page."""
    import os
    from pathlib import Path
    logo = Path("assets/ApexML_Logo.png")

    # ── Hero Header: Logo centred, slogan below ──────────────────────────────
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        if logo.exists():
            st.image(str(logo), use_container_width=True)
        else:
            st.markdown('<h1 style="color:#0F172A;font-size:2.5rem;font-weight:800;text-align:center;margin:0;">ApexML</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #10B981; font-style: italic; font-weight: bold;'>Elevating Data. Maximizing Metrics.</p>", unsafe_allow_html=True)


    st.markdown("---")

    # Feature overview cards
    features = [
        ("📂", "Dataset Overview", "Upload any CSV/Excel. Get instant metadata, quality report, and column statistics."),
        ("🧹", "Smart Preprocessing", "Handle missing values, duplicates, outliers, and encoding — with AI suggestions."),
        ("🔬", "Feature Engineering", "Univariate, bivariate, and multivariate EDA with interactive Plotly charts."),
        ("🤖", "ML Studio", "Train and compare many models, control hyperparameters, or use Competition Boost for the strongest final predictor."),
        ("🏆", "Evaluation Leaderboard", "Compare supervised models or evaluate unsupervised clusters with Silhouette, Davies-Bouldin, profiles, and saved runs."),
        ("🔮", "Inference", "Single, 10-row, and batch inference for supervised models plus supported unsupervised cluster assignment/PCA transform."),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="background: white; border: 1px solid rgba(15,23,42,0.08);
                     border-radius: 14px; padding: 20px; margin-bottom: 16px;
                     box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                     transition: box-shadow 0.2s;
                     border-top: 3px solid #10B981;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
                    <div style="font-weight: 700; color: #0F172A; font-size: 1rem;
                         margin-bottom: 6px;">{title}</div>
                    <div style="color: #64748B; font-size: 0.87rem; line-height: 1.5;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🚀 Getting Started**
        1. Upload your dataset (📂 Dataset Overview)
        2. Clean & encode data (🧹 Preprocessing)
        3. Explore features (🔬 EDA)
        4. Train models (🤖 ML Studio)
        5. Compare results (🏆 Evaluation)
        6. Predict new data (🔮 Inference)
        """)
    with col2:
        st.markdown("""
        **🤖 AI-Powered Features**
        - Smart imputation strategy suggestions
        - Encoding method recommendations
        - Scaler/transform advice
        - Top-2 algorithm recommendations with justification
        - Column-level data insights
        """)
    with col3:
        st.markdown("""
        **📊 Supported Algorithms**
        - **Classification:** LR, NB, DT, RF, SVC, KNN, XGB, LGBM, AdaBoost, ANN
        - **Regression:** Linear, Poly, Ridge, Lasso, DT, RF, SVR, KNN, XGB, LGBM, AdaBoost, ANN
        - **Clustering:** K-Means, Hierarchical, DBSCAN
        - **Dim. Reduction:** PCA, t-SNE
        - **Association:** Apriori
        """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color: #94A3B8; font-size: 0.82rem;'>"
        "ApexML v1.0 · Built for ML Excellence"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Page Router ───────────────────────────────────────────────────────────────
def main():
    init_session_state()
    
    # ── Dark Mode Toggle (Top Right) ───────────────────────────────
    if "theme" not in st.session_state:
        st.session_state["theme"] = "Light"
        
    top_col1, top_col2 = st.columns([9, 1])
    with top_col2:
        theme_toggle = st.toggle("🌙 Dark", value=(st.session_state["theme"] == "Dark"), key="theme_switcher")
        st.session_state["theme"] = "Dark" if theme_toggle else "Light"

    if st.session_state["theme"] == "Dark":
        st.markdown("""
        <style>
        .stApp, .main, section[data-testid="stSidebar"], [data-testid="stHeader"] {
            background-color: #0F172A !important;
        }
        /* Fix cards to dark slate */
        div[data-testid="metric-container"], .stTabs [data-baseweb="tab"], 
        div[style*="background: white"] {
            background-color: #1E293B !important;
            border-color: rgba(255,255,255,0.1) !important;
        }
        h1, h2, h3, p, span, div, label, .stMarkdown, .stText {
            color: #FFFFFF !important;
        }
        /* Dataframes */
        .stDataFrame * {
            color: #FFFFFF !important;
        }
        /* Specific exceptions */
        .apex-slogan, section[data-testid="stSidebar"] .stButton > button:hover {
            color: #10B981 !important;
        }
        .status-ready { color: #6EE7B7 !important; background: rgba(16,185,129,0.2) !important; }
        .status-empty { color: #FCA5A5 !important; background: rgba(239,68,68,0.2) !important; }
        .sb-section-label { color: rgba(255,255,255,0.4) !important; }
        section[data-testid="stSidebar"] .stButton > button {
            color: #FFFFFF !important;
            border-color: rgba(255,255,255,0.1) !important;
        }
        .sb-divider { border-top-color: rgba(255,255,255,0.1) !important; }
        /* Force sidebar text white in dark mode */
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span { color: #FFFFFF !important; }
        /* Except the slogan, which must remain green */
        section[data-testid="stSidebar"] p.sb-slogan { color: #10B981 !important; }
        /* Remove white box from columns in dark mode */
        div[data-testid="stColumn"] {
            background: transparent !important;
        }
        /* Invert logo in dark mode */
        img { filter: brightness(0) invert(1) !important; }
        </style>
        """, unsafe_allow_html=True)

    page = render_sidebar()

    if page == "Home":
        render_home()
    elif page == "Dataset Overview & Validation":
        p1.render()
    elif page == "Smart Preprocessing & Cleaning":
        p2.render()
    elif page == "Feature Engineering & EDA":
        p3.render()
    elif page == "Machine Learning Studio":
        p4.render()
    elif page == "Model Evaluation & Leaderboard":
        p5.render()
    elif page == "Inference / Prediction":
        p6.render()


if __name__ == "__main__":
    main()
