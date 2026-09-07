"""
components/sidebar.py
Professional static sidebar with custom nav buttons and global CSS.
"""

import streamlit as st
from pathlib import Path

LOGO_PATH = str(Path(__file__).parent.parent / "assets" / "ApexML_Logo.png")

# Navigation: (icon, short label, page key that app.py uses for routing)
NAV_ITEMS = [
    ("🏠", "Home",                    "Home"),
    ("📂", "Dataset Overview",         "Dataset Overview & Validation"),
    ("🧹", "Smart Preprocessing",      "Smart Preprocessing & Cleaning"),
    ("🔬", "Feature Engineering & EDA","Feature Engineering & EDA"),
    ("🤖", "ML Studio",               "Machine Learning Studio"),
    ("🏆", "Evaluation & Leaderboard", "Model Evaluation & Leaderboard"),
    ("🔮", "Inference / Prediction",   "Inference / Prediction"),
]

# ─── CSS blocks ──────────────────────────────────────────────────────────────

SIDEBAR_CSS = """
<style>
/* ════ SIDEBAR BASE ════ */

/* ════ DIVIDER ════ */
.sb-divider {
    border: none;
    border-top: 1px solid rgba(15,23,42,0.1);
    margin: 12px 0;
}

/* ════ SECTION LABEL ════ */
.sb-section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(15,23,42,0.4) !important;
    margin-bottom: 8px;
}

/* ════ INACTIVE NAV BUTTONS — strict left-aligned ════ */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(15,23,42,0.05) !important;
    color: #0F172A !important;
    padding: 10px 1rem !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    box-shadow: none !important;
    transform: none !important;
    transition: all 0.2s ease !important;
    margin-bottom: 4px !important;
    display: flex !important;
    justify-content: flex-start !important;
}
/* Force Streamlit's inner text wrappers to left-align */
section[data-testid="stSidebar"] .stButton > button div[data-testid="stMarkdownContainer"] {
    width: 100% !important;
}
section[data-testid="stSidebar"] .stButton > button p {
    text-align: left !important;
    margin: 0 !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(15,23,42,0.04) !important;
    color: #0F172A !important;
    border-color: rgba(15,23,42,0.15) !important;
    border-left: 4px solid #10B981 !important;
}
section[data-testid="stSidebar"] a[data-testid="stPageLink"] {
    justify-content: flex-start !important;
    text-align: left !important;
    padding-left: 1rem !important;
}
</style>
"""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ════ HIDE STREAMLIT CHROME ════ */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stToolbar"]      { display: none !important; }
[data-testid="stDecoration"]   { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
.stDeployButton                { display: none !important; }
[data-testid="stSidebarNav"]   { display: none !important; }
/* Hide native sidebar collapse arrow — makes the sidebar permanently open */
[data-testid="collapsedControl"] { display: none !important; }

/* ════ FORCE SIDEBAR PERMANENTLY OPEN ════ */
section[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    transform: translateX(0) !important;
    min-width: 270px !important;
    width: 270px !important;
}
/* Ensure the main content doesn't overlap the forced sidebar on small screens */
section.main {
    margin-left: 270px !important;
}

/* ════ PAGE CONTAINER ════ */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ════ MAIN CONTENT BUTTONS (not sidebar) ════ */
.main [data-testid="stColumn"] .stButton > button,
.main .stButton > button:not([kind]) {
    background: linear-gradient(135deg, #10B981, #059669) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.main [data-testid="stColumn"] .stButton > button:hover,
.main .stButton > button:not([kind]):hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(16,185,129,0.4) !important;
}

/* ════ TABS ════ */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: rgba(15,23,42,0.06);
    border-radius: 8px;
    border: 1px solid rgba(15,23,42,0.1);
    padding: 8px 18px;
    font-weight: 500;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #10B981 !important;
    color: white !important;
    border-color: #10B981 !important;
}

/* ════ METRIC CARDS ════ */
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid rgba(15,23,42,0.08);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}
div[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 16px rgba(16,185,129,0.15);
}

/* ════ DATAFRAME ════ */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ════ ALERTS ════ */
.stAlert { border-radius: 10px; }

/* ════ HEADINGS ════ */
h1, h2, h3 { color: #0F172A !important; font-weight: 700 !important; }

/* ════ AI SUGGESTION BOX ════ */
.ai-suggestion-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.06), rgba(5,150,105,0.03));
    border-left: 4px solid #10B981;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 12px 0;
}
.ai-suggestion-box .ai-header {
    font-weight: 700;
    color: #059669 !important;
    font-size: 0.9rem;
    margin-bottom: 6px;
}

/* ════ BADGES ════ */
.badge-green {
    background: #10B981; color: white;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.75rem; font-weight: 600;
}
</style>
"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _logo_centered(path: str) -> None:
    """Render logo perfectly centered using native columns."""
    sc1, sc2, sc3 = st.columns([1, 4, 1])
    with sc2:
        if Path(path).exists():
            st.image(path, use_container_width=True)
        else:
            st.markdown(
                '<h1 style="color:#0F172A;font-size:2rem;font-weight:800;'
                'text-align:center;margin:0;">ApexML</h1>',
                unsafe_allow_html=True,
            )


def _active_nav_item(icon: str, label: str) -> None:
    """Render the currently active nav item — always left-aligned."""
    st.markdown(
        f'<div style="background:rgba(16,185,129,0.1); '
        f'border:1px solid rgba(16,185,129,0.2); border-left:4px solid #10B981; '
        f'border-radius:8px; padding:10px 1rem; margin-bottom:4px; font-size:0.9rem; '
        f'font-weight:700; color:#10B981 !important; letter-spacing:0.01em; '
        f'display:flex; align-items:center; justify-content:flex-start; text-align:left;">'
        f'<span style="margin-right:8px; flex-shrink:0;">{icon}</span>'
        f'<span style="color:#10B981 !important;">{label}</span></div>',
        unsafe_allow_html=True,
    )


# ─── Main render ──────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    """
    Render static sidebar.
    Returns the current page key.
    """
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # ── Session state defaults ────────────────────────────────────────────
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home"

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:

        # ── Logo ──────────────────────────────────────────────────────────
        _logo_centered(LOGO_PATH)

        # ── Slogan: always green ──────────────────────────────────────────
        st.sidebar.markdown(
            f"<p class='sb-slogan' style='text-align: center; color: #10B981; font-weight: bold; "
            f"font-style: italic; font-size: 13px; margin-top: -10px;'>"
            f"Elevating Data. Maximizing Metrics.</p>",
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────────────────
        st.markdown('<p class="sb-section-label">Navigation</p>', unsafe_allow_html=True)

        current = st.session_state["current_page"]
        for icon, label, page_key in NAV_ITEMS:
            if page_key == current:
                _active_nav_item(icon, label)
            else:
                if st.button(f"{icon}  {label}", key=f"nav_{page_key}",
                             use_container_width=True):
                    st.session_state["current_page"] = page_key
                    st.rerun()

    return st.session_state["current_page"]
