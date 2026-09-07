"""
components/ai_suggestion.py
Reusable AI Suggestion panel that calls Gemini and renders a styled suggestion box.
"""

import streamlit as st
from utils.llm_utils import get_ai_suggestion


def render_ai_suggestion(context: str, title: str = "🤖 AI Smart Suggestion",
                          expanded: bool = True, key: str = None):
    """
    Render a styled AI suggestion block.
    
    Args:
        context: The full user-context string to send to Gemini.
        title: The expander/header title.
        expanded: Whether to expand by default.
        key: Unique key for the Streamlit button.
    """
    with st.expander(title, expanded=expanded):
        col1, col2 = st.columns([5, 1])
        with col2:
            btn_key = key or f"ai_btn_{hash(context)}"
            refresh = st.button("🔄 Refresh", key=btn_key, help="Regenerate AI suggestion")

        # Cache key in session to avoid re-calling on every render
        cache_key = f"ai_cache_{hash(context)}"
        if cache_key not in st.session_state or refresh:
            with st.spinner("🧠 Generating AI suggestion..."):
                suggestion = get_ai_suggestion(context)
            st.session_state[cache_key] = suggestion
        else:
            suggestion = st.session_state[cache_key]

        # Render header HTML
        st.markdown(
            """
            <div class="ai-suggestion-box">
                <div class="ai-header">🤖 Gemini AI Recommendation</div>
            """,
            unsafe_allow_html=True,
        )
        
        # Render the suggestion text via native markdown so formatting (bold, lists) works
        st.markdown(f"<div style='font-size: 0.9rem; line-height: 1.6; color: #0F172A;'>\n\n{suggestion}\n\n</div>", unsafe_allow_html=True)
        
        # Close the suggestion box div
        st.markdown("</div>", unsafe_allow_html=True)


def render_inline_suggestion(text: str):
    """Render a compact inline AI suggestion (no expander)."""
    st.markdown(
        f"""
        <div class="ai-suggestion-box">
            <div class="ai-header">🤖 AI Suggestion</div>
            <div style="font-size: 0.88rem; line-height: 1.55;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
