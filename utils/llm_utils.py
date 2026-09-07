"""
utils/llm_utils.py
Wraps the Google Gemini API for AI suggestion features across all pages.
Uses the current google.genai SDK (v2+).
Loads the two workflow docs as system-level context so suggestions adhere to standard ML rules.
"""

import os
import streamlit as st
from pathlib import Path

# Load workflow docs once at module level
_DOCS_DIR = Path(__file__).parent.parent / "docs"


def _load_workflow_docs() -> str:
    """Load markdown workflow docs to inject as LLM context."""
    docs_text = ""
    for fname in ["data-analysis-workflow.md", "ml-pipeline-workflow.md"]:
        fpath = _DOCS_DIR / fname
        if fpath.exists():
            docs_text += f"\n\n# {fname}\n" + fpath.read_text(encoding="utf-8")
    return docs_text


_WORKFLOW_CONTEXT = _load_workflow_docs()

_SYSTEM_PROMPT = f"""You are an expert Machine Learning and Data Science assistant embedded in the ApexML dashboard.
Your role is to give SMART, CONCISE suggestions with clear JUSTIFICATIONS at every step of the ML pipeline.

You must ALWAYS:
1. Base your suggestions on the actual column statistics provided (mean, std, skewness, missing %, cardinality, etc.).
2. Provide a PRIMARY recommendation with a clear reason (1-2 sentences).
3. Provide one ALTERNATIVE approach if the primary has trade-offs (1 sentence).
4. Keep your total response under 150 words unless asked for more detail.
5. Use domain knowledge when column names are recognizable (e.g., 'Age', 'Salary', 'Gender').
6. Follow the rules defined in the workflow documents below STRICTLY.

WORKFLOW REFERENCE DOCUMENTS:
{_WORKFLOW_CONTEXT}
"""


def get_all_api_keys():
    """Retrieve all available API keys from environment variables or Streamlit secrets."""
    from dotenv import load_dotenv
    load_dotenv()
    
    keys = []
    
    # Check for comma-separated list in GEMINI_API_KEY
    main_key_env = os.environ.get("GEMINI_API_KEY", "")
    if main_key_env and main_key_env != "your_gemini_api_key_here":
        for k in main_key_env.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
                
    # Also check GEMINI_API_KEY_1 through GEMINI_API_KEY_10
    for i in range(1, 11):
        k = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip()
        if k and k != "your_gemini_api_key_here" and k not in keys:
            keys.append(k)

    # Check Streamlit Cloud secrets (st.secrets)
    try:
        if "GEMINI_API_KEY" in st.secrets:
            val = str(st.secrets["GEMINI_API_KEY"]).strip()
            if val and val != "your_gemini_api_key_here":
                for k in val.split(","):
                    k = k.strip()
                    if k and k not in keys:
                        keys.append(k)
        for i in range(1, 11):
            key_name = f"GEMINI_API_KEY_{i}"
            if key_name in st.secrets:
                k = str(st.secrets[key_name]).strip()
                if k and k != "your_gemini_api_key_here" and k not in keys:
                    keys.append(k)
    except Exception:
        pass
            
    return keys

# Global index to track which key we are currently using
_current_key_index = 0

def get_gemini_client(api_key):
    """Initialize and return a google.genai Client (SDK v2+) for a specific key."""
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_ai_suggestion(user_context: str) -> str:
    """
    Call Gemini API with user_context and return the AI suggestion string.
    Implements automatic fallback across multiple API keys if quota/limits are hit.
    """
    global _current_key_index
    keys = get_all_api_keys()
    
    if not keys:
        return (
            "⚠️ **AI Suggestions Unavailable** — Please add your `GEMINI_API_KEY` (or multiple keys like `GEMINI_API_KEY_1`) "
            "to the `.env` file in the project root to enable this feature."
        )

    from google.genai import types
    
    last_error = None
    num_keys = len(keys)
    
    # Try keys one by one, starting from the current index
    for _ in range(num_keys):
        current_key = keys[_current_key_index]
        client = get_gemini_client(current_key)
        
        if client:
            try:
                gemini_model = "gemini-2.5-flash"
                try:
                    if "GEMINI_MODEL" in st.secrets:
                        gemini_model = str(st.secrets["GEMINI_MODEL"]).strip()
                except Exception:
                    pass
                gemini_model = os.environ.get("GEMINI_MODEL", gemini_model)

                response = client.models.generate_content(
                    model=gemini_model,
                    contents=user_context,
                    config=types.GenerateContentConfig(
                        system_instruction=_SYSTEM_PROMPT,
                        temperature=0.3,
                    ),
                )
                return response.text
            except Exception as e:
                last_error = str(e)
                # Rotate to the next key if an error occurs (e.g., 429 Too Many Requests, Quota Exceeded)
                _current_key_index = (_current_key_index + 1) % num_keys
                continue
        else:
            # If client couldn't be initialized, still rotate
            _current_key_index = (_current_key_index + 1) % num_keys
            continue
            
    return f"⚠️ AI suggestion error (All {num_keys} keys exhausted). Last error: {last_error}"



def build_preprocessing_context(col_name: str, dtype: str, missing_pct: float,
                                 unique_count: int, skewness, mean, std) -> str:
    """Build the user context string for a preprocessing suggestion."""
    return f"""
I need preprocessing suggestions for this column:
- Column Name: '{col_name}'
- Data Type: {dtype}
- Missing Values: {missing_pct}%
- Unique Values: {unique_count}
- Skewness: {skewness}
- Mean: {mean}
- Std Dev: {std}

Please suggest:
1. The best MISSING VALUE imputation strategy (if missing > 0%).
2. The best ENCODING strategy (if categorical).
3. Any OUTLIER treatment recommendation (if numerical).
Justify each recommendation with the column's statistics.
"""


def build_scaler_context(col_name: str, skewness: float, has_outliers: bool,
                          target_models: list) -> str:
    """Build context for scaler/transformation suggestion."""
    return f"""
I need a scaling/transformation suggestion for the feature '{col_name}':
- Skewness: {skewness:.3f}
- Has significant outliers (IQR test): {has_outliers}
- Target ML models planned: {', '.join(target_models) if target_models else 'Not specified'}

Please recommend the best scaling/transformation strategy from: StandardScaler, MinMaxScaler, RobustScaler, Log Transform, or No Scaling.
Justify your recommendation.
"""


def build_algorithm_context(task_type: str, n_rows: int, n_features: int,
                              target_distribution: str, has_imbalance: bool,
                              correlation_summary: str) -> str:
    """Build context for algorithm recommendation."""
    return f"""
I need algorithm recommendations for an ML task:
- Task Type: {task_type}
- Dataset: {n_rows} rows × {n_features} features
- Target Distribution: {target_distribution}
- Class Imbalance Detected: {has_imbalance}
- Feature Correlation Summary: {correlation_summary}

Please recommend the TOP 2 algorithms for this task, explaining WHY each is suitable.
Also state the PRIMARY evaluation metric I should focus on.
Flag any risks or caveats.
"""


def build_eda_context(col_name: str, dtype: str, stats: dict) -> str:
    """Build context for EDA insight suggestion."""
    return f"""
Provide a brief data insight for the column '{col_name}' (type: {dtype}):
Statistics: {stats}

In 2-3 sentences, describe what these statistics reveal about the data and flag any potential issues 
(outliers, skew, high cardinality, etc.) that the analyst should address.
"""
