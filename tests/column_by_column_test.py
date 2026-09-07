from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# preprocessing_utils only needs st.session_state for these unit checks.
fake_streamlit = types.ModuleType("streamlit")
fake_streamlit.session_state = {}
sys.modules.setdefault("streamlit", fake_streamlit)

import streamlit as st
from utils.preprocessing_utils import impute_column, encode_column, treat_outliers


def main():
    st.session_state["preprocessing_log"] = []
    st.session_state["label_encoders"] = {}
    st.session_state["ordinal_encoders"] = {}

    # Mean imputation must alter only the chosen column.
    df = pd.DataFrame({
        "Age": [20.0, np.nan, 40.0, 60.0],
        "Income": [100.0, np.nan, 300.0, 400.0],
        "City": ["A", "B", "A", "C"],
    })
    before_income = df["Income"].copy()
    out = impute_column(df, "Age", "mean")
    assert out["Age"].isna().sum() == 0
    pd.testing.assert_series_equal(out["Income"], before_income, check_names=True)

    # KNN may USE other numeric columns, but it must write only to Age.
    before_income = df["Income"].copy()
    out_knn = impute_column(df, "Age", "knn", knn_neighbors=2)
    assert out_knn["Age"].isna().sum() == 0
    pd.testing.assert_series_equal(out_knn["Income"], before_income, check_names=True)

    # Encoding one categorical column must leave the other original columns intact.
    enc_df = pd.DataFrame({"City": ["A", "B", "A"], "Score": [1, 2, 3]})
    encoded = encode_column(enc_df, "City", "onehot")
    assert "Score" in encoded.columns
    assert encoded["Score"].tolist() == [1, 2, 3]
    assert any(c.startswith("City_") for c in encoded.columns)

    # Outlier cap is selected-column only.
    out_df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 100.0], "B": [10.0, 20.0, 30.0, 40.0]})
    before_b = out_df["B"].copy()
    capped = treat_outliers(out_df, "A", "cap")
    pd.testing.assert_series_equal(capped["B"], before_b, check_names=True)

    # UI source must not expose bulk-cleaning controls or cleaning-strategy CV.
    p2 = (ROOT / "pages" / "p2_preprocessing.py").read_text(encoding="utf-8")
    forbidden = [
        "Quick Encode All Remaining",
        "Encode All Categorical Columns",
        "Apply to All",
        "Compare Cleaning Strategies",
    ]
    for text in forbidden:
        assert text not in p2, f"Forbidden bulk/auto-cleaning UI remains: {text}"

    # Explicit user-facing promise should be present.
    assert "Column-by-Column" in p2
    assert "Cross-Validation is reserved for comparing **models**" in p2

    print("COLUMN-BY-COLUMN TEST PASSED")


if __name__ == "__main__":
    main()
