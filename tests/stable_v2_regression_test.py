"""Regression tests for bugs fixed in Stable v2.

This test mocks only the tiny Streamlit session_state surface used by utility modules,
so it can run even outside `streamlit run`.
"""
from pathlib import Path
import sys, types
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class SessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
    def __setattr__(self, key, value):
        self[key] = value

st = types.ModuleType("streamlit")
st.session_state = SessionState()

def _cache(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    return lambda f: f

st.cache_data = _cache
st.cache_resource = _cache
sys.modules["streamlit"] = st

from utils.preprocessing_utils import impute_column, encode_column
from utils.feature_engineering_utils import apply_scaler
from utils.eda_utils import detect_outliers_iqr
from utils.data_loader import load_dataset, get_dataset_metadata

def main():
    # Exact reported bug: key exists but contains None.
    st.session_state["preprocessing_log"] = None
    df = pd.DataFrame({"Age": [20.0, np.nan, 40.0]})
    out = impute_column(df, "Age", "mean")
    assert out["Age"].isna().sum() == 0
    assert isinstance(st.session_state["preprocessing_log"], list)

    # Legacy state repair.
    st.session_state["label_encoders"] = None
    encode_column(pd.DataFrame({"City": ["A", "B", "A"]}), "City", "label")
    assert isinstance(st.session_state["label_encoders"], dict)

    st.session_state["fitted_scalers"] = None
    apply_scaler(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), ["x"], "StandardScaler")
    assert isinstance(st.session_state["fitted_scalers"], dict)

    # Empty stats and delimiter loading.
    assert detect_outliers_iqr(pd.Series([np.nan, np.nan]))["outlier_pct"] == 0.0
    assert get_dataset_metadata(pd.DataFrame())["missing_pct"] == 0.0

    csv = b"a;b\n1;2\n3;4\n"
    loaded = load_dataset(csv, "semicolon.csv")
    assert loaded.shape == (2, 2)

    print("STABLE V2 REGRESSION TEST PASSED")

if __name__ == "__main__":
    main()
