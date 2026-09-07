"""
pages/2_preprocessing.py
Page 2: Smart Preprocessing & Data Cleaning with AI Suggestions.
"""

import streamlit as st
import pandas as pd
from utils.preprocessing_utils import (
    impute_column, remove_duplicates, treat_outliers, encode_column,
    get_dtype_recommendation,
)
from utils.eda_utils import detect_all_outliers, plot_box
from utils.data_loader import get_column_statistics
from utils.llm_utils import build_preprocessing_context
from components.ai_suggestion import render_ai_suggestion


def _require_df():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ Please upload a dataset on the **Dataset Overview** page first.")
        st.stop()
    return st.session_state["df"]


def render():
    st.title("🧹 Smart Preprocessing & Data Cleaning")
    st.markdown(
        "Clean, encode, and transform your data **one column at a time**, just like the original ApexML workflow."
    )
    st.info(
        "🎯 Manual cleaning here is strictly **Column-by-Column**. "
        "There is no automatic Compare-Cleaning-Strategies step and no Apply-to-All control. "
        "Cross-Validation is reserved for comparing **models** in ML Studio, not for choosing a cleaning method."
    )

    df = _require_df()

    tab_missing, tab_dupes, tab_outliers, tab_encoding, tab_pipeline, tab_log = st.tabs([
        "❓ Missing Values", "🔁 Duplicates", "📦 Outliers",
        "🔠 Encoding", "🛡️ Model Pipeline", "📋 Preprocessing Log"
    ])

    # ── Missing Values ─────────────────────────────────────────────────────────
    with tab_missing:
        df = st.session_state["df"]
        missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

        if not missing_cols:
            st.success("✅ No missing values in the current dataset!")
        else:
            st.info(f"Found **{len(missing_cols)}** column(s) with missing values.")
            col_stats = get_column_statistics(df)

            selected_col = st.selectbox(
                "Select column to handle", missing_cols, key="miss_col"
            )
            row = col_stats[col_stats["Column"] == selected_col]
            missing_pct = float(row["Missing %"].values[0]) if not row.empty else 0
            is_numeric = pd.api.types.is_numeric_dtype(df[selected_col])

            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(f"**Column:** `{selected_col}` | **Missing:** `{missing_pct}%` | **Type:** `{df[selected_col].dtype}`")

                if is_numeric:
                    strategy = st.selectbox(
                        "Imputation strategy",
                        ["mean", "median", "constant", "knn", "drop_rows", "drop_col"],
                        key="miss_strategy",
                    )
                else:
                    strategy = st.selectbox(
                        "Imputation strategy",
                        ["mode", "constant", "drop_rows", "drop_col"],
                        key="miss_strategy_cat",
                    )

                fill_val = None
                if strategy == "constant":
                    if is_numeric:
                        numeric_series = pd.to_numeric(df[selected_col], errors="coerce")
                        default_fill = float(numeric_series.median()) if numeric_series.notna().any() else 0.0
                        fill_val = st.number_input(
                            "Fill value", value=default_fill, key="miss_fill_numeric"
                        )
                    else:
                        fill_val = st.text_input(
                            "Fill value", value="Unknown", key="miss_fill"
                        )

                if st.button("✅ Apply Imputation", key="btn_impute"):
                    try:
                        st.session_state["df"] = impute_column(
                            st.session_state["df"], selected_col,
                            strategy=strategy, fill_value=fill_val,
                        )
                        st.success(f"Applied **{strategy}** imputation to `{selected_col}`.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not apply imputation: {exc}")

            with col2:
                # AI Suggestion
                skew_val = row["Skewness"].values[0] if not row.empty else "—"
                mean_val = row["Mean"].values[0] if not row.empty else "—"
                std_val = row["Std Dev"].values[0] if not row.empty else "—"
                unique_val = int(row["Unique"].values[0]) if not row.empty else 0
                context = build_preprocessing_context(
                    col_name=selected_col,
                    dtype=str(df[selected_col].dtype),
                    missing_pct=missing_pct,
                    unique_count=unique_val,
                    skewness=skew_val,
                    mean=mean_val,
                    std=std_val,
                )
                render_ai_suggestion(context, title="🤖 AI Imputation Suggestion",
                                      key=f"ai_miss_{selected_col}")

    # ── Duplicates ─────────────────────────────────────────────────────────────
    with tab_dupes:
        df = st.session_state["df"]
        n_dupes = df.duplicated().sum()

        if n_dupes == 0:
            st.success("✅ No duplicate rows found!")
        else:
            st.warning(f"⚠️ Found **{n_dupes}** duplicate rows ({n_dupes/len(df)*100:.1f}% of dataset).")

            with st.expander("👁️ Preview duplicate rows", expanded=False):
                st.dataframe(df[df.duplicated(keep="first")].head(20), use_container_width=True)

            if st.button("🗑️ Remove All Duplicates", key="btn_dupes"):
                st.session_state["df"] = remove_duplicates(st.session_state["df"])
                st.success("✅ Duplicates removed!")
                st.rerun()

    # ── Outliers ───────────────────────────────────────────────────────────────
    with tab_outliers:
        df = st.session_state["df"]
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not num_cols:
            st.info("No numerical columns available for outlier analysis.")
        else:
            st.subheader("📊 Outlier Summary (IQR Method)")
            outlier_summary = detect_all_outliers(df)
            outlier_summary_filtered = outlier_summary[outlier_summary["outlier_count"] > 0]

            if outlier_summary_filtered.empty:
                st.success("✅ No significant outliers detected.")
            else:
                st.dataframe(outlier_summary_filtered, use_container_width=True, hide_index=True)

            st.markdown("---")
            selected_out_col = st.selectbox("Inspect column", num_cols, key="out_col")
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_box = plot_box(df, selected_out_col)
                st.plotly_chart(fig_box, use_container_width=True)
            with col2:
                out_method = st.radio(
                    "Treatment method",
                    ["cap", "remove", "log"],
                    format_func=lambda x: {
                        "cap": "🔒 IQR Cap (Winsorize)",
                        "remove": "🗑️ Remove Rows",
                        "log": "📐 Log Transform",
                    }[x],
                    key="out_method",
                )
                if st.button("✅ Apply Treatment", key="btn_outlier"):
                    st.session_state["df"] = treat_outliers(
                        st.session_state["df"], selected_out_col, out_method
                    )
                    st.success(f"Applied **{out_method}** treatment to `{selected_out_col}`.")
                    st.rerun()

                # AI Suggestion
                col_stats = get_column_statistics(df)
                row = col_stats[col_stats["Column"] == selected_out_col]
                missing_pct = float(row["Missing %"].values[0]) if not row.empty else 0
                skew_val = row["Skewness"].values[0] if not row.empty else "—"
                mean_val = row["Mean"].values[0] if not row.empty else "—"
                std_val = row["Std Dev"].values[0] if not row.empty else "—"
                unique_val = int(row["Unique"].values[0]) if not row.empty else 0
                
                outlier_row = outlier_summary_filtered[outlier_summary_filtered["Column"] == selected_out_col]
                outlier_pct = float(outlier_row["outlier_pct"].values[0]) if not outlier_row.empty else 0

                context = build_preprocessing_context(
                    col_name=selected_out_col,
                    dtype=str(df[selected_out_col].dtype),
                    missing_pct=missing_pct,
                    unique_count=unique_val,
                    skewness=skew_val,
                    mean=mean_val,
                    std=std_val,
                )
                context += f"\n- Outlier Percentage: {outlier_pct}%"
                
                render_ai_suggestion(context, title="🤖 AI Outlier Suggestion",
                                      key=f"ai_out_{selected_out_col}")

    # ── Encoding ───────────────────────────────────────────────────────────────
    with tab_encoding:
        df = st.session_state["df"]
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        if not cat_cols:
            st.success("✅ No categorical columns remaining — all encoded!")
        else:
            st.info(f"**{len(cat_cols)}** categorical column(s) need encoding.")

            selected_cat = st.selectbox("Select column", cat_cols, key="enc_col")
            n_unique = df[selected_cat].nunique()
            rec = get_dtype_recommendation(df[selected_cat])

            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown(
                    f"**Column:** `{selected_cat}` | "
                    f"**Unique Values:** `{n_unique}` | "
                    f"**Recommended:** `{rec}`"
                )

                enc_method = st.selectbox(
                    "Encoding method",
                    ["onehot", "label", "ordinal"],
                    index=["onehot", "label", "ordinal"].index(rec) if rec in ["onehot", "label", "ordinal"] else 0,
                    format_func=lambda x: {
                        "onehot": "🔳 One-Hot Encoding",
                        "label": "🏷️ Label Encoding",
                        "ordinal": "📶 Ordinal Encoding",
                    }[x],
                    key="enc_method",
                )

                ordinal_order = None
                if enc_method == "ordinal":
                    unique_vals = df[selected_cat].dropna().unique().tolist()
                    st.markdown(f"**Unique values:** `{unique_vals}`")
                    ordinal_input = st.text_input(
                        "Enter ordinal order (comma-separated, lowest to highest)",
                        value=", ".join(map(str, sorted(unique_vals))),
                        key="ordinal_order",
                    )
                    ordinal_order = [v.strip() for v in ordinal_input.split(",")]

                if st.button("✅ Apply Encoding", key="btn_encode"):
                    st.session_state["df"] = encode_column(
                        st.session_state["df"], selected_cat, enc_method, ordinal_order
                    )
                    st.success(f"Applied **{enc_method}** encoding to `{selected_cat}`.")
                    st.rerun()

                # Preview value counts
                with st.expander("📊 Value Counts", expanded=False):
                    vc = df[selected_cat].value_counts().reset_index()
                    vc.columns = [selected_cat, "Count"]
                    st.dataframe(vc.head(20), use_container_width=True, hide_index=True)

            with col2:
                context = build_preprocessing_context(
                    col_name=selected_cat,
                    dtype=str(df[selected_cat].dtype),
                    missing_pct=round(df[selected_cat].isnull().mean() * 100, 1),
                    unique_count=n_unique,
                    skewness="N/A",
                    mean="N/A",
                    std="N/A",
                )
                render_ai_suggestion(context, title="🤖 AI Encoding Suggestion",
                                      key=f"ai_enc_{selected_cat}")


    # ── Leakage-safe model pipeline configuration ─────────────────────────────
    with tab_pipeline:
        st.subheader("🛡️ Leakage-Safe Model Pipeline")
        st.markdown(
            "Use these settings for the **final model training**. Unlike manual cleaning above, "
            "the learned statistics here are fitted on training folds only, so the held-out data does not leak into the model."
        )
        st.info(
            "💡 The manual tabs are useful for exploration and understanding. For the competition/final result, "
            "ML Studio uses the original uploaded data plus this safe pipeline by default."
        )

        saved = st.session_state.get("pipeline_config") or {}
        c1, c2 = st.columns(2)
        with c1:
            numeric_imputer = st.selectbox(
                "Numeric missing values", ["median", "mean", "most_frequent", "knn"],
                index=["median", "mean", "most_frequent", "knn"].index(saved.get("numeric_imputer", "median")),
                key="safe_num_imp",
            )
            categorical_imputer = st.selectbox(
                "Categorical missing values", ["most_frequent"], key="safe_cat_imp"
            )
            scaler = st.selectbox(
                "Scaling", ["StandardScaler", "RobustScaler", "MinMaxScaler", "None"],
                index=["StandardScaler", "RobustScaler", "MinMaxScaler", "None"].index(saved.get("scaler", "StandardScaler")),
                key="safe_scaler",
            )
            outlier_strategy = st.selectbox(
                "Outlier handling", ["None", "IQR clip"],
                index=["None", "IQR clip"].index(saved.get("outlier_strategy", "None")),
                key="safe_outlier",
            )
            iqr_factor = st.slider("IQR factor", 1.0, 3.0, float(saved.get("iqr_factor", 1.5)), 0.1, key="safe_iqr")

        with c2:
            categorical_encoder = st.selectbox(
                "Categorical encoder", ["OneHot", "Ordinal"],
                index=["OneHot", "Ordinal"].index(saved.get("categorical_encoder", "OneHot")),
                key="safe_encoder",
            )
            max_categories = st.slider("Maximum categories per feature", 10, 250, int(saved.get("max_categories", 60)), key="safe_maxcat")
            remove_zero_variance = st.checkbox(
                "Remove zero-variance features", value=bool(saved.get("remove_zero_variance", True)), key="safe_zero_var"
            )
            feature_selection = st.selectbox(
                "Feature selection", ["None", "SelectKBest"],
                index=["None", "SelectKBest"].index(saved.get("feature_selection", "None")),
                key="safe_fs",
            )
            k_best = st.slider("Keep best K features", 5, 200, int(saved.get("k_best") or 40), key="safe_k") if feature_selection == "SelectKBest" else None
            imbalance_strategy = st.selectbox(
                "Class imbalance handling", ["None", "RandomOverSampler", "SMOTE"],
                index=["None", "RandomOverSampler", "SMOTE"].index(saved.get("imbalance_strategy", "None")),
                key="safe_imbalance",
                help="Used only for classification. If mixed categorical data makes SMOTE unsafe, ML Studio will tell you to use RandomOverSampler/class weights instead.",
            )

        st.markdown("---")
        p1, p2 = st.columns(2)
        with p1:
            test_size = st.slider("Diagnostic test size", 0.10, 0.35, float(saved.get("test_size", 0.20)), 0.05, key="safe_test_size")
        with p2:
            random_state = st.number_input("Random state", 0, 999999, int(saved.get("random_state", 42)), key="safe_random_state")

        if st.button("✅ Save Safe Pipeline", type="primary", key="save_safe_pipeline"):
            st.session_state["pipeline_config"] = {
                "numeric_imputer": numeric_imputer,
                "categorical_imputer": categorical_imputer,
                "scaler": scaler,
                "outlier_strategy": outlier_strategy,
                "iqr_factor": float(iqr_factor),
                "max_categories": int(max_categories),
                "min_category_frequency": None,
                "remove_zero_variance": bool(remove_zero_variance),
                "feature_selection": feature_selection,
                "k_best": int(k_best) if k_best is not None else None,
                "categorical_encoder": categorical_encoder,
                "imbalance_strategy": imbalance_strategy,
                "test_size": float(test_size),
                "random_state": int(random_state),
            }
            st.success("✅ Final-training pipeline settings saved. ML Studio will use them automatically.")

    # ── Preprocessing Log ──────────────────────────────────────────────────────
    with tab_log:
        log = st.session_state.get("preprocessing_log", [])
        if not log:
            st.info("No preprocessing steps applied yet.")
        else:
            st.success(f"**{len(log)}** preprocessing steps applied:")
            for i, step in enumerate(log, 1):
                st.markdown(f"`{i:02d}` {step}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("↩️ Reset to Original Dataset", key="btn_reset"):
                    st.session_state["df"] = st.session_state["df_original"].copy()
                    st.session_state["preprocessing_log"] = []
                    st.success("✅ Dataset reset to original.")
                    st.rerun()
            with col2:
                df_current = st.session_state["df"]
                csv = df_current.to_csv(index=False).encode()
                st.download_button(
                    "💾 Download Processed Dataset",
                    data=csv,
                    file_name="ApexML_processed.csv",
                    mime="text/csv",
                    key="btn_download_proc",
                )
