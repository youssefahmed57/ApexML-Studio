"""
pages/1_dataset_overview.py
Page 1: Dataset Upload, Overview, Validation, and Column Statistics.
"""

import streamlit as st
import hashlib
import pandas as pd
import plotly.express as px
from utils.data_loader import load_dataset, get_dataset_metadata, get_column_statistics
from utils.eda_utils import plot_missing_heatmap
from utils.llm_utils import build_eda_context
from components.ai_suggestion import render_ai_suggestion


def render():
    st.title("📂 Dataset Overview & Validation")
    st.markdown("Upload your dataset to begin. All subsequent pages will reflect this data.")

    # ── Upload ─────────────────────────────────────────────────────────────────
    tab_upload, tab_preview, tab_stats, tab_validation = st.tabs([
        "📤 Upload", "👁️ Data Preview", "📊 Column Statistics", "✅ Validation Report"
    ])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            help="Maximum file size: 500 MB",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            signature = hashlib.sha256(file_bytes).hexdigest()

            # Only reload/reset when a genuinely different file is uploaded.
            # Streamlit reruns the page on every interaction, and the uploader
            # remains populated; resetting state on every rerun would erase
            # preprocessing/model state unexpectedly.
            if st.session_state.get("file_signature") != signature:
                with st.spinner("⏳ Loading dataset..."):
                    df = load_dataset(file_bytes, uploaded_file.name)

                if df is None or df.empty:
                    st.error("The uploaded file contains no data rows.")
                    st.stop()

                st.session_state["df"] = df
                st.session_state["df_original"] = df.copy()
                st.session_state["file_name"] = uploaded_file.name
                st.session_state["file_signature"] = signature

                # Reset each key to its correct data type.
                reset_values = {
                    "preprocessing_log": [],
                    "trained_models": {},
                    "train_test_data": None,
                    "label_encoders": {},
                    "ordinal_encoders": {},
                    "fitted_scalers": {},
                    "dataset_setup": None,
                    "pipeline_config": None,
                    "feature_recipe": [],
                    "model_params": {},
                    "selected_models": [],
                    "model_results": None,
                    "model_bundles": {},
                    "model_details": {},
                    "split_data_safe": None,
                    "final_bundle": None,
                    "competition_run": None,
                    "predictions_10": None,
                    "last_doctor_score": None,
                    "unsup_bundles": {},
                    "unsup_results": [],
                    "unsup_active_run": None,
                    "eval_results": [],
                }
                for _key, _value in reset_values.items():
                    st.session_state[_key] = _value

                st.success(f"✅ **{uploaded_file.name}** loaded successfully!")
            else:
                # Keep all current preprocessing/training state on normal reruns.
                df = st.session_state.get("df")

        # Show metadata if dataset is loaded
        if "df" in st.session_state and st.session_state["df"] is not None:
            df = st.session_state["df"]
            meta = get_dataset_metadata(df)

            st.markdown("---")
            st.subheader("📋 Dataset Snapshot")

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("🗂️ Rows", f"{meta['rows']:,}")
            c2.metric("📐 Columns", meta["columns"])
            c3.metric("❓ Missing Cells", f"{meta['missing_cells']:,}", f"{meta['missing_pct']}%")
            c4.metric("🔁 Duplicates", meta["duplicates"])
            c5.metric("💾 Memory", f"{meta['memory_mb']} MB")

            st.markdown("---")
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown(f"**🔢 Numerical columns ({len(meta['num_cols'])}):**")
                if meta["num_cols"]:
                    st.code(", ".join(meta["num_cols"]))
                else:
                    st.caption("None detected")
            with col_right:
                st.markdown(f"**🔤 Categorical columns ({len(meta['cat_cols'])}):**")
                if meta["cat_cols"]:
                    st.code(", ".join(meta["cat_cols"]))
                else:
                    st.caption("None detected")
        else:
            st.info("👆 Please upload a CSV or Excel file to get started.")

    # ── Preview ────────────────────────────────────────────────────────────────
    with tab_preview:
        if "df" not in st.session_state or st.session_state["df"] is None:
            st.info("Upload a dataset first.")
            return

        df = st.session_state["df"]
        col1, col2 = st.columns([3, 1])
        with col1:
            max_preview = max(1, min(500, len(df)))
            default_preview = min(10, max_preview)
            n_rows = st.slider(
                "Rows to display",
                min_value=1,
                max_value=max_preview,
                value=default_preview,
            )
        with col2:
            view_mode = st.radio("View", ["Head", "Tail", "Sample"], horizontal=True)

        if view_mode == "Head":
            st.dataframe(df.head(n_rows), use_container_width=True)
        elif view_mode == "Tail":
            st.dataframe(df.tail(n_rows), use_container_width=True)
        else:
            st.dataframe(df.sample(n=min(n_rows, len(df)), random_state=42),
                         use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Quick Distribution Overview")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            selected_num = st.selectbox("Select numerical column", num_cols, key="ov_num")
            fig = px.histogram(df, x=selected_num, nbins=30, marginal="box",
                               color_discrete_sequence=["#10B981"],
                               template="plotly_white",
                               title=f"Distribution of {selected_num}")
            st.plotly_chart(fig, use_container_width=True)

    # ── Column Statistics ──────────────────────────────────────────────────────
    with tab_stats:
        if "df" not in st.session_state or st.session_state["df"] is None:
            st.info("Upload a dataset first.")
            return

        df = st.session_state["df"]
        st.subheader("🔍 Per-Column Statistics")

        col_stats = get_column_statistics(df)
        st.dataframe(col_stats, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🤖 AI Column Insight")
        all_cols = df.columns.tolist()
        selected_col = st.selectbox("Choose a column for AI insight", all_cols, key="ai_col_ov")

        if st.button("🧠 Get AI Insight", key="btn_ai_ov"):
            col_row = col_stats[col_stats["Column"] == selected_col].to_dict("records")
            stats_dict = col_row[0] if col_row else {}
            context = build_eda_context(
                col_name=selected_col,
                dtype=str(df[selected_col].dtype),
                stats=stats_dict,
            )
            render_ai_suggestion(context, title=f"🤖 AI Insight — {selected_col}",
                                  key=f"ai_ov_{selected_col}")

    # ── Validation Report ──────────────────────────────────────────────────────
    with tab_validation:
        if "df" not in st.session_state or st.session_state["df"] is None:
            st.info("Upload a dataset first.")
            return

        df = st.session_state["df"]
        meta = get_dataset_metadata(df)

        st.subheader("✅ Data Quality Report")

        # Issues detection
        issues = []
        if meta["missing_cells"] > 0:
            issues.append(f"⚠️ **{meta['missing_cells']:,} missing values** ({meta['missing_pct']}% of all cells)")
        if meta["duplicates"] > 0:
            issues.append(f"⚠️ **{meta['duplicates']} duplicate rows** detected")

        high_missing_cols = [
            col for col in df.columns
            if df[col].isnull().mean() > 0.4
        ]
        if high_missing_cols:
            issues.append(f"🚨 **High missing ratio (>40%)**: {', '.join(high_missing_cols)}")

        high_cardinality = [
            col for col in df.select_dtypes(exclude="number").columns
            if df[col].nunique() > 50
        ]
        if high_cardinality:
            issues.append(f"ℹ️ **High-cardinality categoricals** (>50 unique): {', '.join(high_cardinality)}")

        if not issues:
            st.success("🎉 **Excellent!** No critical data quality issues detected.")
        else:
            for issue in issues:
                st.markdown(issue)

        st.markdown("---")

        # Missing value breakdown
        st.subheader("📊 Missing Value Breakdown")
        missing_series = df.isnull().sum()
        missing_pct = (missing_series / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            "Column": missing_series.index,
            "Missing Count": missing_series.values,
            "Missing %": missing_pct.values,
        }).query("`Missing Count` > 0").sort_values("Missing %", ascending=False)

        if missing_df.empty:
            st.success("✅ No missing values found!")
        else:
            fig_miss = px.bar(
                missing_df, x="Column", y="Missing %",
                color="Missing %",
                color_continuous_scale="RdYlGn_r",
                title="Missing Value % by Column",
                template="plotly_white",
            )
            fig_miss.add_hline(y=40, line_dash="dash", line_color="red",
                               annotation_text="40% threshold")
            st.plotly_chart(fig_miss, use_container_width=True)
            st.dataframe(missing_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🗺️ Missing Value Map")
        if df.isnull().any().any():
            fig_map = plot_missing_heatmap(df)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.success("✅ Dataset is complete — no missing values to map.")
