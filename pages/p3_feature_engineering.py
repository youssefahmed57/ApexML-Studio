"""
pages/3_feature_engineering.py
Page 3: Feature Engineering & EDA — Univariate, Bivariate, Multivariate analysis
plus scaling, transformation, and AI suggestions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.eda_utils import (
    plot_histogram, plot_box, plot_bar_counts, plot_pie,
    plot_scatter, plot_box_grouped, plot_violin,
    plot_correlation_heatmap, plot_pairplot,
    compute_extended_stats, get_feature_target_correlation,
    detect_outliers_iqr,
)
from utils.feature_engineering_utils import (
    apply_scaler, apply_log_transform, create_interaction_feature,
    create_polynomial_features, bin_column, extract_datetime_features,
    plot_pca_variance, get_scaler_recommendation,
)
from utils.llm_utils import build_scaler_context
from components.ai_suggestion import render_ai_suggestion




def _add_feature_recipe(recipe: dict):
    """Store deterministic feature engineering so final training can replay it safely."""
    recipes = st.session_state.get("feature_recipe")
    if not isinstance(recipes, list):
        recipes = []
    if recipe not in recipes:
        recipes.append(recipe)
    st.session_state["feature_recipe"] = recipes


def _require_df():
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()
    return st.session_state["df"]


def render():
    st.title("🔬 Feature Engineering & EDA")
    st.markdown("Explore your data and engineer powerful features to maximise model performance.")
    st.caption("🛡️ Features created here are also saved as a recipe so ML Studio can replay them inside the final training workflow.")

    df = _require_df()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    all_cols = df.columns.tolist()

    tab_uni, tab_bi, tab_multi, tab_scale, tab_create = st.tabs([
        "📊 Univariate", "📈 Bivariate", "🌐 Multivariate",
        "⚖️ Scaling & Transform", "🛠️ Feature Creation"
    ])

    # ── Univariate ─────────────────────────────────────────────────────────────
    with tab_uni:
        st.subheader("📊 Univariate Analysis")
        if not all_cols:
            st.info("No columns available.")
        else:
            selected = st.selectbox("Select column", all_cols, key="uni_col")
            is_num = pd.api.types.is_numeric_dtype(df[selected])

            col1, col2 = st.columns([2, 1])
            with col1:
                if is_num:
                    plot_type = st.radio("Plot type", ["Histogram", "Box Plot"], horizontal=True, key="uni_pt")
                    if plot_type == "Histogram":
                        st.plotly_chart(plot_histogram(df, selected), use_container_width=True, key="hist_uni")
                    else:
                        st.plotly_chart(plot_box(df, selected), use_container_width=True, key="box_uni")
                else:
                    plot_type = st.radio("Plot type", ["Bar Chart", "Pie Chart"], horizontal=True, key="uni_pt_cat")
                    if plot_type == "Bar Chart":
                        st.plotly_chart(plot_bar_counts(df, selected), use_container_width=True, key="bar_uni")
                    else:
                        st.plotly_chart(plot_pie(df, selected), use_container_width=True, key="pie_uni")

            with col2:
                st.markdown("**📋 Statistics**")
                if is_num:
                    stats = compute_extended_stats(df[selected])
                    for k, v in stats.items():
                        st.metric(k.capitalize(), v)
                else:
                    st.metric("Unique Values", df[selected].nunique())
                    st.metric("Most Frequent", df[selected].mode()[0] if not df[selected].empty else "—")
                    st.metric("Missing %", f"{df[selected].isnull().mean()*100:.1f}%")

    # ── Bivariate ──────────────────────────────────────────────────────────────
    with tab_bi:
        st.subheader("📈 Bivariate Analysis")

        col_a = st.selectbox("First variable (X)", all_cols, key="bi_a")
        col_b = st.selectbox("Second variable (Y)", [c for c in all_cols if c != col_a], key="bi_b")

        is_a_num = pd.api.types.is_numeric_dtype(df[col_a])
        is_b_num = pd.api.types.is_numeric_dtype(df[col_b])

        if is_a_num and is_b_num:
            color_col = st.selectbox("Color by (optional)", ["None"] + cat_cols, key="bi_color")
            color = None if color_col == "None" else color_col
            st.plotly_chart(plot_scatter(df, col_a, col_b, color), use_container_width=True, key="scatter_biv")

            # Correlation
            corr_val = df[[col_a, col_b]].corr().iloc[0, 1]
            strength = "strong" if abs(corr_val) > 0.7 else "moderate" if abs(corr_val) > 0.4 else "weak"
            direction = "positive" if corr_val > 0 else "negative"
            st.info(f"**Pearson Correlation:** `{corr_val:.4f}` — {strength} {direction} correlation")

        elif not is_a_num and is_b_num:
            plot_choice = st.radio("Plot type", ["Box Plot", "Violin Plot"], horizontal=True, key="bi_type")
            if plot_choice == "Box Plot":
                st.plotly_chart(plot_box_grouped(df, col_a, col_b), use_container_width=True)
            else:
                st.plotly_chart(plot_violin(df, col_a, col_b), use_container_width=True)

        elif is_a_num and not is_b_num:
            plot_choice = st.radio("Plot type", ["Box Plot", "Violin Plot"], horizontal=True, key="bi_type2")
            if plot_choice == "Box Plot":
                st.plotly_chart(plot_box_grouped(df, col_b, col_a), use_container_width=True)
            else:
                st.plotly_chart(plot_violin(df, col_b, col_a), use_container_width=True)

        else:
            # Both categorical — cross-tab heatmap
            import plotly.express as px
            ct = pd.crosstab(df[col_a], df[col_b])
            fig = px.imshow(ct, text_auto=True, color_continuous_scale="Teal",
                            title=f"Cross-tabulation: {col_a} × {col_b}",
                            template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    # ── Multivariate ───────────────────────────────────────────────────────────
    with tab_multi:
        st.subheader("🌐 Multivariate Analysis")

        mv_tab1, mv_tab2, mv_tab3 = st.tabs(["🔥 Correlation Heatmap", "🔗 Pair Plot", "🎯 Target Correlations"])

        with mv_tab1:
            if len(num_cols) < 2:
                st.info("Need at least 2 numerical columns.")
            else:
                st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

        with mv_tab2:
            if len(num_cols) < 2:
                st.info("Need at least 2 numerical columns.")
            else:
                max_cols = st.multiselect(
                    "Select columns for pair plot (2-8 recommended)",
                    num_cols,
                    default=num_cols[:min(5, len(num_cols))],
                    key="pair_cols",
                )
                color_pair = st.selectbox("Color by (optional)", ["None"] + cat_cols, key="pair_color")
                color_p = None if color_pair == "None" else color_pair
                if max_cols and len(max_cols) >= 2:
                    st.plotly_chart(plot_pairplot(df, max_cols, color_p), use_container_width=True)

        with mv_tab3:
            target_col = st.selectbox("Select target variable", num_cols, key="mv_target")
            if target_col:
                corr_df = get_feature_target_correlation(df, target_col)
                if not corr_df.empty:
                    import plotly.express as px
                    fig_corr = px.bar(
                        corr_df.head(20), x="Correlation", y="Feature",
                        orientation="h",
                        color="Correlation",
                        color_continuous_scale="RdYlGn",
                        range_color=[-1, 1],
                        title=f"Feature Correlations with '{target_col}'",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Target column must be numerical.")

        # PCA Variance
        if len(num_cols) >= 2:
            st.markdown("---")
            st.subheader("🧬 PCA Explained Variance")
            try:
                fig_pca = plot_pca_variance(df, num_cols)
                st.plotly_chart(fig_pca, use_container_width=True)
            except Exception as exc:
                st.info(f"PCA preview is unavailable for the current numeric data: {exc}")

    # ── Scaling & Transform ────────────────────────────────────────────────────
    with tab_scale:
        st.subheader("⚖️ Scaling & Transformation")
        if not num_cols:
            st.info("No numerical columns available.")
        else:
            scale_cols = st.multiselect(
                "Select columns to scale",
                num_cols,
                default=num_cols[:min(3, len(num_cols))],
                key="scale_cols",
            )

            col1, col2 = st.columns([2, 1])
            with col1:
                scaler_choice = st.selectbox(
                    "Scaling method",
                    ["StandardScaler", "MinMaxScaler", "RobustScaler", "Log Transform (log1p)"],
                    key="scaler_choice",
                )

                if scale_cols:
                    if st.button("✅ Apply Scaling", key="btn_scale"):
                        df_work = st.session_state["df"]
                        if scaler_choice == "Log Transform (log1p)":
                            st.session_state["df"] = apply_log_transform(df_work, scale_cols)
                            for _col in scale_cols:
                                _add_feature_recipe({"type": "log", "column": _col})
                            from utils.preprocessing_utils import log_step
                            log_step(f"Applied Log Transform (log1p) to: {scale_cols}")
                        else:
                            scaled_df, _ = apply_scaler(df_work, scale_cols, scaler_choice)
                            st.session_state["df"] = scaled_df
                            from utils.preprocessing_utils import log_step
                            log_step(f"Applied {scaler_choice} to: {scale_cols}")
                        st.success(f"✅ {scaler_choice} applied to {len(scale_cols)} column(s).")
                        st.rerun()

                    # Preview before/after
                    with st.expander("📊 Before/After Distribution Preview"):
                        prev_col = st.selectbox("Column to preview", scale_cols, key="prev_col")
                        st.plotly_chart(plot_histogram(df, prev_col), use_container_width=True, key="hist_scaler_prev")

            with col2:
                # AI Suggestion
                if scale_cols:
                    sample_col = scale_cols[0]
                    skew = round(df[sample_col].skew(), 3) if sample_col in df.columns else 0
                    outlier_info = detect_outliers_iqr(df[sample_col].dropna())
                    has_outliers = outlier_info["outlier_pct"] > 5
                    context = build_scaler_context(
                        col_name=sample_col,
                        skewness=skew,
                        has_outliers=has_outliers,
                        target_models=st.session_state.get("selected_models", []),
                    )
                    render_ai_suggestion(context, title="🤖 AI Scaler Suggestion",
                                          key=f"ai_scale_{sample_col}")

    # ── Feature Creation ───────────────────────────────────────────────────────
    with tab_create:
        st.subheader("🛠️ Feature Creation")

        fc_tab1, fc_tab2, fc_tab3, fc_tab4 = st.tabs([
            "✖️ Interaction", "📐 Polynomial", "📦 Binning", "📅 DateTime"
        ])

        with fc_tab1:
            if len(num_cols) >= 2:
                c1 = st.selectbox("Column A", num_cols, key="int_a")
                c2 = st.selectbox("Column B", [c for c in num_cols if c != c1], key="int_b")
                op = st.selectbox("Operation", ["multiply", "divide", "add", "subtract"], key="int_op")
                if st.button("➕ Create Feature", key="btn_int"):
                    st.session_state["df"] = create_interaction_feature(st.session_state["df"], c1, c2, op)
                    _add_feature_recipe({"type": "interaction", "col1": c1, "col2": c2, "operation": op})
                    from utils.preprocessing_utils import log_step
                    log_step(f"Created interaction feature: {c1}__{op}__{c2}")
                    st.success(f"✅ Created `{c1}__{op}__{c2}`")
                    st.rerun()
            else:
                st.info("Need at least 2 numerical columns.")

        with fc_tab2:
            if num_cols:
                poly_col = st.selectbox("Column", num_cols, key="poly_col")
                poly_deg = st.slider("Degree", 2, 4, 2, key="poly_deg")
                if st.button("📐 Add Polynomial Features", key="btn_poly"):
                    st.session_state["df"] = create_polynomial_features(
                        st.session_state["df"], poly_col, poly_deg
                    )
                    _add_feature_recipe({"type": "polynomial", "column": poly_col, "degree": int(poly_deg)})
                    from utils.preprocessing_utils import log_step
                    log_step(f"Added polynomial features for '{poly_col}' up to degree {poly_deg}")
                    st.success(f"✅ Added polynomial features for `{poly_col}`")
                    st.rerun()

        with fc_tab3:
            if num_cols:
                bin_col_sel = st.selectbox("Column", num_cols, key="bin_col")
                n_bins = st.slider("Number of bins", 2, 20, 5, key="n_bins")
                if st.button("📦 Create Bins", key="btn_bin"):
                    st.session_state["df"] = bin_column(st.session_state["df"], bin_col_sel, n_bins)
                    _add_feature_recipe({"type": "bin", "column": bin_col_sel, "n_bins": int(n_bins)})
                    from utils.preprocessing_utils import log_step
                    log_step(f"Binned '{bin_col_sel}' into {n_bins} bins")
                    st.success(f"✅ Created `{bin_col_sel}_binned`")
                    st.rerun()

        with fc_tab4:
            possible_dt = [c for c in all_cols if
                           "date" in c.lower() or "time" in c.lower() or "year" in c.lower()]
            dt_col = st.selectbox("Date/Time column",
                                   possible_dt if possible_dt else all_cols,
                                   key="dt_col")
            if st.button("📅 Extract DateTime Features", key="btn_dt"):
                st.session_state["df"] = extract_datetime_features(st.session_state["df"], dt_col)
                _add_feature_recipe({"type": "datetime", "column": dt_col, "drop_original": True})
                from utils.preprocessing_utils import log_step
                log_step(f"Extracted datetime features from '{dt_col}'")
                st.success(f"✅ Extracted year, month, day, weekday from `{dt_col}`")
                st.rerun()


        st.markdown("---")
        recipes = st.session_state.get("feature_recipe", [])
        if recipes:
            st.subheader("🧾 Final-Training Feature Recipe")
            st.dataframe(pd.DataFrame(recipes), use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear Feature Recipe", key="clear_feature_recipe"):
                st.session_state["feature_recipe"] = []
                st.success("Feature recipe cleared. The working preview dataset is unchanged until you reset it from Preprocessing.")
                st.rerun()
