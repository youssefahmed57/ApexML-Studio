"""
pages/p6_inference.py
ApexML inference page with a locked final model, single prediction, batch prediction,
and a dedicated exactly-10-cases competition workflow.
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

from core.engine import prepare_inference_features
from core.competition import predict_competition_bundle
from core.unsupervised_engine import predict_or_transform


def _require_model():
    if st.session_state.get("final_bundle") is None:
        st.warning("⚠️ No final model yet. Train models from **ML Studio** first.")
        st.stop()


def _prepare(raw: pd.DataFrame, bundle: dict, setup: dict):
    X = prepare_inference_features(
        raw,
        setup.get("excluded", []),
        setup.get("date_cols", []),
        setup.get("force_numeric", []),
        setup.get("force_cat", []),
    )
    needed = bundle["feature_columns"]
    missing = [c for c in needed if c not in X.columns]
    extras = [c for c in X.columns if c not in needed]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))
    return X[needed], extras


def _predict(bundle: dict, X: pd.DataFrame):
    if bundle.get("competition"):
        return predict_competition_bundle(bundle, X)
    pipe = bundle["pipeline"]
    pred = pipe.predict(X)
    probs = pipe.predict_proba(X) if bundle["task"] == "classification" and hasattr(pipe, "predict_proba") else None
    return pred, probs


def _decode(bundle: dict, pred):
    if bundle["task"] == "classification":
        enc = bundle.get("target_encoder")
        return enc.inverse_transform(np.asarray(pred).astype(int)) if enc is not None else pred
    return np.asarray(pred, dtype=float)


def _raw_feature_columns(df: pd.DataFrame, setup: dict):
    return [c for c in df.columns if c != setup["target"] and c not in setup.get("excluded", [])]


def _default_row(df: pd.DataFrame, columns):
    row = {}
    for col in columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            row[col] = float(num.median()) if num.notna().any() else 0.0
        else:
            mode = s.dropna().astype(str).mode()
            row[col] = str(mode.iloc[0]) if not mode.empty else ""
    return row


def _read_upload(uploaded):
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    return pd.read_excel(io.BytesIO(data))


def _render_supervised_inference():
    st.title("🔮 Inference / Prediction")
    st.markdown("The final winner is selected in ML Studio/Evaluation. Use it here for live prediction or the doctor's 10 hidden cases.")
    _require_model()

    final_bundle = st.session_state["final_bundle"]
    bundles = st.session_state.get("model_bundles", {})
    setup = st.session_state.get("dataset_setup")
    df = st.session_state.get("df_original")
    if not setup or df is None:
        st.error("Dataset setup is missing. Return to ML Studio → Setup.")
        return

    bundle = final_bundle
    best_name = st.session_state.get("best_model_name", "Final Winner")

    c1, c2, c3 = st.columns([2, 1, 1])
    c1.success(f"🏆 **Locked Final Model:** {best_name}")
    c2.metric("Target", bundle["target_name"])
    c3.metric("Task", bundle["task"].title())

    with st.expander("Advanced: use another benchmark model instead", expanded=False):
        if bundles:
            names = ["Final Winner"] + list(bundles.keys())
            chosen = st.selectbox("Inference model", names, key="advanced_inference_model")
            if chosen != "Final Winner":
                bundle = bundles[chosen]
                st.warning("You are overriding the locked final winner for this prediction session.")

    tab_single, tab_ten, tab_batch = st.tabs([
        "📝 Single Prediction", "🎯 Doctor's 10 Cases", "📂 Batch Prediction"
    ])

    raw_cols = _raw_feature_columns(df, setup)

    with tab_single:
        st.subheader("📝 Enter One New Case")
        defaults = _default_row(df, raw_cols)
        values = {}
        cols = st.columns(3)
        for i, feat in enumerate(raw_cols):
            with cols[i % 3]:
                s = df[feat]
                if pd.api.types.is_numeric_dtype(s):
                    num = pd.to_numeric(s, errors="coerce")
                    med = float(num.median()) if num.notna().any() else 0.0
                    values[feat] = st.number_input(feat, value=med, key=f"single_{feat}")
                else:
                    choices = s.dropna().astype(str).value_counts().head(100).index.tolist()
                    values[feat] = st.selectbox(feat, choices if choices else [""], key=f"single_{feat}")

        if st.button("🔮 Generate Prediction", type="primary", key="single_predict"):
            try:
                raw = pd.DataFrame([values])
                X, extras = _prepare(raw, bundle, setup)
                pred, probs = _predict(bundle, X)
                decoded = _decode(bundle, pred)
                st.markdown("---")
                if bundle["task"] == "classification":
                    st.markdown(
                        f'''<div style="background:linear-gradient(135deg,rgba(16,185,129,.12),rgba(5,150,105,.04));border:3px solid #10B981;border-radius:16px;padding:24px;text-align:center;">
                        <div style="font-size:2.5rem;font-weight:800;color:#059669;">{decoded[0]}</div>
                        <div style="color:#6B7280;">Predicted Class</div></div>''',
                        unsafe_allow_html=True,
                    )
                    if probs is not None:
                        enc = bundle.get("target_encoder")
                        labels = [str(x) for x in enc.classes_] if enc is not None else [str(i) for i in range(probs.shape[1])]
                        pframe = pd.DataFrame({"Class": labels, "Probability": probs[0]})
                        st.plotly_chart(px.bar(pframe, x="Probability", y="Class", orientation="h", range_x=[0,1], color="Probability", color_continuous_scale="Teal", template="plotly_white"), use_container_width=True)
                else:
                    st.metric("Predicted Value", f"{float(decoded[0]):,.6g}")
            except Exception as exc:
                st.exception(exc)

    with tab_ten:
        st.subheader("🎯 Final Test — Exactly 10 Unseen Cases")
        st.markdown("Use the locked winner for the 10 cases from the doctor. Upload a file or paste/edit 10 rows directly.")
        source = st.radio("Input method", ["Upload CSV / Excel", "Paste / edit 10 rows"], horizontal=True, key="ten_source")
        raw_cases = None

        if source == "Upload CSV / Excel":
            uploaded = st.file_uploader("Upload the 10 cases", type=["csv", "xlsx", "xls"], key="ten_upload")
            if uploaded is not None:
                try:
                    raw_cases = _read_upload(uploaded)
                    st.dataframe(raw_cases, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.exception(exc)
        else:
            row = _default_row(df, raw_cols)
            initial = pd.DataFrame([row.copy() for _ in range(10)])
            config = {}
            for col in raw_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    choices = df[col].dropna().astype(str).value_counts().head(100).index.tolist()
                    if 0 < len(choices) <= 30:
                        config[col] = st.column_config.SelectboxColumn(col, options=choices)
            raw_cases = st.data_editor(initial, use_container_width=True, hide_index=False, num_rows="fixed", column_config=config, key="ten_editor")

        if raw_cases is not None:
            if len(raw_cases) != 10:
                st.error(f"This final-test mode requires exactly **10 rows**. Current rows: **{len(raw_cases)}**.")
            else:
                st.success("✅ 10/10 cases detected — ready.")
                if st.button("🎯 Predict All 10 Cases", type="primary", use_container_width=True, key="predict_ten"):
                    try:
                        X, extras = _prepare(raw_cases, final_bundle, setup)
                        pred, probs = _predict(final_bundle, X)
                        decoded = _decode(final_bundle, pred)
                        out = raw_cases.copy().reset_index(drop=True)
                        out.insert(0, "Case", np.arange(1, 11))
                        out[f"Prediction_{final_bundle['target_name']}"] = decoded
                        if probs is not None:
                            out["Confidence"] = np.max(probs, axis=1)
                            enc = final_bundle.get("target_encoder")
                            if enc is not None and probs.shape[1] <= 10:
                                for i, label in enumerate(enc.classes_):
                                    out[f"P_{label}"] = probs[:, i]
                        st.session_state["predictions_10"] = out
                        st.balloons()
                    except Exception as exc:
                        st.exception(exc)

        out = st.session_state.get("predictions_10")
        if out is not None:
            st.markdown("---")
            st.subheader("🏁 Final 10 Predictions")
            display = ["Case", f"Prediction_{final_bundle['target_name']}"] + (["Confidence"] if "Confidence" in out.columns else [])
            st.dataframe(out[display], use_container_width=True, hide_index=True,
                         column_config={"Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.1%%")} if "Confidence" in out.columns else None)
            st.download_button("💾 Download 10 Predictions", out.to_csv(index=False).encode("utf-8"), file_name="ApexML_doctor_10_predictions.csv", mime="text/csv", use_container_width=True)

            with st.expander("After the doctor reveals the correct answers — calculate our score"):
                if final_bundle["task"] == "classification":
                    classes = [str(x) for x in final_bundle["target_encoder"].classes_]
                    actual = pd.DataFrame({"Case": np.arange(1,11), "Actual": [classes[0]] * 10})
                    actual = st.data_editor(actual, hide_index=True, num_rows="fixed", column_config={"Actual": st.column_config.SelectboxColumn("Actual", options=classes)}, key="ten_actual")
                    if st.button("Calculate Score", key="score_ten"):
                        pred_col = f"Prediction_{final_bundle['target_name']}"
                        correct = (out[pred_col].astype(str).reset_index(drop=True) == actual["Actual"].astype(str).reset_index(drop=True))
                        st.session_state["last_doctor_score"] = {"Correct": int(correct.sum()), "Accuracy": float(correct.mean())}
                    score = st.session_state.get("last_doctor_score")
                    if score:
                        a, b = st.columns(2)
                        a.metric("Correct", f"{score['Correct']}/10")
                        b.metric("Accuracy", f"{score['Accuracy']:.0%}")
                else:
                    actual = pd.DataFrame({"Case": np.arange(1,11), "Actual": [0.0] * 10})
                    actual = st.data_editor(actual, hide_index=True, num_rows="fixed", key="ten_actual_reg")
                    if st.button("Calculate Regression Score", key="score_ten_reg"):
                        pred_col = f"Prediction_{final_bundle['target_name']}"
                        y_true = pd.to_numeric(actual["Actual"], errors="coerce")
                        y_pred = pd.to_numeric(out[pred_col], errors="coerce")
                        st.session_state["last_doctor_score"] = {
                            "MAE": mean_absolute_error(y_true, y_pred),
                            "RMSE": mean_squared_error(y_true, y_pred) ** .5,
                            "R2": r2_score(y_true, y_pred),
                        }
                    score = st.session_state.get("last_doctor_score")
                    if score:
                        a, b, c = st.columns(3)
                        a.metric("MAE", f"{score['MAE']:.4g}")
                        b.metric("RMSE", f"{score['RMSE']:.4g}")
                        c.metric("R²", f"{score['R2']:.4f}")

    with tab_batch:
        st.subheader("📂 General Batch Prediction")
        uploaded = st.file_uploader("Upload CSV / Excel with the same raw feature columns", type=["csv", "xlsx", "xls"], key="batch_general")
        if uploaded is not None:
            try:
                raw = _read_upload(uploaded)
                X, extras = _prepare(raw, bundle, setup)
                pred, probs = _predict(bundle, X)
                decoded = _decode(bundle, pred)
                out = raw.copy().reset_index(drop=True)
                out[f"Prediction_{bundle['target_name']}"] = decoded
                if probs is not None:
                    out["Confidence"] = np.max(probs, axis=1)
                st.success(f"✅ Predictions generated for **{len(out):,} rows**.")
                st.dataframe(out.head(100), use_container_width=True, hide_index=True)
                st.download_button("💾 Download Predictions CSV", out.to_csv(index=False).encode("utf-8"), file_name="ApexML_predictions.csv", mime="text/csv", use_container_width=True)
            except Exception as exc:
                st.exception(exc)



def _render_unsupervised_inference():
    st.title("🔮 Unsupervised Inference / Assignment")
    st.markdown("Assign new rows to fitted clusters or transform them with a reusable dimensionality-reduction model.")

    bundles = st.session_state.get("unsup_bundles") or {}
    if not bundles:
        st.warning("⚠️ No reusable unsupervised run yet. Train one from **ML Studio → Unsupervised Learning** first.")
        return

    names = list(bundles.keys())
    active = st.session_state.get("unsup_active_run")
    idx = names.index(active) if active in names else len(names) - 1
    run_name = st.selectbox("Unsupervised run", names, index=max(0, idx), key="unsup_infer_run")
    bundle = bundles[run_name]
    algorithm = bundle.get("algorithm", "Unknown")
    features = list(bundle.get("feature_columns") or [])

    a, b, c = st.columns(3)
    a.metric("Algorithm", algorithm)
    b.metric("Required Features", len(features))
    c.metric("Run", run_name)
    if features:
        st.caption("Schema: " + ", ".join(features))

    if algorithm == "t-SNE":
        st.warning("t-SNE cannot transform unseen rows with sklearn. Use the fitted PCA mode when future-row projection is required.")
        return

    if algorithm == "Association Rules (Apriori)":
        st.subheader("🧺 Rule-Based Recommendation")
        rules = bundle.get("rules")
        if not isinstance(rules, pd.DataFrame) or rules.empty:
            st.info("No saved rules are available.")
            return
        selected_items = st.multiselect("Items present in the new transaction", features, key="apriori_present_items")
        if st.button("🔎 Find Matching Rules", type="primary", key="apriori_match"):
            present = set(selected_items)
            matches = []
            for _, row in rules.iterrows():
                ants = {x.strip() for x in str(row.get("antecedents", "")).split(",") if x.strip()}
                cons = {x.strip() for x in str(row.get("consequents", "")).split(",") if x.strip()}
                if ants and ants.issubset(present) and not cons.issubset(present):
                    matches.append(row)
            if matches:
                out = pd.DataFrame(matches)
                sort_cols = [c for c in ["lift", "confidence"] if c in out.columns]
                if sort_cols:
                    out = out.sort_values(sort_cols, ascending=False)
                st.success(f"Found {len(out)} matching rule(s).")
                st.dataframe(out.head(50), use_container_width=True, hide_index=True)
            else:
                st.info("No saved rule matches the selected items.")
        return

    train_frame = bundle.get("train_frame")
    defaults = {}
    if isinstance(train_frame, pd.DataFrame):
        for feat in features:
            s = pd.to_numeric(train_frame[feat], errors="coerce")
            defaults[feat] = float(s.median()) if s.notna().any() else float(bundle.get("fill_values", {}).get(feat, 0.0))
    else:
        defaults = {feat: float(bundle.get("fill_values", {}).get(feat, 0.0)) for feat in features}

    if algorithm == "Hierarchical Clustering":
        st.info("New rows use **nearest-centroid assignment**. This is a reusable approximation because AgglomerativeClustering itself has no predict method.")
    elif algorithm == "DBSCAN":
        st.info("A new row joins the nearest fitted core cluster only when its distance is within **eps**; otherwise Cluster = -1 (Noise).")
    elif algorithm == "K-Means Clustering":
        st.success("K-Means supports native prediction for unseen rows.")
    elif algorithm == "PCA":
        st.success("PCA supports native transform for unseen rows.")

    tab_single, tab_ten, tab_batch = st.tabs(["📝 Single Row", "🔟 Exactly 10 Rows", "📂 Batch File"])

    def show_output(raw: pd.DataFrame, result: pd.DataFrame, prefix: str):
        out = raw.reset_index(drop=True).copy()
        result = result.reset_index(drop=True)
        for col in result.columns:
            out[col] = result[col]
        if "Cluster" in out.columns:
            out["Cluster_Label"] = out["Cluster"].apply(lambda x: "Noise" if int(x) == -1 else f"Cluster {int(x)}")
        st.dataframe(out, use_container_width=True, hide_index=True)
        st.download_button(
            "💾 Download Result CSV", out.to_csv(index=False).encode("utf-8"),
            file_name=f"ApexML_{prefix}_unsupervised.csv", mime="text/csv", use_container_width=True,
            key=f"download_{prefix}_{run_name}",
        )

    with tab_single:
        values = {}
        cols = st.columns(min(3, max(1, len(features))))
        for i, feat in enumerate(features):
            with cols[i % len(cols)]:
                values[feat] = st.number_input(feat, value=float(defaults.get(feat, 0.0)), key=f"unsup_single_{run_name}_{feat}")
        if st.button("🔮 Assign / Transform", type="primary", key=f"unsup_single_predict_{run_name}"):
            try:
                raw = pd.DataFrame([values])
                result = predict_or_transform(raw, bundle)
                show_output(raw, result, "single")
            except Exception as exc:
                st.exception(exc)

    with tab_ten:
        initial = pd.DataFrame([{feat: defaults.get(feat, 0.0) for feat in features} for _ in range(10)])
        edited = st.data_editor(initial, use_container_width=True, hide_index=False, num_rows="fixed", key=f"unsup_ten_editor_{run_name}")
        if len(edited) == 10 and st.button("🔟 Assign / Transform 10 Rows", type="primary", key=f"unsup_ten_predict_{run_name}"):
            try:
                result = predict_or_transform(edited, bundle)
                show_output(edited, result, "10_cases")
            except Exception as exc:
                st.exception(exc)

    with tab_batch:
        uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"], key=f"unsup_batch_{run_name}")
        if uploaded is not None:
            try:
                raw = _read_upload(uploaded)
                missing = [c for c in features if c not in raw.columns]
                if missing:
                    st.error("Missing required columns: " + ", ".join(missing))
                else:
                    result = predict_or_transform(raw, bundle)
                    st.success(f"✅ Processed {len(raw):,} row(s).")
                    show_output(raw, result, "batch")
            except Exception as exc:
                st.exception(exc)


def render():
    has_supervised = st.session_state.get("final_bundle") is not None
    has_unsupervised = bool(st.session_state.get("unsup_bundles"))

    if has_supervised and has_unsupervised:
        view = st.radio("Inference Type", ["🎯 Supervised", "🔍 Unsupervised"], horizontal=True, key="inference_type")
        if "Unsupervised" in view:
            _render_unsupervised_inference()
        else:
            _render_supervised_inference()
    elif has_unsupervised:
        _render_unsupervised_inference()
    else:
        _render_supervised_inference()
