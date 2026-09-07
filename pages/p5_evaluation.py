"""
pages/p5_evaluation.py
ApexML-style evaluation page for safe pipelines and competition model selection.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.engine import feature_importance_table, binary_curve_data, evaluate_baseline, bundle_to_bytes


def _require_results():
    if st.session_state.get("model_results") is None or not st.session_state.get("model_bundles"):
        st.warning("⚠️ No safe-model results yet. Train models from **ML Studio** first.")
        st.stop()


def _best_standard_row(results: pd.DataFrame, task: str, metric: str):
    mapping = {
        "Accuracy": ("Test_Accuracy", False),
        "Balanced Accuracy": ("Test_Balanced_Accuracy", False),
        "Weighted F1": ("Test_F1_Weighted", False),
        "Macro F1": ("Test_F1_Macro", False),
        "RMSE": ("Test_RMSE", True),
        "MAE": ("Test_MAE", True),
        "R²": ("Test_R2", False),
    }
    col, asc = mapping.get(metric, ("Test_F1_Weighted" if task == "classification" else "Test_RMSE", task == "regression"))
    return results.sort_values(col, ascending=asc).iloc[0], col


def _render_supervised_evaluation():
    st.title("🏆 Model Evaluation & Leaderboard")
    st.markdown("Compare models, inspect stability and overfitting, and verify the predictor used for the final 10 cases.")
    _require_results()

    results = st.session_state["model_results"].copy()
    bundles = st.session_state["model_bundles"]
    details = st.session_state.get("model_details", {})
    split = st.session_state.get("split_data_safe")
    task = st.session_state.get("eval_task_type", "classification")
    metric = st.session_state.get("primary_metric") or ("Accuracy" if task == "classification" else "RMSE")
    competition = st.session_state.get("competition_run")
    final_bundle = st.session_state.get("final_bundle")
    best_name = st.session_state.get("best_model_name") or "Best Model"

    tab_lb, tab_diag, tab_curves, tab_feat, tab_nn, tab_export = st.tabs([
        "🏆 Leaderboard", "🔬 Diagnostics", "📈 ROC / Errors",
        "📊 Feature Importance", "🧠 Neural Network", "💾 Export"
    ])

    with tab_lb:
        st.subheader("🏆 Final Model")
        a, b, c, d = st.columns(4)
        a.metric("Winner", best_name)
        b.metric("Primary Metric", metric)
        c.metric("Task", task.title())
        if competition is not None:
            top = competition["leaderboard"].iloc[0]
            d.metric("Repeated CV", f"{float(top['CV_Mean']):.4f}")
        else:
            best_row, metric_col = _best_standard_row(results, task, metric)
            d.metric("Test Score", f"{float(best_row[metric_col]):.4f}")

        if competition is not None:
            st.success("Competition Boost ranks candidates with repeated Cross Validation, then refits the winner on all labeled rows for the external 10-case test.")
            st.subheader("🎯 Competition Ranking")
            lb = competition["leaderboard"].copy()
            show_cols = [c for c in ["Candidate", "Source", "Model_Family", "CV_Mean", "CV_Std", "Stability_Score", "Fit_Seconds"] if c in lb.columns]
            st.dataframe(lb[show_cols], use_container_width=True, hide_index=True)
            fig = px.bar(lb.head(12), x="Candidate", y="CV_Mean",
                         error_y="CV_Std" if "CV_Std" in lb else None,
                         color="Source" if "Source" in lb else None,
                         title=f"Repeated CV — {competition['metric_label']}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            exp = competition.get("expected_correct_out_of_10")
            if exp is not None:
                st.info(f"Repeated-CV estimate: **≈ {exp:.1f}/10 correct** if the hidden cases follow the same distribution. This is an estimate, not a guarantee.")

        st.markdown("---")
        st.subheader("📊 Holdout Model Comparison")
        st.dataframe(results, use_container_width=True, hide_index=True)

        if split is not None:
            X_train, X_test, y_train, y_test = split
            try:
                baseline = evaluate_baseline(X_train, X_test, y_train, y_test, task)
                with st.expander("No-Skill Baseline"):
                    st.dataframe(pd.DataFrame([baseline]), use_container_width=True, hide_index=True)
            except Exception:
                pass

        if task == "classification":
            cols = [c for c in ["Test_Accuracy", "Test_Balanced_Accuracy", "Test_F1_Weighted", "Test_F1_Macro", "Test_ROC_AUC"] if c in results.columns]
            if cols:
                melt = results.melt(id_vars="Model", value_vars=cols, var_name="Metric", value_name="Score")
                st.plotly_chart(px.bar(melt, x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1], template="plotly_white"), use_container_width=True)
        elif "Test_RMSE" in results:
            st.plotly_chart(px.bar(results.sort_values("Test_RMSE"), x="Model", y="Test_RMSE", title="Test RMSE — Lower is Better", template="plotly_white"), use_container_width=True)

    with tab_diag:
        st.subheader("🔬 Generalization, Cross Validation & Cost")
        useful = [c for c in [
            "Model", "Generalization_Gap", "CV_F1_Weighted_Mean", "CV_F1_Weighted_Std",
            "CV_RMSE_Mean", "CV_RMSE_Std", "Fit_Seconds", "Predict_ms_per_1000", "Model_Size_KB"
        ] if c in results.columns]
        st.dataframe(results[useful], use_container_width=True, hide_index=True)

        if "Generalization_Gap" in results:
            fig = px.bar(results, x="Model", y="Generalization_Gap", title="Train–Test Generalization Gap", template="plotly_white")
            fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("A large gap can indicate overfitting. Low train and test performance together can indicate underfitting.")

        failures = details.get("__failures__", []) if isinstance(details, dict) else []
        if failures:
            with st.expander(f"⚠️ Models that could not train ({len(failures)})"):
                st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

        if task == "classification":
            model_name = st.selectbox("Model for confusion matrix / report", list(bundles.keys()), key="diag_model")
            diag = details.get(model_name, {})
            if diag.get("confusion_matrix") is not None:
                enc = bundles[model_name].get("target_encoder")
                labels = [str(x) for x in enc.classes_] if enc is not None else None
                fig = px.imshow(diag["confusion_matrix"], text_auto=True, x=labels, y=labels,
                                labels={"x": "Predicted", "y": "Actual"}, color_continuous_scale="Teal",
                                title=f"Confusion Matrix — {model_name}")
                st.plotly_chart(fig, use_container_width=True)
            if diag.get("classification_report") is not None:
                st.dataframe(pd.DataFrame(diag["classification_report"]).T, use_container_width=True)
        else:
            model_name = st.selectbox("Model for regression errors", list(bundles.keys()), key="diag_reg_model")
            diag = details.get(model_name, {})
            if diag.get("y_test") is not None:
                y_true = np.asarray(diag["y_test"])
                pred = np.asarray(diag["test_pred"])
                c1, c2 = st.columns(2)
                with c1:
                    frame = pd.DataFrame({"Actual": y_true, "Predicted": pred})
                    fig = px.scatter(frame, x="Actual", y="Predicted", opacity=.5, title="Actual vs Predicted", template="plotly_white")
                    lo, hi = float(min(y_true.min(), pred.min())), float(max(y_true.max(), pred.max()))
                    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal"))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    frame = pd.DataFrame({"Predicted": pred, "Residual": y_true - pred})
                    fig = px.scatter(frame, x="Predicted", y="Residual", opacity=.5, title="Residuals", template="plotly_white")
                    fig.add_hline(y=0, line_dash="dash")
                    st.plotly_chart(fig, use_container_width=True)

    with tab_curves:
        if task == "classification" and split is not None:
            _, X_test, _, y_test = split
            if len(np.unique(y_test)) == 2:
                st.subheader("📈 ROC Curves")
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
                for name, bundle in bundles.items():
                    try:
                        curves = binary_curve_data(bundle, X_test, y_test)
                        roc = curves.get("roc")
                        if roc is not None:
                            fig_roc.add_trace(go.Scatter(x=roc["FPR"], y=roc["TPR"], mode="lines", name=name))
                    except Exception:
                        pass
                fig_roc.update_layout(template="plotly_white", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("ROC comparison here is shown for binary classification.")
        elif task == "regression":
            st.info("Regression error diagnostics are available in the Diagnostics tab.")

    with tab_feat:
        st.subheader("📊 Feature Importance / Coefficients")
        name = st.selectbox("Select model", list(bundles.keys()), key="fi_safe")
        fi = feature_importance_table(bundles[name])
        if fi is None:
            st.info("This model does not expose direct feature importance or coefficients.")
        else:
            top = fi.head(25)
            fig = px.bar(top.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Teal", title=f"Top Features — {name}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top, use_container_width=True, hide_index=True)
            st.caption("Feature importance describes model association, not causation.")

    with tab_nn:
        if "Neural Network" not in bundles:
            st.info("Train Neural Network from ML Studio to inspect its architecture and loss curve.")
        else:
            diag = details.get("Neural Network", {})
            nn = diag.get("nn_architecture")
            if nn:
                st.subheader("🧠 Neural Network Architecture")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Input Neurons", nn.get("transformed_input_features", "—"))
                c2.metric("Hidden Layers", len(nn.get("hidden_layer_sizes", ())))
                c3.metric("Output Neurons", nn.get("output_neurons", "—"))
                c4.metric("Trainable Params", f"{nn.get('trainable_parameters', 0):,}")
                if nn.get("all_layers"):
                    st.code("Input → " + " → ".join(map(str, nn["all_layers"][1:-1])) + " → Output")
                st.caption(f"Architecture mode: {nn.get('mode')}")
            loss = diag.get("loss_curve")
            if loss is not None:
                frame = pd.DataFrame({"Iteration": np.arange(1, len(loss) + 1), "Loss": loss})
                st.plotly_chart(px.line(frame, x="Iteration", y="Loss", title="Training Loss", template="plotly_white"), use_container_width=True)
                st.metric("Iterations Used", diag.get("n_iter", "—"))

    with tab_export:
        st.subheader("💾 Export Final Model")
        if final_bundle is not None:
            st.success(f"🏆 Locked final model: **{best_name}**")
            st.download_button("📥 Download Final Pipeline Bundle", data=bundle_to_bytes(final_bundle),
                               file_name="ApexML_final_model_bundle.pkl", mime="application/octet-stream", use_container_width=True)
        st.download_button("📥 Download Leaderboard CSV", data=results.to_csv(index=False).encode("utf-8"),
                           file_name="ApexML_leaderboard.csv", mime="text/csv", use_container_width=True)
        if competition is not None:
            st.download_button("📥 Download Competition CV Ranking", data=competition["leaderboard"].to_csv(index=False).encode("utf-8"),
                               file_name="ApexML_competition_cv_ranking.csv", mime="text/csv", use_container_width=True)



def _render_unsupervised_evaluation():
    import pickle
    from core.unsupervised_engine import cluster_size_table, cluster_profile_table

    st.title("🏆 Unsupervised Evaluation")
    st.markdown("Evaluate clustering quality, inspect cluster profiles, compare saved runs, and export the fitted unsupervised bundle.")

    bundles = st.session_state.get("unsup_bundles") or {}
    rows = st.session_state.get("unsup_results") or []
    if not bundles:
        st.warning("⚠️ No unsupervised run yet. Go to **ML Studio → Unsupervised Learning** and run an algorithm first.")
        return

    results = pd.DataFrame(rows)
    if not results.empty:
        st.subheader("📋 Saved Unsupervised Runs")
        st.dataframe(results, use_container_width=True, hide_index=True)

        cluster_rows = results[results.get("Algorithm", pd.Series(dtype=str)).isin(["K-Means", "Hierarchical", "DBSCAN"])].copy() if "Algorithm" in results else pd.DataFrame()
        if not cluster_rows.empty and "Silhouette" in cluster_rows:
            plot_rows = cluster_rows.dropna(subset=["Silhouette"])
            if not plot_rows.empty:
                fig = px.bar(
                    plot_rows.sort_values("Silhouette", ascending=False), x="Run", y="Silhouette",
                    color="Algorithm", title="Clustering Silhouette — Higher is Better",
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Compare clustering metrics only when runs used the same dataset and feature set.")

    names = list(bundles.keys())
    active = st.session_state.get("unsup_active_run")
    index = names.index(active) if active in names else len(names) - 1
    run_name = st.selectbox("Inspect run", names, index=max(0, index), key="unsup_eval_run")
    bundle = bundles[run_name]
    algorithm = bundle.get("algorithm", "Unknown")
    metrics = bundle.get("metrics") or {}

    st.markdown("---")
    a, b, c = st.columns(3)
    a.metric("Algorithm", algorithm)
    b.metric("Features", len(bundle.get("feature_columns") or []))
    c.metric("Training Rows", len(bundle.get("train_frame", [])) if bundle.get("train_frame") is not None else "—")
    st.caption("Required features: " + ", ".join(bundle.get("feature_columns") or []))

    if algorithm in {"K-Means Clustering", "Hierarchical Clustering", "DBSCAN"}:
        labels = np.asarray(bundle.get("labels"))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clusters", int(metrics.get("Clusters", 0)))
        sil = metrics.get("Silhouette", np.nan)
        db = metrics.get("Davies_Bouldin", np.nan)
        ch = metrics.get("Calinski_Harabasz", np.nan)
        m2.metric("Silhouette ↑", "—" if pd.isna(sil) else f"{float(sil):.4f}")
        m3.metric("Davies-Bouldin ↓", "—" if pd.isna(db) else f"{float(db):.4f}")
        m4.metric("Noise Rate ↓", f"{float(metrics.get('Noise_Rate', 0)):.1%}")
        if not pd.isna(ch):
            st.caption(f"Calinski-Harabasz ↑: **{float(ch):,.2f}**")

        tab_size, tab_profile, tab_vis, tab_rows = st.tabs(["Cluster Sizes", "Cluster Profiles", "2D View", "Assigned Training Rows"])
        with tab_size:
            sizes = cluster_size_table(labels)
            st.dataframe(
                sizes, use_container_width=True, hide_index=True,
                column_config={"Share": st.column_config.ProgressColumn("Share", min_value=0, max_value=1, format="%.1%%")},
            )
        with tab_profile:
            train_frame = bundle.get("train_frame")
            if train_frame is not None:
                profile = cluster_profile_table(train_frame, labels)
                st.dataframe(profile, use_container_width=True, hide_index=True)
                st.caption("Profiles show the mean value of each selected feature inside each cluster.")
        with tab_vis:
            projection = bundle.get("projection_2d")
            if projection is not None and len(projection) == len(labels):
                frame = pd.DataFrame({"PC1": projection[:, 0], "PC2": projection[:, 1], "Cluster": labels.astype(str)})
                st.plotly_chart(px.scatter(frame, x="PC1", y="PC2", color="Cluster", opacity=.7, template="plotly_white", title=f"{run_name} — PCA View"), use_container_width=True)
            else:
                st.info("No stored 2D projection for this run.")
        with tab_rows:
            train_frame = bundle.get("train_frame")
            if train_frame is not None:
                assigned = train_frame.copy().reset_index(drop=True)
                assigned.insert(0, "Cluster", labels)
                idx = bundle.get("training_index")
                if idx is not None and len(idx) == len(assigned):
                    assigned.insert(0, "Original_Index", idx)
                st.dataframe(assigned.head(500), use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download Cluster Assignments CSV", assigned.to_csv(index=False).encode("utf-8"),
                    file_name="ApexML_unsupervised_cluster_assignments.csv", mime="text/csv", use_container_width=True,
                )

        if algorithm == "Hierarchical Clustering":
            st.info("Inference uses nearest-centroid assignment because sklearn AgglomerativeClustering has no native predict method.")
        elif algorithm == "DBSCAN":
            st.info("Inference uses the fitted core points: a new row is assigned only if its nearest core point is within eps; otherwise it is Noise (-1).")

    elif algorithm == "PCA":
        st.subheader("📉 Dimensionality Reduction Evaluation")
        cumulative = metrics.get("Cumulative_Explained_Variance", np.nan)
        st.metric("Cumulative Explained Variance ↑", "—" if pd.isna(cumulative) else f"{float(cumulative):.1%}")
        variance_rows = []
        for key, value in metrics.items():
            if key.startswith("PC") and key.endswith("_Variance"):
                variance_rows.append({"Component": key.replace("_Variance", ""), "Explained Variance": float(value)})
        if variance_rows:
            vf = pd.DataFrame(variance_rows)
            st.plotly_chart(px.bar(vf, x="Component", y="Explained Variance", title="Explained Variance by Component", template="plotly_white"), use_container_width=True)
        projection = bundle.get("projection")
        if projection is not None and np.asarray(projection).shape[1] >= 2:
            pf = pd.DataFrame({"PC1": projection[:, 0], "PC2": projection[:, 1]})
            st.plotly_chart(px.scatter(pf, x="PC1", y="PC2", opacity=.6, title="PCA Training Projection", template="plotly_white"), use_container_width=True)
        st.success("PCA supports native transform for future rows in the Inference page.")

    elif algorithm == "t-SNE":
        trust = metrics.get("Trustworthiness", np.nan)
        st.subheader("🗺️ t-SNE Evaluation")
        st.metric("Trustworthiness ↑", "—" if pd.isna(trust) else f"{float(trust):.4f}")
        projection = bundle.get("projection")
        if projection is not None:
            pf = pd.DataFrame({"Dim 1": projection[:, 0], "Dim 2": projection[:, 1]})
            st.plotly_chart(px.scatter(pf, x="Dim 1", y="Dim 2", opacity=.6, title="t-SNE Projection", template="plotly_white"), use_container_width=True)
        st.warning("t-SNE is mainly a visualization method and does not provide a native out-of-sample transform for new rows in scikit-learn.")

    elif algorithm == "Association Rules (Apriori)":
        st.subheader("🧺 Association Rule Evaluation")
        rules = bundle.get("rules")
        r1, r2, r3 = st.columns(3)
        r1.metric("Rules", int(metrics.get("Rules", 0)))
        conf = metrics.get("Mean_Confidence", np.nan)
        lift = metrics.get("Max_Lift", np.nan)
        r2.metric("Mean Confidence ↑", "—" if pd.isna(conf) else f"{float(conf):.3f}")
        r3.metric("Max Lift ↑", "—" if pd.isna(lift) else f"{float(lift):.3f}")
        if isinstance(rules, pd.DataFrame):
            st.dataframe(rules, use_container_width=True, hide_index=True)
            st.download_button("📥 Download Rules CSV", rules.to_csv(index=False).encode("utf-8"), file_name="ApexML_apriori_rules.csv", mime="text/csv")

    st.markdown("---")
    with st.expander("💾 Export fitted unsupervised bundle", expanded=False):
        st.download_button(
            "📦 Download Unsupervised Bundle (.pkl)", pickle.dumps(bundle),
            file_name="ApexML_unsupervised_bundle.pkl", mime="application/octet-stream", use_container_width=True,
        )
        st.caption("The bundle includes the selected feature schema and the fitted scaler/model needed for supported future-row inference.")


def render():
    has_supervised = st.session_state.get("model_results") is not None and bool(st.session_state.get("model_bundles"))
    has_unsupervised = bool(st.session_state.get("unsup_bundles"))

    if has_supervised and has_unsupervised:
        view = st.radio("Evaluation Type", ["🎯 Supervised", "🔍 Unsupervised"], horizontal=True, key="evaluation_type")
        if "Unsupervised" in view:
            _render_unsupervised_evaluation()
        else:
            _render_supervised_evaluation()
    elif has_unsupervised:
        _render_unsupervised_evaluation()
    else:
        _render_supervised_evaluation()
