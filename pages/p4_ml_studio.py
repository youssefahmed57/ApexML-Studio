"""
pages/p4_ml_studio.py
ApexML-style Machine Learning Studio with leakage-safe training, full model control,
and a competition-focused final model workflow.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.base import clone

from core.engine import (
    DatasetConfig, infer_task, detect_id_like_columns, detect_date_candidates,
    prepare_dataframe, train_models, recommended_models, default_params_for_models,
    data_health_report, leakage_scan, class_balance_report,
)
from core.competition import run_competition_training
from core.unsupervised_engine import (
    evaluate_clustering, cluster_size_table, cluster_profile_table,
)
from ui.model_controls import model_params_ui
from ml_models.unsupervised import (
    run_kmeans_elbow, fit_kmeans, fit_hierarchical, fit_dbscan,
    run_pca, run_tsne, run_apriori,
    plot_elbow, plot_clusters_2d, plot_dendrogram,
)


def _require_df():
    if st.session_state.get("df_original") is None:
        st.warning("⚠️ Please upload a dataset first on the Dataset Overview page.")
        st.stop()
    return st.session_state["df_original"]


def _default_pipeline():
    return {
        "numeric_imputer": "median",
        "categorical_imputer": "most_frequent",
        "scaler": "StandardScaler",
        "outlier_strategy": "None",
        "iqr_factor": 1.5,
        "max_categories": 60,
        "min_category_frequency": None,
        "remove_zero_variance": True,
        "feature_selection": "None",
        "k_best": None,
        "categorical_encoder": "OneHot",
        "imbalance_strategy": "None",
        "test_size": 0.20,
        "random_state": 42,
    }


def _model_options(task):
    if task == "classification":
        return [
            "Logistic Regression", "Naive Bayes", "Decision Tree", "Random Forest",
            "Extra Trees", "Gradient Boosting", "Hist Gradient Boosting", "AdaBoost",
            "KNN", "SVM", "XGBoost", "LightGBM", "Neural Network",
        ]
    return [
        "Linear Regression", "Polynomial Regression", "Ridge", "Lasso", "ElasticNet",
        "Decision Tree", "Random Forest", "Extra Trees", "Gradient Boosting",
        "Hist Gradient Boosting", "AdaBoost", "KNN", "SVM", "XGBoost", "LightGBM",
        "Neural Network",
    ]


def _config(setup, pipe_cfg):
    return DatasetConfig(
        target=setup["target"], task=setup["task"],
        excluded_columns=setup.get("excluded", []),
        date_columns=setup.get("date_cols", []),
        force_numeric=setup.get("force_numeric", []),
        force_categorical=setup.get("force_cat", []),
        test_size=float(pipe_cfg.get("test_size", .20)),
        random_state=int(pipe_cfg.get("random_state", 42)),
        numeric_imputer=pipe_cfg.get("numeric_imputer", "median"),
        categorical_imputer=pipe_cfg.get("categorical_imputer", "most_frequent"),
        scaler=pipe_cfg.get("scaler", "StandardScaler"),
        outlier_strategy=pipe_cfg.get("outlier_strategy", "None"),
        iqr_factor=float(pipe_cfg.get("iqr_factor", 1.5)),
        max_categories=pipe_cfg.get("max_categories", 60),
        min_category_frequency=pipe_cfg.get("min_category_frequency"),
        remove_zero_variance=bool(pipe_cfg.get("remove_zero_variance", True)),
        feature_selection=pipe_cfg.get("feature_selection", "None"),
        k_best=pipe_cfg.get("k_best"),
        categorical_encoder=pipe_cfg.get("categorical_encoder", "OneHot"),
        imbalance_strategy=pipe_cfg.get("imbalance_strategy", "None") if setup["task"] == "classification" else "None",
        feature_recipe=st.session_state.get("feature_recipe", []),
    )


def render():
    st.title("🤖 Machine Learning Studio")
    st.markdown("Configure the problem, control every model if you want, then train a robust final predictor.")
    df = _require_df()

    learning_type = st.radio(
        "**Learning Type**",
        ["🎯 Supervised Learning", "🔍 Unsupervised Learning"],
        horizontal=True,
        key="learning_type",
    )
    if "Supervised" in learning_type:
        _render_supervised(df)
    else:
        _render_unsupervised(st.session_state.get("df") if st.session_state.get("df") is not None else df)


def _render_supervised(df: pd.DataFrame):
    tab_setup, tab_select, tab_controls, tab_train = st.tabs([
        "⚙️ Setup", "🎛️ Algorithms", "🧠 Model Controls", "🚀 Train Best Model"
    ])

    with tab_setup:
        st.subheader("⚙️ Dataset & Target Setup")
        saved = st.session_state.get("dataset_setup") or {}
        cols = df.columns.tolist()
        old_target = saved.get("target")
        idx = cols.index(old_target) if old_target in cols else len(cols) - 1
        target = st.selectbox("🎯 Select Target Variable", cols, index=max(0, idx), key="target_col_safe")
        inferred = infer_task(df[target])
        auto_name = "Classification" if inferred == "classification" else "Regression"
        task_name = st.radio(
            "Task Type (auto-detected)", ["Classification", "Regression"],
            index=0 if inferred == "classification" else 1, horizontal=True, key="task_type_safe",
        )
        task = task_name.lower()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Auto Detected", auto_name)
        m2.metric("Using", task_name)
        m3.metric("Target Unique", int(df[target].nunique(dropna=True)))
        m4.metric("Target Missing", int(df[target].isna().sum()))

        candidates = [c for c in cols if c != target]
        default_ids = [c for c in detect_id_like_columns(df, target) if c in candidates]
        excluded = st.multiselect(
            "Exclude IDs / leakage columns",
            candidates,
            default=[c for c in saved.get("excluded", default_ids) if c in candidates],
            help="IDs, row numbers, or any field that reveals the answer after the event should be excluded.",
        )
        remaining = [c for c in candidates if c not in excluded]
        default_dates = [c for c in detect_date_candidates(df, target) if c in remaining]
        date_cols = st.multiselect("Date/Time columns", remaining, default=[c for c in saved.get("date_cols", default_dates) if c in remaining])

        with st.expander("Advanced column roles", expanded=False):
            role_cols = [c for c in remaining if c not in date_cols]
            c1, c2 = st.columns(2)
            with c1:
                force_numeric = st.multiselect("Force numeric", role_cols, default=[c for c in saved.get("force_numeric", []) if c in role_cols])
            with c2:
                force_cat = st.multiselect("Force categorical", [c for c in role_cols if c not in force_numeric], default=[c for c in saved.get("force_cat", []) if c in role_cols])

        if task == "classification":
            st.markdown("**Class Balance**")
            try:
                st.dataframe(class_balance_report(df[target].dropna()), use_container_width=True, hide_index=True)
            except Exception:
                pass

        with st.expander("🛡️ Quick Leakage Scan", expanded=True):
            try:
                leak = leakage_scan(df, target, task)
                risky = leak[leak["Risk"].isin(["Critical", "High"])] if not leak.empty else leak
                if risky is not None and not risky.empty:
                    st.error("Possible target leakage detected. Review/exclude suspicious columns before final training.")
                    st.dataframe(risky, use_container_width=True, hide_index=True)
                else:
                    st.success("No obvious Critical/High leakage signal detected by the automatic scanner.")
            except Exception as exc:
                st.caption(f"Leakage scan unavailable: {exc}")

        if st.button("✅ Save Setup", type="primary", key="save_ml_setup"):
            st.session_state["dataset_setup"] = {
                "target": target, "task": task, "excluded": excluded,
                "date_cols": date_cols, "force_numeric": force_numeric, "force_cat": force_cat,
            }
            if st.session_state.get("pipeline_config") is None:
                st.session_state["pipeline_config"] = _default_pipeline()
            st.session_state["task_type"] = task_name
            st.session_state["target_col"] = target
            st.success("✅ Setup saved. Go to Algorithms and choose what you want to train.")

    setup = st.session_state.get("dataset_setup")
    if not setup:
        with tab_select: st.info("Save the Setup first.")
        with tab_controls: st.info("Save the Setup first.")
        with tab_train: st.info("Save the Setup first.")
        return

    task = setup["task"]
    options = _model_options(task)

    with tab_select:
        st.subheader(f"🎛️ {setup['task'].title()} Algorithm Selection")
        default_models = recommended_models(task, len(df), max(1, len(df.columns)-1), full=False)
        default_models = [m for m in default_models if m in options][:6]
        selected = st.multiselect(
            "Select algorithms to train",
            options,
            default=st.session_state.get("selected_models") or default_models,
            key="selected_models_safe",
        )
        st.session_state["selected_models"] = selected

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Recommended starting set**")
            st.write(", ".join(default_models) if default_models else "Choose at least 2 diverse model families.")
        with c2:
            st.markdown("**Why compare multiple families?**")
            st.caption("The best algorithm depends on the dataset. Linear, tree, boosting, distance, SVM and neural models have different inductive biases.")

    with tab_controls:
        selected = st.session_state.get("selected_models", [])
        if not selected:
            st.info("Choose algorithms first.")
        else:
            st.subheader("🧠 Full Model Control")
            st.caption("You can leave defaults for an easy run, or open any model and control its hyperparameters. Neural Network supports Auto or Custom hidden layers.")
            params = {}
            for name in selected:
                params[name] = model_params_ui(name, task, prefix=f"apex_{task}_{name.replace(' ', '_')}")
            st.session_state["model_params"] = params

    with tab_train:
        selected = st.session_state.get("selected_models", [])
        if not selected:
            st.info("Choose algorithms first.")
            return

        pipe_cfg = st.session_state.get("pipeline_config") or _default_pipeline()
        st.subheader("🚀 Train & Select the Final Model")
        method = st.radio(
            "Training strategy",
            ["Standard Benchmark", "Competition Boost"],
            horizontal=True,
            help="Standard is faster. Competition Boost ranks candidates with repeated CV, tunes strong models, tests an ensemble, and refits the winner on all labeled data.",
        )
        if task == "classification":
            metric = st.selectbox("Primary metric", ["Accuracy", "Balanced Accuracy", "Weighted F1", "Macro F1"], index=0)
        else:
            metric = st.selectbox("Primary metric", ["RMSE", "MAE", "R²"], index=0)

        if method == "Standard Benchmark":
            cv_folds = st.slider("Cross Validation folds", 2, 10, 5)
            st.caption("Fast and transparent: one holdout split + K-fold Cross Validation for every selected model.")
        else:
            competition_mode = st.selectbox("Competition strength", ["Fast", "Competition", "Maximum"], index=1)
            st.caption("Competition = repeated CV + tuning of top models + voting ensemble + full-data refit. Maximum also tries stacking and a wider search.")

        with st.expander("Current safe pipeline", expanded=False):
            st.json(pipe_cfg)
            recipes = st.session_state.get("feature_recipe", [])
            if recipes:
                st.markdown("**Feature recipe from Feature Engineering page**")
                st.dataframe(pd.DataFrame(recipes), use_container_width=True, hide_index=True)

        if st.button("🚀 Start Training", type="primary", key="start_safe_training"):
            try:
                X, y, _ = prepare_dataframe(
                    df, setup["target"], setup["excluded"], setup["date_cols"],
                    setup["force_numeric"], setup["force_cat"],
                )
                cfg = _config(setup, pipe_cfg)
                params = st.session_state.get("model_params") or default_params_for_models(selected, task)

                with st.status("Training models...", expanded=True) as status:
                    if method == "Standard Benchmark":
                        st.write("Building leakage-safe pipelines and fitting selected models...")
                        results, bundles, details, split = train_models(
                            X, y, task, selected, params, cfg, cv_folds=int(cv_folds), run_cv=True,
                        )
                        metric_col = {
                            "Accuracy": "Test_Accuracy", "Balanced Accuracy": "Test_Balanced_Accuracy",
                            "Weighted F1": "Test_F1_Weighted", "Macro F1": "Test_F1_Macro",
                            "RMSE": "Test_RMSE", "MAE": "Test_MAE", "R²": "Test_R2",
                        }[metric]
                        ascending = metric in {"RMSE", "MAE"}
                        best_name = results.sort_values(metric_col, ascending=ascending).iloc[0]["Model"]
                        # Refit the selected standard winner on ALL labeled rows before inference.
                        final_bundle = dict(bundles[best_name])
                        final_pipe = clone(bundles[best_name]["pipeline"])
                        if task == "classification":
                            _enc = bundles[best_name].get("target_encoder")
                            _y_full = _enc.transform(y.astype(str))
                        else:
                            _y_full = pd.to_numeric(y, errors="coerce")
                            _valid = _y_full.notna()
                            X = X.loc[_valid].reset_index(drop=True)
                            _y_full = _y_full.loc[_valid].to_numpy(dtype=float)
                        final_pipe.fit(X, _y_full)
                        final_bundle["pipeline"] = final_pipe
                        final_bundle["final_refit"] = {"trained_on_all_labeled_rows": True, "rows": int(len(X))}

                        st.session_state["model_results"] = results
                        st.session_state["model_bundles"] = bundles
                        st.session_state["model_details"] = details
                        st.session_state["split_data_safe"] = split
                        st.session_state["final_bundle"] = final_bundle
                        st.session_state["competition_run"] = None
                        st.session_state["primary_metric"] = metric
                        st.session_state["best_model_name"] = best_name
                        st.write(f"Selected best holdout model and refit it on all labeled rows: {best_name}")
                    else:
                        st.write("Running repeated Cross Validation and competition model selection...")
                        run = run_competition_training(
                            X, y, task, cfg, metric_choice=metric, mode=competition_mode,
                            selected_models=selected, custom_params=params,
                        )
                        st.session_state["competition_run"] = run
                        st.session_state["model_results"] = run["holdout_results"]
                        st.session_state["model_bundles"] = run["holdout_bundles"]
                        st.session_state["model_details"] = run["holdout_details"]
                        st.session_state["split_data_safe"] = run["split_data"]
                        st.session_state["final_bundle"] = run["final_bundle"]
                        st.session_state["primary_metric"] = metric
                        st.session_state["best_model_name"] = run["winner_name"]
                        st.write(f"Competition winner: {run['winner_name']}")
                    st.session_state["eval_results"] = st.session_state["model_results"].to_dict("records")
                    st.session_state["eval_task_type"] = task
                    status.update(label="✅ Training complete", state="complete")
                st.success("🏆 Final model is ready. Open Evaluation to inspect it, then Inference for the 10 new cases.")
            except Exception as exc:
                st.exception(exc)

        if st.session_state.get("final_bundle") is not None:
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            c1.metric("🏆 Final Model", st.session_state.get("best_model_name", "Ready"))
            c2.metric("Target", setup["target"])
            c3.metric("Task", task.title())
            if st.session_state.get("competition_run") is not None:
                exp = st.session_state["competition_run"].get("expected_correct_out_of_10")
                if exp is not None:
                    st.info(f"Repeated-CV estimate ≈ **{exp:.1f}/10** correct cases if the hidden cases follow the same distribution. This is an estimate, not a guarantee.")


def _render_unsupervised(df: pd.DataFrame):
    """Train reusable unsupervised runs and save them for Evaluation/Inference."""
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if not num_cols:
        st.error("Unsupervised learning requires numerical columns. Please encode categoricals first.")
        return

    st.caption(
        "Unsupervised runs are now saved as reusable bundles. Clustering runs can be evaluated "
        "and assigned to new cases from the existing Evaluation and Inference pages."
    )

    mode = st.selectbox(
        "Unsupervised Mode",
        ["K-Means Clustering", "Hierarchical Clustering", "DBSCAN", "PCA", "t-SNE", "Association Rules (Apriori)"],
        key="unsup_mode",
    )

    feature_cols = st.multiselect(
        "Select features", num_cols, default=num_cols[:min(5, len(num_cols))], key="unsup_feat"
    )
    if len(feature_cols) < 2:
        st.warning("Please select at least 2 numeric features.")
        return

    from sklearn.preprocessing import StandardScaler

    unsup_frame = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(unsup_frame) < 3:
        st.error("At least 3 complete rows are required after removing missing values.")
        return

    X_raw = unsup_frame.to_numpy(dtype=float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    fill_values = {
        c: float(pd.to_numeric(unsup_frame[c], errors="coerce").median())
        for c in feature_cols
    }

    st.info(
        f"📐 Working with **{X.shape[0]:,} complete rows × {X.shape[1]} features** "
        f"(auto-scaled; {len(df) - len(unsup_frame):,} incomplete row(s) excluded for this analysis)"
    )

    def store_run(run_name: str, bundle: dict, result_row: dict):
        bundles = dict(st.session_state.get("unsup_bundles") or {})
        bundles[run_name] = bundle
        st.session_state["unsup_bundles"] = bundles
        st.session_state["unsup_active_run"] = run_name

        rows = list(st.session_state.get("unsup_results") or [])
        rows = [r for r in rows if r.get("Run") != run_name]
        rows.append({"Run": run_name, **result_row})
        st.session_state["unsup_results"] = rows
        st.success(f"✅ Saved **{run_name}** for Evaluation and Inference.")

    def show_cluster_evaluation(metrics: dict, labels: np.ndarray):
        st.markdown("### 📊 Internal Evaluation")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clusters", int(metrics.get("Clusters", 0)))
        sil = metrics.get("Silhouette", np.nan)
        db = metrics.get("Davies_Bouldin", np.nan)
        ch = metrics.get("Calinski_Harabasz", np.nan)
        c2.metric("Silhouette ↑", "—" if pd.isna(sil) else f"{float(sil):.4f}")
        c3.metric("Davies-Bouldin ↓", "—" if pd.isna(db) else f"{float(db):.4f}")
        c4.metric("Noise", f"{int(metrics.get('Noise_Points', 0))} ({float(metrics.get('Noise_Rate', 0)):.1%})")
        if not pd.isna(ch):
            st.caption(f"Calinski-Harabasz: **{float(ch):,.2f}** (higher is better when comparing runs on the same data).")
        st.dataframe(
            cluster_size_table(labels), use_container_width=True, hide_index=True,
            column_config={"Share": st.column_config.ProgressColumn("Share", min_value=0, max_value=1, format="%.1%%")},
        )

    # ── K-Means ────────────────────────────────────────────────────────────────
    if mode == "K-Means Clustering":
        st.subheader("K-Means Clustering")
        with st.spinner("Running elbow & silhouette analysis..."):
            k_vals, inertias, sil_scores = run_kmeans_elbow(X)
        st.plotly_chart(plot_elbow(k_vals, inertias, sil_scores), use_container_width=True)

        valid_sil = np.asarray(sil_scores, dtype=float)
        best_k = k_vals[int(np.nanargmax(valid_sil))] if np.isfinite(valid_sil).any() else k_vals[0]
        max_k = max(2, min(15, len(X) - 1))
        n_clusters = st.slider("Number of clusters (K)", 2, max_k, min(best_k, max_k), key="kmeans_k")
        st.caption(f"💡 Highest silhouette score in the scan at K = **{best_k}**")

        if st.button("🚀 Run K-Means", key="btn_kmeans"):
            model = fit_kmeans(X, n_clusters)
            labels = model.labels_
            metrics = evaluate_clustering(X, labels, inertia=float(model.inertia_))
            df_vis = unsup_frame.copy()
            df_vis["Cluster"] = labels.astype(int)
            X_pca, _ = run_pca(X, n_components=2)

            st.plotly_chart(plot_clusters_2d(X_pca, labels, f"K-Means Clusters (K={n_clusters})"), use_container_width=True)
            show_cluster_evaluation(metrics, labels)

            run_name = f"K-Means | K={n_clusters}"
            bundle = {
                "algorithm": "K-Means Clustering",
                "run_name": run_name,
                "feature_columns": feature_cols,
                "scaler": scaler,
                "fill_values": fill_values,
                "model": model,
                "labels": np.asarray(labels),
                "X_train_scaled": np.asarray(X),
                "train_frame": unsup_frame.reset_index(drop=True),
                "training_index": unsup_frame.index.to_numpy(),
                "metrics": metrics,
                "projection_2d": X_pca,
                "inference_kind": "native_cluster_predict",
            }
            store_run(run_name, bundle, {"Algorithm": "K-Means", **metrics})

    # ── Hierarchical ───────────────────────────────────────────────────────────
    elif mode == "Hierarchical Clustering":
        st.subheader("Hierarchical Clustering")
        linkage_method = st.selectbox("Linkage method", ["ward", "complete", "average", "single"], key="hier_link")
        st.plotly_chart(plot_dendrogram(X, method=linkage_method), use_container_width=True)

        max_hier = max(2, min(15, len(X)))
        n_clusters = st.slider("Number of clusters", 2, max_hier, min(3, max_hier), key="hier_k")
        if st.button("🚀 Run Hierarchical", key="btn_hier"):
            model = fit_hierarchical(X, n_clusters, linkage_method)
            labels = model.labels_
            metrics = evaluate_clustering(X, labels)
            X_pca, _ = run_pca(X, n_components=2)
            st.plotly_chart(plot_clusters_2d(X_pca, labels, f"Hierarchical Clusters (K={n_clusters})"), use_container_width=True)
            show_cluster_evaluation(metrics, labels)

            run_name = f"Hierarchical | K={n_clusters} | {linkage_method}"
            bundle = {
                "algorithm": "Hierarchical Clustering",
                "run_name": run_name,
                "feature_columns": feature_cols,
                "scaler": scaler,
                "fill_values": fill_values,
                "model": model,
                "labels": np.asarray(labels),
                "X_train_scaled": np.asarray(X),
                "train_frame": unsup_frame.reset_index(drop=True),
                "training_index": unsup_frame.index.to_numpy(),
                "metrics": metrics,
                "projection_2d": X_pca,
                "linkage": linkage_method,
                "inference_kind": "nearest_centroid_assignment",
            }
            store_run(run_name, bundle, {"Algorithm": "Hierarchical", **metrics})
            st.caption("New-row assignment uses the nearest fitted cluster centroid because AgglomerativeClustering has no native predict method.")

    # ── DBSCAN ─────────────────────────────────────────────────────────────────
    elif mode == "DBSCAN":
        st.subheader("DBSCAN Clustering")
        col1, col2 = st.columns(2)
        with col1:
            eps = st.slider("eps (neighborhood radius)", 0.1, 5.0, 0.5, 0.1, key="dbscan_eps")
        with col2:
            min_samples = st.slider("min_samples", 2, 20, 5, key="dbscan_ms")

        if st.button("🚀 Run DBSCAN", key="btn_dbscan"):
            model = fit_dbscan(X, eps, min_samples)
            labels = model.labels_
            metrics = evaluate_clustering(X, labels)
            X_pca, _ = run_pca(X, n_components=2)
            st.plotly_chart(plot_clusters_2d(X_pca, labels, "DBSCAN Clusters"), use_container_width=True)
            show_cluster_evaluation(metrics, labels)

            run_name = f"DBSCAN | eps={eps:g} | min={min_samples}"
            bundle = {
                "algorithm": "DBSCAN",
                "run_name": run_name,
                "feature_columns": feature_cols,
                "scaler": scaler,
                "fill_values": fill_values,
                "model": model,
                "labels": np.asarray(labels),
                "X_train_scaled": np.asarray(X),
                "train_frame": unsup_frame.reset_index(drop=True),
                "training_index": unsup_frame.index.to_numpy(),
                "metrics": metrics,
                "projection_2d": X_pca,
                "eps": float(eps),
                "min_samples": int(min_samples),
                "inference_kind": "nearest_core_assignment",
            }
            store_run(run_name, bundle, {"Algorithm": "DBSCAN", **metrics})
            st.caption("For new rows, DBSCAN assigns the nearest fitted core-point cluster only when the point is within eps; otherwise it returns Noise (-1).")

    # ── PCA ────────────────────────────────────────────────────────────────────
    elif mode == "PCA":
        st.subheader("Principal Component Analysis")
        max_comp = min(len(feature_cols), len(X), 10)
        if max_comp < 2:
            st.warning("PCA needs at least two rows and two selected features.")
            return
        n_comp = st.slider("Number of components", 2, max_comp, 2, key="pca_comp")
        if st.button("🚀 Run PCA", key="btn_pca"):
            X_pca, pca_model = run_pca(X, n_comp)
            explained = pca_model.explained_variance_ratio_
            import plotly.express as px
            fig_var = px.bar(
                x=[f"PC{i+1}" for i in range(len(explained))], y=explained * 100,
                color=explained * 100, color_continuous_scale="Teal",
                labels={"x": "Component", "y": "Variance Explained (%)"},
                title="PCA Explained Variance", template="plotly_white",
            )
            st.plotly_chart(fig_var, use_container_width=True)
            cumulative = float(explained.sum())
            st.metric(f"Cumulative Variance ({n_comp} PCs)", f"{cumulative:.1%}")
            if n_comp >= 2:
                fig2 = px.scatter(
                    x=X_pca[:, 0], y=X_pca[:, 1], labels={"x": "PC1", "y": "PC2"},
                    title="PCA — 2D Projection", template="plotly_white",
                    color_discrete_sequence=["#10B981"], opacity=0.7,
                )
                st.plotly_chart(fig2, use_container_width=True)

            run_name = f"PCA | {n_comp} components"
            metrics = {
                "Components": int(n_comp),
                "Cumulative_Explained_Variance": cumulative,
                **{f"PC{i+1}_Variance": float(v) for i, v in enumerate(explained)},
            }
            bundle = {
                "algorithm": "PCA", "run_name": run_name, "feature_columns": feature_cols,
                "scaler": scaler, "fill_values": fill_values, "model": pca_model,
                "train_frame": unsup_frame.reset_index(drop=True), "metrics": metrics,
                "projection": X_pca, "inference_kind": "native_transform",
            }
            store_run(run_name, bundle, {"Algorithm": "PCA", **metrics})

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    elif mode == "t-SNE":
        st.subheader("t-SNE Dimensionality Reduction")
        max_perplexity = max(2, min(50, len(X) - 1))
        default_perplexity = min(30, max_perplexity)
        perplexity = st.slider("Perplexity", 2, max_perplexity, default_perplexity, key="tsne_perp")
        color_col = st.selectbox("Color by (optional)", ["None"] + df.columns.tolist(), key="tsne_color")
        if st.button("🚀 Run t-SNE", key="btn_tsne"):
            with st.spinner("Running t-SNE (may take a moment)..."):
                X_tsne = run_tsne(X, n_components=2, perplexity=perplexity)
            import plotly.express as px
            color_vals = df.loc[unsup_frame.index, color_col].values if color_col != "None" else None
            fig_tsne = px.scatter(
                x=X_tsne[:, 0], y=X_tsne[:, 1],
                color=color_vals.astype(str) if color_vals is not None else None,
                labels={"x": "Dim 1", "y": "Dim 2"}, title="t-SNE 2D Projection",
                template="plotly_white", color_discrete_sequence=px.colors.qualitative.Bold, opacity=0.7,
            )
            st.plotly_chart(fig_tsne, use_container_width=True)

            try:
                from sklearn.manifold import trustworthiness
                n_eval = min(2000, len(X))
                trust = float(trustworthiness(X[:n_eval], X_tsne[:n_eval], n_neighbors=min(5, max(1, n_eval // 10))))
            except Exception:
                trust = np.nan
            if not pd.isna(trust):
                st.metric("Trustworthiness ↑", f"{trust:.4f}")

            run_name = f"t-SNE | perplexity={perplexity}"
            metrics = {"Trustworthiness": trust, "Perplexity": float(perplexity)}
            bundle = {
                "algorithm": "t-SNE", "run_name": run_name, "feature_columns": feature_cols,
                "scaler": scaler, "fill_values": fill_values, "projection": X_tsne,
                "train_frame": unsup_frame.reset_index(drop=True), "metrics": metrics,
                "inference_kind": "not_supported",
            }
            store_run(run_name, bundle, {"Algorithm": "t-SNE", **metrics})
            st.info("t-SNE is visualization-oriented and has no native transform for future rows. Use PCA if you need reusable projection.")

    # ── Apriori ────────────────────────────────────────────────────────────────
    elif mode == "Association Rules (Apriori)":
        st.subheader("Apriori — Association Rules")
        st.info("This requires a binary/transactional dataset (columns are items, values are 0/1).")
        col1, col2 = st.columns(2)
        with col1:
            min_support = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01, key="apr_sup")
        with col2:
            min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.5, 0.05, key="apr_conf")
        if st.button("🚀 Run Apriori", key="btn_apriori"):
            binary_df = df[feature_cols].fillna(0).astype(bool)
            rules = run_apriori(binary_df, min_support, min_confidence)
            if rules.empty:
                st.warning("No association rules found. Try lowering min_support or min_confidence.")
            elif "Error" in rules.columns:
                st.error(str(rules.iloc[0]["Error"]))
            else:
                st.success(f"Found **{len(rules)}** rules!")
                st.dataframe(rules, use_container_width=True, hide_index=True)
                run_name = f"Apriori | support={min_support:g} | confidence={min_confidence:g}"
                metrics = {
                    "Rules": int(len(rules)),
                    "Mean_Confidence": float(rules["confidence"].mean()) if "confidence" in rules else np.nan,
                    "Max_Lift": float(rules["lift"].max()) if "lift" in rules else np.nan,
                }
                bundle = {
                    "algorithm": "Association Rules (Apriori)", "run_name": run_name,
                    "feature_columns": feature_cols, "rules": rules, "metrics": metrics,
                    "inference_kind": "rules",
                }
                store_run(run_name, bundle, {"Algorithm": "Apriori", **metrics})
