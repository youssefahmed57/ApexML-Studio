from __future__ import annotations

import copy
import math
import re
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    RepeatedStratifiedKFold,
    RepeatedKFold,
    cross_val_score,
    cross_val_predict,
    GridSearchCV,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from core.engine import (
    DatasetConfig,
    prepare_dataframe,
    train_models,
    default_tuning_grid,
    recommended_models,
    default_params_for_models,
    safe_cv_folds,
)


def competition_metric(task: str, metric: str) -> tuple[str, bool, str]:
    """Return sklearn scoring name, whether higher is better, and display label."""
    if task == "classification":
        mapping = {
            "Accuracy": ("accuracy", True, "Accuracy"),
            "Balanced Accuracy": ("balanced_accuracy", True, "Balanced Accuracy"),
            "Weighted F1": ("f1_weighted", True, "Weighted F1"),
            "Macro F1": ("f1_macro", True, "Macro F1"),
        }
        return mapping.get(metric, mapping["Accuracy"])

    mapping = {
        "RMSE": ("neg_root_mean_squared_error", False, "RMSE"),
        "MAE": ("neg_mean_absolute_error", False, "MAE"),
        "R²": ("r2", True, "R²"),
    }
    return mapping.get(metric, mapping["RMSE"])


def _encode_target(y: pd.Series, task: str, target_encoder=None):
    if task == "classification":
        if target_encoder is None:
            raise ValueError("Classification target encoder is missing.")
        return target_encoder.transform(y.astype(str))
    numeric = pd.to_numeric(y, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Regression target contains non-numeric values after setup.")
    return np.asarray(numeric, dtype=float)


def _cv_splitter(task: str, y, folds: int, repeats: int, seed: int):
    folds = safe_cv_folds(task, y, folds)
    repeats = max(1, int(repeats))
    if task == "classification":
        if repeats == 1:
            return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), folds
        return RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed), folds
    if repeats == 1:
        return KFold(n_splits=folds, shuffle=True, random_state=seed), folds
    return RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed), folds


def _normalize_cv_scores(raw_scores: np.ndarray, scoring: str):
    arr = np.asarray(raw_scores, dtype=float)
    if scoring.startswith("neg_"):
        arr = -arr
    return arr


def _stability_score(mean: float, std: float, higher_is_better: bool, penalty: float = 0.10) -> float:
    if higher_is_better:
        return float(mean - penalty * std)
    # lower errors are better, so ranking score is negated error + stability penalty
    return float(-(mean + penalty * std))


def _safe_name(name: str, idx: int) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower()).strip("_")
    return f"m{idx}_{cleaned[:35]}"


def _candidate_cv(estimator, X, y, task: str, scoring: str, folds: int, repeats: int, seed: int):
    splitter, actual_folds = _cv_splitter(task, y, folds, repeats, seed)
    t0 = time.perf_counter()
    raw = cross_val_score(estimator, X, y, cv=splitter, scoring=scoring, n_jobs=1)
    elapsed = time.perf_counter() - t0
    scores = _normalize_cv_scores(raw, scoring)
    return {
        "CV_Mean": float(np.mean(scores)),
        "CV_Std": float(np.std(scores)),
        "CV_Min": float(np.min(scores)),
        "CV_Max": float(np.max(scores)),
        "CV_Folds": int(actual_folds),
        "CV_Repeats": int(repeats),
        "CV_Seconds": float(elapsed),
    }


def _tune_candidate(estimator, model_name: str, task: str, X, y, scoring: str, folds: int, seed: int, size: str):
    grid = default_tuning_grid(model_name, task, size=size)
    if not grid:
        return None

    cv, actual_folds = _cv_splitter(task, y, folds, 1, seed)
    search = GridSearchCV(
        estimator=clone(estimator),
        param_grid=grid,
        scoring=scoring,
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - t0

    # Convert negative loss scores back to positive error values for display.
    best_display = float(-search.best_score_) if scoring.startswith("neg_") else float(search.best_score_)
    best_idx = int(search.best_index_)
    std_raw = float(search.cv_results_["std_test_score"][best_idx])
    best_std = std_raw

    return {
        "estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "cv_mean": best_display,
        "cv_std": best_std,
        "seconds": float(elapsed),
        "folds": int(actual_folds),
    }


class CompetitionEnsemble:
    """Small fitted wrapper used as the final external-test predictor."""

    def __init__(self, task: str, fitted_models: list[Any], weights: list[float] | None = None, threshold: float | None = None):
        self.task = task
        self.fitted_models = fitted_models
        self.weights = np.asarray(weights if weights is not None else np.ones(len(fitted_models)), dtype=float)
        if self.weights.sum() <= 0:
            self.weights = np.ones(len(fitted_models), dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.threshold = threshold

    def predict_proba(self, X):
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification.")
        probs = [m.predict_proba(X) for m in self.fitted_models]
        return np.average(np.stack(probs, axis=0), axis=0, weights=self.weights)

    def predict(self, X):
        if self.task == "classification":
            probs = self.predict_proba(X)
            if probs.shape[1] == 2 and self.threshold is not None:
                return (probs[:, 1] >= self.threshold).astype(int)
            return np.argmax(probs, axis=1)
        preds = np.stack([m.predict(X) for m in self.fitted_models], axis=0)
        return np.average(preds, axis=0, weights=self.weights)


def _optimize_binary_threshold(estimator, X, y, scoring_choice: str, folds: int, seed: int):
    if not hasattr(estimator, "predict_proba"):
        return None
    unique = np.unique(y)
    if len(unique) != 2:
        return None

    actual_folds = safe_cv_folds("classification", y, folds)
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
    probs = cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]

    thresholds = np.linspace(0.20, 0.80, 121)
    rows = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        if scoring_choice == "Balanced Accuracy":
            score = balanced_accuracy_score(y, pred)
        elif scoring_choice == "Weighted F1":
            score = f1_score(y, pred, average="weighted", zero_division=0)
        elif scoring_choice == "Macro F1":
            score = f1_score(y, pred, average="macro", zero_division=0)
        else:
            score = accuracy_score(y, pred)
        rows.append((float(t), float(score)))

    frame = pd.DataFrame(rows, columns=["Threshold", "OOF_Score"])
    best = frame.sort_values(["OOF_Score", "Threshold"], ascending=[False, True]).iloc[0]
    default_score = float(frame.iloc[(frame["Threshold"] - 0.5).abs().argsort()[:1]]["OOF_Score"].iloc[0])
    best_threshold = float(best["Threshold"])
    best_score = float(best["OOF_Score"])

    # Avoid changing threshold for meaningless tiny gains.
    if best_score < default_score + 0.001:
        best_threshold = 0.5
        best_score = default_score

    return {
        "threshold": best_threshold,
        "oof_score": best_score,
        "default_score": default_score,
        "curve": frame,
        "folds": actual_folds,
    }


def _weights_from_scores(scores: list[float], higher_is_better: bool) -> list[float]:
    arr = np.asarray(scores, dtype=float)
    if higher_is_better:
        shifted = arr - np.nanmin(arr) + 0.02
        return shifted.tolist()
    # lower error -> higher inverse weight
    safe = np.maximum(arr, 1e-9)
    inv = 1.0 / safe
    return inv.tolist()


def run_competition_training(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    config: DatasetConfig,
    metric_choice: str,
    mode: str = "Competition",
    selected_models: list[str] | None = None,
    custom_params: dict[str, dict[str, Any]] | None = None,
):
    """Train specifically for an external 10-case evaluation.

    Strategy:
    1. Holdout benchmark for diagnostics.
    2. Repeated CV on the full labeled dataset for robust ranking.
    3. Tune only the strongest few model families.
    4. Evaluate a top-model voting ensemble.
    5. Optionally evaluate stacking in Maximum mode.
    6. Tune binary decision threshold from out-of-fold predictions.
    7. Refit the selected predictor on ALL labeled rows before external prediction.
    """
    scoring, higher_is_better, metric_label = competition_metric(task, metric_choice)

    # Normalize regression target once so every CV/tuning stage sees the exact same rows.
    if task == "regression":
        y_numeric = pd.to_numeric(y, errors="coerce")
        valid_target = y_numeric.notna()
        if not valid_target.all():
            X = X.loc[valid_target].reset_index(drop=True)
            y = y_numeric.loc[valid_target].reset_index(drop=True)
        else:
            y = y_numeric.reset_index(drop=True)
            X = X.reset_index(drop=True)
    else:
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    profile = {
        "Fast": {"folds": 5, "repeats": 1, "full_models": False, "tune_top": 1, "tune_size": "Quick", "stack": False},
        "Competition": {"folds": 5, "repeats": 2, "full_models": False, "tune_top": 2, "tune_size": "Quick", "stack": False},
        "Maximum": {"folds": 5, "repeats": 3, "full_models": True, "tune_top": 3, "tune_size": "Standard", "stack": True},
    }.get(mode, None)
    if profile is None:
        raise ValueError(f"Unknown competition mode: {mode}")

    if selected_models is None:
        selected_models = recommended_models(task, len(X), X.shape[1], full=profile["full_models"])
    selected_models = list(dict.fromkeys(selected_models))
    params = default_params_for_models(selected_models, task)
    if custom_params:
        for k, v in custom_params.items():
            params.setdefault(k, {}).update(v)

    # For clear class imbalance, use class weighting where supported rather than global oversampling by default.
    if task == "classification":
        counts = y.astype(str).value_counts()
        if len(counts) > 1 and counts.max() / max(counts.min(), 1) >= 2.0:
            for name in ["Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees", "SVM", "LightGBM"]:
                if name in params:
                    params[name]["class_weight"] = "balanced"

    # Existing safe train/holdout benchmark is still useful as a diagnostic view.
    holdout_results, bundles, details, split = train_models(
        X, y, task, selected_models, params, config,
        cv_folds=profile["folds"], run_cv=True,
    )

    first_bundle = next(iter(bundles.values()))
    y_model = _encode_target(y, task, first_bundle.get("target_encoder"))

    # Robust full-data CV ranking. This is the main competition selector because the real test is external.
    cv_rows = []
    estimator_versions: dict[str, Any] = {}
    for name, bundle in bundles.items():
        estimator = clone(bundle["pipeline"])
        try:
            stats = _candidate_cv(
                estimator, X, y_model, task, scoring,
                profile["folds"], profile["repeats"], config.random_state,
            )
            stats.update({
                "Candidate": name,
                "Source": "Baseline",
                "Model_Family": name,
                "Stability_Score": _stability_score(stats["CV_Mean"], stats["CV_Std"], higher_is_better),
            })
            cv_rows.append(stats)
            estimator_versions[name] = estimator
        except Exception as exc:
            cv_rows.append({
                "Candidate": name,
                "Source": "Baseline",
                "Model_Family": name,
                "CV_Mean": np.nan,
                "CV_Std": np.nan,
                "Stability_Score": -np.inf,
                "Error": f"{type(exc).__name__}: {str(exc)[:300]}",
            })

    cv_df = pd.DataFrame(cv_rows)
    valid = cv_df[pd.notna(cv_df["CV_Mean"])].sort_values("Stability_Score", ascending=False)
    if valid.empty:
        raise RuntimeError("No model completed competition cross-validation successfully.")

    # Tune only strongest single-model families.
    tuned_records = []
    top_for_tune = valid.head(profile["tune_top"])["Candidate"].tolist()
    for name in top_for_tune:
        try:
            tune = _tune_candidate(
                estimator_versions[name], name, task, X, y_model, scoring,
                profile["folds"], config.random_state, profile["tune_size"],
            )
            if not tune:
                continue
            stats = _candidate_cv(
                clone(tune["estimator"]), X, y_model, task, scoring,
                profile["folds"], profile["repeats"], config.random_state + 97,
            )
            candidate_name = f"Tuned {name}"
            stats.update({
                "Candidate": candidate_name,
                "Source": "Tuned",
                "Model_Family": name,
                "Best_Params": tune["best_params"],
                "Tune_Seconds": tune["seconds"],
                "Stability_Score": _stability_score(stats["CV_Mean"], stats["CV_Std"], higher_is_better),
            })
            tuned_records.append(stats)
            estimator_versions[candidate_name] = clone(tune["estimator"])
        except Exception as exc:
            tuned_records.append({
                "Candidate": f"Tuned {name}",
                "Source": "Tuned",
                "Model_Family": name,
                "CV_Mean": np.nan,
                "CV_Std": np.nan,
                "Stability_Score": -np.inf,
                "Error": f"{type(exc).__name__}: {str(exc)[:300]}",
            })

    if tuned_records:
        cv_df = pd.concat([cv_df, pd.DataFrame(tuned_records)], ignore_index=True, sort=False)

    # Re-rank before ensemble construction.
    valid = cv_df[pd.notna(cv_df["CV_Mean"])].sort_values("Stability_Score", ascending=False)

    # Top-3 voting ensemble: only add it if it can actually be evaluated.
    # Favor diversity: keep only the strongest version of each model family.
    top_candidates = (
        valid.sort_values("Stability_Score", ascending=False)
        .drop_duplicates(subset=["Model_Family"], keep="first")
        .head(min(3, len(valid)))
        .copy()
    )
    if len(top_candidates) >= 2:
        names = top_candidates["Candidate"].tolist()
        estimators = [(_safe_name(n, i), clone(estimator_versions[n])) for i, n in enumerate(names)]
        weights = _weights_from_scores(top_candidates["CV_Mean"].tolist(), higher_is_better)
        try:
            if task == "classification":
                ensemble_est = VotingClassifier(estimators=estimators, voting="soft", weights=weights, flatten_transform=True)
            else:
                ensemble_est = VotingRegressor(estimators=estimators, weights=weights)
            stats = _candidate_cv(
                ensemble_est, X, y_model, task, scoring,
                profile["folds"], profile["repeats"], config.random_state + 211,
            )
            stats.update({
                "Candidate": "CV Weighted Ensemble",
                "Source": "Ensemble",
                "Model_Family": " + ".join(names),
                "Stability_Score": _stability_score(stats["CV_Mean"], stats["CV_Std"], higher_is_better),
            })
            cv_df = pd.concat([cv_df, pd.DataFrame([stats])], ignore_index=True, sort=False)
            estimator_versions["CV Weighted Ensemble"] = ensemble_est
        except Exception:
            pass

    # Maximum mode also tests stacking; it wins only if CV says it wins.
    if profile["stack"]:
        valid_now = cv_df[pd.notna(cv_df["CV_Mean"])].sort_values("Stability_Score", ascending=False)
        top_stack = (
            valid_now[~valid_now["Source"].eq("Ensemble")]
            .sort_values("Stability_Score", ascending=False)
            .drop_duplicates(subset=["Model_Family"], keep="first")
            .head(min(3, len(valid_now)))
        )
        if len(top_stack) >= 2:
            stack_names = top_stack["Candidate"].tolist()
            stack_estimators = [(_safe_name(n, i), clone(estimator_versions[n])) for i, n in enumerate(stack_names)]
            try:
                if task == "classification":
                    stack_est = StackingClassifier(
                        estimators=stack_estimators,
                        final_estimator=LogisticRegression(max_iter=3000),
                        stack_method="predict_proba",
                        cv=3,
                        n_jobs=None,
                    )
                else:
                    stack_est = StackingRegressor(
                        estimators=stack_estimators,
                        final_estimator=Ridge(alpha=1.0),
                        cv=3,
                        n_jobs=None,
                    )
                stats = _candidate_cv(
                    stack_est, X, y_model, task, scoring,
                    profile["folds"], max(1, profile["repeats"] - 1), config.random_state + 409,
                )
                stats.update({
                    "Candidate": "Stacking Ensemble",
                    "Source": "Stacking",
                    "Model_Family": " + ".join(stack_names),
                    "Stability_Score": _stability_score(stats["CV_Mean"], stats["CV_Std"], higher_is_better),
                })
                cv_df = pd.concat([cv_df, pd.DataFrame([stats])], ignore_index=True, sort=False)
                estimator_versions["Stacking Ensemble"] = stack_est
            except Exception:
                pass

    leaderboard = cv_df[pd.notna(cv_df["CV_Mean"])].sort_values("Stability_Score", ascending=False).reset_index(drop=True)
    winner_name = str(leaderboard.iloc[0]["Candidate"])
    winner_estimator = clone(estimator_versions[winner_name])

    threshold_info = None
    if task == "classification" and metric_choice in {"Accuracy", "Balanced Accuracy", "Weighted F1", "Macro F1"}:
        try:
            threshold_info = _optimize_binary_threshold(
                winner_estimator, X, y_model, metric_choice,
                profile["folds"], config.random_state + 701,
            )
        except Exception:
            threshold_info = None

    # Refit winner on ALL labeled rows. This is what will be used on the doctor's 10 unseen cases.
    fit_t0 = time.perf_counter()
    winner_estimator.fit(X, y_model)
    full_fit_seconds = time.perf_counter() - fit_t0

    final_bundle = copy.deepcopy(first_bundle)
    final_bundle["pipeline"] = winner_estimator
    final_bundle["feature_columns"] = X.columns.tolist()
    final_bundle["competition"] = {
        "winner": winner_name,
        "metric_choice": metric_choice,
        "metric_label": metric_label,
        "mode": mode,
        "cv_mean": float(leaderboard.iloc[0]["CV_Mean"]),
        "cv_std": float(leaderboard.iloc[0]["CV_Std"]),
        "stability_score": float(leaderboard.iloc[0]["Stability_Score"]),
        "threshold": None if threshold_info is None else float(threshold_info["threshold"]),
        "trained_on_all_labeled_rows": True,
        "labeled_rows": int(len(X)),
        "full_fit_seconds": float(full_fit_seconds),
        "selection_note": "Winner selected by repeated cross-validation and then refit on all labeled rows for the external 10-case test.",
    }

    expected_correct = None
    if task == "classification" and metric_choice == "Accuracy":
        expected_correct = float(np.clip(leaderboard.iloc[0]["CV_Mean"], 0, 1) * 10.0)

    return {
        "leaderboard": leaderboard,
        "holdout_results": holdout_results,
        "holdout_bundles": bundles,
        "holdout_details": details,
        "split_data": split,
        "final_bundle": final_bundle,
        "winner_name": winner_name,
        "threshold_info": threshold_info,
        "expected_correct_out_of_10": expected_correct,
        "metric_choice": metric_choice,
        "metric_label": metric_label,
        "higher_is_better": higher_is_better,
        "mode": mode,
        "selected_models": selected_models,
        "params": params,
    }


def predict_competition_bundle(bundle: dict, X: pd.DataFrame):
    pipe = bundle["pipeline"]
    task = bundle["task"]
    competition = bundle.get("competition", {})
    threshold = competition.get("threshold")

    if task == "classification":
        probs = pipe.predict_proba(X) if hasattr(pipe, "predict_proba") else None
        if probs is not None and probs.shape[1] == 2 and threshold is not None:
            pred = (probs[:, 1] >= float(threshold)).astype(int)
        else:
            pred = pipe.predict(X)
        return pred, probs
    return pipe.predict(X), None
