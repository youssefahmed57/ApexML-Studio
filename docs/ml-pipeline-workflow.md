# Machine Learning Pipeline Workflow

## Overview
This document defines the standard ML pipeline for supervised and unsupervised learning tasks. The AI suggestion engine must strictly follow these guidelines when recommending algorithms and hyperparameters.

---

## Phase 1: Problem Definition
1. Determine task type:
   - **Supervised Classification:** Target is categorical.
   - **Supervised Regression:** Target is continuous.
   - **Unsupervised Clustering:** No target; find natural groups.
   - **Dimensionality Reduction:** Reduce feature space for visualization or compression.
   - **Association Rules:** Discover item co-occurrence patterns.

2. Define the primary evaluation metric:
   - Classification (balanced): F1-Score (macro).
   - Classification (imbalanced): ROC-AUC or F1-Score (weighted).
   - Regression: RMSE (penalizes large errors) or MAE (robust to outliers).
   - Clustering: Silhouette Score.

## Phase 2: Data Splitting (CRITICAL — No Leakage)
### Rules:
- **ALWAYS** split BEFORE fitting ANY preprocessor (scaler, encoder, imputer).
- Use `train_test_split` with a fixed `random_state` for reproducibility.
- For classification with class imbalance, use `stratify=y`.
- Recommended default: 80% train / 20% test.
- Fit preprocessors on TRAIN set only. Transform both TRAIN and TEST.

## Phase 3: Algorithm Selection Guide

### Supervised — Classification

| Algorithm | Best When | Avoid When |
|-----------|-----------|------------|
| Logistic Regression | Linearly separable, interpretability needed | Complex non-linear boundaries |
| Naive Bayes | Text classification, very high-dimensional, small data | Strong feature dependencies |
| Decision Tree | Interpretability critical, mixed data types | Overfitting on noisy data |
| Random Forest | General purpose, robust to overfitting | Very high-dimensional sparse data |
| SVC | High-dimensional, small-medium datasets | Very large datasets (slow) |
| KNN | Simple baseline, non-linear | Large datasets, high dimensions |
| XGBoost | Tabular data, competitions, maximum accuracy | Very small datasets (<100 rows) |
| LightGBM | Large datasets, fast training, high cardinality | Very small datasets |
| AdaBoost | Weak learner boosting, low bias data | Noisy data (sensitive to outliers) |
| MLP (ANN) | Complex patterns, sufficient data (>1000 rows) | Small datasets, interpretability needed |

### Supervised — Regression

| Algorithm | Best When | Avoid When |
|-----------|-----------|------------|
| Linear Regression | Linear relationship, interpretability | Non-linear relationships |
| Polynomial Regression | Curved/non-linear relationships (low degree) | High degree (overfitting risk) |
| Ridge | Multicollinearity present, many features | Very sparse data |
| Lasso | Feature selection needed (sparse solution) | Correlated features (drops one arbitrarily) |
| Decision Tree | Non-linear, mixed types, interpretability | Overfitting on noisy data |
| Random Forest | General purpose, handles missing values | Memory-intensive for large forests |
| SVR | Non-linear, medium datasets, robust to outliers | Very large datasets |
| KNN | Local patterns matter | Large datasets, high dimensions |
| XGBoost | Maximum accuracy, tabular data | Very small datasets |
| LightGBM | Speed + accuracy, large datasets | Very small datasets |
| AdaBoost | Sequential error correction | Noisy data |
| MLP (ANN) | Complex non-linear regression | Insufficient data |

### AI Algorithm Recommendation Logic:
1. **Dataset size < 500 rows:** Prefer Logistic/Linear Regression, Naive Bayes, Decision Tree. Avoid XGBoost/LightGBM.
2. **Dataset size 500–10,000 rows:** Random Forest and XGBoost are strong defaults.
3. **Dataset size > 10,000 rows:** LightGBM preferred (speed). ANN viable.
4. **High-dimensional data (features > samples):** SVC, Logistic Regression with L2, Naive Bayes.
5. **Strong non-linearity (detected via correlation analysis):** XGBoost, LightGBM, Random Forest, ANN.
6. **Interpretability required:** Logistic Regression, Decision Tree.
7. **Class imbalance detected:** Apply SMOTE + prefer F1/ROC-AUC metrics. Recommend Random Forest or XGBoost with `scale_pos_weight`.

### Unsupervised — Clustering

| Algorithm | Best When | Parameters |
|-----------|-----------|------------|
| K-Means | Spherical clusters, known K | n_clusters (use elbow + silhouette) |
| Hierarchical | Unknown K, non-spherical | linkage method, distance threshold |
| DBSCAN | Arbitrary shape clusters, noise/outliers | eps, min_samples (use k-distance graph) |

### Unsupervised — Dimensionality Reduction

| Algorithm | Best When |
|-----------|-----------|
| PCA | Linear relationships, compression, preprocessing |
| t-SNE | Non-linear visualization, cluster exploration (2D/3D) |

### Unsupervised — Association Rules
- **Apriori:** Use for market basket / transactional data. Set min_support ≥ 0.05 and min_confidence ≥ 0.5.

## Phase 4: Model Training
1. Use `Pipeline` from sklearn to chain preprocessor + model (prevents leakage).
2. For XGBoost/LightGBM: enable early stopping with a validation set.
3. For ANN (MLP): use `max_iter=500`, `early_stopping=True`, `validation_fraction=0.1`.
4. Cache trained models in session state to avoid retraining on UI interactions.

## Phase 5: Model Evaluation

### Classification Metrics (compute ALL of these):
- **Accuracy:** Overall correctness. Misleading for imbalanced data.
- **Precision:** Of predicted positives, how many are correct. (Minimize false positives)
- **Recall:** Of actual positives, how many are detected. (Minimize false negatives)
- **F1-Score:** Harmonic mean of Precision and Recall. Best single metric for imbalanced data.
- **ROC-AUC:** Area under the ROC curve. Model's discrimination ability. Higher = better.
- **Confusion Matrix:** Visual breakdown of TP, FP, TN, FN.

### Regression Metrics (compute ALL of these):
- **RMSE:** Root Mean Squared Error. Penalizes large errors. Primary metric.
- **MAE:** Mean Absolute Error. Robust to outliers. Secondary metric.
- **R²:** Proportion of variance explained. 1.0 = perfect, 0 = baseline mean predictor.

### Leaderboard Rules:
- Primary sort: F1-Score (classification) or RMSE ascending (regression).
- Highlight the top model in green.
- Show delta vs. second-best model.

## Phase 6: Interpretability
1. **Feature Importance:** Available for tree-based models natively.
2. **SHAP Values:** Use `shap.TreeExplainer` for tree models, `shap.LinearExplainer` for linear models.
   - Show SHAP summary bar plot (global importance).
   - Show SHAP waterfall plot for a single prediction (local explanation).
3. **ROC Curve:** Plot for all classifiers on the same chart for comparison.
4. **Residual Plot:** For regression — plot (y_pred - y_true) vs. y_pred.

## Phase 7: Inference
1. Load the BEST model (highest primary metric on test set).
2. Apply the SAME preprocessing pipeline used during training (fitted on train set).
3. Validate user input: enforce column types, reject out-of-range values with warnings.
4. Display prediction with confidence (probability for classification, value for regression).

---

## AI Suggestion Rules for Algorithm Selection
The LLM must always:
1. State the task type (classification/regression/clustering) with reasoning.
2. Recommend the TOP 2 algorithms with specific justification referencing:
   - Dataset shape (n_rows, n_features).
   - Target variable distribution (class balance for classification, skewness for regression).
   - Detected non-linearity or feature correlations.
3. Mention the PRIMARY evaluation metric the user should focus on.
4. Flag any risks (e.g., "Warning: only 200 rows detected — XGBoost may overfit; use cross-validation").
5. Keep response under 150 words.
