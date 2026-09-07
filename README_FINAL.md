# ApexML Final — Simple Pro

This edition keeps the **original ApexML look and 7-page navigation** while adding the final-course features that matter most for model quality and the 10 hidden cases.

## Same Simple Navigation

1. Home
2. Dataset Overview
3. Smart Preprocessing
4. Feature Engineering & EDA
5. ML Studio
6. Evaluation & Leaderboard
7. Inference / Prediction

## What Changed Inside the Pages

### Dataset Overview
- CSV / Excel upload
- Preview, column statistics, validation, missing-value map
- Resets previous experiment state safely when a new dataset is loaded

### Smart Preprocessing
The original interactive cleaning tools remain available. A new **Model Pipeline** tab stores leakage-safe final-training settings:

- numeric imputation: median / mean / most frequent / KNN
- categorical imputation
- Standard / Robust / MinMax / no scaling
- IQR clipping
- One-Hot / Ordinal encoding
- category cap
- zero-variance removal
- SelectKBest
- RandomOverSampler / SMOTE for Classification where compatible
- internal diagnostic test split

The manual cleaning tabs are useful for learning and EDA. Final supervised training uses the original uploaded data plus the safe pipeline so learned preprocessing is fitted on training folds only.

### Feature Engineering & EDA
The original charts stay available. Feature creation is also stored as a **final-training recipe**:

- interaction features
- polynomial powers
- log features
- binning
- datetime extraction

The recipe is replayed inside the fitted ML workflow.

### ML Studio
The page keeps Supervised / Unsupervised modes. Supervised now contains four simple tabs:

- Setup
- Algorithms
- Model Controls
- Train Best Model

Classification models include Logistic Regression, Naive Bayes, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, Hist Gradient Boosting, AdaBoost, KNN, SVM, XGBoost, LightGBM and Neural Network.

Regression includes Linear, Polynomial, Ridge, Lasso, ElasticNet and the corresponding tree/boosting/KNN/SVM/XGBoost/LightGBM/Neural families.

#### Full model control
Every selected model exposes its important hyperparameters. Neural Network supports:

- Auto or Custom architecture
- 1–8 hidden layers
- neurons per layer
- activation
- solver
- L2 alpha
- learning rate and schedule
- batch size
- epochs / max iterations
- early stopping
- validation fraction
- patience

#### Training strategies
**Standard Benchmark**
- safe pipeline
- holdout test
- K-Fold CV
- failure isolation
- select best model by chosen metric
- refit selected winner on all labeled data before inference

**Competition Boost**
- repeated CV ranking
- tune strongest candidate families
- compare voting ensemble
- Maximum mode can test stacking
- optimize binary threshold when applicable
- refit the final winner on all labeled rows

### Evaluation
- final winner
- holdout leaderboard
- repeated-CV leaderboard for Competition Boost
- no-skill baseline
- generalization gap
- CV mean/std
- fit time / prediction time / model size
- confusion matrix / classification report
- regression actual-vs-predicted / residuals
- ROC curves for binary tasks
- feature importance / coefficients
- Neural Network architecture + loss curve
- final pipeline export

### Inference
Three tabs:

- Single Prediction
- **Doctor's 10 Cases**
- General Batch Prediction

The final winner is locked by default. The 10-case tab requires exactly 10 rows, applies the exact fitted pipeline, outputs predictions and confidence, and can calculate the final score after labels are revealed.

## Why Cleaning Does Not Cheat the Test

Manual cleaning is separated from final model preprocessing. The final supervised pipeline learns imputation, encoding, scaling, outlier clipping, feature selection and optional resampling from training folds instead of fitting those transformations on the held-out test data.

## Security

The uploaded ApexML archive contained Gemini API keys in a `.env` file. Those keys are **not included** in this edition. Add your own key locally using `.env.example`. You should rotate/revoke the keys that were present in the uploaded archive.

## Run

```powershell
python -m pip install -r requirements.txt
python launch.py
```

Or double-click `run_streamlit.bat`.

`launch.py` chooses an available port automatically.


## Stable Bugfix Edition v2.0

This build includes a reliability pass focused on the exact runtime issues found during interactive use.

### Fixed
- `preprocessing_log = None` causing `AttributeError: 'NoneType' object has no attribute 'append'`.
- Dataset reloads resetting preprocessing/training state on every Streamlit rerun.
- Wrong session-state types after loading a new dataset (`None` instead of list/dict).
- Missing `Model Pipeline` tab variable in Smart Preprocessing.
- Numeric constant imputation accepting text such as `"Unknown"`.
- More robust mean/median/mode/KNN imputation with clear validation errors.
- Label/ordinal/scaler session containers now self-repair if a legacy session contains `None`.
- CSV delimiter/encoding loading is more robust.
- Empty/small dataset metadata and preview controls no longer divide by zero or create invalid sliders.
- IQR/outlier statistics safely handle empty/all-missing numeric columns.
- PCA preview handles missing/constant columns more safely.
- Scatter plots fall back gracefully when optional OLS trendline support is unavailable.
- `SelectKBest` now caps `k` to the actual transformed feature count.
- Zero-variance filtering no longer collapses the whole feature matrix.
- t-SNE updated for current scikit-learn (`max_iter`) and automatically caps perplexity.
- K-Means/Hierarchical/PCA controls adapt to small datasets.
- Unsupervised visualizations preserve row alignment after dropping incomplete rows.

### Reliability checks
The build was tested with:
- Titanic classification
- Employee Attrition classification with missing + categorical data
- House Prices regression
- Mall Customers unsupervised clustering / PCA / t-SNE
- Standard Benchmark
- Competition Boost
- exact 10-case inference
- manual imputation regression test reproducing the reported bug
