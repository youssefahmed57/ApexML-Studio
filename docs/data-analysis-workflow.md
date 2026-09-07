# End-to-End Data Analysis Workflow

## Overview
This document defines the standard workflow for performing end-to-end data analysis. The AI suggestion engine must adhere strictly to these steps and their recommended methodologies.

---

## Phase 1: Data Ingestion & Understanding
1. Load the dataset (CSV, Excel, JSON, SQL).
2. Inspect shape, data types, and memory usage.
3. Display the first/last rows and a statistical summary.
4. Identify and document:
   - Numerical vs. Categorical columns.
   - Target variable (if supervised).
   - Date/Time columns.

## Phase 2: Data Quality Assessment
1. **Missing Values:** Count and percentage per column. Flag columns with >40% missing as candidates for removal.
2. **Duplicates:** Detect exact duplicate rows. Report count and percentage.
3. **Outliers:** Use IQR method for numerical columns. Flag values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
4. **Data Type Errors:** Detect columns stored as wrong types (e.g., numbers as strings).
5. **Cardinality Check:** Flag categorical columns with very high cardinality (>50 unique values).

## Phase 3: Data Cleaning & Preprocessing
### Missing Value Strategy (AI should choose based on distribution):
- **Mean imputation:** Use ONLY when distribution is approximately normal (|skewness| < 0.5).
- **Median imputation:** Use when distribution is skewed (|skewness| >= 0.5). Robust to outliers.
- **Mode imputation:** Use for categorical columns with a clear dominant value.
- **KNN imputation:** Use when missingness is not completely random and other features can predict it.
- **Drop column:** Only if >60% values are missing AND the column has no strong correlation with the target.
- **Drop row:** Only if <1% of total rows have missing values in a critical column.

### Encoding Strategy (AI should choose based on column semantics):
- **One-Hot Encoding (OHE):** Use for NOMINAL categorical variables (no inherent order). Best for low cardinality (<15 unique values).
- **Label Encoding:** Use for BINARY categorical variables or when the model is tree-based (handles arbitrary integers).
- **Ordinal Encoding:** Use for ORDINAL categorical variables (e.g., Low/Medium/High). Requires specifying the order.
- **Target Encoding:** Use for high-cardinality nominal features when paired with a supervised model. Requires careful cross-validation to avoid leakage.

### Outlier Treatment Strategy:
- **IQR Capping (Winsorization):** Cap values at [Q1 - 1.5*IQR, Q3 + 1.5*IQR]. Best when outliers are errors.
- **Log Transform:** Apply to heavily right-skewed positive data. Compresses the tail naturally.
- **Keep as-is:** If outliers are genuine data points (e.g., high-value customers in sales data).
- **Remove rows:** Only if outliers represent clear measurement errors.

## Phase 4: Exploratory Data Analysis (EDA)
### Univariate Analysis:
- **Numerical:** Histogram, Box Plot, KDE Plot. Report: mean, median, std, skewness, kurtosis.
- **Categorical:** Bar Chart (value counts). Report: mode, unique count.

### Bivariate Analysis:
- **Numerical vs. Numerical:** Scatter Plot, Correlation (Pearson if linear, Spearman if monotonic).
- **Categorical vs. Numerical:** Box Plot, Violin Plot, Grouped Bar Chart.
- **Categorical vs. Categorical:** Heatmap of cross-tabulation, Stacked Bar Chart.

### Multivariate Analysis:
- Correlation Heatmap (all numerical features).
- Pairplot (for datasets with <20 features).
- PCA Biplot to visualize explained variance.

## Phase 5: Feature Engineering
### Scaling Strategy (AI should choose based on distribution and model):
- **StandardScaler (Z-score):** Use for normally distributed data. Required by: Logistic Regression, SVC, KNN, Linear/Ridge/Lasso Regression.
- **MinMaxScaler:** Use when you need bounded [0,1] output. Sensitive to outliers. Best for: Neural Networks, KNN (if no outliers).
- **RobustScaler:** Use when data has significant outliers. Uses median and IQR. Best for: most models when outliers exist.
- **Log Transform (np.log1p):** Apply BEFORE scaling to right-skewed features (skewness > 1.0).
- **No scaling needed:** Tree-based models (Decision Tree, Random Forest, XGBoost, LightGBM) are scale-invariant.

### Feature Creation:
- **Interaction features:** Multiply or divide related numerical features.
- **Polynomial features:** Add squared/cubic terms for non-linear relationships.
- **Binning:** Convert continuous variables into categorical bins (pd.cut).
- **Date decomposition:** Extract year, month, day, weekday, is_weekend from datetime columns.

## Phase 6: Validation & Reporting
1. After all preprocessing, verify: no remaining NaN values, all columns are numerical.
2. Check for data leakage: target variable must NOT appear in feature set.
3. Check class imbalance (for classification): if minority class < 10%, consider SMOTE.
4. Document all transformations applied for reproducibility.

---

## AI Suggestion Rules
The LLM must always:
1. Reference the column's actual statistics (mean, std, skewness, missing %, cardinality) in its justification.
2. Provide a PRIMARY recommendation with a clear reason.
3. Provide one ALTERNATIVE if the primary has trade-offs.
4. Keep suggestions concise: 3-5 sentences maximum.
5. Use domain knowledge if detectable (e.g., 'Age' → median imputation, 'Gender' → One-Hot Encoding).
