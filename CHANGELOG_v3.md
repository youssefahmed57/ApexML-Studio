# ApexML Final — Column-by-Column Edition v3

## Requested behavior

Manual preprocessing is intentionally performed **one selected column at a time**.

### Removed

- Bulk `Quick Encode All Remaining` workflow.
- `Encode All Categorical Columns` button.
- Any cleaning-strategy comparison by Cross Validation.
- Any `Apply to All Numeric/Categorical Columns` workflow.

### Kept

- Missing-value treatment for one selected column.
- Outlier treatment for one selected numeric column.
- Encoding for one selected categorical column.
- Preprocessing log and reset-to-original workflow.
- Leakage-safe model-pipeline settings for final training.
- Cross Validation for **model evaluation and model selection**.

### Important KNN fix

KNN imputation may use other numeric columns as neighbour context, but it now writes imputed values back **only to the selected column**. Missing values in other columns stay unchanged until the user selects those columns explicitly.

## Validation

Passed:

- `stable_v2_regression_test.py`
- `final_smoke_test.py`
- `column_by_column_test.py`
