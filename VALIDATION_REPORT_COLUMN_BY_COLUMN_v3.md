# Validation Report — Column-by-Column Edition v3

## Requested change

The preprocessing experience was returned to the original ApexML style: the user chooses and handles one column at a time.

## Checks

- No `Quick Encode All Remaining` UI.
- No `Encode All Categorical Columns` button.
- No `Compare Cleaning Strategies by Cross-Validation` feature.
- No bulk apply-to-all numeric/categorical cleaning action.
- Mean/median/etc. imputation modifies the selected column.
- KNN imputation uses neighbouring features but writes only to the selected column.
- One-hot/label/ordinal encoding begins from one selected column.
- IQR cap/log operates on one selected column.
- Cross Validation remains available for model selection.
- Final competition workflow still supports 10-case prediction.

## Automated tests

- Stable-v2 regression test: PASS
- Final supervised/competition smoke test: PASS
- New column-by-column behavior test: PASS
