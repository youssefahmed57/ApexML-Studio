# Validation Report — Unsupervised v4

## Static checks
- 34 Python files parsed successfully with Python AST.
- ML Studio, Evaluation and Inference modules imported successfully using a Streamlit test stub.

## Existing regression tests
- `final_smoke_test.py` — PASS
- `stable_v2_regression_test.py` — PASS
- `column_by_column_test.py` — PASS

## New unsupervised integration test
- `unsupervised_evaluation_inference_test.py` — PASS

Covered:
- K-Means metrics and native prediction.
- Hierarchical internal metrics and nearest-centroid assignment.
- DBSCAN metrics and core-point assignment.
- PCA out-of-sample transform.
- Cluster size/profile exports.
- Evaluation and Inference page integration markers.

## Methodology note
For algorithms without a native future-row prediction API, the UI states the exact assignment rule instead of presenting it as native model prediction. t-SNE future-row inference is deliberately disabled.
