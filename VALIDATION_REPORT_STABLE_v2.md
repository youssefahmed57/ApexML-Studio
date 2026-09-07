# ApexML Final — Simple Pro
## Stable Bugfix Edition v2.0 — Validation Report

### Reported runtime error reproduced and fixed

The reported error was:

```text
AttributeError: 'NoneType' object has no attribute 'append'
```

Root cause:

`Dataset Overview` reset `preprocessing_log` to `None`, while the preprocessing helper expected a list and called `.append()`.

Fixes applied:

- reset state using the correct types
- make preprocessing helpers repair legacy `None` state automatically
- prevent the same uploaded file from resetting the whole session on every Streamlit rerun

### Additional UI/runtime issues found and fixed

- Smart Preprocessing referenced an undefined `tab_pipeline`
- small datasets could create an invalid preview slider
- numeric constant imputation could accidentally turn a numeric column into text
- KNN imputation edge cases were not validated
- scaler / encoder dictionaries could also be `None` in old sessions
- empty/all-missing outlier calculations could divide by zero
- PCA preview could fail on missing or constant numeric data
- `SelectKBest(k)` could request more features than existed after encoding
- variance filtering could remove every feature
- t-SNE used the removed/deprecated `n_iter` parameter with current scikit-learn
- t-SNE perplexity and clustering K values could exceed small-dataset limits
- t-SNE color labels could become misaligned after missing rows were dropped
- CSV delimiter handling was improved

### Automated tests completed

1. Python syntax / compile check: PASS
2. Original final smoke test: PASS
3. Exact imputation-button regression test: PASS
4. Dataset rerun state-preservation test: PASS
5. Smart Preprocessing no-click render test: PASS
6. Feature Engineering no-click render test: PASS
7. Supervised ML Studio no-click render test: PASS
8. Unsupervised ML Studio no-click render test: PASS
9. Evaluation page render test: PASS
10. Inference page render test: PASS
11. Multi-dataset classification test: PASS
12. Multi-dataset regression test: PASS
13. K-Means / Hierarchical / PCA / t-SNE regression tests: PASS
14. SelectKBest with intentionally oversized K: PASS
15. Competition Boost + exactly 10 predictions: PASS

### Datasets used

- Titanic — binary classification
- Employee Attrition — mixed categorical/numeric classification with missing values
- House Prices — regression
- Mall Customers — unsupervised analysis

### Final status

**STABLE FOR COURSE DEMO / FINAL PROJECT USE**

A Machine Learning application can never guarantee compatibility with every malformed or semantically invalid dataset. The application now handles the tested structural/runtime edge cases gracefully and provides clearer errors for invalid ML conditions.
