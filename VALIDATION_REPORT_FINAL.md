# Validation Report

## Static checks
- 29 Python files parsed successfully.
- No Gemini API secrets are included in the final project.
- The original ApexML 7-page navigation is preserved.

## Runtime engine checks

### Classification
Tested on the included Titanic dataset with:
- Logistic Regression
- Random Forest
- Neural Network
- safe preprocessing
- 3-fold Cross Validation

PASS.

### Competition Boost
Tested on Titanic with:
- repeated CV
- tuning
- weighted voting ensemble
- final full-data refit

PASS.

### Regression
Tested on the included House Prices/Boston-style dataset with:
- Linear Regression
- Random Forest
- safe preprocessing
- Cross Validation

PASS.

### Final 10-case flow
Tested a Competition Boost winner with a feature recipe and exactly ten unseen-like input rows.

- 10 predictions returned
- class probabilities returned
- schema-preserving inference passed

PASS.

## UI runtime note
The artifact environment used for packaging does not currently have Streamlit installed, so the browser server itself could not be launched here. Install `requirements.txt` locally and use `python launch.py`.
