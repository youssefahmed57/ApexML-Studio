from __future__ import annotations
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    ('streamlit','streamlit'),('pandas','pandas'),('numpy','numpy'),
    ('scikit-learn','sklearn'),('plotly','plotly'),('openpyxl','openpyxl'),('joblib','joblib'),
]
RECOMMENDED = [
    ('xgboost','xgboost'),('lightgbm','lightgbm'),('imbalanced-learn','imblearn'),
]

print('ApexML Final — Simple Pro — Environment Doctor')
print('Python:', sys.version)
print('Project:', ROOT)

ok = True
print('\nRequired packages')
for label, module in REQUIRED:
    try:
        mod = importlib.import_module(module)
        print(f'[OK] {label}:', getattr(mod,'__version__','installed'))
    except Exception as exc:
        ok = False
        print(f'[MISSING] {label}: {exc}')

print('\nRecommended competition packages')
for label, module in RECOMMENDED:
    try:
        mod = importlib.import_module(module)
        print(f'[OK] {label}:', getattr(mod,'__version__','installed'))
    except Exception as exc:
        print(f'[OPTIONAL MISSING] {label}: {exc}')

print('\nEnvironment status:', 'READY' if ok else 'INSTALL REQUIREMENTS')
if not ok:
    print('Run: python -m pip install -r requirements.txt')
