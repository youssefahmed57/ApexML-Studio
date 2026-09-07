from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.engine import DatasetConfig, prepare_dataframe, prepare_inference_features, train_models
from core.competition import run_competition_training, predict_competition_bundle


def cfg(target, task, excluded):
    return DatasetConfig(
        target=target, task=task, excluded_columns=excluded, date_columns=[],
        force_numeric=[], force_categorical=[], test_size=.2, random_state=42,
        numeric_imputer='median', categorical_imputer='most_frequent',
        scaler='StandardScaler', outlier_strategy='None', iqr_factor=1.5,
        max_categories=60, remove_zero_variance=True, feature_selection='None',
        k_best=None, categorical_encoder='OneHot', imbalance_strategy='None',
        feature_recipe=[],
    )


def main():
    df = pd.read_csv(ROOT / 'datasets_tests' / 'titanic.csv')
    target = 'Survived'
    excluded = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    X, y, _ = prepare_dataframe(df, target, excluded, [], [], [])
    conf = cfg(target, 'classification', excluded)

    results, bundles, details, split = train_models(
        X, y, 'classification', ['Logistic Regression', 'Random Forest'], {}, conf,
        cv_folds=3, run_cv=True,
    )
    assert len(results) == 2
    assert 'Test_Accuracy' in results.columns

    run = run_competition_training(
        X, y, 'classification', conf, 'Accuracy', mode='Fast',
        selected_models=['Logistic Regression', 'Random Forest'], custom_params={},
    )
    assert run['final_bundle'] is not None

    raw10 = df.drop(columns=[target] + excluded).head(10).copy()
    X10 = prepare_inference_features(raw10, excluded, [], [], [])
    X10 = X10[run['final_bundle']['feature_columns']]
    pred, probs = predict_competition_bundle(run['final_bundle'], X10)
    assert len(pred) == 10

    print('APEXML FINAL SIMPLE PRO — SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
