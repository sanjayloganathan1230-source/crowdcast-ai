import json

import numpy as np
import pandas as pd
import pytest

from backend.app.ml.data_generation import generate_dataset
from backend.app.ml.features import FEATURE_COLUMNS, make_training_table
from backend.app.ml.model_store import (ModelMismatchError, ModelNotFoundError, load_bundle,
                                        load_metrics, save_artifacts)
from backend.app.ml.training import split_chronological, train_and_evaluate

FAST = dict(n_estimators=15, min_samples_leaf=8)


@pytest.fixture(scope="module")
def df():
    return generate_dataset(seed=42, days=70)


@pytest.fixture(scope="module")
def result(df):
    return train_and_evaluate(df, seed=42, test_days=14, rf_params=FAST)


def test_split_is_strictly_chronological_and_disjoint(df):
    X, y, meta = make_training_table(df)
    s = split_chronological(X, y, meta, test_days=14)
    assert s.meta_train.timestamp.max() < s.meta_test.timestamp.min()
    assert set(s.X_train.index).isdisjoint(s.X_test.index)
    assert len(s.X_train) + len(s.X_test) == len(X)
    assert s.meta_test.timestamp.min() == s.test_start
    assert s.meta_test.timestamp.max() == df.timestamp.max()
    assert s.meta_test["location_id"].nunique() == 4  # every location appears in test


def test_split_rejects_bad_args(df):
    X, y, meta = make_training_table(df)
    with pytest.raises(ValueError):
        split_chronological(X, y, meta, test_days=0)
    with pytest.raises(ValueError):
        split_chronological(X, y, meta, test_days=500)  # leaves no training data


def test_train_features_never_use_test_period_values(df):
    """Zeroing every test-period observation must not change any TRAIN feature."""
    X, y, meta = make_training_table(df)
    s = split_chronological(X, y, meta, test_days=14)
    tampered = df.copy()
    tampered.loc[tampered.timestamp >= s.test_start, "crowd_count"] = 0
    X2, _, _ = make_training_table(tampered)
    pd.testing.assert_frame_equal(s.X_train, X2.loc[s.X_train.index])


def test_metrics_are_computed_not_hardcoded(result):
    o = result.metrics["overall"]
    assert set(o) == {"seasonal_naive_168h", "mean_last_4_weeks", "random_forest"}
    sp = result.split
    obs = sp.y_test.to_numpy()
    naive = sp.X_test["lag_168"].to_numpy()
    assert o["seasonal_naive_168h"]["mae"] == pytest.approx(np.mean(np.abs(naive - obs)))
    assert o["seasonal_naive_168h"]["rmse"] == pytest.approx(np.sqrt(np.mean((naive - obs) ** 2)))
    tp = result.test_predictions
    assert o["random_forest"]["mae"] == pytest.approx(np.mean(np.abs(tp["random_forest"] - tp["observed"])))
    assert all(o[k]["rmse"] >= o[k]["mae"] for k in o)  # RMSE >= MAE always


def test_model_predictions_within_bounds(result):
    tp = result.test_predictions
    cap = result.split.X_test["capacity"].to_numpy()
    assert (tp["random_forest"] >= 0).all() and (tp["random_forest"].to_numpy() <= cap).all()


def test_training_is_deterministic(df, result):
    again = train_and_evaluate(df, seed=42, test_days=14, rf_params=FAST)
    assert again.metrics["overall"] == result.metrics["overall"]
    pd.testing.assert_frame_equal(again.test_predictions, result.test_predictions)


def test_save_load_roundtrip_gives_identical_predictions(result, tmp_path):
    mp, jp = tmp_path / "m.joblib", tmp_path / "metrics.json"
    save_artifacts(result, mp, jp)
    bundle = load_bundle(mp)
    assert bundle["feature_columns"] == FEATURE_COLUMNS
    X = result.split.X_test[bundle["feature_columns"]]
    np.testing.assert_array_equal(bundle["model"].predict(X), result.model.predict(X))
    saved = load_metrics(jp)
    assert saved["overall"]["random_forest"]["mae"] == result.metrics["overall"]["random_forest"]["mae"]
    assert "DEMO DATA" in saved["data_label"]
    json.dumps(saved)  # serialisable


def test_load_missing_and_mismatched(result, tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_bundle(tmp_path / "none.joblib")
    with pytest.raises(ModelNotFoundError):
        load_metrics(tmp_path / "none.json")
    import joblib
    bad = tmp_path / "bad.joblib"
    joblib.dump({"model": result.model, "feature_columns": FEATURE_COLUMNS[:-1],
                 "sklearn_version": "0.0"}, bad)
    with pytest.raises(ModelMismatchError):
        load_bundle(bad)
