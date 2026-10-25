"""Training and honest evaluation of the crowd model vs. simple baselines.

Split policy (no leakage)
-------------------------
* The split is CHRONOLOGICAL and global: every test row is strictly later than
  every train row, for all locations. Rows are never shuffled.
* Features only look backwards (lags of 1-4 weeks), so no train feature reads a
  test-period value. Test-row lags may read earlier observations; that is
  legitimate because those are known at forecast time (horizon <= 168 h).
* Hyperparameters are fixed constants chosen for model size/stability, NOT tuned
  on the test set.

Evaluation
----------
MAE and RMSE (in people) on the held-out test period for:
  * seasonal_naive_168h : "same hour last week" (primary baseline)
  * mean_last_4_weeks   : mean of the same hour over the previous 4 weeks
  * random_forest       : the model (predicts deviation from the 4-week same-hour mean; see model.py)
Nothing here is hardcoded; numbers come from the run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from ..core.config import DEMO_DATA_TAG, LAG_HOURS, MAX_HORIZON_HOURS
from .features import FEATURE_COLUMNS, make_training_table
from .locations import LOCATION_IDS
from .model import TARGET_MODE, make_target, predict_counts

DEFAULT_TEST_DAYS = 28
MIN_TRAIN_ROWS = 500

# Fixed, untuned. min_samples_leaf/n_estimators keep the saved file small.
RF_PARAMS = dict(n_estimators=100, min_samples_leaf=8, max_features=0.6, n_jobs=-1)


@dataclass
class Split:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame
    test_start: pd.Timestamp


@dataclass
class TrainingResult:
    model: RandomForestRegressor
    metrics: dict
    split: Split
    test_predictions: pd.DataFrame  # timestamp, location_id, observed, model, baselines


def split_chronological(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                        test_days: int = DEFAULT_TEST_DAYS) -> Split:
    if test_days < 1:
        raise ValueError("test_days must be >= 1")
    last = meta["timestamp"].max()
    test_start = last + pd.Timedelta(hours=1) - pd.Timedelta(days=test_days)
    is_test = meta["timestamp"] >= test_start
    if is_test.sum() == 0 or (~is_test).sum() < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Split leaves too little data (train={int((~is_test).sum())}, test={int(is_test.sum())}); "
            "generate more days or reduce test_days.")
    split = Split(X[~is_test], y[~is_test], X[is_test], y[is_test],
                  meta[~is_test], meta[is_test], test_start)
    assert split.meta_train["timestamp"].max() < split.meta_test["timestamp"].min(), "train/test overlap"
    return split


def _scores(obs: np.ndarray, pred: np.ndarray) -> dict:
    err = np.asarray(pred, float) - np.asarray(obs, float)
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}


def _evaluate(split: Split, preds: dict[str, np.ndarray]) -> dict:
    obs = split.y_test.to_numpy()
    loc = split.meta_test["location_id"].to_numpy()
    cap = split.X_test["capacity"].to_numpy()
    overall, per_location = {}, {}
    for name, p in preds.items():
        overall[name] = _scores(obs, p)
        overall[name]["mae_pct_of_capacity"] = float(np.mean(np.abs(p - obs) / cap) * 100)
        for lid in LOCATION_IDS:
            m = loc == lid
            per_location.setdefault(lid, {})[name] = _scores(obs[m], p[m])
    base = overall["seasonal_naive_168h"]
    rf = overall["random_forest"]
    return {
        "overall": overall,
        "per_location": per_location,
        "improvement_vs_seasonal_naive": {
            "mae_pct": float((base["mae"] - rf["mae"]) / base["mae"] * 100),
            "rmse_pct": float((base["rmse"] - rf["rmse"]) / base["rmse"] * 100),
        },
    }


def train_and_evaluate(df: pd.DataFrame, seed: int = 42, test_days: int = DEFAULT_TEST_DAYS,
                       rf_params: dict | None = None) -> TrainingResult:
    X, y, meta = make_training_table(df)
    split = split_chronological(X, y, meta, test_days)

    params = {**RF_PARAMS, **(rf_params or {})}
    model = RandomForestRegressor(random_state=seed, **params)
    model.fit(split.X_train[FEATURE_COLUMNS], make_target(split.X_train, split.y_train))

    rf_pred = predict_counts(model, split.X_test)  # the exact function the API uses
    preds = {
        "seasonal_naive_168h": split.X_test["lag_168"].to_numpy(),
        "mean_last_4_weeks": split.X_test["same_slot_mean"].to_numpy(),
        "random_forest": rf_pred,
    }
    metrics = _evaluate(split, preds)
    metrics.update({
        "data_label": DEMO_DATA_TAG + " (synthetic; not live sensor data)",
        "seed": seed,
        "split": {
            "type": "chronological",
            "train_rows": int(len(split.X_train)), "test_rows": int(len(split.X_test)),
            "train_start": str(split.meta_train["timestamp"].min()),
            "train_end": str(split.meta_train["timestamp"].max()),
            "test_start": str(split.meta_test["timestamp"].min()),
            "test_end": str(split.meta_test["timestamp"].max()),
        },
        "model": {"type": "RandomForestRegressor", "params": {**params, "random_state": seed},
                  "sklearn_version": sklearn.__version__,
                  "feature_columns": FEATURE_COLUMNS, "lag_hours": list(LAG_HOURS),
                  "max_horizon_hours": MAX_HORIZON_HOURS, "predictions_clipped_to": "[0, capacity]",
                  "target_mode": TARGET_MODE},
        "feature_importance": {c: float(v) for c, v in sorted(
            zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda t: -t[1])},
        "last_observation": str(pd.to_datetime(df["timestamp"]).max()),
    })

    test_pred = split.meta_test.copy()
    test_pred["observed"] = split.y_test.to_numpy()
    for k, v in preds.items():
        test_pred[k] = v
    return TrainingResult(model, metrics, split, test_pred.reset_index(drop=True))
