"""Feature engineering shared by training and inference (single code path).

Leakage rules
-------------
* Every lag feature for target time t reads the observation at t - k hours
  (k in LAG_HOURS, all >= 168 h), by timestamp lookup, never from t or later.
* At inference a forecast `origin` (last observed time) can be given; history
  after the origin is masked, so a lag that would not yet be known becomes NaN
  instead of silently using future data.
* `event_indicator` and `capacity` are treated as known in advance (scheduled
  events, fixed venue capacity), not as measurements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.config import LAG_HOURS
from .locations import LOCATION_IDS
from .preprocessing import DataValidationError, validate_observations

LAG_COLUMNS = [f"lag_{k}" for k in LAG_HOURS]
FEATURE_COLUMNS: list[str] = (
    ["hour_sin", "hour_cos", "weekday", "is_weekend", "event_indicator", "capacity"]
    + [f"loc_{loc}" for loc in LOCATION_IDS]
    + LAG_COLUMNS
    + ["same_slot_mean"]
)
TARGET_COLUMN = "crowd_count"
TARGET_REQUIRED = ["timestamp", "location_id", "capacity", "event_indicator"]


def build_feature_matrix(targets: pd.DataFrame, history: pd.DataFrame,
                         origin: pd.Timestamp | str | None = None) -> pd.DataFrame:
    """Build model features for `targets` using only `history` (optionally cut at `origin`).

    `targets` needs: timestamp, location_id, capacity, event_indicator.
    `history` needs: timestamp, location_id, crowd_count (validated, unique keys).
    Returns a float frame with columns FEATURE_COLUMNS and the index of `targets`.
    Lag columns may be NaN where history is unavailable.
    """
    missing = [c for c in TARGET_REQUIRED if c not in targets.columns]
    if missing:
        raise DataValidationError(f"targets missing columns: {missing}")
    unknown = sorted(set(targets["location_id"]) - set(LOCATION_IDS))
    if unknown:
        raise DataValidationError(f"Unknown location_id values: {unknown}")

    ts = pd.to_datetime(targets["timestamp"]).astype("datetime64[ns]")
    hist = history[["timestamp", "location_id", "crowd_count"]].copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"]).astype("datetime64[ns]")
    if origin is not None:
        hist = hist[hist["timestamp"] <= pd.Timestamp(origin)]
    if hist.duplicated(["location_id", "timestamp"]).any():
        raise DataValidationError("history has duplicate (location_id, timestamp) rows")
    series = hist.set_index(["location_id", "timestamp"])["crowd_count"].astype(float)

    feats = pd.DataFrame(index=targets.index)
    hour = ts.dt.hour.to_numpy()
    weekday = ts.dt.day_of_week.to_numpy()
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feats["weekday"] = weekday.astype(float)
    feats["is_weekend"] = (weekday >= 5).astype(float)
    feats["event_indicator"] = targets["event_indicator"].to_numpy().astype(float)
    feats["capacity"] = targets["capacity"].to_numpy().astype(float)
    for loc in LOCATION_IDS:
        feats[f"loc_{loc}"] = (targets["location_id"].to_numpy() == loc).astype(float)

    loc_values = targets["location_id"].to_numpy()
    for k, col in zip(LAG_HOURS, LAG_COLUMNS):
        keys = pd.MultiIndex.from_arrays([loc_values, (ts - pd.Timedelta(hours=k)).to_numpy()])
        feats[col] = series.reindex(keys).to_numpy()
    feats["same_slot_mean"] = feats[LAG_COLUMNS].mean(axis=1, skipna=True)  # NaN only if all lags missing

    return feats[FEATURE_COLUMNS].astype(float)


def make_training_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Validate `df`, build features, drop warm-up rows lacking the 1- and 2-week lags.

    Returns (X, y, meta) with a shared index. `meta` holds timestamp / location_id
    so a chronological split can be done on it in Stage 3.
    """
    clean = validate_observations(df)
    X = build_feature_matrix(clean, clean)
    keep = X["lag_168"].notna() & X["lag_336"].notna()
    X = X.loc[keep]
    y = clean.loc[X.index, TARGET_COLUMN].astype(float)
    meta = clean.loc[X.index, ["timestamp", "location_id"]]
    return X, y, meta
