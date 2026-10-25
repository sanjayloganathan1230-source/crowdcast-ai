"""Empirical leakage check: forecasting from a truncated history must reproduce the batch evaluation exactly.

If any feature secretly used information from after the forecast origin, predictions made with history cut at the
origin would differ from predictions made with the full history. They must be identical.
"""
import numpy as np
import pandas as pd
import pytest

from backend.app.ml.model_store import load_bundle, load_metrics
from backend.app.ml.predict import PredictionError, Predictor


@pytest.fixture(scope="module")
def parts(env):
    s = env["settings"]
    return env["df"], env["result"], load_bundle(s.model_path), load_metrics(s.metrics_path)


@pytest.mark.parametrize("offset_hours", [-1, 72, 216])  # origins at the start, early, and late in the test period
def test_forecast_from_truncated_history_equals_batch_evaluation(parts, offset_hours):
    df, result, bundle, metrics = parts
    test_start = result.split.meta_test["timestamp"].min()
    origin = test_start + pd.Timedelta(hours=offset_hours)

    truncated = df[df["timestamp"] <= origin]
    assert truncated["timestamp"].max() == origin and len(truncated) < len(df)
    predictor = Predictor(bundle, truncated, metrics)
    assert predictor.origin == origin  # the predictor genuinely cannot see anything later

    horizon = df[(df["timestamp"] > origin) & (df["timestamp"] <= origin + pd.Timedelta(hours=48))]
    targets = horizon[["timestamp", "location_id", "capacity", "event_indicator"]]
    rolling = pd.Series(predictor._predict_frame(targets, origin=origin), index=horizon.index)

    batch = result.test_predictions.set_index(["timestamp", "location_id"])["random_forest"]
    expected = batch.loc[list(zip(horizon["timestamp"], horizon["location_id"]))].to_numpy()
    assert len(expected) == 48 * 4
    np.testing.assert_allclose(rolling.to_numpy(), expected, rtol=0, atol=1e-9)


def test_targets_beyond_the_horizon_are_refused_not_leaked(parts):
    """A target 200 h past the origin would need a lag that is not yet known: must fail, never peek."""
    df, result, bundle, metrics = parts
    origin = result.split.meta_test["timestamp"].min()
    predictor = Predictor(bundle, df[df["timestamp"] <= origin], metrics)
    far = df[(df["timestamp"] == origin + pd.Timedelta(hours=200)) & (df["location_id"] == "bus_terminal")]
    with pytest.raises(PredictionError, match="Insufficient history"):
        predictor._predict_frame(far[["timestamp", "location_id", "capacity", "event_indicator"]], origin=origin)


def test_predictions_are_not_constant(parts):
    df, _, bundle, metrics = parts
    p = Predictor(bundle, df, metrics)
    ts = [p.origin + pd.Timedelta(hours=h) for h in range(1, 49)]
    out = p.predict("railway_station", ts)["predicted_count"]
    assert out.nunique() > 10 and out.std() > 50  # follows the daily pattern, not a fixed number
    other = p.predict("college_campus", ts)["predicted_count"]
    assert not np.array_equal(out.to_numpy(), other.to_numpy())
