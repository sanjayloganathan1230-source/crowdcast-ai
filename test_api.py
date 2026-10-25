import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import create_app
import dataclasses

ORIGIN = pd.Timestamp("2026-05-10 23:00")  # last obs of the 70-day fixture dataset


def iso(h):
    return (ORIGIN + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%S")


def test_health(client):
    r = client.get("/api/health"); j = r.json()
    assert r.status_code == 200 and j["status"] == "ok" and j["observations"] == 70 * 24 * 4
    assert "DEMO DATA" in j["data_label"]


def test_config(client):
    j = client.get("/api/config").json()
    assert j["last_observation"] == "2026-05-10 23:00:00" and j["max_horizon_hours"] == 168
    assert j["default_thresholds"] == {"low_max": 40, "moderate_max": 70, "high_max": 90}


def test_locations(client):
    j = client.get("/api/locations").json()
    assert len(j["locations"]) == 4 and "Illustrative" in j["coordinates_note"]
    assert {l["id"] for l in j["locations"]} == {"railway_station", "bus_terminal", "shopping_mall", "college_campus"}
    assert all(-90 <= l["lat"] <= 90 and -180 <= l["lon"] <= 180 for l in j["locations"])


def test_observations_filters_and_pagination(client):
    j = client.get("/api/observations", params={"location_id": "shopping_mall", "limit": 10}).json()
    assert j["total"] == 70 * 24 and len(j["items"]) == 10 and "DEMO DATA" in j["data_label"]
    p2 = client.get("/api/observations", params={"location_id": "shopping_mall", "limit": 10, "offset": 10}).json()
    assert p2["items"][0]["timestamp"] != j["items"][0]["timestamp"]
    d = client.get("/api/observations", params={"start": "2026-03-10T00:00:00", "end": "2026-03-10T23:00:00"}).json()
    assert d["total"] == 24 * 4


@pytest.mark.parametrize("params", [
    {"location_id": "nowhere"}, {"limit": 0}, {"limit": 99999}, {"offset": -1},
    {"start": "2026-03-11T00:00:00", "end": "2026-03-10T00:00:00"}, {"start": "not-a-date"},
    {"start": "2026-03-10T00:00:00Z"}])
def test_observations_bad_params(client, params):
    assert client.get("/api/observations", params=params).status_code == 422


def test_export_csv(client):
    r = client.get("/api/observations/export", params={"location_id": "bus_terminal"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    lines = r.text.strip().splitlines()
    assert lines[0].startswith("timestamp,location_id") and len(lines) == 70 * 24 + 1
    assert "DEMO DATA" in lines[1]
    assert client.get("/api/observations/export", params={"location_id": "zzz"}).status_code == 422


def test_hourly_profile_and_peaks(client):
    j = client.get("/api/analytics/hourly-profile", params={"location_id": "railway_station", "day_type": "weekday"}).json()
    assert len(j["profile"]) == 24 and len(j["peak_hours"]) == 3
    assert set(j["peak_hours"]) & {7, 8, 9, 17, 18, 19}
    assert client.get("/api/analytics/hourly-profile", params={"location_id": "railway_station", "day_type": "x"}).status_code == 422
    assert client.get("/api/analytics/hourly-profile", params={"location_id": "nope"}).status_code == 422


def test_location_comparison(client):
    j = client.get("/api/analytics/location-comparison").json()
    assert len(j["items"]) == 4 and all(0 <= i["avg_occupancy_pct"] <= 100 and 0 <= i["peak_hour"] <= 23 for i in j["items"])


def test_model_metrics_read_from_file(client, env):
    j = client.get("/api/model/metrics").json()
    assert j["overall"]["random_forest"]["mae"] == env["result"].metrics["overall"]["random_forest"]["mae"]


def test_metrics_endpoint_serves_the_file_not_constants(env, tmp_path):
    """Edit the metrics file on disk: the API must reflect the edit (proves nothing is hardcoded)."""
    import json
    m = json.loads(env["settings"].metrics_path.read_text())
    m["overall"]["random_forest"]["mae"] = 12345.678
    f = tmp_path / "edited_metrics.json"; f.write_text(json.dumps(m))
    s = dataclasses.replace(env["settings"], metrics_path=f)
    with TestClient(create_app(s)) as c:
        assert c.get("/api/model/metrics").json()["overall"]["random_forest"]["mae"] == 12345.678


def test_next_hour(client):
    j = client.get("/api/predictions/next-hour", params={"location_id": "railway_station"}).json()
    p = j["predictions"][0]
    assert p["timestamp"] == iso(1) and j["origin"] == iso(0)
    assert 0 <= p["predicted_count"] <= p["capacity"] == 5000
    assert p["level"] in ("Low", "Moderate", "High", "Very High") and "DEMO DATA" in j["data_label"]


def test_predictions_deterministic_and_consistent(client):
    body = {"location_id": "bus_terminal", "timestamps": [iso(3), iso(30), iso(100)]}
    a, b = client.post("/api/predictions", json=body).json(), client.post("/api/predictions", json=body).json()
    assert a == b and [p["timestamp"] for p in a["predictions"]] == [iso(3), iso(30), iso(100)]
    nh = client.get("/api/predictions/next-hour", params={"location_id": "bus_terminal"}).json()
    one = client.post("/api/predictions", json={"location_id": "bus_terminal", "timestamps": [iso(1)]}).json()
    assert nh["predictions"] == one["predictions"]


def test_thresholds_and_capacity_override_change_class_only(client):
    base = {"location_id": "college_campus", "timestamps": [iso(11)]}
    d = client.post("/api/predictions", json=base).json()["predictions"][0]
    s = client.post("/api/predictions", json={**base, "thresholds": {"low_max": 1, "moderate_max": 2, "high_max": 3}}).json()
    assert s["predictions"][0]["predicted_count"] == d["predicted_count"]
    assert s["thresholds_used"]["high_max"] == 3
    if d["predicted_count"] > 0:
        assert s["predictions"][0]["level"] == "Very High"
    o = client.post("/api/predictions", json={**base, "capacity_override": 10}).json()["predictions"][0]
    assert o["predicted_count"] == d["predicted_count"] and o["capacity"] == 10
    if d["predicted_count"] > 0:
        assert o["level"] == "Very High"


@pytest.mark.parametrize("body", [
    {"location_id": "nowhere", "timestamps": [iso(1)]},
    {"location_id": "bus_terminal", "timestamps": []},
    {"location_id": "bus_terminal", "timestamps": [iso(0)]},                  # not future
    {"location_id": "bus_terminal", "timestamps": [iso(169)]},                # beyond horizon
    {"location_id": "bus_terminal", "timestamps": [iso(-48)]},                # past
    {"location_id": "bus_terminal", "timestamps": ["2026-05-11T01:30:00"]},   # not on the hour
    {"location_id": "bus_terminal", "timestamps": ["2026-05-11T01:00:00Z"]},  # tz-aware
    {"location_id": "bus_terminal", "timestamps": [iso(1), iso(1)]},          # duplicate
    {"location_id": "bus_terminal", "timestamps": ["garbage"]},
    {"location_id": "bus_terminal", "timestamps": [iso(1)], "event_indicator": 2},
    {"location_id": "bus_terminal", "timestamps": [iso(1)], "capacity_override": 0},
    {"location_id": "bus_terminal", "timestamps": [iso(1)], "capacity_override": -5},
    {"location_id": "bus_terminal", "timestamps": [iso(1)], "thresholds": {"low_max": 80, "moderate_max": 50, "high_max": 90}},
    {"location_id": "bus_terminal", "timestamps": [iso(1)], "thresholds": {"low_max": 40, "moderate_max": 70, "high_max": 120}},
    {"timestamps": [iso(1)]},
])
def test_prediction_validation(client, body):
    assert client.post("/api/predictions", json=body).status_code == 422


@pytest.mark.parametrize("flag", ["0", "1"])
def test_next_hour_accepts_explicit_event_indicator(client, flag):
    """Regression: an explicit valid event_indicator used to be rejected with 422 (found by the browser test)."""
    r = client.get("/api/predictions/next-hour", params={"location_id": "college_campus", "event_indicator": flag})
    assert r.status_code == 200
    assert r.json()["event_indicator"] == int(flag)


def test_next_hour_bad_params(client):
    for p in [{"location_id": "zzz"}, {"location_id": "bus_terminal", "event_indicator": 5},
              {"location_id": "bus_terminal", "low_max": 90, "moderate_max": 50}]:
        assert client.get("/api/predictions/next-hour", params=p).status_code == 422


def test_backtest(client, env):
    j = client.get("/api/predictions/backtest", params={"location_id": "railway_station", "days": 3}).json()
    assert len(j["points"]) == 72 and "Held-out" in j["note"]
    test_start = env["result"].metrics["split"]["test_start"]
    assert j["points"][0]["timestamp"] >= test_start.replace(" ", "T")
    assert client.get("/api/predictions/backtest", params={"location_id": "railway_station", "days": 30}).status_code == 422
    assert client.get("/api/predictions/backtest", params={"location_id": "railway_station", "days": 0}).status_code == 422


def test_503_when_model_missing(env, tmp_path):
    s = dataclasses.replace(env["settings"], model_path=tmp_path / "none.joblib")
    with TestClient(create_app(s)) as c:
        assert c.get("/api/health").json()["model_file_present"] is False
        r = c.get("/api/predictions/next-hour", params={"location_id": "bus_terminal"})
        assert r.status_code == 503 and "train_model" in r.json()["detail"]
        assert c.get("/api/locations").status_code == 200  # non-model endpoints still work


def test_empty_db_is_auto_seeded_with_demo_data(env, tmp_path):
    s = dataclasses.replace(env["settings"], db_path=tmp_path / "fresh.db")
    with TestClient(create_app(s)) as c:
        assert c.get("/api/health").json()["observations"] == 182 * 24 * 4


def test_cors_headers(client):
    r = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    r2 = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r2.headers
