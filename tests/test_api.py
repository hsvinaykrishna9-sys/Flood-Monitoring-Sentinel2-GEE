"""
Integration tests for the FastAPI backend - boots the real app (trains
both models for real, same as a real server start) and hits every
endpoint except /weather/{location}, which needs live network (same
limitation as test_data_collection.py's live-API test).
"""

import datetime
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "api"))

from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models_loaded"] is True


def test_predict_today(client):
    r = client.get("/predict/today")
    assert r.status_code == 200
    body = r.json()
    assert body["risk_band"] in ("LOW", "MODERATE", "HIGH")
    assert 0 <= body["flood_risk_probability"] <= 1
    # "today" must be bridged to the real calendar date, not whatever the
    # most recent (structurally lagged) row in the dataset happens to be.
    assert body["date"] == datetime.date.today().isoformat()
    assert "bridge_days" in body
    assert "most_recent_real_data_date" in body


def test_predict_outlook(client):
    r = client.get("/predict/outlook")
    assert r.status_code == 200
    body = r.json()
    assert len(body["days"]) == 16  # Open-Meteo's real live-forecast horizon
    assert body["days"][0]["date"] == datetime.date.today().isoformat()
    for day in body["days"]:
        assert day["risk_band"] in ("LOW", "MODERATE", "HIGH")
        assert 0 <= day["flood_risk_probability"] <= 1


def test_predict_taluk_outlook(client):
    r = client.get("/predict/taluk/outlook")
    assert r.status_code == 200
    body = r.json()
    assert len(body["days"]) == 16
    assert body["days"][0]["date"] == datetime.date.today().isoformat()
    for day in body["days"]:
        taluks = {t["taluk"] for t in day["taluks"]}
        assert taluks == {"Madikeri", "Virajpet", "Somwarpet"}
        for t in day["taluks"]:
            assert t["risk_band"] in ("LOW", "MODERATE", "HIGH")
            assert 0 <= t["flood_risk_probability"] <= 1
            assert isinstance(t["is_live_input"], bool)


def test_predict_historical_known_flood_day(client):
    r = client.get("/predict/historical/2023-07-05")
    assert r.status_code == 200
    body = r.json()
    assert body["risk_band"] == "HIGH"
    assert body["actual_outcome"] == "flood_risk_day"
    assert body["model_matched_actual"] is True


def test_predict_historical_out_of_range_returns_404(client):
    r = client.get("/predict/historical/1800-01-01")
    assert r.status_code == 404


def test_predict_historical_invalid_date_returns_400(client):
    r = client.get("/predict/historical/not-a-date")
    assert r.status_code == 400


def test_predict_taluk_today_covers_all_three(client):
    r = client.get("/predict/taluk/today")
    assert r.status_code == 200
    taluks = {t["taluk"] for t in r.json()["taluks"]}
    assert taluks == {"Madikeri", "Virajpet", "Somwarpet"}


def test_predict_taluk_historical_differentiated_event(client):
    # real event: Madikeri was LOW while Somwarpet/Virajpet were HIGH
    r = client.get("/predict/taluk/historical/2022-08-14")
    assert r.status_code == 200
    by_taluk = {t["taluk"]: t for t in r.json()["taluks"]}
    assert by_taluk["Madikeri"]["risk_band"] == "LOW"
    assert by_taluk["Somwarpet"]["risk_band"] == "HIGH"
    assert by_taluk["Virajpet"]["risk_band"] == "HIGH"


def test_models_comparison(client):
    r = client.get("/models/comparison")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 8
    assert any(row["Model"] == "XGBoost" for row in rows)


def test_api_info_moved_off_root(client):
    # "/" is owned by the dashboard (StaticFiles mount), not the JSON info
    # endpoint - that lives at "/api" instead
    r = client.get("/api")
    assert r.status_code == 200
    assert r.json()["name"] == "Kodagu Flood Risk API"


def test_dashboard_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Kodagu" in r.text


def test_dashboard_assets_served(client):
    for path in ("/style.css", "/app.js"):
        r = client.get(path)
        assert r.status_code == 200, f"{path} should be served as a static asset"


def test_weather_by_coords_rejects_invalid_coordinates(client):
    r = client.get("/weather-by-coords?lat=999&lon=0")
    assert r.status_code == 400


def test_weather_by_coords_fails_gracefully_without_network(client):
    # this sandbox has no route to Open-Meteo, so this must return a clean
    # 502, never crash the server - on a real deployment with network this
    # would return 200 with real weather data instead
    r = client.get("/weather-by-coords?lat=12.4244&lon=75.7382")
    assert r.status_code in (200, 502)


def test_geocode_search_returns_empty_list_without_network(client):
    # never errors, even with no network - always a 200 with a (possibly
    # empty) results list, since it's called on every keystroke
    r = client.get("/geocode-search?q=Mysore")
    assert r.status_code == 200
    assert isinstance(r.json()["results"], list)


def test_build_weather_response_includes_all_dashboard_fields():
    # extra weather fields (UV, visibility, pressure, dew point,
    # sunrise/sunset, moon phase) against a mocked Open-Meteo-shaped
    # response, since this sandbox can't reach the live API
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "api"))
    import main as api_module

    fake_data = {
        "current": {"time": "2026-10-27T09:34", "temperature_2m": 23.4, "relative_humidity_2m": 78,
                     "precipitation": 0.0, "weather_code": 3, "wind_speed_10m": 10.2,
                     "apparent_temperature": 24.1, "dew_point_2m": 15.0, "surface_pressure": 1014.0},
        "hourly": {"time": [f"2026-10-27T{h:02d}:00" for h in range(24)],
                   "temperature_2m": [23.0] * 24, "precipitation_probability": [0] * 24,
                   "weather_code": [3] * 24, "visibility": [19000] * 24, "uv_index": [4] * 24},
        "daily": {"time": ["2026-10-27", "2026-10-28"],
                  "temperature_2m_max": [29, 29], "temperature_2m_min": [18, 17],
                  "precipitation_sum": [0, 0], "weather_code": [3, 3], "wind_speed_10m_max": [13, 18],
                  "uv_index_max": [4, 5], "sunrise": ["2026-10-27T06:12"], "sunset": ["2026-10-27T17:55"]},
    }
    loc = {"name": "Bengaluru", "admin1": "Karnataka", "country": "India", "lat": 12.9716, "lon": 77.5946}

    resp = api_module._build_weather_response(loc, fake_data)
    cur = resp["current"]
    assert cur["feels_like_C"] == 24.1
    assert cur["dew_point_C"] == 15.0
    assert cur["pressure_hpa"] == 1014.0
    assert cur["visibility_km"] == 19.0
    assert cur["uv_index"] == 4
    assert resp["sun"]["sunrise"] == "2026-10-27T06:12"
    assert resp["moon"]["name"]
    assert 0 <= resp["moon"]["illumination_pct"] <= 100
    assert resp["flood_risk_available"] is False
    assert "flood_risk" not in resp
