"""
Integration tests for the FastAPI backend - boots the real app (trains
both models for real, same as a real server start) and hits every
endpoint except /weather/{location}, which needs live network (same
limitation as test_data_collection.py's live-API test).
"""

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
    assert len(body["explanation"]) == 5


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
