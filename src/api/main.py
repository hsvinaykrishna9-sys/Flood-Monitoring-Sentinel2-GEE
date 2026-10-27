"""
FastAPI backend exposing the trained flood-risk models as REST
endpoints - wraps the same models and logic used by predict_daily.py,
predict_taluk.py, and weather_app.py, no duplicated model logic.

Models are trained ONCE at startup and cached in memory (training is a
few seconds on ~28k/~9.5k rows - fine to pay once, not per request).

Run:
    uvicorn src.api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs
(FastAPI generates this automatically).
"""

import datetime
import os
import sys
import warnings
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from xgboost import XGBClassifier

from feature_engineering_daily import FEATURE_COLUMNS as DAILY_FEATURES, run as build_daily_features
from feature_engineering_taluk import FEATURE_COLUMNS as TALUK_FEATURES, TALUKS, run as build_taluk_features

MODELS = {}


def train_daily_model():
    df = build_daily_features()
    X, y = df[DAILY_FEATURES].values, df["flood_risk"].values
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", random_state=42)
    xgb.fit(X, y)
    return xgb, df


def train_taluk_model():
    df = build_taluk_features()
    X, y = df[TALUK_FEATURES].values, df["flood_risk"].values
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", random_state=42)
    xgb.fit(X, y)
    return xgb, df


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Training daily (whole-district) model...")
    MODELS["daily_xgb"], MODELS["daily_df"] = train_daily_model()
    print("Training per-taluk (spatial) model...")
    MODELS["taluk_xgb"], MODELS["taluk_df"] = train_taluk_model()
    print("Models ready.")
    yield
    MODELS.clear()


app = FastAPI(
    title="Kodagu Flood Risk API",
    description="Real day-ahead flood-risk prediction for Kodagu district, Karnataka, "
                 "built on 76 years of real ERA5-Land weather data.",
    version="1.0.0",
    lifespan=lifespan,
)

# Permissive CORS: this is a student project API with no auth/private data,
# and the dashboard may be opened from a different origin than the API
# (e.g. a local file, or a separately-hosted static page) during testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def risk_band(prob):
    if prob < 0.33:
        return "LOW"
    elif prob < 0.66:
        return "MODERATE"
    return "HIGH"


def shap_explanation(xgb, X_row, feature_names, top_n=5):
    import shap
    import numpy as np
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_row)[0]
    order = np.argsort(-np.abs(shap_values))[:top_n]
    return [
        {"feature": feature_names[i], "impact": round(float(shap_values[i]), 3),
         "direction": "up" if shap_values[i] > 0 else "down"}
        for i in order
    ]


@app.get("/api")
def root():
    # NOT mounted at "/" - the dashboard (src/frontend/index.html) owns the
    # root path via the StaticFiles mount at the bottom of this file.
    return {
        "name": "Kodagu Flood Risk API",
        "docs": "/docs",
        "endpoints": [
            "/health", "/predict/today", "/predict/historical/{date}",
            "/predict/taluk/today", "/predict/taluk/historical/{date}",
            "/weather/{location}", "/weather-by-coords?lat=&lon=",
            "/geocode-search?q=", "/models/comparison",
        ],
        "dashboard": "/",
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(MODELS)}


@app.get("/predict/today")
def predict_today():
    xgb, df = MODELS["daily_xgb"], MODELS["daily_df"]
    latest = df.sort_values("date").iloc[-1]
    X_row = latest[DAILY_FEATURES].values.reshape(1, -1)
    prob = float(xgb.predict_proba(X_row)[0, 1])
    return {
        "most_recent_real_data_date": str(latest["date"].date()),
        "flood_risk_probability": round(prob, 4),
        "risk_band": risk_band(prob),
        "explanation": shap_explanation(xgb, X_row, DAILY_FEATURES),
        "note": "This is the most recent available day's conditions, not bridged to "
                "today's calendar date. For a live-forecast-bridged 10-day outlook, "
                "use predict_daily.py directly.",
    }


@app.get("/predict/historical/{date}")
def predict_historical(date: str):
    xgb, df = MODELS["daily_xgb"], MODELS["daily_df"]
    try:
        target_date = pd.Timestamp(date)
    except ValueError:
        raise HTTPException(400, f"Invalid date: {date}. Use YYYY-MM-DD.")

    match = df[df["date"] == target_date]
    if match.empty:
        raise HTTPException(
            404,
            f"No data for {target_date.date()}. Available range: "
            f"{df['date'].min().date()} to {df['date'].max().date()}.",
        )

    row = match.iloc[0]
    X_row = row[DAILY_FEATURES].values.reshape(1, -1)
    prob = float(xgb.predict_proba(X_row)[0, 1])
    actual = int(row["flood_risk"])
    return {
        "date": str(target_date.date()),
        "flood_risk_probability": round(prob, 4),
        "risk_band": risk_band(prob),
        "actual_outcome": "flood_risk_day" if actual else "normal_day",
        "model_matched_actual": (prob > 0.5) == bool(actual),
        "explanation": shap_explanation(xgb, X_row, DAILY_FEATURES),
        "note": "This date is inside the model's training data (recalling a fitted "
                "pattern), not out-of-sample generalization.",
    }


@app.get("/predict/taluk/today")
def predict_taluk_today():
    xgb, df = MODELS["taluk_xgb"], MODELS["taluk_df"]
    results = []
    for taluk in TALUKS:
        sub = df[df["taluk"] == taluk].sort_values("date")
        latest = sub.iloc[-1]
        X_row = latest[TALUK_FEATURES].values.reshape(1, -1)
        prob = float(xgb.predict_proba(X_row)[0, 1])
        results.append({
            "taluk": taluk,
            "most_recent_real_data_date": str(latest["date"].date()),
            "flood_risk_probability": round(prob, 4),
            "risk_band": risk_band(prob),
            "elevation_m": round(float(latest["elevation_m"]), 1),
            "slope_degrees": round(float(latest["slope_degrees"]), 1),
        })
    return {"taluks": results}


@app.get("/predict/taluk/historical/{date}")
def predict_taluk_historical(date: str):
    xgb, df = MODELS["taluk_xgb"], MODELS["taluk_df"]
    try:
        target_date = pd.Timestamp(date)
    except ValueError:
        raise HTTPException(400, f"Invalid date: {date}. Use YYYY-MM-DD.")

    match = df[df["date"] == target_date]
    if match.empty:
        raise HTTPException(
            404,
            f"No data for {target_date.date()}. Available range: "
            f"{df['date'].min().date()} to {df['date'].max().date()}.",
        )

    results = []
    for _, row in match.sort_values("taluk").iterrows():
        X_row = row[TALUK_FEATURES].values.reshape(1, -1)
        prob = float(xgb.predict_proba(X_row)[0, 1])
        actual = int(row["flood_risk"])
        results.append({
            "taluk": row["taluk"],
            "flood_risk_probability": round(prob, 4),
            "risk_band": risk_band(prob),
            "actual_outcome": "flood_risk_day" if actual else "normal_day",
            "model_matched_actual": (prob > 0.5) == bool(actual),
        })
    return {"date": str(target_date.date()), "taluks": results}


def _build_weather_response(loc, data):
    from weather_app import is_in_kodagu, wmo_text, moon_phase, fetch_air_quality

    times = data["hourly"]["time"]
    current_time = data["current"].get("time", "")
    now_idx = next((i for i, t in enumerate(times) if t >= current_time), 0)
    cur = data["current"]
    daily = data["daily"]

    response = {
        "location": {"name": loc["name"], "admin1": loc["admin1"], "country": loc["country"],
                      "lat": loc["lat"], "lon": loc["lon"]},
        "current": {
            "temperature_C": cur["temperature_2m"],
            "feels_like_C": cur.get("apparent_temperature"),
            "condition": wmo_text(cur["weather_code"]),
            "humidity_pct": cur["relative_humidity_2m"],
            "dew_point_C": cur.get("dew_point_2m"),
            "wind_speed_kmh": cur["wind_speed_10m"],
            "pressure_hpa": cur.get("surface_pressure"),
            "visibility_km": (data["hourly"]["visibility"][now_idx] / 1000
                               if "visibility" in data.get("hourly", {}) else None),
            "uv_index": daily.get("uv_index_max", [None])[0],
        },
        "hourly_outlook": [
            {"time": times[i], "temp_C": data["hourly"]["temperature_2m"][i],
             "rain_chance_pct": data["hourly"]["precipitation_probability"][i],
             "condition": wmo_text(data["hourly"]["weather_code"][i])}
            for i in range(now_idx, min(now_idx + 8, len(times)))
        ],
        "daily_outlook": [
            {"date": daily["time"][i],
             "temp_min_C": daily["temperature_2m_min"][i],
             "temp_max_C": daily["temperature_2m_max"][i],
             "rain_mm": daily["precipitation_sum"][i],
             "condition": wmo_text(daily["weather_code"][i])}
            for i in range(len(daily["time"]))
        ],
        "sun": {"sunrise": daily.get("sunrise", [None])[0], "sunset": daily.get("sunset", [None])[0]},
        "moon": moon_phase(),
        "flood_risk_available": is_in_kodagu(loc["lat"], loc["lon"]),
    }

    air = fetch_air_quality(loc["lat"], loc["lon"])
    if air is not None:
        response["air_quality"] = air

    if response["flood_risk_available"]:
        response["flood_risk"] = predict_today()

    return response


@app.get("/geocode-search")
def geocode_search_endpoint(q: str):
    """Search-as-you-type autocomplete for the dashboard's search box -
    returns up to 8 matches (name + state/admin1 + country), never an
    error for 'no results' (just an empty list), since this is meant to
    be called on every keystroke."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
    from weather_app import geocode_search
    return {"results": geocode_search(q)}


@app.get("/weather/{location}")
def weather(location: str):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
    from weather_app import geocode, fetch_weather

    loc = geocode(location)
    if loc is None:
        raise HTTPException(404, f"Couldn't find '{location}'.")

    data = fetch_weather(loc["lat"], loc["lon"])
    if data is None:
        raise HTTPException(502, "Found the location but couldn't reach the weather service.")

    return _build_weather_response(loc, data)


@app.get("/weather-by-coords")
def weather_by_coords(lat: float, lon: float, name: str = "Current Location",
                       admin1: str = "", country: str = ""):
    """Coordinates-based weather lookup. Used two ways by the dashboard:
    1. Browser GPS 'current location' mode - real coordinates from
       navigator.geolocation, name left as the default 'Current Location'
       (no reverse-geocoding lookup).
    2. An autocomplete suggestion was clicked - the dashboard already has
       the name/admin1/country/lat/lon from /geocode-search, so it passes
       them straight through here instead of a redundant second geocode
       lookup by name."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
    from weather_app import fetch_weather

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(400, "lat must be in [-90, 90] and lon in [-180, 180].")

    data = fetch_weather(lat, lon)
    if data is None:
        raise HTTPException(502, "Couldn't reach the weather service.")

    loc = {"name": name, "admin1": admin1, "country": country, "lat": lat, "lon": lon}
    return _build_weather_response(loc, data)


@app.get("/models/comparison")
def models_comparison():
    path = "reports/model_comparison_daily.csv"
    if not os.path.exists(path):
        raise HTTPException(404, f"{path} not found.")
    df = pd.read_csv(path)
    return JSONResponse(df.to_dict(orient="records"))


# Serve the dashboard (src/frontend/) as static files, mounted LAST so it
# never shadows the API routes declared above - only paths that don't match
# an API route fall through to the static files here. html=True serves
# index.html automatically at "/".
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
