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
import math
import os
import sys
import time
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
from predict_daily import compute_outlook as compute_daily_outlook
from predict_taluk import TALUK_COORDS, compute_outlook_taluk

MODELS = {}
OUTLOOK_CACHE_TTL = 600  # seconds - avoids hitting Open-Meteo on every page load


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


def nearest_taluk(lat, lon):
    """Which of the 3 taluks' HQ is closest to a given point - used so a
    search for a specific place inside Kodagu (e.g. 'Virajpet') shows that
    taluk's own bridged prediction in the gauge, matching the marker
    already highlighted for it on the per-taluk map, instead of a
    different whole-district number."""
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))
    return min(TALUK_COORDS, key=lambda t: haversine_km(lat, lon, *TALUK_COORDS[t]))


def get_daily_outlook():
    """Cached (10 min TTL) bridged-to-today outlook for the whole
    district - computing it calls the live Open-Meteo forecast API, so
    this avoids doing that on every single page load."""
    cached = MODELS.get("daily_outlook_cache")
    if cached and time.time() - cached["ts"] < OUTLOOK_CACHE_TTL:
        return cached["outlook"], cached["last_real"], cached["bridge_days"], cached["live_forecast"]
    xgb, df = MODELS["daily_xgb"], MODELS["daily_df"]
    outlook, last_real, bridge_days, live_forecast = compute_daily_outlook(df, xgb)
    MODELS["daily_outlook_cache"] = {
        "ts": time.time(), "outlook": outlook, "last_real": last_real,
        "bridge_days": bridge_days, "live_forecast": live_forecast,
    }
    return outlook, last_real, bridge_days, live_forecast


def get_taluk_outlook(taluk):
    """Same caching idea as get_daily_outlook(), keyed per taluk (each
    one fetches its own live forecast at its own coordinates)."""
    cache_key = f"taluk_outlook_cache:{taluk}"
    cached = MODELS.get(cache_key)
    if cached and time.time() - cached["ts"] < OUTLOOK_CACHE_TTL:
        return (cached["outlook"], cached["last_real"], cached["bridge_days"],
                cached["live_forecast"], cached["terrain"])
    xgb, df = MODELS["taluk_xgb"], MODELS["taluk_df"]
    outlook, last_real, bridge_days, live_forecast, terrain = compute_outlook_taluk(df, xgb, taluk)
    MODELS[cache_key] = {
        "ts": time.time(), "outlook": outlook, "last_real": last_real,
        "bridge_days": bridge_days, "live_forecast": live_forecast, "terrain": terrain,
    }
    return outlook, last_real, bridge_days, live_forecast, terrain


@app.get("/api")
def root():
    # NOT mounted at "/" - the dashboard (src/frontend/index.html) owns the
    # root path via the StaticFiles mount at the bottom of this file.
    return {
        "name": "Kodagu Flood Risk API",
        "docs": "/docs",
        "endpoints": [
            "/health", "/predict/today", "/predict/outlook", "/predict/climatology", "/predict/historical/{date}",
            "/predict/taluk/today", "/predict/taluk/outlook", "/predict/taluk/historical/{date}",
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
    outlook, last_real, bridge_days, live_forecast = get_daily_outlook()
    today = outlook[0]
    return {
        "date": str(today["date"].date()),
        "flood_risk_probability": round(float(today["prob"]), 4),
        "risk_band": risk_band(today["prob"]),
        "is_live_input": today["is_live_input"],
        "bridge_days": bridge_days,
        "live_forecast_used": live_forecast,
        "most_recent_real_data_date": str(last_real["date"].date()),
    }


@app.get("/predict/outlook")
def predict_outlook():
    """16-day bridged flood-risk outlook for the whole district, starting
    today (16 days = Open-Meteo's real live-forecast horizon, the actual
    limit of what's a genuine forecast rather than trend-projection)."""
    outlook, last_real, bridge_days, live_forecast = get_daily_outlook()
    return {
        "bridge_days": bridge_days,
        "live_forecast_used": live_forecast,
        "most_recent_real_data_date": str(last_real["date"].date()),
        "days": [
            {"date": str(o["date"].date()), "flood_risk_probability": round(float(o["prob"]), 4),
             "risk_band": risk_band(o["prob"]), "is_live_input": o["is_live_input"]}
            for o in outlook
        ],
    }


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@app.get("/predict/climatology")
def climatology():
    """Seasonal flood-risk pattern by calendar month - computed directly
    from the real OBSERVED outcome (flood_risk) across all 76 years of
    ERA5-Land data, not from model predictions. This answers "which
    months are historically riskiest" honestly: no model (this one or
    any weather service's) can skillfully predict actual day-specific
    rainfall more than ~2 weeks out, so a genuine months-ahead answer
    has to come from historical pattern, not a day-by-day forecast."""
    df = MODELS["daily_df"]
    months = pd.to_datetime(df["date"]).dt.month
    years_span = int(pd.to_datetime(df["date"]).dt.year.max() - pd.to_datetime(df["date"]).dt.year.min() + 1)

    rows = []
    for m in range(1, 13):
        sub = df[months == m]
        rows.append({
            "month": MONTH_NAMES[m - 1],
            "flood_risk_rate_pct": round(float(sub["flood_risk"].mean() * 100), 1),
            "avg_rainfall_mm": round(float(sub["rainfall_mm"].mean()), 1),
            "n_days": int(len(sub)),
        })
    return {"years_of_data": years_span, "months": rows}


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


@app.get("/predict/taluk/outlook")
def predict_taluk_outlook():
    """16-day bridged outlook for all 3 taluks, re-shaped per day (each
    day lists all 3 taluks' numbers together) so the dashboard can build
    a per-day, per-taluk map for any day a user picks from the outlook -
    not just today."""
    per_taluk = {}
    for taluk in TALUKS:
        outlook, _, _, _, _ = get_taluk_outlook(taluk)
        per_taluk[taluk] = outlook

    n_days = len(per_taluk[TALUKS[0]])
    days = []
    for i in range(n_days):
        date_str = str(per_taluk[TALUKS[0]][i]["date"].date())
        days.append({
            "date": date_str,
            "taluks": [
                {"taluk": t, "flood_risk_probability": round(float(per_taluk[t][i]["prob"]), 4),
                 "risk_band": risk_band(per_taluk[t][i]["prob"]),
                 "is_live_input": per_taluk[t][i]["is_live_input"]}
                for t in TALUKS
            ],
        })
    return {"days": days}


@app.get("/predict/taluk/today")
def predict_taluk_today():
    results = []
    for taluk in TALUKS:
        outlook, last_real, bridge_days, live_forecast, terrain = get_taluk_outlook(taluk)
        today = outlook[0]
        results.append({
            "taluk": taluk,
            "date": str(today["date"].date()),
            "flood_risk_probability": round(float(today["prob"]), 4),
            "risk_band": risk_band(today["prob"]),
            "is_live_input": today["is_live_input"],
            "bridge_days": bridge_days,
            "most_recent_real_data_date": str(last_real["date"].date()),
            "elevation_m": round(terrain["elevation_m"], 1),
            "slope_degrees": round(terrain["slope_degrees"], 1),
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
        taluk = nearest_taluk(loc["lat"], loc["lon"])
        outlook, last_real, bridge_days, live_forecast, terrain = get_taluk_outlook(taluk)
        today = outlook[0]
        response["flood_risk"] = {
            "taluk": taluk,
            "date": str(today["date"].date()),
            "flood_risk_probability": round(float(today["prob"]), 4),
            "risk_band": risk_band(today["prob"]),
            "is_live_input": today["is_live_input"],
            "bridge_days": bridge_days,
            "most_recent_real_data_date": str(last_real["date"].date()),
        }

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
