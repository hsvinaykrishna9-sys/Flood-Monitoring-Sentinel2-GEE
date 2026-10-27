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
from fastapi.responses import JSONResponse
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


@app.get("/")
def root():
    return {
        "name": "Kodagu Flood Risk API",
        "docs": "/docs",
        "endpoints": [
            "/health", "/predict/today", "/predict/historical/{date}",
            "/predict/taluk/today", "/predict/taluk/historical/{date}",
            "/weather/{location}", "/models/comparison",
        ],
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


@app.get("/weather/{location}")
def weather(location: str):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
    from weather_app import geocode, fetch_weather, is_in_kodagu, wmo_text

    loc = geocode(location)
    if loc is None:
        raise HTTPException(404, f"Couldn't find '{location}'.")

    data = fetch_weather(loc["lat"], loc["lon"])
    if data is None:
        raise HTTPException(502, "Found the location but couldn't reach the weather service.")

    response = {
        "location": {"name": loc["name"], "admin1": loc["admin1"], "country": loc["country"],
                      "lat": loc["lat"], "lon": loc["lon"]},
        "current": {
            "temperature_C": data["current"]["temperature_2m"],
            "condition": wmo_text(data["current"]["weather_code"]),
            "humidity_pct": data["current"]["relative_humidity_2m"],
            "wind_speed_kmh": data["current"]["wind_speed_10m"],
        },
        "daily_outlook": [
            {"date": data["daily"]["time"][i],
             "temp_min_C": data["daily"]["temperature_2m_min"][i],
             "temp_max_C": data["daily"]["temperature_2m_max"][i],
             "rain_mm": data["daily"]["precipitation_sum"][i],
             "condition": wmo_text(data["daily"]["weather_code"][i])}
            for i in range(len(data["daily"]["time"]))
        ],
        "flood_risk_available": is_in_kodagu(loc["lat"], loc["lon"]),
    }

    if response["flood_risk_available"]:
        today = predict_today()
        response["flood_risk"] = today

    return response


@app.get("/models/comparison")
def models_comparison():
    path = "reports/model_comparison_daily.csv"
    if not os.path.exists(path):
        raise HTTPException(404, f"{path} not found.")
    df = pd.read_csv(path)
    return JSONResponse(df.to_dict(orient="records"))
