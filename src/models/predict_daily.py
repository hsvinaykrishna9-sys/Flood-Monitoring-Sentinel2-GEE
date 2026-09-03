"""
Live day-ahead flood-risk prediction using the SVM model, on real daily
ERA5-Land data for Kodagu district (through 2026-08-26).

Predicts flood risk for TOMORROW (the day after the most recent real
data), using only already-known information (no leakage) - a genuine
one-day-ahead forecast. Also prints an approximate 7-day outlook by
recursively projecting forward assuming near-term conditions persist
near the recent rolling average - this part is a labeled simplification,
not measured data, and is explicitly flagged as such in the output.

Usage:
    python src/models/predict_daily.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_engineering_daily import FEATURE_COLUMNS, run as build_features

N_OUTLOOK_DAYS = 10

# ANSI colors (most terminals support this; harmless if not)
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def risk_band(prob):
    if prob < 0.33:
        return "LOW", GREEN
    elif prob < 0.66:
        return "MODERATE", YELLOW
    return "HIGH", RED


def risk_bar(prob, width=24):
    filled = round(prob * width)
    _, color = risk_band(prob)
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"


def build_next_day_row(history, day_offset=1):
    """history: list of dicts with keys rainfall_mm, runoff_mm, soil_moisture_m3m3,
    temperature_C, relative_humidity_pct, wind_speed_ms, date (most recent last).
    Builds the feature row for the day right after history[-1]."""
    last = history[-1]
    prev = history[-2]
    prev2 = history[-3]
    rainfall_last3 = [h["rainfall_mm"] for h in history[-3:]]
    rainfall_last7 = [h["rainfall_mm"] for h in history[-7:]]
    runoff_last3 = [h["runoff_mm"] for h in history[-3:]]

    next_date = last["date"] + pd.Timedelta(days=1)
    doy = next_date.dayofyear

    row = {
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "rainfall_lag1": last["rainfall_mm"],
        "rainfall_lag2": prev["rainfall_mm"],
        "rainfall_lag3": prev2["rainfall_mm"],
        "rainfall_roll3": float(np.mean(rainfall_last3)),
        "rainfall_roll7": float(np.mean(rainfall_last7)),
        "runoff_lag1": last["runoff_mm"],
        "runoff_roll3": float(np.mean(runoff_last3)),
        "soil_moisture_lag1": last["soil_moisture_m3m3"],
        "temperature_lag1": last["temperature_C"],
        "humidity_lag1": last["relative_humidity_pct"],
        "wind_speed_lag1": last["wind_speed_ms"],
        "rain_soil_interaction_lag1": last["rainfall_mm"] * last["soil_moisture_m3m3"],
    }
    return row, next_date


def run():
    df = build_features()

    X = df[FEATURE_COLUMNS].values
    y = df["flood_risk"].values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_scaled, y)

    # rolling "known history" buffer, starts as the real observed data
    raw_cols = ["date", "rainfall_mm", "runoff_mm", "soil_moisture_m3m3",
                "temperature_C", "relative_humidity_pct", "wind_speed_ms"]
    history = df[raw_cols].to_dict("records")

    last_real = history[-1]
    print("=" * 66)
    print("FloodPredict - Kodagu District - SVM Day-Ahead Flood Forecast")
    print("=" * 66)
    print(f"Latest real data: {last_real['date'].date()}")
    print(f"  rainfall: {last_real['rainfall_mm']:.1f} mm   runoff: {last_real['runoff_mm']:.1f} mm"
          f"   soil moisture: {last_real['soil_moisture_m3m3']:.3f} m3/m3")
    print("-" * 66)

    outlook = []
    for step in range(N_OUTLOOK_DAYS):
        row, next_date = build_next_day_row(history)
        X_next = pd.DataFrame([row])[FEATURE_COLUMNS].values
        X_next_scaled = scaler.transform(X_next)
        prob = svm.predict_proba(X_next_scaled)[0, 1]
        pred = int(prob > 0.5)
        outlook.append((next_date, prob, pred))

        if step == 0:
            print(f"TOMORROW ({next_date.date()}) - based on real observed data through "
                  f"{last_real['date'].date()}:")
            print(f"  Predicted flood-risk probability: {prob:.1%}")
            print(f"  Predicted risk level: {'HIGH' if pred else 'LOW'}")
            print("-" * 66)
            print(f"Approximate {N_OUTLOOK_DAYS - 1}-day extended outlook")
            print("(projects forward assuming near-term rainfall stays close to the")
            print(" recent 7-day average - NOT measured data, confidence drops with")
            print(" each extra day, same as any real weather forecast):")

        # extend the "known history" with a projected day (persistence of recent
        # rolling averages) so the next iteration's lag features are defined
        projected_rainfall = float(np.mean([h["rainfall_mm"] for h in history[-7:]]))
        projected_runoff = float(np.mean([h["runoff_mm"] for h in history[-3:]]))
        history.append({
            "date": next_date,
            "rainfall_mm": projected_rainfall,
            "runoff_mm": projected_runoff,
            "soil_moisture_m3m3": history[-1]["soil_moisture_m3m3"],
            "temperature_C": history[-1]["temperature_C"],
            "relative_humidity_pct": history[-1]["relative_humidity_pct"],
            "wind_speed_ms": history[-1]["wind_speed_ms"],
        })

    for i, (date, prob, pred) in enumerate(outlook[1:], start=2):
        print(f"  Day +{i} ({date.date()}): {prob:.1%} probability -> {'HIGH' if pred else 'LOW'}")

    print("=" * 66)
    print("Model: SVM (RBF kernel), trained on all 3,153 daily records")
    print("(2018-2026). Validated out-of-sample AUC: 0.874")
    print("NOTE: On this daily dataset, Random Forest scored higher (AUC 0.933,")
    print("recall 51%) than SVM (AUC 0.874, recall 49%). SVM is shown here because")
    print("it was the chosen model; be ready to explain this trade-off if asked.")
    print("See reports/model_comparison_daily.csv for the full comparison.")
    print("=" * 66)


if __name__ == "__main__":
    run()
