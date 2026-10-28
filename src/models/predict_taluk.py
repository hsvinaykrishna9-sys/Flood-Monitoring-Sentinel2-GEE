"""
Spatial flood-risk prediction: one risk number PER TALUK (Madikeri,
Virajpet, Somwarpet) instead of a single district-wide number - the
whole point of collecting real per-taluk terrain (elevation, slope,
TWI) and weather. Uses the same XGBoost approach as predict_daily.py,
trained on data/processed/kodagu_taluk_daily_features_2018_2026.csv.

Taluk boundaries are an approximation (Voronoi split around each
taluk's headquarters, not an official administrative polygon) - see
src/preprocessing/feature_engineering_taluk.py for details.

Usage:
    python src/models/predict_taluk.py                 # live, all 3 taluks today
    python src/models/predict_taluk.py 2023-07-05       # historical check, all 3 taluks
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

import datetime

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from feature_engineering_taluk import FEATURE_COLUMNS, TALUKS, run as build_features
from predict_daily import fetch_live_forecast

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

N_OUTLOOK_DAYS = 16  # same horizon as predict_daily.py - Open-Meteo's real live-forecast limit

# Approximate taluk headquarters coordinates (same Voronoi-split anchor
# points used by the GEE terrain/weather extraction scripts) - used here
# only to fetch a live, location-specific forecast for bridging each
# taluk's outlook to today. Not an official administrative boundary.
TALUK_COORDS = {
    "Madikeri": (12.4244, 75.7382),
    "Virajpet": (12.1936, 75.8064),
    "Somwarpet": (12.5985, 75.8517),
}


def build_next_day_row_taluk(history, elevation_m, slope_degrees, twi):
    """Same recursive feature-row construction as
    predict_daily.build_next_day_row, plus the taluk's own (static -
    terrain doesn't change day to day) elevation/slope/TWI columns."""
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
        "elevation_m": elevation_m,
        "slope_degrees": slope_degrees,
        "twi": twi,
    }
    return row, next_date


def compute_outlook_taluk(df, xgb, taluk, n_outlook_days=N_OUTLOOK_DAYS):
    """Per-taluk version of predict_daily.compute_outlook: bridges that
    taluk's own real data up to today's actual calendar date, using a
    live forecast fetched at that taluk's own coordinates (so Virajpet's
    bridged days reflect Virajpet's forecast, not Madikeri's). Returns
    (outlook, last_real, bridge_days, live_forecast_used, terrain) -
    terrain is the taluk's static {elevation_m, slope_degrees, twi}."""
    sub = df[df["taluk"] == taluk].sort_values("date")
    raw_cols = ["date", "rainfall_mm", "runoff_mm", "soil_moisture_m3m3",
                "temperature_C", "relative_humidity_pct", "wind_speed_ms"]
    history = sub[raw_cols].to_dict("records")
    history_live = [True] * len(history)
    last_real = history[-1]

    terrain_row = sub.iloc[-1]
    terrain = {
        "elevation_m": float(terrain_row["elevation_m"]),
        "slope_degrees": float(terrain_row["slope_degrees"]),
        "twi": float(terrain_row["twi"]),
    }
    lat, lon = TALUK_COORDS[taluk]

    real_today = pd.Timestamp(datetime.date.today())
    bridge_days = max(0, (real_today - last_real["date"]).days)
    today_index = max(bridge_days - 1, 0)
    total_steps = today_index + n_outlook_days

    live_forecast = fetch_live_forecast(
        latitude=lat, longitude=lon, past_days=bridge_days, forecast_days=n_outlook_days + 2
    )

    all_predictions = []
    all_X_rows = []
    input_live_flags = []
    for step in range(total_steps):
        row, next_date = build_next_day_row_taluk(
            history, terrain["elevation_m"], terrain["slope_degrees"], terrain["twi"]
        )
        input_live_flags.append(history_live[-1])
        X_next = pd.DataFrame([row])[FEATURE_COLUMNS].values
        prob = xgb.predict_proba(X_next)[0, 1]
        pred = int(prob > 0.5)
        all_predictions.append((next_date, prob, pred))
        all_X_rows.append(X_next)

        fc = live_forecast.get(next_date.strftime("%Y-%m-%d")) if live_forecast else None
        if fc:
            new_rainfall = fc["rainfall_mm"]
            new_temp = fc["temperature_C"]
            new_humidity = fc["relative_humidity_pct"]
            new_wind = fc["wind_speed_ms"]
        else:
            new_rainfall = float(np.mean([h["rainfall_mm"] for h in history[-7:]]))
            new_temp = history[-1]["temperature_C"]
            new_humidity = history[-1]["relative_humidity_pct"]
            new_wind = history[-1]["wind_speed_ms"]
        projected_runoff = float(np.mean([h["runoff_mm"] for h in history[-3:]]))

        history.append({
            "date": next_date,
            "rainfall_mm": new_rainfall,
            "runoff_mm": projected_runoff,
            "soil_moisture_m3m3": history[-1]["soil_moisture_m3m3"],
            "temperature_C": new_temp,
            "relative_humidity_pct": new_humidity,
            "wind_speed_ms": new_wind,
        })
        history_live.append(fc is not None)

    outlook_predictions = all_predictions[today_index:today_index + n_outlook_days]
    outlook_X_rows = all_X_rows[today_index:today_index + n_outlook_days]
    outlook_live_flags = input_live_flags[today_index:today_index + n_outlook_days]

    assert outlook_predictions[0][0].date() == real_today.date(), \
        f"expected first outlook day to be today ({real_today.date()}), got {outlook_predictions[0][0].date()}"

    outlook = [
        {"date": d, "prob": p, "pred": pr, "is_live_input": live, "X_row": xr}
        for (d, p, pr), xr, live in zip(outlook_predictions, outlook_X_rows, outlook_live_flags)
    ]
    return outlook, last_real, bridge_days, live_forecast is not None, terrain


def risk_band(prob):
    if prob < 0.33:
        return "LOW", GREEN
    elif prob < 0.66:
        return "MODERATE", YELLOW
    return "HIGH", RED


def train():
    df = build_features()
    X = df[FEATURE_COLUMNS].values
    y = df["flood_risk"].values
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", random_state=42)
    xgb.fit(X, y)
    return xgb, df


def print_taluk_table(rows):
    print("┌" + "─" * 62 + "┐")
    print(f"│ {BOLD}KODAGU - REGIONAL FLOOD RISK{RESET}" + " " * 33 + "│")
    print("├" + "─" * 62 + "┤")
    for taluk, prob, elevation in rows:
        band, color = risk_band(prob)
        bar_len = round(prob * 20)
        bar = f"{color}{'█' * bar_len}{DIM}{'░' * (20 - bar_len)}{RESET}"
        line = f" {taluk:<11}{color}{band:<9}{RESET}{prob:>5.0%}  {bar}  ({elevation:.0f}m elev)"
        visible_len = len(f" {taluk:<11}{band:<9}{prob:>5.0%}  {'#'*20}  ({elevation:.0f}m elev)")
        pad = max(0, 62 - visible_len)
        print(f"│{line}{' ' * pad}│")
    print("└" + "─" * 62 + "┘")


def run_live():
    xgb, df = train()
    last_dates = df.groupby("taluk")["date"].max()

    rows = []
    for taluk in TALUKS:
        sub = df[df["taluk"] == taluk].sort_values("date")
        latest = sub.iloc[-1]
        X_row = latest[FEATURE_COLUMNS].values.reshape(1, -1)
        prob = xgb.predict_proba(X_row)[0, 1]
        rows.append((taluk, prob, latest["elevation_m"]))

    print(f" {DIM}Most recent real data: {last_dates.max().date()} "
          f"(per-taluk, 2018-2026 record){RESET}")
    print(f" {DIM}Note: this shows the most recent AVAILABLE day's conditions per taluk,")
    print(f" not bridged to today's date (see predict_daily.py for the live-forecast-")
    print(f" bridged single-point version). Taluk split is an approximation - see")
    print(f" feature_engineering_taluk.py.{RESET}")
    print()
    print_taluk_table(rows)
    print()
    print(f" {DIM}Model: XGBoost, trained on {len(df):,} taluk-days (3 taluks x 2018-2026)")
    print(f" Test AUC: 0.975 overall | elevation is the 2nd most important feature{RESET}")


def run_historical(date_str):
    xgb, df = train()
    target_date = pd.Timestamp(date_str)
    match = df[df["date"] == target_date]
    if match.empty:
        print(f"No data for {target_date.date()}. Try a date between "
              f"{df['date'].min().date()} and {df['date'].max().date()}.")
        return

    print("=" * 64)
    print(f"HISTORICAL REGIONAL CHECK - {target_date.date()}")
    print("=" * 64)
    rows = []
    for _, row in match.sort_values("taluk").iterrows():
        X_row = row[FEATURE_COLUMNS].values.reshape(1, -1)
        prob = xgb.predict_proba(X_row)[0, 1]
        actual = int(row["flood_risk"])
        pred = int(prob > 0.5)
        band, color = risk_band(prob)
        match_mark = "✓" if pred == actual else "✗"
        print(f" {row['taluk']:<11}{color}{band:<9}{RESET}{prob:>5.0%}   "
              f"actual: {'FLOOD' if actual else 'normal':<6} {match_mark}   "
              f"(elev {row['elevation_m']:.0f}m, slope {row['slope_degrees']:.1f}°)")
        rows.append((row["taluk"], prob, row["elevation_m"]))
    print("=" * 64)
    print(f"{DIM}Note: inside training data (recalling a fitted pattern), see")
    print(f"reports/model_comparison_taluk_by_taluk.csv for held-out test accuracy.{RESET}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_historical(sys.argv[1])
    else:
        run_live()
