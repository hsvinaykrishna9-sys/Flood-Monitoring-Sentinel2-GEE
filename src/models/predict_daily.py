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

import datetime
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


def run_historical_check(target_date_str):
    """Show the model's prediction on a real historical date, alongside the
    real (ground-truth) outcome for that day. Demonstrates that the model
    genuinely varies its output with real conditions, using real data -
    not a live forecast."""
    df = build_features()
    df["date"] = pd.to_datetime(df["date"])

    X = df[FEATURE_COLUMNS].values
    y = df["flood_risk"].values
    scaler = StandardScaler().fit(X)
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(scaler.transform(X), y)

    target_date = pd.Timestamp(target_date_str)
    match = df[df["date"] == target_date]
    if match.empty:
        print(f"No data for {target_date.date()}. Try a date between "
              f"{df['date'].min().date()} and {df['date'].max().date()}.")
        return

    row = match.iloc[0]
    X_row = scaler.transform(row[FEATURE_COLUMNS].values.reshape(1, -1))
    prob = svm.predict_proba(X_row)[0, 1]
    pred = int(prob > 0.5)
    actual = int(row["flood_risk"])
    band, color = risk_band(prob)

    print("=" * 66)
    print(f"HISTORICAL CHECK - {target_date.date()} ({DAY_NAMES[target_date.weekday()]})")
    print("=" * 66)
    print(f"Conditions going into this day: prior-day rainfall "
          f"{row['rainfall_lag1']:.1f}mm, prior-day runoff {row['runoff_lag1']:.1f}mm")
    print(f"Model prediction:  {color}{BOLD}{band} RISK{RESET}  ({prob:.1%} probability)")
    print(f"Actual outcome:    {'FLOOD-RISK DAY' if actual else 'normal day'} "
          f"(runoff exceeded the 85th-percentile threshold: {'yes' if actual else 'no'})")
    print(f"Model {'correctly matched' if pred == actual else 'did NOT match'} the real outcome.")
    print("-" * 66)
    print(f"{DIM}Note: this date is inside the model's training data (it was trained")
    print(f"on all 3,153 days), so this shows the model recalling a fitted pattern,")
    print(f"not out-of-sample generalization. For genuine held-out accuracy, see")
    print(f"reports/model_comparison_daily.csv (test-set AUC 0.874 for SVM).{RESET}")
    print("=" * 66)


def run():
    if len(sys.argv) > 1:
        run_historical_check(sys.argv[1])
        return

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
    W = 74

    def box_line(visible_text, colored_text=None):
        colored_text = colored_text if colored_text is not None else visible_text
        pad = max(0, (W - 2) - len(visible_text))
        print(f"│{colored_text}{' ' * pad}│")

    print("┌" + "─" * (W - 2) + "┐")
    box_line(" MY LOCATION", f" {BOLD}MY LOCATION{RESET}")
    box_line(" Kodagu District, Karnataka", f" {BOLD}Kodagu District, Karnataka{RESET}")
    box_line("")
    box_line(f"  Last observed: {last_real['date'].date()}")
    cond_line = (f"  Rainfall {last_real['rainfall_mm']:.1f}mm   "
                 f"Runoff {last_real['runoff_mm']:.1f}mm   "
                 f"Soil moisture {last_real['soil_moisture_m3m3']:.2f} m3/m3")
    box_line(cond_line)
    print("└" + "─" * (W - 2) + "┘")
    print()

    # Bridge the gap between the last real ERA5-Land data point and the
    # actual current date (ERA5-Land always lags real-time by roughly a
    # week or more) so the displayed outlook is anchored to TODAY, not to
    # whatever date the satellite/reanalysis data happened to stop at.
    real_today = pd.Timestamp(datetime.date.today())
    bridge_days = max(0, (real_today - last_real["date"]).days)
    # all_predictions[k] will hold the date (last_real_date + k+1 days), so the
    # entry for "today" sits at index (bridge_days - 1); floor at 0 for the
    # (practically impossible) case where the data is already fully current.
    today_index = max(bridge_days - 1, 0)
    total_steps = today_index + N_OUTLOOK_DAYS

    all_predictions = []
    for step in range(total_steps):
        row, next_date = build_next_day_row(history)
        X_next = pd.DataFrame([row])[FEATURE_COLUMNS].values
        X_next_scaled = scaler.transform(X_next)
        prob = svm.predict_proba(X_next_scaled)[0, 1]
        pred = int(prob > 0.5)
        all_predictions.append((next_date, prob, pred))

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

    # only the last N_OUTLOOK_DAYS (today onward) are actually displayed;
    # the bridge days were needed to advance the lag/rolling features but
    # aren't shown individually
    outlook = all_predictions[today_index:today_index + N_OUTLOOK_DAYS]
    assert outlook[0][0].date() == real_today.date(), \
        f"expected first outlook day to be today ({real_today.date()}), got {outlook[0][0].date()}"

    if bridge_days > 0:
        print(f" {DIM}Real satellite/weather data currently ends {last_real['date'].date()}"
              f" ({bridge_days} day(s) behind today, due to ERA5-Land's normal")
        print(f" processing lag). The {bridge_days} day(s) since then were auto-bridged"
              f" using recent trends so this outlook lines up with today's date.{RESET}")
        print()

    today_prob, today_pred = outlook[0][1], outlook[0][2]
    band, color = risk_band(today_prob)
    print(f" {BOLD}TODAY{RESET}  ({real_today.date()}, {DAY_NAMES[real_today.weekday()]})"
          f"   {color}{BOLD}{band} RISK{RESET}   ({today_prob:.0%} probability)")
    print()
    print(f" {BOLD}{N_OUTLOOK_DAYS}-DAY FLOOD RISK OUTLOOK (today + next {N_OUTLOOK_DAYS - 1} days){RESET}")
    print(f" {DIM}Projects forward from recent rainfall trends - not measured data,")
    print(f" confidence drops the further out you go, same as any weather forecast.{RESET}")
    print("─" * W)
    for i, (date, prob, pred) in enumerate(outlook, start=0):
        band, color = risk_band(prob)
        label = "Today" if i == 0 else f"Day +{i}"
        day_str = f"{date.strftime('%b %d')} ({DAY_NAMES[date.weekday()]})"
        print(f" {label:<10}{day_str:<14}{color}{band:<9}{RESET}{prob:>5.0%}   {risk_bar(prob)}")
    print("─" * W)
    print()
    print(f" {DIM}Model: SVM (RBF kernel) | trained on 3,153 real daily records, 2018-2026{RESET}")
    print(f" {DIM}Validated out-of-sample AUC: 0.874{RESET}")
    print(f" {YELLOW}Note: Random Forest scored higher on this dataset (AUC 0.933, recall 51%){RESET}")
    print(f" {YELLOW}vs SVM (AUC 0.874, recall 49%). Full comparison: reports/model_comparison_daily.csv{RESET}")


if __name__ == "__main__":
    run()
