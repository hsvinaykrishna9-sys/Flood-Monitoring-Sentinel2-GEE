"""
Live day-ahead flood-risk prediction using the XGBoost model, on
real daily ERA5-Land data for Kodagu district (1950-2026, through the
most recently processed day).

Predicts flood risk for TOMORROW (the day after the most recent real
data), using only already-known information (no leakage) - a genuine
one-day-ahead forecast. For the rest of the outlook window, it first
tries to fetch a REAL rainfall/temperature/humidity/wind forecast from
Open-Meteo's live forecast API (the same API already used in
weather_collector.py). Where that succeeds, later days use real
forecasted weather instead of a guess. Where it's unavailable (no
network, or a day beyond forecast range), those days fall back to
recursively projecting forward from recent rolling averages - a labeled
simplification, not measured data, flagged as such in the output either
way. Soil moisture and runoff are always trend-projected for future
days, since Open-Meteo does not forecast those.

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
from xgboost import XGBClassifier

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


def fetch_live_forecast(latitude=12.4244, longitude=75.7382, past_days=0, forecast_days=10):
    """Fetch a real weather forecast (rainfall, temperature, humidity, wind)
    from Open-Meteo's live forecast API, keyed by ISO date string. Returns
    None on ANY failure (no network, bad response, unexpected schema) so
    callers can safely fall back to the trend-projection method - this must
    never crash the live demo."""
    try:
        import requests
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean",
            "timezone": "Asia/Kolkata",
            "past_days": max(0, min(past_days, 92)),
            "forecast_days": max(1, min(forecast_days, 16)),
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        daily = resp.json()["daily"]
        out = {}
        for i, date_str in enumerate(daily["time"]):
            out[date_str] = {
                "rainfall_mm": daily["precipitation_sum"][i],
                "temperature_C": daily["temperature_2m_mean"][i],
                "relative_humidity_pct": daily["relative_humidity_2m_mean"][i],
                "wind_speed_ms": daily["wind_speed_10m_mean"][i],
            }
        return out
    except Exception:
        return None


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


def detect_watch_periods(dated_probs, min_days=2):
    """Given a list of (date, prob) in chronological order, find runs of
    MODERATE-or-worse days (prob >= 0.33) at least min_days long - the same
    kind of "sustained multi-day risk" a weather app groups into a single
    Watch/Warning banner instead of listing day by day."""
    periods = []
    run_start = None
    for i, (date, prob) in enumerate(dated_probs):
        if prob >= 0.33:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_days:
                periods.append((run_start, i - 1))
            run_start = None
    if run_start is not None and len(dated_probs) - run_start >= min_days:
        periods.append((run_start, len(dated_probs) - 1))
    return periods


def print_watch_banner(dated_probs, periods, W):
    for s, e in periods:
        start_date, end_date = dated_probs[s][0], dated_probs[e][0]
        peak = max(p for _, p in dated_probs[s:e + 1])
        n_days = e - s + 1
        _, color = risk_band(peak)
        level = "WARNING" if peak >= 0.66 else "WATCH"
        print(f"{color}{BOLD}⚠ FLOOD {level}{RESET}{color}: elevated risk for {n_days} consecutive "
              f"days, {start_date.strftime('%b %d')}-{end_date.strftime('%b %d')} "
              f"(peak {peak:.0%} probability){RESET}")
    if periods:
        print("─" * W)
        print()


def run_historical_range_check(start_str, end_str):
    """Show day-by-day predictions across a real historical date range, plus
    a sustained-risk Watch/Warning banner when the run is long enough - the
    'we saw this coming days in advance' view, on real past data."""
    df = build_features()
    df["date"] = pd.to_datetime(df["date"])

    X = df[FEATURE_COLUMNS].values
    y = df["flood_risk"].values
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", random_state=42)
    xgb.fit(X, y)

    start_date, end_date = pd.Timestamp(start_str), pd.Timestamp(end_str)
    window = df[(df["date"] >= start_date) & (df["date"] <= end_date)].reset_index(drop=True)
    if window.empty:
        print(f"No data in range {start_date.date()} to {end_date.date()}. Try dates between "
              f"{df['date'].min().date()} and {df['date'].max().date()}.")
        return

    X_window = window[FEATURE_COLUMNS].values
    probs = xgb.predict_proba(X_window)[:, 1]
    dated_probs = list(zip(window["date"], probs))
    W = 70

    print("=" * W)
    print(f"HISTORICAL RANGE CHECK - {start_date.date()} to {end_date.date()}")
    print("=" * W)

    periods = detect_watch_periods(dated_probs)
    print_watch_banner(dated_probs, periods, W)

    n_correct = 0
    for i, (date, prob) in enumerate(dated_probs):
        band, color = risk_band(prob)
        actual = int(window.loc[i, "flood_risk"])
        pred = int(prob > 0.5)
        n_correct += (pred == actual)
        match_mark = "✓" if pred == actual else "✗"
        print(f" {date.strftime('%Y-%m-%d')} ({DAY_NAMES[date.weekday()]})  "
              f"{color}{band:<9}{RESET}{prob:>5.0%}   {risk_bar(prob)}  "
              f"actual: {'FLOOD' if actual else 'normal':<6} {match_mark}")
    print("-" * W)
    print(f"{DIM}Matched real outcome on {n_correct}/{len(dated_probs)} days in this window.")
    print(f"Note: these dates are inside training data (recalling a fitted pattern,")
    print(f"not out-of-sample generalization - see reports/model_comparison_daily.csv")
    print(f"for genuine held-out test-set accuracy, AUC 0.977 for XGBoost).{RESET}")
    print("=" * W)


def run_historical_check(target_date_str):
    """Show the model's prediction on a real historical date, alongside the
    real (ground-truth) outcome for that day. Demonstrates that the model
    genuinely varies its output with real conditions, using real data -
    not a live forecast."""
    df = build_features()
    df["date"] = pd.to_datetime(df["date"])

    X = df[FEATURE_COLUMNS].values
    y = df["flood_risk"].values
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", random_state=42)
    xgb.fit(X, y)

    target_date = pd.Timestamp(target_date_str)
    match = df[df["date"] == target_date]
    if match.empty:
        print(f"No data for {target_date.date()}. Try a date between "
              f"{df['date'].min().date()} and {df['date'].max().date()}.")
        return

    row = match.iloc[0]
    X_row = row[FEATURE_COLUMNS].values.reshape(1, -1)
    prob = xgb.predict_proba(X_row)[0, 1]
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
    print(f"on all {len(df):,} days), so this shows the model recalling a fitted pattern,")
    print(f"not out-of-sample generalization. For genuine held-out accuracy, see")
    print(f"reports/model_comparison_daily.csv (test-set AUC 0.977 for XGBoost).{RESET}")
    print("=" * 66)


def run():
    if len(sys.argv) > 2:
        run_historical_range_check(sys.argv[1], sys.argv[2])
        return
    if len(sys.argv) > 1:
        run_historical_check(sys.argv[1])
        return

    df = build_features()

    X = df[FEATURE_COLUMNS].values
    y = df["flood_risk"].values

    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", random_state=42)
    xgb.fit(X, y)

    # rolling "known history" buffer, starts as the real observed data
    raw_cols = ["date", "rainfall_mm", "runoff_mm", "soil_moisture_m3m3",
                "temperature_C", "relative_humidity_pct", "wind_speed_ms"]
    history = df[raw_cols].to_dict("records")
    history_live = [True] * len(history)  # all real observed data so far

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

    # try to fetch a real rainfall/temp/humidity/wind forecast to replace the
    # trend-projection for as many of these days as possible; None (no
    # network, API error) just means every day below falls back exactly as
    # before - never a crash, never a regression
    live_forecast = fetch_live_forecast(past_days=bridge_days, forecast_days=N_OUTLOOK_DAYS + 2)

    all_predictions = []
    input_live_flags = []  # was the lag-1 day feeding THIS prediction real/live data?
    for step in range(total_steps):
        row, next_date = build_next_day_row(history)
        input_live_flags.append(history_live[-1])
        X_next = pd.DataFrame([row])[FEATURE_COLUMNS].values
        prob = xgb.predict_proba(X_next)[0, 1]
        pred = int(prob > 0.5)
        all_predictions.append((next_date, prob, pred))

        # extend the "known history": use the real forecast for this day if
        # we have it, otherwise fall back to trend projection (persistence of
        # recent rolling averages) so the next iteration's lag features are
        # always defined. Soil moisture and runoff are always trend-projected -
        # Open-Meteo doesn't forecast either.
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

    # only the last N_OUTLOOK_DAYS (today onward) are actually displayed;
    # the bridge days were needed to advance the lag/rolling features but
    # aren't shown individually
    outlook = all_predictions[today_index:today_index + N_OUTLOOK_DAYS]
    outlook_live_flags = input_live_flags[today_index:today_index + N_OUTLOOK_DAYS]
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
    if live_forecast:
        n_live = sum(outlook_live_flags)
        print(f" {DIM}[live] = used a real Open-Meteo rainfall/weather forecast."
              f" [trend] = that day's forecast wasn't available, so recent")
        print(f" rolling averages were projected forward instead."
              f" ({n_live}/{N_OUTLOOK_DAYS} days used live data.){RESET}")
    else:
        print(f" {DIM}Live forecast unavailable (no network reached Open-Meteo from here) -")
        print(f" projecting forward from recent rainfall trends instead - not measured")
        print(f" data, confidence drops the further out you go, same as any weather forecast.{RESET}")
    print("─" * W)

    outlook_dated_probs = [(date, prob) for date, prob, _ in outlook]
    watch_periods = detect_watch_periods(outlook_dated_probs)
    print_watch_banner(outlook_dated_probs, watch_periods, W)

    for i, (date, prob, pred) in enumerate(outlook, start=0):
        band, color = risk_band(prob)
        label = "Today" if i == 0 else f"Day +{i}"
        day_str = f"{date.strftime('%b %d')} ({DAY_NAMES[date.weekday()]})"
        src_tag = f"{DIM}[live]{RESET} " if outlook_live_flags[i] else f"{DIM}[trend]{RESET}"
        print(f" {label:<10}{day_str:<14}{color}{band:<9}{RESET}{prob:>5.0%}   {risk_bar(prob)}  {src_tag}")
    print("─" * W)
    print()
    print(f" {DIM}Model: XGBoost (300 trees) | trained on {len(df):,} real daily records, 1950-2026{RESET}")
    print(f" {DIM}Validated out-of-sample AUC: 0.977, F1: 0.838 (best F1 of 8 models compared){RESET}")
    print(f" {DIM}Full comparison: reports/model_comparison_daily.csv{RESET}")


if __name__ == "__main__":
    run()
