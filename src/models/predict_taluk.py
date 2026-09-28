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

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from feature_engineering_taluk import FEATURE_COLUMNS, TALUKS, run as build_features

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


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
