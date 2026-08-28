"""
Feature engineering for the Kodagu flood-risk dataset.

Label definition
-----------------
The repo does not yet contain a verified historical flood-event dataset
(README lists "Historical flood records from Karnataka SDMA" as a future
source). Until that is available, flood risk is approximated from the
Sentinel-2 water index itself: a month is labelled high-water (flood_risk=1)
when its NDWI mean exceeds the upper-tercile NDWI value observed in the
training period. NDWI is a standard remote-sensing proxy for surface-water
extent, so this is a reasonable stand-in, but it is a proxy label, not a
verified ground-truth flood record. Replace `build_label()` once real flood
event data is added.

Forecasting setup (no same-month leakage)
------------------------------------------
The project's stated goal is predicting flood risk *in advance* (7-30 days /
roughly a month), not detecting it as it happens. So every feature used to
predict month t's flood_risk is built only from information available
through month t-1: lagged NDWI/NDVI/rainfall/temperature/humidity, a
3-month rolling rainfall average ending at t-1, and the calendar month
(known in advance). No same-month rainfall, humidity, NDVI or NDWI is used
as a feature — an earlier version of this script did include same-month
rainfall/humidity, which let the model detect "is this the monsoon" from
another monsoon signal and produced trivially perfect (and meaningless)
scores. This version is intentionally harder and reflects an actual
forecasting task.
"""

import numpy as np
import pandas as pd

IN_PATH = "data/processed/kodagu_final_dataset_2018_2025.csv"
OUT_PATH = "data/processed/kodagu_features_2018_2025.csv"

FEATURE_COLUMNS = [
    "month_sin",
    "month_cos",
    "ndvi_mean_lag1",
    "ndwi_mean_lag1",
    "ndwi_mean_lag2",
    "rainfall_lag1",
    "rainfall_lag2",
    "rainfall_roll3",
    "temperature_lag1",
    "humidity_lag1",
    "rain_humidity_interaction_lag1",
]


def load_raw(path=IN_PATH):
    df = pd.read_csv(path)
    df["month"] = df["month"].astype(int)
    df["year"] = df["year"].astype(int)
    df = df.rename(columns={"rainfall_mm_y": "rainfall_mm"})
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    return df


def add_temporal_features(df):
    # calendar month is knowable in advance, so it's a legitimate forecast feature
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_and_rolling_features(df):
    df["ndvi_mean_lag1"] = df["ndvi_mean"].shift(1)
    df["ndwi_mean_lag1"] = df["ndwi_mean"].shift(1)
    df["ndwi_mean_lag2"] = df["ndwi_mean"].shift(2)
    df["rainfall_lag1"] = df["rainfall_mm"].shift(1)
    df["rainfall_lag2"] = df["rainfall_mm"].shift(2)
    # rolling mean of the 3 months strictly before the current one (no leakage)
    df["rainfall_roll3"] = df["rainfall_mm"].shift(1).rolling(window=3).mean()
    df["temperature_lag1"] = df["temperature_c"].shift(1)
    df["humidity_lag1"] = df["humidity_percent"].shift(1)
    df["rain_humidity_interaction_lag1"] = df["rainfall_lag1"] * df["humidity_lag1"]
    return df


def build_label(df, train_frac=0.8):
    n_train = int(len(df) * train_frac)
    threshold = df["ndwi_mean"].iloc[:n_train].quantile(2 / 3)
    df["flood_risk"] = (df["ndwi_mean"] > threshold).astype(int)
    return df, threshold


def run():
    df = load_raw()
    df = add_temporal_features(df)
    df = add_lag_and_rolling_features(df)
    df, threshold = build_label(df)

    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["flood_risk"]).reset_index(drop=True)
    dropped = before - len(df)

    df.to_csv(OUT_PATH, index=False)

    print(f"NDWI flood-risk threshold (train-derived): {threshold:.4f}")
    print(f"Rows before feature engineering: {before}, dropped (lag warm-up): {dropped}")
    print(f"Rows after feature engineering: {len(df)}")
    print(f"Positive class (flood_risk=1) rate: {df['flood_risk'].mean():.2%}")
    print(f"Saved: {OUT_PATH}")
    return df


if __name__ == "__main__":
    run()
