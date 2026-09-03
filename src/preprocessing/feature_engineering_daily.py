"""
Daily-resolution feature engineering for real-time (day-ahead) flood risk.

Data source: ECMWF ERA5-Land daily aggregates for the whole Kodagu district,
pulled live via Google Earth Engine (data/raw/kodagu_daily_era5_2018_2026.csv),
2018-01-01 through the most recent day ERA5-Land had processed at export time.

Label definition
-----------------
No NDWI/satellite water index exists at daily resolution, so the label here
is a physical proxy instead: a day is flagged flood_risk=1 when its surface
RUNOFF (water that isn't absorbed by soil - the direct physical precursor to
surface flooding) exceeds the 85th-percentile runoff value observed in the
training period. This is a different, arguably more direct proxy than the
NDWI-tercile label used in the monthly pipeline, and is documented here for
the same reason: it's a proxy, not a verified historical flood-event record.

Forecasting setup (no same-day leakage)
-----------------------------------------
Every feature used to predict day t's flood_risk comes only from day t-1
and earlier (lagged rainfall/runoff/soil-moisture/wind/temperature/humidity,
plus day-of-year seasonality, which is known in advance). This is a genuine
one-day-ahead forecast, not same-day detection.
"""

import numpy as np
import pandas as pd

IN_PATH = "data/raw/kodagu_daily_era5_2018_2026.csv"
OUT_PATH = "data/processed/kodagu_daily_features_2018_2026.csv"

FEATURE_COLUMNS = [
    "doy_sin",
    "doy_cos",
    "rainfall_lag1",
    "rainfall_lag2",
    "rainfall_lag3",
    "rainfall_roll3",
    "rainfall_roll7",
    "runoff_lag1",
    "runoff_roll3",
    "soil_moisture_lag1",
    "temperature_lag1",
    "humidity_lag1",
    "wind_speed_lag1",
    "rain_soil_interaction_lag1",
]


def load_raw(path=IN_PATH):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"precipitation_mm": "rainfall_mm"})
    return df


def add_temporal_features(df):
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def add_lag_and_rolling_features(df):
    df["rainfall_lag1"] = df["rainfall_mm"].shift(1)
    df["rainfall_lag2"] = df["rainfall_mm"].shift(2)
    df["rainfall_lag3"] = df["rainfall_mm"].shift(3)
    df["rainfall_roll3"] = df["rainfall_mm"].shift(1).rolling(window=3).mean()
    df["rainfall_roll7"] = df["rainfall_mm"].shift(1).rolling(window=7).mean()
    df["runoff_lag1"] = df["runoff_mm"].shift(1)
    df["runoff_roll3"] = df["runoff_mm"].shift(1).rolling(window=3).mean()
    df["soil_moisture_lag1"] = df["soil_moisture_m3m3"].shift(1)
    df["temperature_lag1"] = df["temperature_C"].shift(1)
    df["humidity_lag1"] = df["relative_humidity_pct"].shift(1)
    df["wind_speed_lag1"] = df["wind_speed_ms"].shift(1)
    df["rain_soil_interaction_lag1"] = df["rainfall_lag1"] * df["soil_moisture_lag1"]
    return df


def build_label(df, train_frac=0.8):
    n_train = int(len(df) * train_frac)
    threshold = df["runoff_mm"].iloc[:n_train].quantile(0.85)
    df["flood_risk"] = (df["runoff_mm"] > threshold).astype(int)
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

    print(f"Runoff flood-risk threshold (85th pct, train-derived): {threshold:.2f} mm/day")
    print(f"Rows before feature engineering: {before}, dropped (lag warm-up): {dropped}")
    print(f"Rows after feature engineering: {len(df)}")
    print(f"Positive class (flood_risk=1) rate: {df['flood_risk'].mean():.2%}")
    print(f"Saved: {OUT_PATH}")
    return df


if __name__ == "__main__":
    run()
