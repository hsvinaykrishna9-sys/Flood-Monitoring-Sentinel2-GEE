"""
Per-taluk (sub-district) daily feature engineering - the spatial version
of feature_engineering_daily.py. Instead of one number for the whole
district, this builds a dataset covering 3 taluks (Madikeri, Virajpet,
Somwarpet) separately, each with its own weather time series AND real
terrain features (elevation, slope, TWI) that actually vary between
them - terrain is collected but never used as a model feature in the
whole-district pipeline, since a single district-wide average can't
vary.

Taluk boundaries are an approximation (Voronoi split around each
taluk's headquarters coordinates, done in the GEE extraction script),
not official administrative polygons - documented here so it's never
presented as more precise than it is.

Data sources:
  data/raw/kodagu_taluk_daily_era5_2018_2026.csv  (3 taluks x 3,185 days each)
  data/raw/kodagu_taluk_terrain.csv               (1 row per taluk)

Label definition
-----------------
Each taluk gets its OWN flood_risk threshold (85th percentile of ITS
OWN runoff distribution, train-period only) rather than one shared
district-wide threshold - "unusually wet for Madikeri" and "unusually
wet for Somwarpet" are different absolute values, since Madikeri is
structurally wetter. This is a deliberate modeling choice, not an
oversight, and is noted wherever results are reported.
"""

import numpy as np
import pandas as pd

WEATHER_PATH = "data/raw/kodagu_taluk_daily_era5_2018_2026.csv"
TERRAIN_PATH = "data/raw/kodagu_taluk_terrain.csv"
OUT_PATH = "data/processed/kodagu_taluk_daily_features_2018_2026.csv"

TALUKS = ["Madikeri", "Virajpet", "Somwarpet"]

FEATURE_COLUMNS = [
    "doy_sin", "doy_cos",
    "rainfall_lag1", "rainfall_lag2", "rainfall_lag3",
    "rainfall_roll3", "rainfall_roll7",
    "runoff_lag1", "runoff_roll3",
    "soil_moisture_lag1", "temperature_lag1", "humidity_lag1",
    "wind_speed_lag1", "rain_soil_interaction_lag1",
    "elevation_m", "slope_degrees", "twi",
]


def load_raw():
    weather = pd.read_csv(WEATHER_PATH)
    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.rename(columns={"precipitation_mm": "rainfall_mm"})
    terrain = pd.read_csv(TERRAIN_PATH)
    df = weather.merge(terrain, on="taluk", how="left")
    df = df.sort_values(["taluk", "date"]).reset_index(drop=True)
    return df


def add_temporal_features(df):
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def add_lag_and_rolling_features(df):
    # every lag/rolling feature is computed WITHIN each taluk's own time
    # series (groupby), never mixing one taluk's history into another's
    g = df.groupby("taluk", group_keys=False)
    df["rainfall_lag1"] = g["rainfall_mm"].shift(1)
    df["rainfall_lag2"] = g["rainfall_mm"].shift(2)
    df["rainfall_lag3"] = g["rainfall_mm"].shift(3)
    df["rainfall_roll3"] = g["rainfall_mm"].apply(lambda s: s.shift(1).rolling(3).mean())
    df["rainfall_roll7"] = g["rainfall_mm"].apply(lambda s: s.shift(1).rolling(7).mean())
    df["runoff_lag1"] = g["runoff_mm"].shift(1)
    df["runoff_roll3"] = g["runoff_mm"].apply(lambda s: s.shift(1).rolling(3).mean())
    df["soil_moisture_lag1"] = g["soil_moisture_m3m3"].shift(1)
    df["temperature_lag1"] = g["temperature_C"].shift(1)
    df["humidity_lag1"] = g["relative_humidity_pct"].shift(1)
    df["wind_speed_lag1"] = g["wind_speed_ms"].shift(1)
    df["rain_soil_interaction_lag1"] = df["rainfall_lag1"] * df["soil_moisture_lag1"]
    return df


def build_label(df, train_frac=0.8):
    thresholds = {}
    df["flood_risk"] = 0
    for taluk in TALUKS:
        mask = df["taluk"] == taluk
        sub = df.loc[mask].sort_values("date")
        n_train = int(len(sub) * train_frac)
        thresh = sub["runoff_mm"].iloc[:n_train].quantile(0.85)
        thresholds[taluk] = thresh
        df.loc[mask, "flood_risk"] = (df.loc[mask, "runoff_mm"] > thresh).astype(int)
    return df, thresholds


def run():
    df = load_raw()
    df = add_temporal_features(df)
    df = add_lag_and_rolling_features(df)
    df, thresholds = build_label(df)

    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["flood_risk"]).reset_index(drop=True)
    dropped = before - len(df)

    df.to_csv(OUT_PATH, index=False)

    print("Per-taluk runoff flood-risk thresholds (85th pct, train-derived):")
    for t, v in thresholds.items():
        print(f"  {t}: {v:.2f} mm/day")
    print(f"Rows before feature engineering: {before}, dropped (lag warm-up): {dropped}")
    print(f"Rows after feature engineering: {len(df)}")
    for t in TALUKS:
        rate = df.loc[df["taluk"] == t, "flood_risk"].mean()
        print(f"  {t} positive class rate: {rate:.2%}")
    print(f"Saved: {OUT_PATH}")
    return df


if __name__ == "__main__":
    run()
