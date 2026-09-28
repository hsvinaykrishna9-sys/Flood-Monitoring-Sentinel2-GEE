"""
Satellite + weather fusion feature set - extends the daily ERA5-Land
pipeline (feature_engineering_daily.py) with real Sentinel-2 NDVI/NDWI
signal, so the model that actually makes predictions uses both data
sources instead of just weather (the original daily model used zero
satellite data despite this being a "Sentinel2-GEE" project).

Leakage handling
-----------------
Monthly NDVI/NDWI (data/processed/kodagu_satellite_monthly_2018_2025.csv)
is an aggregate over an ENTIRE calendar month, so using the CURRENT
month's value on an early day of that month would leak information from
days later in the same month that haven't happened yet. Every satellite
feature here therefore uses the PREVIOUS calendar month's value, which
is fully known by any day in the current month - no leakage.

Coverage handling
------------------
Real Sentinel-2 observations only exist 2018-01 to 2025-12. For dates
outside that range (the 1950-2017 extension, and 2026), each day gets
a MONTHLY CLIMATOLOGY value instead - the average NDVI/NDWI for that
calendar month across all real observed years (2018-2025). This is
standard practice for filling a short remote-sensing record against a
longer weather record, and is clearly separated in the output
(`satellite_is_real` flag) so it's never presented as an actual
observation where it isn't.
"""

import numpy as np
import pandas as pd

from feature_engineering_daily import (
    load_raw, add_temporal_features, add_lag_and_rolling_features, build_label,
    FEATURE_COLUMNS as BASE_FEATURE_COLUMNS,
)

SATELLITE_PATH = "data/processed/kodagu_satellite_monthly_2018_2025.csv"
OUT_PATH = "data/processed/kodagu_daily_features_fused_1950_2026.csv"

SATELLITE_COLS = ["ndvi_mean", "ndwi_mean", "ndwi_max"]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [
    "ndvi_lag_month", "ndwi_mean_lag_month", "ndwi_max_lag_month",
]


def load_satellite_climatology():
    sat = pd.read_csv(SATELLITE_PATH)
    sat["month_num"] = sat["year_month"].str.slice(5, 7).astype(int)
    sat["year_num"] = sat["year_month"].str.slice(0, 4).astype(int)
    real_lookup = sat.set_index("year_month")[SATELLITE_COLS].to_dict("index")
    climatology = sat.groupby("month_num")[SATELLITE_COLS].mean().to_dict("index")
    return real_lookup, climatology


def satellite_features_for_month(year_month_str, real_lookup, climatology):
    if year_month_str in real_lookup:
        vals = real_lookup[year_month_str]
        return vals["ndvi_mean"], vals["ndwi_mean"], vals["ndwi_max"], True
    month_num = int(year_month_str[5:7])
    vals = climatology[month_num]
    return vals["ndvi_mean"], vals["ndwi_mean"], vals["ndwi_max"], False


def add_satellite_features(df):
    real_lookup, climatology = load_satellite_climatology()

    # previous calendar month for each row - the only month guaranteed
    # fully known (no leakage) as of any day in the current month
    prev_month = (df["date"].dt.to_period("M") - 1).astype(str)

    results = [satellite_features_for_month(ym, real_lookup, climatology) for ym in prev_month]
    df["ndvi_lag_month"] = [r[0] for r in results]
    df["ndwi_mean_lag_month"] = [r[1] for r in results]
    df["ndwi_max_lag_month"] = [r[2] for r in results]
    df["satellite_is_real"] = [r[3] for r in results]
    return df


def run():
    df = load_raw()
    df = add_temporal_features(df)
    df = add_lag_and_rolling_features(df)
    df = add_satellite_features(df)
    df, threshold = build_label(df)

    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["flood_risk"]).reset_index(drop=True)
    dropped = before - len(df)

    df.to_csv(OUT_PATH, index=False)

    real_pct = df["satellite_is_real"].mean()
    print(f"Runoff flood-risk threshold (85th pct, train-derived): {threshold:.2f} mm/day")
    print(f"Rows before feature engineering: {before}, dropped (lag warm-up): {dropped}")
    print(f"Rows after feature engineering: {len(df)}")
    print(f"Positive class (flood_risk=1) rate: {df['flood_risk'].mean():.2%}")
    print(f"Rows using REAL satellite observation (not climatology): {real_pct:.1%}")
    print(f"Saved: {OUT_PATH}")
    return df


if __name__ == "__main__":
    run()
