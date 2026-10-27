"""
Validation tests for the spatial per-taluk pipeline and model.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "preprocessing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models"))

from feature_engineering_taluk import FEATURE_COLUMNS, TALUKS, run as build_features

COMPARISON_PATH = "reports/model_comparison_taluk.csv"
BY_TALUK_PATH = "reports/model_comparison_taluk_by_taluk.csv"
WEATHER_PATH = "data/raw/kodagu_taluk_daily_era5_2018_2026.csv"
TERRAIN_PATH = "data/raw/kodagu_taluk_terrain.csv"


def test_raw_taluk_weather_has_no_gaps_per_taluk():
    df = pd.read_csv(WEATHER_PATH)
    df["date"] = pd.to_datetime(df["date"])
    for taluk in TALUKS:
        sub = df[df["taluk"] == taluk]
        full_range = pd.date_range(sub["date"].min(), sub["date"].max(), freq="D")
        missing = full_range.difference(sub["date"])
        assert len(missing) == 0, f"{taluk}: expected 0 missing days, found {len(missing)}"


def test_terrain_values_differ_across_taluks():
    # the whole point of the spatial model is that terrain actually varies -
    # if these were ever identical across taluks, the Voronoi split or the
    # GEE extraction broke
    terrain = pd.read_csv(TERRAIN_PATH)
    assert terrain["taluk"].nunique() == len(TALUKS)
    assert terrain["elevation_m"].nunique() == len(TALUKS), "elevation should differ across all 3 taluks"


def test_no_missing_values_in_taluk_feature_columns():
    df = build_features()
    nulls = df[FEATURE_COLUMNS + ["flood_risk"]].isna().sum().sum()
    assert nulls == 0, f"expected 0 nulls in taluk feature columns, found {nulls}"


def test_lag_features_never_cross_taluk_boundaries():
    # a day's rainfall_lag1 must equal the SAME taluk's own previous-day raw
    # rainfall, never a different taluk's - this is the regression test for
    # the groupby("taluk") requirement in add_lag_and_rolling_features
    df = build_features()
    raw = pd.read_csv(WEATHER_PATH)
    raw["date"] = pd.to_datetime(raw["date"])

    for taluk in TALUKS:
        sub = df[df["taluk"] == taluk].sort_values("date").reset_index(drop=True)
        raw_sub = raw[raw["taluk"] == taluk].sort_values("date").set_index("date")["precipitation_mm"]

        prev_day = sub["date"] - pd.Timedelta(days=1)
        expected_lag1 = raw_sub.reindex(prev_day).values
        assert sub["date"].is_monotonic_increasing
        assert ((sub["rainfall_lag1"].values - expected_lag1) ** 2 < 1e-9).all(), \
            f"{taluk}: rainfall_lag1 doesn't match this taluk's own previous-day rainfall"


def test_all_taluk_models_beat_random_guess_baseline():
    assert os.path.exists(COMPARISON_PATH), (
        f"{COMPARISON_PATH} not found - run `python src/models/train_models_taluk.py` first"
    )
    comparison = pd.read_csv(COMPARISON_PATH, index_col=0)
    assert (comparison["AUC"] > 0.5).all()


def test_by_taluk_results_cover_all_three_taluks():
    by_taluk = pd.read_csv(BY_TALUK_PATH, index_col="taluk")
    assert set(by_taluk.index) == set(TALUKS)
    assert (by_taluk["AUC"] > 0.5).all()
