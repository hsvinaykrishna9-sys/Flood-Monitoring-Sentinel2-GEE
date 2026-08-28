"""
Validation tests for the satellite/weather ETL and feature-engineering output.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "preprocessing"))

from feature_engineering import FEATURE_COLUMNS, run as build_features

MERGED_PATH = "data/processed/kodagu_final_dataset_2018_2025.csv"


def test_merge_produces_full_monthly_coverage():
    # regression test for the year_month zero-padding bug that used to make
    # this merge produce 0 rows (see combine_satellite.py fix)
    df = pd.read_csv(MERGED_PATH)
    assert len(df) == 96, f"expected 96 monthly records (2018-2025), got {len(df)}"


def test_ndvi_ndwi_within_valid_range():
    df = pd.read_csv(MERGED_PATH)
    for col in ["ndvi_mean", "ndwi_mean", "ndwi_max"]:
        assert df[col].between(-1, 1).all(), f"{col} has values outside [-1, 1]"


def test_no_missing_values_in_feature_columns_after_cleaning():
    df = build_features()
    nulls = df[FEATURE_COLUMNS + ["flood_risk"]].isna().sum().sum()
    assert nulls == 0, f"expected 0 nulls in feature columns after cleaning, found {nulls}"


def test_flood_risk_label_is_binary():
    df = build_features()
    assert set(df["flood_risk"].unique()) <= {0, 1}
