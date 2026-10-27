"""
Validation tests for the daily (1950-2026, whole-district) feature
pipeline and the model comparison it feeds - mirrors
test_data_processing.py / test_model_training.py, but for the daily
pipeline those only ever covered the monthly one.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "preprocessing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models"))

from feature_engineering_daily import FEATURE_COLUMNS, run as build_features

COMPARISON_PATH = "reports/model_comparison_daily.csv"
RAW_PATH = "data/raw/kodagu_daily_era5_1950_2026.csv"


def test_raw_daily_data_has_no_gaps():
    df = pd.read_csv(RAW_PATH)
    df["date"] = pd.to_datetime(df["date"])
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing = full_range.difference(df["date"])
    assert len(missing) == 0, f"expected 0 missing days in the daily record, found {len(missing)}"
    assert df["date"].duplicated().sum() == 0


def test_no_missing_values_in_daily_feature_columns():
    df = build_features()
    nulls = df[FEATURE_COLUMNS + ["flood_risk"]].isna().sum().sum()
    assert nulls == 0, f"expected 0 nulls in daily feature columns, found {nulls}"


def test_daily_flood_risk_label_is_binary():
    df = build_features()
    assert set(df["flood_risk"].unique()) <= {0, 1}


def test_daily_chronological_split_has_no_leakage():
    # chronological_split resets each frame's index (0..n-1), same as the
    # monthly version's split - so leakage is checked by date coverage,
    # not raw index values (those always "overlap" by construction)
    from train_models_daily import chronological_split
    df = build_features()
    train, test = chronological_split(df)
    assert train["date"].max() < test["date"].min()
    assert set(train["date"]).isdisjoint(set(test["date"]))


def test_all_daily_models_beat_random_guess_baseline():
    assert os.path.exists(COMPARISON_PATH), (
        f"{COMPARISON_PATH} not found - run `python src/models/train_models_daily.py` first"
    )
    comparison = pd.read_csv(COMPARISON_PATH, index_col="Model")
    below_random = comparison[comparison["AUC"] <= 0.5]
    assert below_random.empty, f"models at or below random-guess AUC (0.5): {list(below_random.index)}"


def test_chosen_model_meets_documented_performance():
    # regression guard: if XGBoost's reported numbers drop meaningfully
    # below what's documented in the README, something upstream changed
    # (new data, a code change) and the README/deck need re-checking.
    comparison = pd.read_csv(COMPARISON_PATH, index_col="Model")
    xgb_row = comparison.loc["XGBoost"]
    assert xgb_row["AUC"] >= 0.95, f"XGBoost AUC dropped to {xgb_row['AUC']} (README documents 0.977)"
    assert xgb_row["F1"] >= 0.75, f"XGBoost F1 dropped to {xgb_row['F1']} (README documents 0.838)"
