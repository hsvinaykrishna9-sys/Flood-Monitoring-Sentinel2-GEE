"""
Sanity checks for the model-training pipeline.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "preprocessing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models"))

from feature_engineering import run as build_features
from train_models import chronological_split

COMPARISON_PATH = "reports/model_comparison.csv"


def test_train_test_split_has_no_leakage():
    df = build_features()
    train, test = chronological_split(df)

    # chronological_split resets each frame's index (0..n-1), so the split
    # itself is verified by month coverage, not raw index values: every
    # calendar month in train must come strictly before every month in test,
    # and no month should appear in both.
    assert train["year_month"].max() < test["year_month"].min()
    assert set(train["year_month"]).isdisjoint(set(test["year_month"]))


def test_all_models_beat_random_guess_baseline():
    assert os.path.exists(COMPARISON_PATH), (
        f"{COMPARISON_PATH} not found — run `python src/models/train_models.py` first"
    )
    comparison = pd.read_csv(COMPARISON_PATH, index_col="Model")
    below_random = comparison[comparison["AUC"] <= 0.5]
    assert below_random.empty, f"models at or below random-guess AUC (0.5): {list(below_random.index)}"
