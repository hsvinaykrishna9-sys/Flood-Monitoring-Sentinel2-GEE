"""
Sanity checks for the live prediction / explainability / weather-app
layer - things that would silently break the demo without failing
loudly otherwise.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "preprocessing"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models"))


def test_shap_explanation_matches_prediction_direction():
    # for a known real flood day, the top SHAP contributor should push
    # risk UP (positive), not down - catches a sign/indexing bug in
    # explain_prediction silently
    from feature_engineering_daily import FEATURE_COLUMNS, run as build_features
    from predict_daily import explain_prediction
    from xgboost import XGBClassifier
    import pandas as pd

    df = build_features()
    df["date"] = pd.to_datetime(df["date"])
    X = df[FEATURE_COLUMNS].values
    y = df["flood_risk"].values
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", random_state=42)
    xgb.fit(X, y)

    row = df[df["date"] == pd.Timestamp("2023-07-05")].iloc[0]
    X_row = row[FEATURE_COLUMNS].values.reshape(1, -1)
    prob = xgb.predict_proba(X_row)[0, 1]
    assert prob > 0.9, "known real flood day should score HIGH risk"

    lines = explain_prediction(xgb, X_row)
    assert len(lines) == 5
    top_val = lines[0][1]
    assert top_val > 0, "top SHAP contributor for a confident HIGH-risk day should push risk UP"


def test_watch_period_detection():
    from predict_daily import detect_watch_periods
    import pandas as pd

    dates = pd.date_range("2023-07-01", periods=10)
    probs = [0.05, 0.03, 0.68, 0.91, 1.0, 1.0, 1.0, 1.0, 0.05, 0.02]
    periods = detect_watch_periods(list(zip(dates, probs)))
    assert periods == [(2, 7)], f"expected one period spanning indices 2-7, got {periods}"


def test_kodagu_boundary_includes_all_taluks_excludes_other_cities():
    from weather_app import is_in_kodagu

    assert is_in_kodagu(12.4244, 75.7382)  # Madikeri
    assert is_in_kodagu(12.1936, 75.8064)  # Virajpet
    assert is_in_kodagu(12.5985, 75.8517)  # Somwarpet
    assert not is_in_kodagu(12.9716, 77.5946)  # Bengaluru
    assert not is_in_kodagu(12.2958, 76.6394)  # Mysore
