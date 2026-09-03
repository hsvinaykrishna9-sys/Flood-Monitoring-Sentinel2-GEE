"""
Live flood-risk prediction using the SVM model, for demo purposes.

Predicts flood risk for the month AFTER the most recent month in the
dataset, using only information already on hand (no future/unknown
inputs) — this is a genuine one-month-ahead forecast, consistent with
the "no same-month leakage" design in feature_engineering.py.

NOTE ON LEAD TIME: the satellite-derived NDWI/NDVI features (the core
water-index predictors) were only collected at monthly resolution, even
though the raw weather data is daily. So this model predicts month-ahead
risk, not day-level risk. Extending to weekly/daily NDWI composites is
future work, not a redesign.

Usage:
    python src/models/predict.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_engineering import FEATURE_COLUMNS, run as build_features

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def build_next_month_features(df):
    """Construct the feature row for the month right after the dataset's
    last row, using the same lag/rolling definitions as feature_engineering.py."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    last3_rainfall = df["rainfall_mm"].iloc[-3:].mean()

    next_month = int(last["month"]) + 1
    next_year = int(last["year"])
    if next_month > 12:
        next_month = 1
        next_year += 1

    row = {
        "month_sin": np.sin(2 * np.pi * next_month / 12),
        "month_cos": np.cos(2 * np.pi * next_month / 12),
        "ndvi_mean_lag1": last["ndvi_mean"],
        "ndwi_mean_lag1": last["ndwi_mean"],
        "ndwi_mean_lag2": prev["ndwi_mean"],
        "rainfall_lag1": last["rainfall_mm"],
        "rainfall_lag2": prev["rainfall_mm"],
        "rainfall_roll3": last3_rainfall,
        "temperature_lag1": last["temperature_c"],
        "humidity_lag1": last["humidity_percent"],
        "rain_humidity_interaction_lag1": last["rainfall_mm"] * last["humidity_percent"],
    }
    return row, next_month, next_year


def run():
    df = build_features()

    # Final deployed model: retrain on ALL historical data (not just the
    # train split) once the methodology is already validated via the
    # held-out test set in train_models.py (see model_comparison.csv).
    X = df[FEATURE_COLUMNS].values
    y = df["flood_risk"].values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_scaled, y)

    next_row, next_month, next_year = build_next_month_features(df)
    X_next = pd.DataFrame([next_row])[FEATURE_COLUMNS].values
    X_next_scaled = scaler.transform(X_next)

    prob = svm.predict_proba(X_next_scaled)[0, 1]
    pred = int(prob > 0.5)

    last = df.iloc[-1]
    month_label = f"{MONTH_NAMES[next_month - 1]} {next_year}"

    print("=" * 60)
    print("FloodPredict - Kodagu District - SVM Flood-Risk Forecast")
    print("=" * 60)
    print(f"Latest available data: {MONTH_NAMES[int(last['month']) - 1]} {int(last['year'])}")
    print(f"  rainfall: {last['rainfall_mm']:.1f} mm   temperature: {last['temperature_c']:.1f} C"
          f"   humidity: {last['humidity_percent']:.1f}%   NDWI: {last['ndwi_mean']:.3f}")
    print("-" * 60)
    print(f"Forecast target month: {month_label}")
    print(f"Predicted flood-risk probability: {prob:.1%}")
    print(f"Predicted risk level: {'HIGH' if pred else 'LOW'}")
    print("=" * 60)
    print("Model: SVM (RBF kernel), trained on all 93 historical monthly")
    print("records (2018-2025). Validated out-of-sample AUC: 0.977")
    print("(see reports/model_comparison.csv for the full comparison).")
    print("=" * 60)


if __name__ == "__main__":
    run()
