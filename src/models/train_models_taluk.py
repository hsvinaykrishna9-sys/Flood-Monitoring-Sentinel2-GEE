"""
Trains the per-taluk (spatial) flood-risk model: one XGBoost model
shared across all 3 taluks, using real terrain features (elevation,
slope, TWI) that vary by location - so the model can learn that terrain
itself modulates flood risk, not just weather.

Chronological split: the same date cutoff is used across all 3 taluks
(not per-taluk row count), so no taluk's "future" leaks into another
taluk's "past" via a misaligned split.

Outputs:
  reports/model_comparison_taluk.csv          (overall test metrics)
  reports/model_comparison_taluk_by_taluk.csv (metrics broken out per taluk)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from feature_engineering_taluk import FEATURE_COLUMNS, TALUKS, run as build_features

RANDOM_STATE = 42


def chronological_split(df, train_frac=0.8):
    dates = sorted(df["date"].unique())
    cutoff = dates[int(len(dates) * train_frac)]
    train = df[df["date"] < cutoff].reset_index(drop=True)
    test = df[df["date"] >= cutoff].reset_index(drop=True)
    assert train["date"].max() < test["date"].min()
    return train, test, cutoff


def evaluate(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan"),
    }


def run():
    df = build_features()
    train, test, cutoff = chronological_split(df)
    print(f"Chronological split at {cutoff.date()}: {len(train)} train rows, {len(test)} test rows")

    X_train, y_train = train[FEATURE_COLUMNS].values, train["flood_risk"].values
    X_test, y_test = test[FEATURE_COLUMNS].values, test["flood_risk"].values

    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    score = xgb.predict_proba(X_test)[:, 1]
    pred = (score > 0.5).astype(int)

    overall = evaluate(y_test, pred, score)
    print("\nOverall (all 3 taluks pooled):")
    for k, v in overall.items():
        print(f"  {k}: {v:.3f}")

    os.makedirs("reports", exist_ok=True)
    pd.DataFrame([overall], index=["XGBoost (spatial, per-taluk)"]).round(3) \
        .to_csv("reports/model_comparison_taluk.csv")

    by_taluk = []
    for t in TALUKS:
        mask = test["taluk"].values == t
        m = evaluate(y_test[mask], pred[mask], score[mask])
        m["taluk"] = t
        m["n_test_days"] = int(mask.sum())
        by_taluk.append(m)
    by_taluk_df = pd.DataFrame(by_taluk).set_index("taluk").round(3)
    by_taluk_df.to_csv("reports/model_comparison_taluk_by_taluk.csv")
    print("\nPer-taluk breakdown:")
    print(by_taluk_df.to_string())

    imp = sorted(zip(FEATURE_COLUMNS, xgb.feature_importances_.tolist()), key=lambda x: -x[1])
    print("\nFeature importances (top 8):")
    for name, val in imp[:8]:
        print(f"  {name:30s} {val:.4f}")
    terrain_ranks = [(i + 1, n, v) for i, (n, v) in enumerate(imp) if n in ("elevation_m", "slope_degrees", "twi")]
    print("\nTerrain feature ranks (out of", len(imp), "):")
    for rank, name, val in terrain_ranks:
        print(f"  rank {rank}: {name} = {val:.4f}")

    return xgb, overall, by_taluk_df


if __name__ == "__main__":
    run()
