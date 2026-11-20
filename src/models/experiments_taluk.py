"""
Per-taluk (spatial) model under the same chronological 70/10/20
train/validation/test protocol used for the daily model, plus:

  * SMOTE (training data only)
  * a decision threshold tuned on the validation set
  * a terrain ablation - the same model WITHOUT elevation/slope/TWI - to
    test directly whether terrain actually helps, rather than only citing
    feature importance

Splits by calendar date (all three taluks share the same dates), so no
taluk's future leaks into another taluk's training period. Each taluk's
flood threshold is the 85th percentile of ITS OWN training-period runoff.

  python src/models/experiments_taluk.py
  -> reports/experiment_taluk.csv
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier

from feature_engineering_taluk import (
    FEATURE_COLUMNS, TALUKS, add_lag_and_rolling_features, add_temporal_features,
    build_label, load_raw,
)

RANDOM_STATE = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10
TERRAIN = ["elevation_m", "slope_degrees", "twi"]


def make_frame():
    df = add_lag_and_rolling_features(add_temporal_features(load_raw()))
    df, thresholds = build_label(df, train_frac=TRAIN_FRAC)
    df = df.dropna(subset=FEATURE_COLUMNS + ["flood_risk"]).reset_index(drop=True)
    return df, thresholds


def date_split(df):
    dates = sorted(df["date"].unique())
    n = len(dates)
    cut1 = dates[int(n * TRAIN_FRAC)]
    cut2 = dates[int(n * (TRAIN_FRAC + VAL_FRAC))]
    train = df[df["date"] < cut1].reset_index(drop=True)
    val = df[(df["date"] >= cut1) & (df["date"] < cut2)].reset_index(drop=True)
    test = df[df["date"] >= cut2].reset_index(drop=True)
    assert train["date"].max() < val["date"].min() and val["date"].max() < test["date"].min()
    return train, val, test, cut1, cut2


def metrics(y, pred, score):
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "AUC": roc_auc_score(y, score),
    }


def fit_xgb(X, y):
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", random_state=RANDOM_STATE).fit(X, y)


def tune_threshold(y_val, score_val):
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[int(np.argmax([f1_score(y_val, (score_val > t).astype(int), zero_division=0) for t in grid]))])


def run():
    df, thresholds = make_frame()
    train, val, test, cut1, cut2 = date_split(df)
    print(f"train < {cut1.date()} ({len(train)} rows) | validation < {cut2.date()} ({len(val)}) | test ({len(test)})")
    print("per-taluk runoff thresholds (85th pct of training years):",
          {k: round(float(v), 2) for k, v in thresholds.items()})

    arms = {}
    for label, feats in (("with terrain", FEATURE_COLUMNS),
                         ("no terrain (ablation)", [c for c in FEATURE_COLUMNS if c not in TERRAIN])):
        Xtr, ytr = train[feats].values, train["flood_risk"].values
        model = fit_xgb(Xtr, ytr)
        s_val, s_test = model.predict_proba(val[feats].values)[:, 1], model.predict_proba(test[feats].values)[:, 1]
        arms[f"default | {label}"] = (s_val, s_test, 0.5)

        if label == "with terrain":
            Xr, yr = SMOTE(random_state=RANDOM_STATE).fit_resample(Xtr, ytr)
            sm = fit_xgb(Xr, yr)
            arms["SMOTE | with terrain"] = (sm.predict_proba(val[feats].values)[:, 1],
                                            sm.predict_proba(test[feats].values)[:, 1], 0.5)
            t = tune_threshold(val["flood_risk"].values, s_val)
            arms["tuned threshold | with terrain"] = (s_val, s_test, t)

    y_val, y_test = val["flood_risk"].values, test["flood_risk"].values
    rows = []
    for arm, (s_val, s_test, t) in arms.items():
        overall = metrics(y_test, (s_test > t).astype(int), s_test)
        rows.append({"arm": arm, "scope": "all taluks", "decision_threshold": round(t, 2),
                     "val_F1": round(f1_score(y_val, (s_val > t).astype(int)), 3),
                     **{k: round(v, 3) for k, v in overall.items()}})
        for taluk in TALUKS:
            m = (test["taluk"] == taluk).values
            r = metrics(y_test[m], (s_test[m] > t).astype(int), s_test[m])
            rows.append({"arm": arm, "scope": taluk, "decision_threshold": round(t, 2), "val_F1": np.nan,
                         **{k: round(v, 3) for k, v in r.items()}})
    out = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    out.to_csv("reports/experiment_taluk.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print("\nSaved reports/experiment_taluk.csv")


if __name__ == "__main__":
    run()
