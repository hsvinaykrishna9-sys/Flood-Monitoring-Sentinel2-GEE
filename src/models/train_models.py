"""
Trains and compares flood-risk classifiers on the Kodagu monthly dataset.

IMPORTANT CAVEATS (read before presenting these numbers):
  * Only 93 monthly rows are available for a single district (Kodagu),
    after dropping rows lost to lag/rolling feature warm-up. This is a
    small-sample demo of the modeling pipeline, not a validated flood
    model — treat metrics as illustrative, not production-grade.
  * The target `flood_risk` is a proxy label derived from NDWI (see
    src/preprocessing/feature_engineering.py docstring), not a verified
    historical flood record.
  * The split is chronological (not shuffled) 70/10/20 train/validation/test
    to respect the time-series nature of the data and avoid look-ahead
    leakage. With only 93 rows the validation set is about 9 months, far too
    small to pick a model reliably - the selection is shown for protocol
    consistency with the daily pipeline, not as evidence one model is better.

Outputs:
  reports/model_comparison.csv
  reports/model_comparison.png
  reports/roc_curves.png
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from feature_engineering import FEATURE_COLUMNS, run as build_features

RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10  # the remaining 0.20 is the test set
SELECTION_METRIC = "F1"
SEQ_LEN = 3


def chronological_split(df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    n_train = int(len(df) * train_frac)
    n_val = int(len(df) * val_frac)
    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val:].reset_index(drop=True)
    return train, val, test


def evaluate(name, y_true, y_pred, y_score):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan"),
    }


def rainfall_baseline(train):
    best_thresh, best_acc = None, -1
    for q in np.linspace(0.1, 0.9, 33):
        thresh = train["rainfall_lag1"].quantile(q)
        pred = (train["rainfall_lag1"] > thresh).astype(int)
        acc = accuracy_score(train["flood_risk"], pred)
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh
    return best_thresh


def make_sequences(feature_matrix, labels, seq_len=SEQ_LEN):
    X_seq, y_seq = [], []
    for i in range(seq_len - 1, len(feature_matrix)):
        X_seq.append(feature_matrix[i - seq_len + 1: i + 1])
        y_seq.append(labels[i])
    return np.array(X_seq), np.array(y_seq)


def train_lstm(X_train_seq, y_train_seq):
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.utils.set_random_seed(RANDOM_STATE)

    model = keras.Sequential([
        layers.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        layers.LSTM(16, activation="tanh"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train_seq, y_train_seq, epochs=60, batch_size=8, verbose=0)
    return model


def run():
    df = build_features()
    train, val, test = chronological_split(df)
    for name, part in (("train", train), ("validation", val), ("test", test)):
        print(f"{name:<10} {len(part):>4} rows  {part['year_month'].min()} to {part['year_month'].max()}")

    X_train, y_train = train[FEATURE_COLUMNS].values, train["flood_risk"].values
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    thresh = rainfall_baseline(train)
    print(f"Baseline rainfall threshold (train-tuned): {thresh:.2f} mm")
    rf = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=RANDOM_STATE).fit(X_train, y_train)
    xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                        eval_metric="logloss", random_state=RANDOM_STATE).fit(X_train, y_train)
    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE).fit(X_train_s, y_train)
    X_train_seq, y_train_seq = make_sequences(X_train_s, y_train)
    lstm = train_lstm(X_train_seq, y_train_seq)

    def predict_all(frame):
        X, y = frame[FEATURE_COLUMNS].values, frame["flood_risk"].values
        X_s = scaler.transform(X)
        out = {}
        base = frame["rainfall_lag1"].values
        out["Baseline (Rainfall Threshold)"] = (y, (base > thresh).astype(int), base)
        rf_s = rf.predict_proba(X)[:, 1]
        out["Random Forest"] = (y, (rf_s > 0.5).astype(int), rf_s)
        xgb_s = xgb.predict_proba(X)[:, 1]
        out["XGBoost"] = (y, (xgb_s > 0.5).astype(int), xgb_s)
        svm_s = svm.predict_proba(X_s)[:, 1]
        out["SVM"] = (y, (svm_s > 0.5).astype(int), svm_s)
        ens = (rf_s + xgb_s + svm_s) / 3
        out["Ensemble (RF+XGB+SVM)"] = (y, (ens > 0.5).astype(int), ens)
        X_seq, y_seq = make_sequences(X_s, y)
        lstm_s = lstm.predict(X_seq, verbose=0).ravel()
        out["LSTM"] = (y_seq, (lstm_s > 0.5).astype(int), lstm_s)
        return out

    val_preds, test_preds = predict_all(val), predict_all(test)
    val_table = pd.DataFrame([evaluate(n, *v) for n, v in val_preds.items()]).set_index("Model").round(3)
    comparison = pd.DataFrame([evaluate(n, *v) for n, v in test_preds.items()]).set_index("Model").round(3)
    roc_data = {n: roc_curve(v[0], v[2])[:2] for n, v in test_preds.items()}

    os.makedirs("reports", exist_ok=True)
    val_table.to_csv("reports/model_comparison_validation.csv")
    comparison.to_csv("reports/model_comparison.csv")
    # With ~9 validation months and almost no flood months in them, the models
    # often tie (F1 = 0 for all). Naming a winner from a tie would be arbitrary.
    scores = val_table[SELECTION_METRIC]
    selected = scores.idxmax() if scores.max() > 0 and scores.nunique() > 1 else None

    print("\nVALIDATION set (used to pick the model):")
    print(val_table.to_string())
    if selected is None:
        print(f"\nValidation {SELECTION_METRIC} cannot separate the models (validation set too small); "
              "no model selected.")
    else:
        print(f"\nSelected by validation {SELECTION_METRIC}: {selected}")
    print("\nTEST set (held out, reported once):")
    print(comparison.to_string())

    best_model = comparison["AUC"].idxmax()
    with open("reports/model_comparison_meta.json", "w") as f:
        json.dump({
            "split": "chronological 70/10/20 (train/validation/test)",
            "n_total_rows": int(len(df)),
            "n_train": int(len(train)),
            "n_validation": int(len(val)),
            "n_test": int(len(test)),
            "positive_rate_test": float(test["flood_risk"].mean()),
            "rainfall_threshold_mm": float(thresh),
            "selection_metric": f"validation {SELECTION_METRIC}",
            "selected_model": selected if selected is not None else "inconclusive (validation set too small)",
            "best_model_by_test_auc": best_model,
            "caveat": "validation set is ~9 months; too small for reliable model selection",
        }, f, indent=2)

    # --- Plots ---
    ax = comparison[["Accuracy", "Precision", "Recall", "F1", "AUC"]].plot(
        kind="bar", figsize=(10, 5.5), ylim=(0, 1), rot=20,
    )
    ax.set_title("Model Comparison — Kodagu Flood-Risk Classification (test set)")
    ax.set_ylabel("Score")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random guess (0.5)")
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/model_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6.5, 6))
    for name, (fpr, tpr) in roc_data.items():
        auc = comparison.loc[name, "AUC"]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Flood-Risk Classifiers")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/roc_curves.png", dpi=150)
    plt.close()

    print("\nSaved: reports/model_comparison.csv, reports/model_comparison.png, reports/roc_curves.png")
    return comparison


if __name__ == "__main__":
    run()
