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
  * The split is chronological (not shuffled) to respect the time-series
    nature of the data and avoid look-ahead leakage.

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
TRAIN_FRAC = 0.8
SEQ_LEN = 3


def chronological_split(df, train_frac=TRAIN_FRAC):
    n_train = int(len(df) * train_frac)
    train_idx = df.index[:n_train]
    test_idx = df.index[n_train:]
    assert set(train_idx).isdisjoint(set(test_idx)), "train/test indices overlap"
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


def evaluate(name, y_true, y_pred, y_score):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan"),
    }


def rainfall_baseline(train, test):
    best_thresh, best_acc = None, -1
    for q in np.linspace(0.1, 0.9, 33):
        thresh = train["rainfall_lag1"].quantile(q)
        pred = (train["rainfall_lag1"] > thresh).astype(int)
        acc = accuracy_score(train["flood_risk"], pred)
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh

    y_pred = (test["rainfall_lag1"] > best_thresh).astype(int)
    y_score = test["rainfall_lag1"]  # continuous score for AUC ranking
    return best_thresh, y_pred.values, y_score.values


def make_sequences(feature_matrix, labels, seq_len=SEQ_LEN):
    X_seq, y_seq = [], []
    for i in range(seq_len - 1, len(feature_matrix)):
        X_seq.append(feature_matrix[i - seq_len + 1: i + 1])
        y_seq.append(labels[i])
    return np.array(X_seq), np.array(y_seq)


def train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq):
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

    y_score = model.predict(X_test_seq, verbose=0).ravel()
    y_pred = (y_score > 0.5).astype(int)
    return y_pred, y_score


def run():
    df = build_features()
    train, test = chronological_split(df)

    X_train = train[FEATURE_COLUMNS].values
    y_train = train["flood_risk"].values
    X_test = test[FEATURE_COLUMNS].values
    y_test = test["flood_risk"].values

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []
    roc_data = {}

    # --- Baseline: rainfall threshold ---
    thresh, base_pred, base_score = rainfall_baseline(train, test)
    results.append(evaluate("Baseline (Rainfall Threshold)", y_test, base_pred, base_score))
    fpr, tpr, _ = roc_curve(y_test, base_score)
    roc_data["Baseline (Rainfall Threshold)"] = (fpr, tpr)
    print(f"Baseline rainfall threshold (train-tuned): {thresh:.2f} mm")

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_score = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)
    results.append(evaluate("Random Forest", y_test, rf_pred, rf_score))
    roc_data["Random Forest"] = roc_curve(y_test, rf_score)[:2]

    # --- XGBoost ---
    xgb = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_STATE,
    )
    xgb.fit(X_train, y_train)
    xgb_score = xgb.predict_proba(X_test)[:, 1]
    xgb_pred = xgb.predict(X_test)
    results.append(evaluate("XGBoost", y_test, xgb_pred, xgb_score))
    roc_data["XGBoost"] = roc_curve(y_test, xgb_score)[:2]

    # --- SVM ---
    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train_s, y_train)
    svm_score = svm.predict_proba(X_test_s)[:, 1]
    svm_pred = svm.predict(X_test_s)
    results.append(evaluate("SVM", y_test, svm_pred, svm_score))
    roc_data["SVM"] = roc_curve(y_test, svm_score)[:2]

    # --- Ensemble (soft-voting: RF + XGBoost + SVM) ---
    ens_score = (rf_score + xgb_score + svm_score) / 3
    ens_pred = (ens_score > 0.5).astype(int)
    results.append(evaluate("Ensemble (RF+XGB+SVM)", y_test, ens_pred, ens_score))
    roc_data["Ensemble (RF+XGB+SVM)"] = roc_curve(y_test, ens_score)[:2]

    # --- LSTM (sequence model, built on the same chronological split) ---
    X_train_seq, y_train_seq = make_sequences(X_train_s, y_train)
    X_test_seq, y_test_seq = make_sequences(X_test_s, y_test)
    lstm_pred, lstm_score = train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq)
    results.append(evaluate("LSTM", y_test_seq, lstm_pred, lstm_score))
    roc_data["LSTM"] = roc_curve(y_test_seq, lstm_score)[:2]

    comparison = pd.DataFrame(results).set_index("Model").round(3)
    comparison.to_csv("reports/model_comparison.csv")
    print("\n" + comparison.to_string())

    best_model = comparison["AUC"].idxmax()
    print(f"\nBest model by AUC: {best_model} (AUC={comparison.loc[best_model, 'AUC']:.3f})")

    with open("reports/model_comparison_meta.json", "w") as f:
        json.dump({
            "n_total_rows": int(len(df)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_test_lstm": int(len(y_test_seq)),
            "positive_rate_test": float(np.mean(y_test)),
            "rainfall_threshold_mm": float(thresh),
            "best_model_by_auc": best_model,
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
