"""
Trains and compares day-ahead flood-risk classifiers on real daily
ERA5-Land data for Kodagu district (2018-01-01 to 2026-08-26).

Same six-model comparison as train_models.py (monthly), applied to the
daily dataset: a runoff-persistence baseline, Random Forest, XGBoost,
SVM, an LSTM sequence model, and a soft-voting ensemble.

CAVEATS:
  * The flood_risk label is a proxy (runoff > 85th percentile), not a
    verified historical flood-event record — see
    feature_engineering_daily.py for why.
  * Chronological split (not shuffled) to avoid look-ahead leakage.

Outputs:
  reports/model_comparison_daily.csv
  reports/model_comparison_daily.png
  reports/roc_curves_daily.png
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from feature_engineering_daily import FEATURE_COLUMNS, run as build_features

RANDOM_STATE = 42
TRAIN_FRAC = 0.8
SEQ_LEN = 7  # one week of lagged context for the LSTM


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


def runoff_baseline(train, test):
    best_thresh, best_acc = None, -1
    for q in np.linspace(0.5, 0.98, 49):
        thresh = train["runoff_lag1"].quantile(q)
        pred = (train["runoff_lag1"] > thresh).astype(int)
        acc = accuracy_score(train["flood_risk"], pred)
        if acc > best_acc:
            best_acc, best_thresh = acc, thresh

    y_pred = (test["runoff_lag1"] > best_thresh).astype(int)
    y_score = test["runoff_lag1"]
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
        layers.LSTM(24, activation="tanh"),
        layers.Dense(12, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train_seq, y_train_seq, epochs=25, batch_size=64, verbose=0)

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

    # --- Baseline: yesterday's runoff threshold ---
    thresh, base_pred, base_score = runoff_baseline(train, test)
    results.append(evaluate("Baseline (Runoff Persistence)", y_test, base_pred, base_score))
    roc_data["Baseline (Runoff Persistence)"] = roc_curve(y_test, base_score)[:2]
    print(f"Baseline runoff threshold (train-tuned): {thresh:.2f} mm/day")

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_score = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)
    results.append(evaluate("Random Forest", y_test, rf_pred, rf_score))
    roc_data["Random Forest"] = roc_curve(y_test, rf_score)[:2]

    # --- XGBoost ---
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
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

    # --- Logistic Regression (classic linear classification baseline) ---
    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg.fit(X_train_s, y_train)
    logreg_score = logreg.predict_proba(X_test_s)[:, 1]
    logreg_pred = logreg.predict(X_test_s)
    results.append(evaluate("Logistic Regression", y_test, logreg_pred, logreg_score))
    roc_data["Logistic Regression"] = roc_curve(y_test, logreg_score)[:2]

    # --- Ridge Regression: genuine regression on continuous runoff_mm, ---
    # --- then thresholded at the same cutoff used to define flood_risk ---
    y_train_runoff = train["runoff_mm"].values
    y_test_runoff = test["runoff_mm"].values
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X_train_s, y_train_runoff)
    ridge_pred_runoff = ridge.predict(X_test_s)

    reg_mae = mean_absolute_error(y_test_runoff, ridge_pred_runoff)
    reg_rmse = np.sqrt(mean_squared_error(y_test_runoff, ridge_pred_runoff))
    reg_r2 = r2_score(y_test_runoff, ridge_pred_runoff)
    print(f"\nRidge Regression - continuous runoff_mm prediction (not a 0/1 classifier):")
    print(f"  MAE: {reg_mae:.2f} mm   RMSE: {reg_rmse:.2f} mm   R2: {reg_r2:.3f}")

    ridge_class_pred = (ridge_pred_runoff > thresh).astype(int)
    results.append(evaluate("Ridge Regression (thresholded)", y_test, ridge_class_pred, ridge_pred_runoff))
    roc_data["Ridge Regression (thresholded)"] = roc_curve(y_test, ridge_pred_runoff)[:2]

    # --- Ensemble ---
    ens_score = (rf_score + xgb_score + svm_score) / 3
    ens_pred = (ens_score > 0.5).astype(int)
    results.append(evaluate("Ensemble (RF+XGB+SVM)", y_test, ens_pred, ens_score))
    roc_data["Ensemble (RF+XGB+SVM)"] = roc_curve(y_test, ens_score)[:2]

    # --- LSTM ---
    X_train_seq, y_train_seq = make_sequences(X_train_s, y_train)
    X_test_seq, y_test_seq = make_sequences(X_test_s, y_test)
    lstm_pred, lstm_score = train_lstm(X_train_seq, y_train_seq, X_test_seq, y_test_seq)
    results.append(evaluate("LSTM", y_test_seq, lstm_pred, lstm_score))
    roc_data["LSTM"] = roc_curve(y_test_seq, lstm_score)[:2]

    comparison = pd.DataFrame(results).set_index("Model").round(3)
    os.makedirs("reports", exist_ok=True)
    comparison.to_csv("reports/model_comparison_daily.csv")
    print("\n" + comparison.to_string())

    best_model = comparison["AUC"].idxmax()
    print(f"\nBest model by AUC: {best_model} (AUC={comparison.loc[best_model, 'AUC']:.3f})")

    with open("reports/model_comparison_daily_meta.json", "w") as f:
        json.dump({
            "n_total_rows": int(len(df)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_test_lstm": int(len(y_test_seq)),
            "positive_rate_test": float(np.mean(y_test)),
            "runoff_threshold_mm": float(thresh),
            "best_model_by_auc": best_model,
            "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
            "ridge_regression_continuous_runoff": {
                "mae_mm": float(reg_mae), "rmse_mm": float(reg_rmse), "r2": float(reg_r2),
            },
        }, f, indent=2)

    ax = comparison[["Accuracy", "Precision", "Recall", "F1", "AUC"]].plot(
        kind="bar", figsize=(10, 5.5), ylim=(0, 1), rot=20,
    )
    ax.set_title("Daily Model Comparison - Kodagu Flood-Risk Classification (test set)")
    ax.set_ylabel("Score")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random guess (0.5)")
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/model_comparison_daily.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6.5, 6))
    for name, (fpr, tpr) in roc_data.items():
        auc = comparison.loc[name, "AUC"]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Daily Flood-Risk Classifiers")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/roc_curves_daily.png", dpi=150)
    plt.close()

    print("\nSaved: reports/model_comparison_daily.csv, model_comparison_daily.png, roc_curves_daily.png")
    return comparison


if __name__ == "__main__":
    run()
