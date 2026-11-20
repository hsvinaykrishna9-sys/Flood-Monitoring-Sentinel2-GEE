"""
Trains and compares day-ahead flood-risk classifiers on real daily
ERA5-Land data for Kodagu district (1950-2026).

Eight-model comparison: a runoff-persistence baseline, Random Forest,
XGBoost, SVM, Logistic Regression, thresholded Ridge Regression, a
soft-voting ensemble, and an LSTM sequence model.

Evaluation protocol - chronological 70 / 10 / 20 split:
  * train      (oldest 70%)  - every model is fit on this only
  * validation (next 10%)    - used to CHOOSE the best model (by F1)
  * test       (newest 20%)  - touched once, to report the chosen
                               model's final, unbiased performance
Choosing the winner on the test set itself (as an 80/20 split forces you
to) makes the reported test score slightly optimistic; the separate
validation set removes that selection bias. The split is chronological,
never shuffled, so no future weather leaks into training.

CAVEAT: the flood_risk label is a proxy (runoff > 85th percentile of the
training years), not a verified historical flood-event record - see
feature_engineering_daily.py for why.

Outputs:
  reports/model_comparison_daily.csv             (test-set metrics)
  reports/model_comparison_daily_validation.csv  (validation-set metrics)
  reports/model_comparison_daily_meta.json
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
    accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from feature_engineering_daily import FEATURE_COLUMNS, run as build_features

RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10  # the remaining 0.20 is the test set
SELECTION_METRIC = "F1"
SEQ_LEN = 7  # one week of lagged context for the LSTM
CV_FOLDS = 5
LSTM_EPOCHS = 25
# Models whose score is a calibrated probability, so cross-entropy (log loss)
# is meaningful. The baseline and Ridge output raw runoff in mm, not a probability.
PROBABILISTIC = {"Random Forest", "XGBoost", "SVM", "Logistic Regression",
                 "Ensemble (RF+XGB+SVM)", "LSTM"}


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
        "LogLoss": (log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7), labels=[0, 1])
                    if name in PROBABILISTIC else float("nan")),
    }


def tune_runoff_threshold(train):
    best_thresh, best_acc = None, -1
    for q in np.linspace(0.5, 0.98, 49):
        thresh = train["runoff_lag1"].quantile(q)
        pred = (train["runoff_lag1"] > thresh).astype(int)
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


def train_lstm(X_train_seq, y_train_seq, X_val_seq=None, y_val_seq=None):
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
    # validation_data only RECORDS validation loss each epoch; it never
    # changes the weights, so the fitted model is identical either way.
    validation = (X_val_seq, y_val_seq) if X_val_seq is not None else None
    history = model.fit(X_train_seq, y_train_seq, epochs=LSTM_EPOCHS, batch_size=64,
                        verbose=0, validation_data=validation)
    return model, history.history


class FittedModels:
    """All 8 models fit on one training frame; predict() scores any other
    frame (validation, test, or a cross-validation fold) the same way."""

    def __init__(self, train, val_for_curves=None):
        X_train, y_train = train[FEATURE_COLUMNS].values, train["flood_risk"].values
        self.scaler = StandardScaler().fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        self.thresh = tune_runoff_threshold(train)

        self.rf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                         random_state=RANDOM_STATE).fit(X_train, y_train)

        self.xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                 eval_metric="logloss", random_state=RANDOM_STATE)
        if val_for_curves is not None:
            # eval_set only records per-round train/validation log loss
            self.xgb.fit(X_train, y_train, verbose=False, eval_set=[
                (X_train, y_train),
                (val_for_curves[FEATURE_COLUMNS].values, val_for_curves["flood_risk"].values),
            ])
            self.xgb_curves = self.xgb.evals_result()
        else:
            self.xgb.fit(X_train, y_train)
            self.xgb_curves = None

        self.svm = SVC(kernel="rbf", probability=True,
                       random_state=RANDOM_STATE).fit(X_train_s, y_train)
        self.logreg = LogisticRegression(max_iter=1000,
                                         random_state=RANDOM_STATE).fit(X_train_s, y_train)
        # Ridge: genuine regression on continuous runoff_mm, thresholded afterwards
        self.ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X_train_s, train["runoff_mm"].values)

        X_train_seq, y_train_seq = make_sequences(X_train_s, y_train)
        X_val_seq = y_val_seq = None
        if val_for_curves is not None:
            X_val_s = self.scaler.transform(val_for_curves[FEATURE_COLUMNS].values)
            X_val_seq, y_val_seq = make_sequences(X_val_s, val_for_curves["flood_risk"].values)
        self.lstm, self.lstm_curves = train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq)

    def predict(self, frame):
        """name -> (y_true, y_pred, y_score), plus Ridge's continuous-runoff metrics."""
        X, y = frame[FEATURE_COLUMNS].values, frame["flood_risk"].values
        X_s = self.scaler.transform(X)
        out = {}

        base = frame["runoff_lag1"].values
        out["Baseline (Runoff Persistence)"] = (y, (base > self.thresh).astype(int), base)

        rf_score = self.rf.predict_proba(X)[:, 1]
        out["Random Forest"] = (y, (rf_score > 0.5).astype(int), rf_score)

        xgb_score = self.xgb.predict_proba(X)[:, 1]
        out["XGBoost"] = (y, (xgb_score > 0.5).astype(int), xgb_score)

        svm_score = self.svm.predict_proba(X_s)[:, 1]
        out["SVM"] = (y, (svm_score > 0.5).astype(int), svm_score)

        lr_score = self.logreg.predict_proba(X_s)[:, 1]
        out["Logistic Regression"] = (y, (lr_score > 0.5).astype(int), lr_score)

        ridge_runoff = self.ridge.predict(X_s)
        out["Ridge Regression (thresholded)"] = (y, (ridge_runoff > self.thresh).astype(int), ridge_runoff)
        y_runoff = frame["runoff_mm"].values
        ridge_metrics = {
            "mae_mm": float(mean_absolute_error(y_runoff, ridge_runoff)),
            "rmse_mm": float(np.sqrt(mean_squared_error(y_runoff, ridge_runoff))),
            "r2": float(r2_score(y_runoff, ridge_runoff)),
        }

        ens = (rf_score + xgb_score + svm_score) / 3
        out["Ensemble (RF+XGB+SVM)"] = (y, (ens > 0.5).astype(int), ens)

        X_seq, y_seq = make_sequences(X_s, y)
        lstm_score = self.lstm.predict(X_seq, verbose=0).ravel()
        out["LSTM"] = (y_seq, (lstm_score > 0.5).astype(int), lstm_score)
        return out, ridge_metrics


def to_table(predictions):
    rows = [evaluate(name, *vals) for name, vals in predictions.items()]
    return pd.DataFrame(rows).set_index("Model").round(3)


def cross_validate(dev):
    """Expanding-window (forward-chaining) cross-validation on all the data
    BEFORE the test set. Ordinary shuffled k-fold would train on future
    years and test on past ones, so each fold here trains only on days
    that come before the days it is scored on."""
    folds = []
    for i, (tr_idx, te_idx) in enumerate(TimeSeriesSplit(n_splits=CV_FOLDS).split(dev), start=1):
        tr, te = dev.iloc[tr_idx].reset_index(drop=True), dev.iloc[te_idx].reset_index(drop=True)
        print(f"  CV fold {i}/{CV_FOLDS}: train {tr['date'].min().date()}..{tr['date'].max().date()}"
              f"  ->  test {te['date'].min().date()}..{te['date'].max().date()}", flush=True)
        preds, _ = FittedModels(tr).predict(te)
        folds.append(to_table(preds))
    stacked = pd.concat(folds)
    grouped = stacked.groupby(level=0)
    summary = pd.DataFrame({
        f"{m}_{stat}": getattr(grouped[m], stat)()
        for m in ("F1", "AUC", "LogLoss") for stat in ("mean", "std")
    }).round(3)
    return summary.loc[folds[0].index]


def plot_loss_curves(fitted):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    xc = fitted.xgb_curves
    axes[0].plot(xc["validation_0"]["logloss"], label="training loss")
    axes[0].plot(xc["validation_1"]["logloss"], label="validation loss")
    axes[0].set_title("XGBoost - log loss per boosting round")
    axes[0].set_xlabel("boosting round")
    axes[0].set_ylabel("cross-entropy (log loss)")
    axes[0].legend()

    lc = fitted.lstm_curves
    axes[1].plot(lc["loss"], label="training loss")
    axes[1].plot(lc["val_loss"], label="validation loss")
    axes[1].set_title("LSTM - binary cross-entropy per epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("cross-entropy (log loss)")
    axes[1].legend()

    fig.suptitle("Training vs validation loss (chronological validation set)", fontsize=11)
    plt.tight_layout()
    plt.savefig("reports/validation_loss_curves_daily.png", dpi=150)
    plt.close()


def run():
    df = build_features()
    train, val, test = chronological_split(df)
    for name, part in (("train", train), ("validation", val), ("test", test)):
        print(f"{name:<10} {len(part):>6} rows  {part['date'].min().date()} to {part['date'].max().date()}")

    fitted = FittedModels(train, val_for_curves=val)
    print(f"Baseline runoff threshold (train-tuned): {fitted.thresh:.2f} mm/day")

    val_preds, val_ridge = fitted.predict(val)
    test_preds, test_ridge = fitted.predict(test)
    val_table, test_table = to_table(val_preds), to_table(test_preds)

    # Model selection happens on the VALIDATION set only.
    selected = val_table[SELECTION_METRIC].idxmax()

    os.makedirs("reports", exist_ok=True)
    val_table.to_csv("reports/model_comparison_daily_validation.csv")
    test_table.to_csv("reports/model_comparison_daily.csv")
    plot_loss_curves(fitted)

    print("\nVALIDATION set (used to pick the model):")
    print(val_table.to_string())
    print(f"\nSelected by validation {SELECTION_METRIC}: {selected} "
          f"({SELECTION_METRIC}={val_table.loc[selected, SELECTION_METRIC]:.3f})")
    print("\nTEST set (held out, reported once):")
    print(test_table.to_string())
    print(f"\nFinal test performance of {selected}: "
          f"F1={test_table.loc[selected, 'F1']:.3f}, AUC={test_table.loc[selected, 'AUC']:.3f}")
    print(f"\nRidge Regression - continuous runoff_mm on test: MAE {test_ridge['mae_mm']:.2f} mm, "
          f"RMSE {test_ridge['rmse_mm']:.2f} mm, R2 {test_ridge['r2']:.3f}")

    print(f"\n{CV_FOLDS}-fold expanding-window cross-validation on train+validation (pre-test data):")
    dev = pd.concat([train, val]).reset_index(drop=True)
    cv_table = cross_validate(dev)
    cv_table.to_csv("reports/cross_validation_daily.csv")
    print(cv_table.to_string())

    with open("reports/model_comparison_daily_meta.json", "w") as f:
        json.dump({
            "split": "chronological 70/10/20 (train/validation/test)",
            "n_total_rows": int(len(df)),
            "n_train": int(len(train)),
            "n_validation": int(len(val)),
            "n_test": int(len(test)),
            "train_dates": [str(train["date"].min().date()), str(train["date"].max().date())],
            "validation_dates": [str(val["date"].min().date()), str(val["date"].max().date())],
            "test_dates": [str(test["date"].min().date()), str(test["date"].max().date())],
            "positive_rate_validation": float(val["flood_risk"].mean()),
            "positive_rate_test": float(test["flood_risk"].mean()),
            "runoff_threshold_mm": float(fitted.thresh),
            "selection_metric": f"validation {SELECTION_METRIC}",
            "selected_model": selected,
            "selected_model_test_metrics": test_table.loc[selected].to_dict(),
            "cross_validation": {"scheme": f"expanding-window TimeSeriesSplit, {CV_FOLDS} folds",
                                 "summary_file": "reports/cross_validation_daily.csv"},
            "ridge_regression_continuous_runoff": {"validation": val_ridge, "test": test_ridge},
        }, f, indent=2)

    ax = test_table[["Accuracy", "Precision", "Recall", "F1", "AUC"]].plot(
        kind="bar", figsize=(10, 5.5), ylim=(0, 1), rot=20,
    )
    ax.set_title(f"Daily Model Comparison - held-out test set (newest 20%) | selected on validation: {selected}",
                 fontsize=10)
    ax.set_ylabel("Score")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random guess (0.5)")
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/model_comparison_daily.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6.5, 6))
    for name, (y_true, _, y_score) in test_preds.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={test_table.loc[name, 'AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Daily Flood-Risk Classifiers (test set)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("reports/roc_curves_daily.png", dpi=150)
    plt.close()

    print("\nSaved to reports/: model_comparison_daily.csv, model_comparison_daily_validation.csv, "
          "cross_validation_daily.csv, model_comparison_daily_meta.json, model_comparison_daily.png, "
          "roc_curves_daily.png, validation_loss_curves_daily.png")
    return val_table, test_table, selected


if __name__ == "__main__":
    run()
