"""
Robustness experiments for the daily flood-risk model comparison, all under
the same chronological 70/10/20 train/validation/test protocol as
train_models_daily.py.

Two modes (run them separately; each takes a while because it re-fits all
eight models):

  python src/models/experiments_daily.py threshold
      How sensitive are the results to the flood-label percentile? The label
      is "runoff above the Nth percentile of the TRAINING years"; N=85 is a
      modelling choice, so this sweeps 75/80/85/90/95.
      -> reports/experiment_label_percentile.csv

  python src/models/experiments_daily.py arms
      Three ways of handling the ~15% class imbalance, on the standard label:
        default      - no resampling, decision threshold 0.5
        smote        - SMOTE on the TRAINING data only, threshold 0.5
        tuned        - no resampling; decision threshold tuned on the
                       VALIDATION set to maximise F1
      plus block-bootstrap 95% confidence intervals on the test set for the
      default arm, to show which differences between models are real.
      -> reports/experiment_imbalance_arms.csv
         reports/experiment_bootstrap_ci.csv

Caveat on the threshold sweep: changing the percentile changes the task
(different positive rate), so F1 is not comparable across percentiles. Read
AUC across percentiles, and compare models WITHIN a percentile.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "preprocessing"))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from feature_engineering_daily import (
    FEATURE_COLUMNS, add_lag_and_rolling_features, add_temporal_features,
    build_label, load_raw,
)
from train_models_daily import FittedModels, SEQ_LEN, chronological_split, evaluate, to_table

PERCENTILES = [0.75, 0.80, 0.85, 0.90, 0.95]
PROBABILISTIC_MODELS = ["Random Forest", "XGBoost", "SVM", "Logistic Regression",
                        "Ensemble (RF+XGB+SVM)", "LSTM"]
BOOTSTRAP_BLOCK_DAYS = 30
BOOTSTRAP_RESAMPLES = 500
SEED = 42


def make_frame(quantile):
    df = add_lag_and_rolling_features(add_temporal_features(load_raw()))
    df, threshold = build_label(df, quantile=quantile)
    df = df.dropna(subset=FEATURE_COLUMNS + ["flood_risk"]).reset_index(drop=True)
    return df, threshold


def run_threshold_sweep():
    rows = []
    for q in PERCENTILES:
        df, thr = make_frame(q)
        train, val, test = chronological_split(df)
        print(f"\n=== label percentile {q:.2f}: runoff > {thr:.2f} mm/day, "
              f"positive rate train {train['flood_risk'].mean():.1%} / test {test['flood_risk'].mean():.1%}", flush=True)
        fitted = FittedModels(train)
        val_t = to_table(fitted.predict(val)[0])
        test_t = to_table(fitted.predict(test)[0])
        for model in test_t.index:
            rows.append({
                "percentile": q, "runoff_threshold_mm": round(float(thr), 2),
                "positive_rate_test": round(float(test["flood_risk"].mean()), 3),
                "Model": model,
                "val_F1": val_t.loc[model, "F1"], "test_F1": test_t.loc[model, "F1"],
                "test_AUC": test_t.loc[model, "AUC"],
                "test_Precision": test_t.loc[model, "Precision"], "test_Recall": test_t.loc[model, "Recall"],
            })
        print(test_t[["Precision", "Recall", "F1", "AUC"]].to_string(), flush=True)
    out = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    out.to_csv("reports/experiment_label_percentile.csv", index=False)
    print("\nTest AUC by percentile:")
    print(out.pivot(index="Model", columns="percentile", values="test_AUC").to_string())
    print("\nTest F1 by percentile (not comparable across columns - different task):")
    print(out.pivot(index="Model", columns="percentile", values="test_F1").to_string())
    print("\nSaved reports/experiment_label_percentile.csv")


def tune_decision_threshold(y_val, score_val):
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_val, (score_val > t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def block_bootstrap_indices(n, rng):
    n_blocks = int(np.ceil(n / BOOTSTRAP_BLOCK_DAYS))
    starts = rng.integers(0, n - BOOTSTRAP_BLOCK_DAYS + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + BOOTSTRAP_BLOCK_DAYS) for s in starts])
    return idx[:n]


def run_bootstrap(test_preds):
    """Block bootstrap over the test period (30-day blocks, because weather
    days are strongly autocorrelated and an i.i.d. bootstrap would be too
    optimistic). All models are aligned on the same days (the LSTM needs a
    7-day warm-up, so the first SEQ_LEN-1 days are dropped for everyone)."""
    aligned = {}
    for name, (y, pred, score) in test_preds.items():
        drop = 0 if name == "LSTM" else SEQ_LEN - 1
        aligned[name] = (np.asarray(y)[drop:], np.asarray(pred)[drop:], np.asarray(score)[drop:])
    n = len(next(iter(aligned.values()))[0])
    rng = np.random.default_rng(SEED)
    f1s = {m: [] for m in aligned}
    aucs = {m: [] for m in aligned}
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = block_bootstrap_indices(n, rng)
        for m, (y, pred, score) in aligned.items():
            yb = y[idx]
            f1s[m].append(f1_score(yb, pred[idx], zero_division=0))
            aucs[m].append(roc_auc_score(yb, score[idx]) if len(set(yb)) > 1 else np.nan)
    rows = []
    for m in aligned:
        f1a, auca = np.array(f1s[m]), np.array(aucs[m])
        d = f1a - np.array(f1s["XGBoost"])
        rows.append({
            "Model": m,
            "F1_point": round(f1_score(aligned[m][0], aligned[m][1], zero_division=0), 3),
            "F1_CI95_low": round(float(np.percentile(f1a, 2.5)), 3),
            "F1_CI95_high": round(float(np.percentile(f1a, 97.5)), 3),
            "AUC_CI95_low": round(float(np.nanpercentile(auca, 2.5)), 3),
            "AUC_CI95_high": round(float(np.nanpercentile(auca, 97.5)), 3),
            "F1_minus_XGBoost_CI95_low": round(float(np.percentile(d, 2.5)), 3),
            "F1_minus_XGBoost_CI95_high": round(float(np.percentile(d, 97.5)), 3),
            "P(F1 > XGBoost)": round(float(np.mean(d > 0)), 2),
        })
    return pd.DataFrame(rows).set_index("Model")


def run_imbalance_arms():
    df, thr = make_frame(0.85)
    train, val, test = chronological_split(df)
    print(f"label: runoff > {thr:.2f} mm/day; positive rate train {train['flood_risk'].mean():.1%}", flush=True)

    print("\nFitting default models ...", flush=True)
    plain = FittedModels(train)
    val_p, _ = plain.predict(val)
    test_p, _ = plain.predict(test)

    print("Fitting SMOTE models ...", flush=True)
    smote = FittedModels(train, oversample=True)
    val_s, _ = smote.predict(val)
    test_s, _ = smote.predict(test)

    # threshold-tuned arm: reuse the default models' scores, no refit
    test_tuned, tuned_thresholds = {}, {}
    for name in PROBABILISTIC_MODELS:
        y_v, _, score_v = val_p[name]
        t = tune_decision_threshold(y_v, score_v)
        tuned_thresholds[name] = t
        y_t, _, score_t = test_p[name]
        test_tuned[name] = (y_t, (score_t > t).astype(int), score_t)
    val_tuned = {n: (val_p[n][0], (val_p[n][2] > tuned_thresholds[n]).astype(int), val_p[n][2])
                 for n in PROBABILISTIC_MODELS}

    arms = {
        "default": (to_table(val_p), to_table(test_p)),
        "smote": (to_table(val_s), to_table(test_s)),
        "tuned_threshold": (to_table(val_tuned), to_table(test_tuned)),
    }
    rows = []
    for arm, (vt, tt) in arms.items():
        for model in tt.index:
            rows.append({
                "arm": arm, "Model": model,
                "decision_threshold": round(tuned_thresholds.get(model, 0.5), 2) if arm == "tuned_threshold" else 0.5,
                "val_F1": vt.loc[model, "F1"],
                "test_Precision": tt.loc[model, "Precision"], "test_Recall": tt.loc[model, "Recall"],
                "test_F1": tt.loc[model, "F1"], "test_AUC": tt.loc[model, "AUC"],
                "test_LogLoss": tt.loc[model, "LogLoss"],
            })
    out = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    out.to_csv("reports/experiment_imbalance_arms.csv", index=False)

    pd.set_option("display.width", 220)
    for metric in ("test_F1", "test_Recall", "test_Precision", "test_AUC", "test_LogLoss"):
        print(f"\n{metric}:")
        print(out.pivot(index="Model", columns="arm", values=metric).to_string())
    print("\nValidation-tuned decision thresholds:", {k: round(v, 2) for k, v in tuned_thresholds.items()})

    print("\nBlock-bootstrap 95% CIs on the test set (default arm):", flush=True)
    ci = run_bootstrap(test_p)
    ci.to_csv("reports/experiment_bootstrap_ci.csv")
    print(ci.to_string())
    print("\nSaved reports/experiment_imbalance_arms.csv, experiment_bootstrap_ci.csv")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "threshold":
        run_threshold_sweep()
    elif mode == "arms":
        run_imbalance_arms()
    else:
        sys.exit("usage: experiments_daily.py threshold | arms")
