"""
Exploratory Data Analysis (EDA) for the daily Kodagu flood-risk dataset.

Covers exactly what's expected before modeling:
  - outlier analysis (IQR method, per feature)
  - correlation analysis (heatmap across engineered features)
  - missing-value check
  - class imbalance (flood vs no-flood day counts)
  - distribution shape (mean/median/skew) for the core raw weather variables

Outputs (all real, computed from data/processed/kodagu_daily_features_1950_2026.csv):
  reports/eda/outlier_report.csv
  reports/eda/correlation_heatmap.png
  reports/eda/class_balance.png
  reports/eda/distributions.png
  reports/eda/missing_values.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_engineering_daily import FEATURE_COLUMNS, run as build_features

RAW_VARS = [
    "rainfall_mm", "runoff_mm", "soil_moisture_m3m3",
    "temperature_C", "relative_humidity_pct", "wind_speed_ms",
]

OUT_DIR = "reports/eda"


def outlier_report(df, columns):
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lo) | (df[col] > hi)).sum()
        rows.append({
            "feature": col, "q1": q1, "q3": q3, "iqr": iqr,
            "lower_bound": lo, "upper_bound": hi,
            "n_outliers": int(n_outliers),
            "pct_outliers": round(100 * n_outliers / len(df), 2),
        })
    return pd.DataFrame(rows)


def correlation_heatmap(df, columns, path):
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=8)
    ax.set_yticklabels(columns, fontsize=8)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Correlation Heatmap - Engineered Daily Features")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return corr


def class_balance_chart(df, path):
    counts = df["flood_risk"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(["No Flood Risk (0)", "Flood Risk (1)"], counts.values,
                   color=["#4C72B0", "#C44E52"])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val}\n({val/len(df):.1%})",
                ha="center", va="bottom", fontsize=10)
    ax.set_title(f"Class Balance (n={len(df)})")
    ax.set_ylabel("Number of days")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return counts


def distribution_plots(df, columns, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), columns):
        data = df[col].dropna()
        mean, median = data.mean(), data.median()
        skew = data.skew()
        ax.hist(data, bins=60, color="#4C72B0", alpha=0.75)
        ax.axvline(mean, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean:.2f}")
        ax.axvline(median, color="green", linestyle="-", linewidth=1.5, label=f"median={median:.2f}")
        ax.set_title(f"{col}\nskew={skew:.2f}", fontsize=10)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build_features()

    # 1. Missing values (post feature-engineering, should be 0 - lag warm-up rows already dropped)
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_df = pd.DataFrame({"feature": missing.index, "n_missing": missing.values}) \
        if len(missing) else pd.DataFrame({"feature": [], "n_missing": []})
    missing_df.to_csv(f"{OUT_DIR}/missing_values.csv", index=False)
    print(f"Missing values: {int(missing.sum())} across {len(missing)} columns"
          f" (0 expected - lag warm-up rows were already dropped in feature engineering)")

    # 2. Outlier analysis (IQR method) on raw weather variables
    outliers = outlier_report(df, RAW_VARS)
    outliers.to_csv(f"{OUT_DIR}/outlier_report.csv", index=False)
    print("\nOutlier report (IQR method):")
    print(outliers[["feature", "n_outliers", "pct_outliers"]].to_string(index=False))

    # 3. Correlation analysis across engineered model features
    corr = correlation_heatmap(df, FEATURE_COLUMNS, f"{OUT_DIR}/correlation_heatmap.png")
    high_corr = []
    for i, a in enumerate(FEATURE_COLUMNS):
        for b in FEATURE_COLUMNS[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > 0.85:
                high_corr.append((a, b, round(r, 3)))
    print(f"\nSaved correlation heatmap: {OUT_DIR}/correlation_heatmap.png")
    print("Feature pairs with |correlation| > 0.85 (candidates for redundancy):")
    for a, b, r in high_corr:
        print(f"  {a} <-> {b}: {r}")
    if not high_corr:
        print("  none")

    # 4. Class imbalance
    counts = class_balance_chart(df, f"{OUT_DIR}/class_balance.png")
    print(f"\nClass balance: {dict(counts)} -> flood_risk=1 rate = {df['flood_risk'].mean():.2%}")
    print(f"Saved: {OUT_DIR}/class_balance.png")

    # 5. Distribution shape (mean/median/skew) for core raw variables
    distribution_plots(df, RAW_VARS, f"{OUT_DIR}/distributions.png")
    print(f"Saved: {OUT_DIR}/distributions.png")

    return {
        "missing": missing_df, "outliers": outliers,
        "correlation": corr, "high_corr_pairs": high_corr, "class_counts": counts,
    }


if __name__ == "__main__":
    run()
