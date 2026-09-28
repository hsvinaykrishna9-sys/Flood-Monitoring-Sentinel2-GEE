# Flood-Monitoring-Sentinel2-GEE

Day-ahead flood-risk prediction for Kodagu district, Karnataka, built on real ERA5-Land weather/hydrology data (1950-2026), Sentinel-2 satellite indices, and terrain data — all pulled via Google Earth Engine. Compares 8 machine learning models, explains every prediction with SHAP, and includes a spatial (per-taluk) risk model that uses real elevation and slope data, not just weather.

---

## 📋 Project Overview

Kodagu is one of the most flood-prone districts in Karnataka, particularly during the June-September monsoon. This project builds a **genuine day-ahead flood-risk classifier**: given only what's already known as of today (rainfall, runoff, soil moisture trends), it predicts whether tomorrow is likely to be a flood-risk day — with no lookahead into future data.

**What makes this different from a typical rainfall-alert app:** it doesn't just forecast rain — it combines historical antecedent conditions (recent rainfall, runoff, soil saturation trends) with a trained classifier to estimate flood risk specifically, explains *why* each prediction was made (SHAP feature attribution), and — uniquely — breaks risk down **per sub-region (taluk)** using real terrain differences, instead of one number for the whole district.

---

## 🎯 Objectives

* Build a **day-ahead flood-risk classifier** using real historical weather/hydrology data, with no data leakage (strict chronological train/test split, lagged features only)
* Compare multiple ML approaches — tree ensembles, linear/regression baselines, kernel methods, and a sequence model — on identical data and evaluation, not just pick one algorithm
* Explain every prediction (SHAP feature attribution), not just output a bare probability
* Extend beyond a single-point model to **spatial, per-region risk** using real terrain (elevation, slope, TWI)
* Explore fusing satellite-derived vegetation/water indices (NDVI/NDWI) with weather data
* Provide a live terminal demo that pulls real forecast data and predicts the next 10 days, including multi-day Flood Watch/Warning alerts

---

## 📊 Dataset Used

### Weather / Hydrology (primary dataset)
* **ECMWF ERA5-Land daily aggregates** — temperature, dewpoint, humidity, wind speed, precipitation, runoff, soil moisture, surface pressure, evaporation
* Source: Copernicus Climate Data Store, accessed via Google Earth Engine
* **Whole-district**: 1950-01-02 to 2026-09-20 (28,021 real daily records)
* **Per-taluk** (Madikeri, Virajpet, Somwarpet — approximate Voronoi split): 2018-01-01 to 2026-09-20 (9,555 records, 3 regions × 3,185 days)

### Satellite Data
* **Sentinel-2 optical imagery** (10m resolution), via Google Earth Engine
* NDVI (vegetation) and NDWI (water) monthly indices, 2018-2025

### Terrain Data
* **SRTM Digital Elevation Model** (elevation, slope), plus a Topographic Wetness Index (TWI) derived from flow accumulation
* Whole-district average, and — for the spatial model — per-taluk values (elevation ranges 849-990m, slope 9-13.6° across the 3 taluks)

### Study Region
* **District:** Kodagu, Karnataka
* **Time period:** 1950-2026 (whole-district weather); 2018-2026 (per-taluk weather, satellite indices)
* **Prediction window:** day-ahead (1 day), with a 10-day outlook using live forecast data where available

**Label definition (important caveat):** No verified historical flood-event record exists for Kodagu, so `flood_risk` is a physical proxy — a day is flagged 1 when its runoff exceeds the 85th percentile of the training-period runoff distribution (per-taluk in the spatial model, since Madikeri is structurally wetter than Somwarpet). This is documented, not hidden, throughout the code and reports.

---

## 🔬 Methodology

### 1. Data Acquisition
* Daily and per-taluk ERA5-Land weather pulled via Google Earth Engine (`src/data_collection/`, plus GEE scripts run directly in the Earth Engine code editor for the larger historical/spatial pulls)
* Monthly Sentinel-2 NDVI/NDWI and SRTM terrain, also via GEE

### 2. Exploratory Data Analysis (`src/preprocessing/eda_daily.py`)
* Outlier analysis (IQR method)
* Correlation heatmap across engineered features (flagged `rainfall_lag1` / `rain_soil_interaction_lag1` as 99.9% correlated — near-redundant)
* Missing-value check
* Class balance (flood-risk days are ~15% of the record — a real, structural imbalance, not a data artifact)
* Distribution shape (mean/median/skew) for core weather variables

### 3. Feature Engineering (no same-day leakage)
* Every feature used to predict day *t* comes only from day *t-1* and earlier: lagged rainfall/runoff/soil-moisture/temperature/humidity/wind, 3-day and 7-day rolling averages, a rainfall × soil-moisture interaction term, and day-of-year seasonality (sin/cos encoding)
* Chronological (never random) train/test split, with an explicit assertion that train and test indices never overlap

### 4. Model Training & Comparison (`src/models/train_models_daily.py`)
Eight models trained and evaluated identically on the same 76-year dataset, chronological 80/20 split:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline (Runoff Persistence) | 0.936 | 0.835 | 0.781 | 0.808 | 0.968 |
| Random Forest | 0.943 | 0.863 | 0.790 | 0.825 | 0.976 |
| **XGBoost (chosen)** | **0.945** | **0.855** | **0.821** | **0.838** | **0.977** |
| SVM | 0.943 | 0.878 | 0.773 | 0.822 | 0.957 |
| Logistic Regression | 0.941 | 0.859 | 0.785 | 0.820 | 0.975 |
| Ridge Regression (thresholded) | 0.934 | 0.785 | 0.846 | 0.814 | 0.972 |
| Ensemble (RF+XGB+SVM) | 0.944 | 0.876 | 0.785 | 0.828 | 0.977 |
| LSTM | 0.942 | 0.846 | 0.813 | 0.829 | 0.978 |

**XGBoost** was chosen: best F1 and Recall among the classification models, near-best AUC, and gives interpretable feature importances.

**Class imbalance (SMOTE), tested and not adopted:** SMOTE was tried on the training set only; it substantially raises Recall (up to ~93%) but consistently lowers Precision and F1 across every model tested, on both the original 8-year and the full 76-year dataset. The better fix turned out to be more real data, not synthetic oversampling — confirmed by comparing results before/after extending the dataset from 8 to 76 years (Recall rose from ~51% to ~82% on Random Forest from data volume alone).

**Satellite fusion, tested and not adopted:** NDVI/NDWI (previous-month-lagged, to avoid leakage) were added as extra features (`src/preprocessing/feature_engineering_daily_fused.py`). They rank 4th and 5th of 17 features by importance — genuine signal — but don't improve overall test metrics, since they're largely redundant with runoff once runoff is already a feature.

### 5. Spatial / Per-Taluk Model (the real differentiator)
Terrain data (elevation, slope, TWI) was collected early in the project but was a single whole-district average — useless as a model feature since it can't vary. `src/preprocessing/feature_engineering_taluk.py` and `src/models/train_models_taluk.py` fix this: real per-taluk weather and terrain for Madikeri, Virajpet, and Somwarpet, each taluk's own flood-risk threshold, one XGBoost model trained across all three.

| Taluk | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Madikeri | 0.947 | 0.857 | 0.788 | 0.821 | 0.975 |
| Virajpet | 0.954 | 0.818 | 0.808 | 0.813 | 0.976 |
| Somwarpet | 0.939 | 0.759 | 0.750 | 0.755 | 0.972 |
| **Overall** | **0.947** | **0.814** | **0.782** | **0.798** | **0.975** |

`elevation_m` ranks **2nd of 17 features** by importance, right after `runoff_lag1` — real confirmation that terrain genuinely modulates flood risk once it varies spatially.

### 6. Explainability
Every single-day prediction is explained with **SHAP** (TreeExplainer, exact for XGBoost) — the top features that pushed that day's risk up or down, not just a bare probability.

### 7. Live Prediction & Alerts (`src/models/predict_daily.py`, `predict_taluk.py`)
* Genuine day-ahead forecast for "today," using only already-known data
* 10-day outlook that pulls a real rainfall/temperature/humidity/wind forecast from the Open-Meteo API where available, clearly tagged `[live]` vs `[trend]` (trend-projected) per day
* Multi-day **Flood Watch / Warning banners** — consecutive elevated-risk days are grouped into a single alert, the way a weather service issues a multi-day warning instead of a bare list of daily numbers
* Historical single-date and date-range check modes, for validating predictions against real past outcomes
* Per-taluk live/historical mode showing risk broken out by region

---

## 🏗️ High-Level Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                          │
│  ERA5-Land (GEE)  │  Sentinel-2 NDVI/NDWI (GEE)  │  SRTM DEM   │
│                    │  Open-Meteo (live forecast)               │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                 PREPROCESSING & EDA                            │
│  Lag/rolling features (no leakage)  │  Outlier/correlation/    │
│  Per-taluk terrain merge            │  class-balance analysis  │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                MODEL TRAINING & COMPARISON                    │
│  8 models (whole-district)  │  Spatial per-taluk XGBoost       │
│  SMOTE tested  │  Satellite fusion tested  │  SHAP explainer   │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                  LIVE PREDICTION (CLI)                        │
│  Day-ahead + 10-day outlook  │  Flood Watch/Warning banner     │
│  Per-taluk regional risk     │  SHAP "why" explanation         │
└───────────────────────────────────────────────────────────────┘
```

*Planned, not yet built:* a REST API and web dashboard layer on top of the existing prediction pipeline (see Roadmap).

---

## 🛠️ Technology Stack

### Data Processing
* **Google Earth Engine (GEE)** — satellite/reanalysis data extraction
* **Python 3.11**, **Pandas**, **NumPy**
* **Requests** — Open-Meteo live forecast API

### Machine Learning
* **scikit-learn** — Random Forest, Logistic Regression, Ridge Regression, SVM, preprocessing, metrics
* **XGBoost** — chosen model
* **TensorFlow / Keras** — LSTM sequence model
* **imbalanced-learn** — SMOTE (tested, not used in the final pipeline)
* **SHAP** — per-prediction explainability

### Visualization / Reporting
* **Matplotlib** — EDA plots, ROC curves, model comparison charts

### Testing
* **pytest**

---

## 📁 Repository Structure
```
Flood-Monitoring-Sentinel2-GEE/
│
├── src/
│   ├── data_collection/
│   │   ├── weather_collector.py         # Open-Meteo historical weather
│   │   ├── combine_satellite.py         # Sentinel-2 NDVI/NDWI monthly merge
│   │   └── merge_datasets.py            # satellite + weather join
│   │
│   ├── preprocessing/
│   │   ├── feature_engineering.py       # monthly pipeline (original)
│   │   ├── feature_engineering_daily.py # daily pipeline, whole-district
│   │   ├── feature_engineering_daily_fused.py  # + satellite fusion (tested)
│   │   ├── feature_engineering_taluk.py # per-taluk spatial pipeline
│   │   └── eda_daily.py                 # outlier/correlation/imbalance/distribution EDA
│   │
│   └── models/
│       ├── train_models.py              # monthly 6-model comparison
│       ├── train_models_daily.py        # daily 8-model comparison
│       ├── train_models_taluk.py        # spatial per-taluk model
│       ├── predict.py                   # monthly live prediction
│       ├── predict_daily.py             # daily live prediction, SHAP, alerts
│       └── predict_taluk.py             # per-taluk live/historical prediction
│
├── data/
│   ├── raw/                             # real ERA5-Land, terrain, satellite CSVs
│   └── processed/                       # engineered feature sets
│
├── reports/                             # model comparison tables, ROC curves, EDA plots
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📈 Results Summary

* **28,014** real daily weather/hydrology records (1950-2026) after feature engineering, whole-district
* **9,534** real per-taluk records (2018-2026) across 3 regions
* **8 models compared** identically; **XGBoost chosen** (F1 0.838, AUC 0.977, best Recall/F1 of the classifiers tested)
* **SMOTE tested and rejected** for the final pipeline — hurts F1/Precision on this dataset; more real data was the actual fix
* **Satellite fusion tested** — NDVI/NDWI rank 4th/5th by importance but don't improve headline metrics; documented, not adopted
* **Spatial per-taluk model**: AUC 0.975 overall, elevation ranks 2nd of 17 features — real, demonstrated proof that terrain matters once it can vary
* **Every prediction is explainable** (SHAP) and **every multi-day risk period is alerted** (Watch/Warning banner), not just a bare number

---

## 🚀 Installation & Usage

### Prerequisites
```bash
# Python 3.11+
# A Google Earth Engine account (only needed to re-run the data extraction scripts)
```

### Setup
```bash
git clone https://github.com/hsvinaykrishna9-sys/Flood-Monitoring-Sentinel2-GEE.git
cd Flood-Monitoring-Sentinel2-GEE
pip install -r requirements.txt
```

### Run the EDA
```bash
python3 src/preprocessing/eda_daily.py
```

### Train and compare all 8 models
```bash
python3 src/models/train_models_daily.py
```

### Live day-ahead prediction (whole district)
```bash
python3 src/models/predict_daily.py                       # today + 10-day outlook
python3 src/models/predict_daily.py 2023-07-05             # historical single-day check
python3 src/models/predict_daily.py 2023-07-01 2023-07-12  # historical range + Watch/Warning banner
```

### Spatial per-taluk prediction
```bash
python3 src/models/predict_taluk.py                # today, all 3 taluks
python3 src/models/predict_taluk.py 2022-08-14      # historical regional check
python3 src/models/train_models_taluk.py            # retrain the spatial model
```

---

## 👥 Team Roles

| Name | ID | Role |
|------|----|----- |
| **Vinay Krishna H S** | PES2UG23CS691 | ML pipeline design, model training/comparison, live prediction system, system architecture |
| **Sujay M** | PES2UG23CS620 | Data collection & preprocessing, ETL pipeline |
| **Karthik P** | PES2UG24CS811 | Dashboard/visualization design (planned) |
| **Sudeep A Biradar** | PES2UG23CS609 | QA & documentation, testing, reports and presentations |

**Project Guide:** Prof. Lenish Pramiee
**Institution:** PES University, Department of CSE (UE23CS441A)

---

## 📅 Development Roadmap

### ✅ Completed
- [x] Real ERA5-Land data pipeline, whole-district (1950-2026) and per-taluk (2018-2026)
- [x] Full EDA (outliers, correlation, class balance, distributions)
- [x] 8-model comparison with a strict no-leakage chronological split
- [x] Class-imbalance handling explored (SMOTE) — tested, documented, not adopted
- [x] Satellite (NDVI/NDWI) fusion explored — tested, documented, not adopted
- [x] Spatial per-taluk model using real terrain features
- [x] SHAP explainability on every prediction
- [x] Live CLI prediction with real forecast integration and multi-day alerting

### 🚧 In Progress / Next
- [ ] FastAPI backend exposing the trained models as an endpoint
- [ ] Web dashboard for visualization and alerts
- [ ] End-to-end testing of the full pipeline

### 🔮 Future Enhancements
- [ ] Verified historical flood-event ground truth (replacing the runoff-percentile proxy label)
- [ ] Higher-resolution spatial risk (beyond 3 approximate taluk zones)
- [ ] SMS/push alert system
- [ ] Integration with Karnataka disaster management systems

---

## 📄 License

This project is developed for **academic and research purposes** using open-source data from the Copernicus Programme (ERA5-Land, Sentinel-2), NASA/USGS SRTM, and the Open-Meteo API.

Licensed under the MIT License - see [LICENSE](LICENSE) file for details.
