# Flood-Monitoring-Sentinel2-GEE

Day-ahead flood-risk prediction for Kodagu district, Karnataka, built on real ERA5-Land weather/hydrology data (1950-2026), Sentinel-2 satellite indices, and terrain data — all pulled via Google Earth Engine. Compares 8 machine learning models, explains every prediction with SHAP, and includes a spatial (per-taluk) risk model that uses real elevation and slope data, not just weather.

---

## 📋 Project Overview

Kodagu is one of the most flood-prone districts in Karnataka, particularly during the June-September monsoon. This project builds a **genuine day-ahead flood-risk classifier**: given only what's already known as of today (rainfall, runoff, soil moisture trends), it predicts whether tomorrow is likely to be a flood-risk day — with no lookahead into future data.

**What makes this different from a typical rainfall-alert app:** it doesn't just forecast rain — it combines historical antecedent conditions (recent rainfall, runoff, soil saturation trends) with a trained classifier to estimate flood risk specifically, explains *why* each prediction was made (SHAP feature attribution), and — uniquely — breaks risk down **per sub-region (taluk)** using real terrain differences, instead of one number for the whole district.

---

## 🎯 Objectives

* Build a **day-ahead flood-risk classifier** using real historical weather/hydrology data, with no data leakage (strict chronological train/validation/test split, lagged features only)
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
* Chronological (never random) 70/10/20 train/validation/test split, with tests asserting the three date ranges never overlap
* The flood-label runoff threshold (85th percentile) is computed from the training years only, so no validation- or test-period data influences the labels

### 4. Model Training & Comparison (`src/models/train_models_daily.py`)
Eight models trained and evaluated identically on the same 76-year dataset, using a **chronological 70 / 10 / 20 split**:

| Split | Share | Dates | Days |
|---|---|---|---|
| Train | 70% | 1950-01 to 2003-09 | 19,609 |
| Validation | 10% | 2003-09 to 2011-05 | 2,801 |
| Test | 20% | 2011-05 to 2026-09 | 5,604 |

Models are fit on the training set only. The **validation set is used to choose the model** (by F1); the **test set is scored once** to report final performance. Choosing a winner by looking at test scores would make the test result optimistic, so the two roles are kept separate. The split is chronological, never shuffled, so no future weather leaks into training.

**Held-out test set results** (`reports/model_comparison_daily.csv`; log loss = cross-entropy):

| Model | Accuracy | Precision | Recall | F1 | AUC | Log loss |
|---|---|---|---|---|---|---|
| Baseline (Runoff Persistence) | 0.937 | 0.841 | 0.786 | 0.813 | 0.968 | n/a |
| Random Forest | 0.942 | 0.867 | 0.789 | 0.826 | 0.976 | 0.145 |
| **XGBoost (deployed)** | 0.941 | 0.849 | 0.809 | 0.828 | 0.976 | 0.146 |
| SVM | 0.940 | 0.885 | 0.756 | 0.816 | 0.953 | 0.175 |
| Logistic Regression | 0.940 | 0.862 | 0.783 | 0.821 | 0.975 | 0.154 |
| Ridge Regression (thresholded) | 0.934 | 0.792 | 0.843 | 0.817 | 0.972 | n/a |
| Ensemble (RF+XGB+SVM) | 0.945 | 0.880 | 0.792 | 0.834 | 0.976 | 0.146 |
| LSTM | 0.942 | 0.862 | 0.796 | 0.828 | 0.978 | 0.141 |

**Validation loss and overfitting** (`reports/validation_loss_curves_daily.png`): XGBoost's validation log loss bottoms out around boosting round 80-100 (about 0.148) and then rises slightly while training loss keeps falling; the LSTM shows the same pattern after about epoch 12. This is mild overfitting, not a collapse, and the loss gap is small.

**Cross-validation** (`reports/cross_validation_daily.csv`): 5-fold *expanding-window* (forward-chaining) cross-validation on all pre-test data - each fold trains only on days before the days it is scored on, because ordinary shuffled k-fold would train on future years. Mean +/- std across folds:

| Model | F1 | AUC |
|---|---|---|
| Random Forest | 0.812 +/- 0.047 | 0.979 +/- 0.005 |
| XGBoost | 0.811 +/- 0.042 | 0.978 +/- 0.003 |
| Ensemble (RF+XGB+SVM) | 0.815 +/- 0.045 | 0.979 +/- 0.004 |
| LSTM | 0.804 +/- 0.035 | 0.979 +/- 0.003 |
| SVM | 0.807 +/- 0.039 | 0.953 +/- 0.011 |
| Logistic Regression | 0.796 +/- 0.038 | 0.976 +/- 0.004 |

**Which model, honestly:** the top models are statistically indistinguishable. Selecting on validation F1 picks the LSTM (0.794 vs XGBoost 0.791), and on the test set the ensemble has the highest F1 (0.834), but every gap between the leading models is smaller than the fold-to-fold standard deviation (about 0.04 F1). **XGBoost is deployed** for practical reasons rather than a statistically significant accuracy win: it trains in seconds, needs no GPU or TensorFlow, serves quickly from a small API, and supports exact SHAP explanations.

**Robustness checks** (`src/models/experiments_daily.py`, `experiments_taluk.py`; all under the same 70/10/20 protocol, results in `reports/experiment_*.csv`):

* **Which model is best?** No single model is significantly best. 30-day block-bootstrap 95% confidence intervals on the test set are about +/-0.03 F1 wide, and the differences between the seven ML models are inside that noise. Only the simple runoff-persistence baseline is (barely) credibly worse than XGBoost (F1 difference -0.030 to -0.001). Honest corollary: the ML models add a small gain over "yesterday's runoff" (test F1 0.813 vs 0.828; AUC 0.968 vs 0.976).
* **Label percentile (75 / 80 / 85 / 90 / 95).** The 85th percentile is a modelling choice, so it was swept. Test AUC changes little (XGBoost 0.987 / 0.982 / 0.976 / 0.973 / 0.974) and the ranking of the top models is stable; SVM degrades fastest as floods get rarer (AUC 0.971 to 0.924). F1 falls for every model as the label gets rarer because the task gets harder, so F1 is only comparable within one percentile. XGBoost has the best test F1 at the 95th percentile (0.666 vs 0.638-0.642 for the next models), but that is one test set.
* **SMOTE on every model: not adopted.** Applied to training data only. It raises Recall (to 0.89-0.95) but lowers Precision (0.64-0.73), and F1 falls for every model (-0.03 to -0.06); log loss gets worse (e.g. XGBoost 0.146 to 0.185) because the probabilities become miscalibrated; AUC is unchanged. The one exception is the SVM, whose AUC rises 0.953 to 0.969 but stays below the others. (Ridge and the baseline are unaffected: they are not classifiers trained on labels.) For the LSTM, SMOTE was applied to flattened 7-day windows, so its synthetic sequences are interpolations, not real weather.
* **Validation-tuned decision threshold: a better imbalance fix.** Choosing the cut-off on the validation set instead of 0.5 gives F1 +0.000 to +0.024 and raises Recall (XGBoost 0.809 to 0.845 at threshold 0.40) while AUC and log loss are unchanged. Gains are within the noise above, so this is a trade-off option rather than a proven improvement.
* **Per-taluk model, 70/10/20.** Overall F1 0.791 / AUC 0.972. A terrain ablation (same model without elevation, slope, TWI) scores F1 0.783 / AUC 0.970, a difference too small to claim terrain helps on this short test period (about 1,900 rows); earlier claims that terrain "matters" rested on feature importance alone. SMOTE leaves per-taluk F1 unchanged (0.791).

**Satellite fusion, tested and not adopted:** NDVI/NDWI (previous-month-lagged, to avoid leakage) were added as extra features (`src/preprocessing/feature_engineering_daily_fused.py`). They rank 4th and 5th of 17 features by importance — genuine signal — but don't improve overall test metrics, since they're largely redundant with runoff once runoff is already a feature.

### 5. Spatial / Per-Taluk Model (the real differentiator)
Terrain data (elevation, slope, TWI) was collected early in the project but was a single whole-district average — useless as a model feature since it can't vary. `src/preprocessing/feature_engineering_taluk.py` and `src/models/train_models_taluk.py` fix this: real per-taluk weather and terrain for Madikeri, Virajpet, and Somwarpet, each taluk's own flood-risk threshold, one XGBoost model trained across all three.

Evaluated with the same chronological **70 / 10 / 20** train / validation / test split as the daily model (cut by calendar date, shared across taluks, so no taluk's future leaks into another's past). The model has fixed hyperparameters and nothing is tuned, so the validation set only serves as a stability check (validation F1 0.790, AUC 0.966).

Held-out test results (`reports/model_comparison_taluk*.csv`):

| Taluk | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Madikeri | 0.948 | 0.851 | 0.808 | 0.829 | 0.973 |
| Virajpet | 0.943 | 0.787 | 0.747 | 0.766 | 0.970 |
| Somwarpet | 0.940 | 0.768 | 0.768 | 0.768 | 0.974 |
| **Overall** | **0.944** | **0.805** | **0.777** | **0.791** | **0.972** |

`elevation_m` ranks 2nd of 17 features by importance, right after `runoff_lag1`, but `twi` ranks 16th and `slope_degrees` has zero importance. Importance alone does not show that terrain helps: a direct ablation (same model without the three terrain features) scores F1 0.783 / AUC 0.970 versus 0.791 / 0.972 with them, a gap too small to call real on this short test period. Terrain's value here is suggestive, not proven.

**Monthly model (original pipeline, `train_models.py`):** also moved to the 70/10/20 split (65 / 9 / 19 months). With only 93 months, the 9-month validation set has almost no flood months, so every model scores F1 = 0 on it and it cannot select a model; the script reports "inconclusive" rather than naming one. Treat the monthly comparison as an illustration of the pipeline, not a result.

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
│  Weather-app-style location search (any location worldwide)   │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                      REST API (FastAPI)                       │
│  /predict/today  │  /predict/historical/{date}                │
│  /predict/taluk/*  │  /weather/{location}  │  /models/*        │
└───────────────────────────────────────────────────────────────┘
```

*Planned, not yet built:* a visual web dashboard on top of this API (see Roadmap).

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

### Backend
* **FastAPI** — REST API exposing every trained model, also serves the dashboard
* **Uvicorn** — ASGI server

### Frontend
* **Vanilla HTML/CSS/JS** — no framework, no build step, no external libraries or CDN dependencies (nothing outside the deployment itself can fail to load)
* **Browser Geolocation API** — real current-location detection (not IP-based guessing)

### Visualization / Reporting
* **Matplotlib** — EDA plots, ROC curves, model comparison charts

### Testing
* **pytest**, **httpx** (FastAPI TestClient) — 40 automated tests

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
│   ├── models/
│   │   ├── train_models.py              # monthly 6-model comparison (70/10/20)
│   │   ├── train_models_daily.py        # daily 8-model comparison
│   │   ├── train_models_taluk.py        # spatial per-taluk model
│   │   ├── predict.py                   # monthly live prediction
│   │   ├── predict_daily.py             # daily live prediction, SHAP, alerts
│   │   ├── predict_taluk.py             # per-taluk live/historical prediction
│   │   └── weather_app.py               # current/searched-location weather + Kodagu flood risk
│   │
│   ├── api/
│   │   └── main.py                      # FastAPI backend wrapping all trained models + serves the dashboard
│   │
│   └── frontend/                        # dashboard (vanilla HTML/CSS/JS, no build step, no external libs)
│       ├── index.html
│       ├── style.css
│       └── app.js
│
├── data/
│   ├── raw/                             # real ERA5-Land, terrain, satellite CSVs
│   └── processed/                       # engineered feature sets
│
├── reports/                             # model comparison tables, ROC curves, EDA plots
│
├── tests/                               # pytest suite - data, model, API regression tests
│
├── requirements.txt                     # full dev install (includes TensorFlow, matplotlib)
├── requirements-api.txt                 # slim install for deploying just the API + dashboard
├── render.yaml                          # one-click Render Blueprint deploy config
├── DEPLOY.md                            # step-by-step deployment instructions
├── README.md
└── LICENSE
```

---

## 📈 Results Summary

* **28,014** real daily weather/hydrology records (1950-2026) after feature engineering, whole-district
* **9,534** real per-taluk records (2018-2026) across 3 regions
* **8 models compared** identically on a chronological 70/10/20 split with 5-fold expanding-window cross-validation; the top models are statistically tied (test F1 0.826-0.834, AUC 0.976-0.978); **XGBoost deployed** (test F1 0.828, AUC 0.976) for speed, simplicity and SHAP support
* **SMOTE re-tested on every model under the 70/10/20 split and rejected** — raises Recall but lowers F1, Precision and calibration; a validation-tuned decision threshold is the better imbalance lever
* **Satellite fusion tested** — NDVI/NDWI rank 4th/5th by importance but don't improve headline metrics; documented, not adopted
* **Spatial per-taluk model**: AUC 0.972 overall under the 70/10/20 split; elevation ranks high by feature importance, but a terrain ablation shows only a marginal gain (F1 0.791 vs 0.783), so terrain's value is suggestive, not proven
* **Every prediction is explainable** (SHAP) and **every multi-day risk period is alerted** (Watch/Warning banner), not just a bare number
* **40 automated tests** (pytest) covering data integrity, no-leakage splits, label validity, geographic boundary logic, and every API/dashboard endpoint
* **A live dashboard** styled after a phone weather app — real browser-GPS current location or search, live weather for anywhere worldwide, and the flood-risk card specifically inside Kodagu — served by the same API, one deployable service
* **A REST API** (FastAPI) exposing every trained model as a live endpoint, plus a weather-app-style CLI that works for any location worldwide and adds flood-risk prediction specifically inside Kodagu district

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

### Weather-app-style location search
```bash
python3 src/models/weather_app.py              # auto-detects current location
python3 src/models/weather_app.py "Kodagu"      # any location - flood risk shown only if inside Kodagu
python3 src/models/weather_app.py "Bengaluru"
```

### REST API + Dashboard
```bash
pip install -r requirements-api.txt
uvicorn src.api.main:app --reload --reload-dir src --port 8000
# then open http://localhost:8000 for the dashboard,
# or http://localhost:8000/docs for interactive API docs
```
`--reload-dir src` matters: on startup, the API writes its processed feature CSVs to `data/processed/` (same files the CLI tools produce). Without `--reload-dir src`, `--reload` watches the whole project by default, sees those CSVs appear, and reloads the server - which writes them again, which triggers another reload, looping forever ("Reloading..." spamming the terminal, page never finishes loading). Restricting the watch to `src/` (your actual code) fixes it.

API endpoints: `/health`, `/predict/today`, `/predict/historical/{date}`, `/predict/taluk/today`, `/predict/taluk/historical/{date}`, `/weather/{location}`, `/weather-by-coords?lat=&lon=`, `/geocode-search?q=`, `/models/comparison`.

The dashboard (`src/frontend/`) is a single page, no build step, no external JS/CSS libraries — it asks for your browser's location (like a phone weather app's "Home" screen) or lets you search any city, and shows the flood-risk card only when that location is inside Kodagu district.

**To deploy this publicly** (not just localhost), see [DEPLOY.md](DEPLOY.md) — one free Render.com service hosts both the API and the dashboard together.

### Run the test suite
```bash
pytest tests/ -v
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
- [x] FastAPI backend exposing every trained model as a live endpoint
- [x] Weather-app-style location search/auto-detect (any location worldwide; flood risk inside Kodagu)
- [x] Web dashboard (visual UI) on top of the API — real browser GPS, search, live weather, flood-risk card
- [x] Automated test suite (34 tests) covering data integrity, no-leakage splits, and every API/dashboard endpoint
- [x] Deployment config prepared and verified (slim dependency set tested in an isolated environment, `render.yaml` ready)

### 🚧 In Progress / Next
- [ ] Actually deploying to a public URL — this needs a hosting account (Render.com, free), which is the one step that has to be done by hand rather than automated; see [DEPLOY.md](DEPLOY.md)

### 🔮 Future Enhancements
- [ ] Verified historical flood-event ground truth (replacing the runoff-percentile proxy label)
- [ ] Higher-resolution spatial risk (beyond 3 approximate taluk zones)
- [ ] SMS/push alert system
- [ ] Integration with Karnataka disaster management systems

---

## 📄 License

This project is developed for **academic and research purposes** using open-source data from the Copernicus Programme (ERA5-Land, Sentinel-2), NASA/USGS SRTM, and the Open-Meteo API.

Licensed under the MIT License - see [LICENSE](LICENSE) file for details.
