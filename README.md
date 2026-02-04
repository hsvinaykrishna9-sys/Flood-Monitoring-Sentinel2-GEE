# Flood-Monitoring-Sentinel2-GEE

Real-time flood prediction and monitoring system using Sentinel-2 satellite imagery, Google Earth Engine, and LSTM neural networks. This project predicts flood risk 7-30 days in advance using multi-modal data fusion, automated machine learning pipelines, and interactive web dashboards to support proactive disaster management and regional planning.

---

## 📋 Project Overview

This project focuses on **flood prediction and risk assessment** using Sentinel-2 satellite imagery, meteorological data, and machine learning. Floods are one of the most frequent natural disasters in Karnataka, causing severe damage to agriculture, infrastructure, and livelihoods. The objective is to predict flood probability at district level by analyzing time-series patterns in satellite data, weather conditions, and terrain features using LSTM neural networks and automated cloud-based processing.

**Key Innovation:** Unlike traditional reactive flood monitoring, this system provides **predictive forecasts** 7-30 days in advance, enabling proactive disaster response.

---

## 🎯 Objectives

* To develop an **AI-powered flood prediction system** using LSTM neural networks
* To integrate **multi-modal data** (satellite imagery + weather + terrain)
* To predict flood probability with **>85% accuracy** for 7-30 day windows
* To build an **automated data pipeline** using Google Earth Engine
* To create a **REST API** for real-time predictions
* To develop an **interactive web dashboard** for visualization and alerts
* To generate high-resolution flood risk maps for disaster management authorities

---

## 📊 Dataset Used

### Satellite Data
* **Sentinel-2 Optical Satellite Data** (10m resolution)
  * Source: European Space Agency (ESA) – Copernicus Programme
  * Accessed via: Google Earth Engine
  * Indices: NDWI (water), NDVI (vegetation)

### Meteorological Data
* **Weather Data** (daily)
  * Source: Open-Meteo API
  * Parameters: Rainfall (mm), Temperature (°C), Humidity (%)

### Terrain Data
* **Digital Elevation Model (DEM)** (30m resolution)
  * Source: SRTM (Shuttle Radar Topography Mission)

### Study Region
* **Districts:** Bangalore Urban, Mysore, Kodagu (expandable)
* **Time Period:** 2019-2024 (5 years historical data)
* **Prediction Window:** 7-30 days ahead

---

## 🔬 Methodology

### 1. Data Acquisition
* Sentinel-2 images filtered by date, region, and cloud coverage
* Weather data fetched from meteorological APIs
* Historical flood records from Karnataka SDMA

### 2. Preprocessing
* Cloud filtering (<30% cloud coverage)
* Image clipping using district boundaries
* Monthly median composite generation
* Feature normalization and scaling

### 3. Feature Engineering
* **Satellite Indices:** NDWI, NDVI calculation
* **Temporal Features:** Month, season, day of year
* **Lag Features:** 1-2 month historical values
* **Rolling Statistics:** 3-month moving averages
* **Interaction Terms:** NDWI × Rainfall, etc.

### 4. Model Training
* **Architecture:** LSTM neural network (2 layers, 128+64 units)
* **Input:** 12-month time-series sequences (16 features)
* **Output:** Flood probability [0-1]
* **Training:** 70/15/15 train/val/test split
* **Optimization:** Adam optimizer, early stopping

### 5. Prediction & API
* Real-time flood probability prediction
* Risk classification (Low, Medium, High, Critical)
* RESTful API endpoints for integration
* Confidence scores and feature importance

### 6. Visualization
* Interactive web dashboard (React.js)
* Flood risk maps with Leaflet/Mapbox
* Time-series charts and trend analysis
* Alert system for high-risk predictions

### 7. Export
* Predictions stored in PostgreSQL database
* GeoTIFF exports for GIS analysis
* PDF reports with visualizations

---

## 🏗️ High Level Architecture Diagram
```
┌───────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                          │
│    Sentinel-2  │  Weather API  │  DEM  │  Historical Floods   │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING LAYER                      │
│  Google Earth Engine  │  Feature Engineering  │  PostgreSQL   │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                   MACHINE LEARNING LAYER                      │
│     LSTM Model  │  Training Pipeline  │  Model Registry       │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                         │
│    FastAPI Backend  │  Prediction Service  │  Authentication  │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                         │
│   React Dashboard  │  Interactive Maps  │  Charts & Alerts    │
└───────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Data Processing
* **Google Earth Engine (GEE)** – satellite data processing
* **Python 3.10+** – core programming language
* **Pandas, NumPy** – data manipulation
* **GeoPandas, Rasterio** – geospatial data handling

### Machine Learning
* **TensorFlow 2.15+** – deep learning framework
* **Keras** – neural network API
* **Scikit-learn** – preprocessing and metrics
* **LSTM Networks** – time-series prediction

### Backend
* **FastAPI** – REST API framework
* **PostgreSQL + PostGIS** – database with geospatial support
* **SQLAlchemy** – ORM
* **Pydantic** – data validation

### Frontend
* **React 18** – UI framework
* **Leaflet / Mapbox** – interactive maps
* **Chart.js** – data visualization
* **Axios** – HTTP client

### DevOps
* **Docker** – containerization
* **GitHub Actions** – CI/CD pipeline
* **Railway / Render** – cloud hosting
* **pytest** – automated testing

---

## 📁 Repository Structure
```
Flood-Monitoring-Sentinel2-GEE/
│
├── src/
│   ├── data_collection/       # GEE and weather data collectors
│   ├── preprocessing/          # Feature engineering pipeline
│   ├── models/                 # LSTM model and training scripts
│   ├── api/                    # FastAPI backend
│   └── frontend/               # React dashboard
│
├── data/
│   ├── raw/                    # Raw satellite and weather data
│   ├── processed/              # Preprocessed datasets
│   └── geotiffs/               # Exported flood maps
│
├── notebooks/                  # Jupyter notebooks for EDA
├── tests/                      # Unit and integration tests
├── config/                     # Configuration files
├── scripts/                    # Setup and deployment scripts
├── docs/                       # Documentation and reports
├── docker/                     # Docker configuration
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
└── .gitignore                  # Git ignore rules
```

---

## 📈 Output

### Model Performance
* **Accuracy:** >85% (Target: 87%+)
* **Precision:** >80%
* **Recall:** >80%
* **F1-Score:** >0.80
* **Prediction Window:** 7-30 days ahead
* **Processing Time:** <5 minutes per prediction

### Deliverables
* ✅ Trained LSTM model with >85% accuracy
* ✅ Flood probability predictions (0-1 scale)
* ✅ Risk level classification (Low/Medium/High/Critical)
* ✅ Time-series flood risk charts
* ✅ High-resolution GeoTIFF flood maps
* ✅ RESTful API for integration
* ✅ Interactive web dashboard
* ✅ PDF reports with visualizations
* ✅ Feature importance analysis

---

## 🚀 Installation & Usage

### Prerequisites
```bash
# Python 3.10+, Node.js 18+, PostgreSQL 14+
# Google Earth Engine account
```

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/Flood-Monitoring-Sentinel2-GEE.git
cd Flood-Monitoring-Sentinel2-GEE

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp config/.env.example .env
# Edit .env with your configurations

# Collect data
python src/data_collection/gee_collector.py

# Train model
python src/models/trainer.py

# Start API
uvicorn src.api.main:app --reload

# Make predictions
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"district_id": 1, "target_date": "2024-07-15"}'
```

**Full documentation:** See [docs/installation.md](docs/installation.md)

---

## 👥 Team Roles

| Name | ID | Role |
|------|----|----- |
| **Vinay Krishna H S** | PES2UG23CS691 | Lead Developer - Designs ML pipeline, implements LSTM model, develops API backend, manages system architecture |
| **Sujay M** | PES2UG23CS620 | Data Engineer - Collects and preprocesses satellite & weather data, builds ETL pipeline |
| **Karthik P** | PES2UG24CS811 | Frontend Developer - Creates dashboard, implements visualizations, designs UI/UX |
| **Sudeep A Biradar** | PES2UG23CS609 | QA & Documentation - Conducts testing, literature survey, prepares reports and presentations |

**Project Guide:** Prof. Pavitra  
**Institution:** PES University, Department of CSE

---

## 📅 Development Roadmap

### ✅ Phase 1: Foundation (Completed)
- [x] System architecture design
- [x] Database schema design
- [x] Technology stack selection

### 🚧 Phase 2: Core Implementation (In Progress)
- [ ] Data collection module
- [ ] Preprocessing pipeline
- [ ] LSTM model training
- [ ] API backend development

### 📅 Phase 3: Integration (Planned)
- [ ] Frontend dashboard
- [ ] End-to-end testing
- [ ] Deployment

### 🔮 Future Enhancements
- [ ] Mobile application
- [ ] Real-time satellite data streaming
- [ ] Multi-hazard support (droughts, landslides)
- [ ] SMS alert system
- [ ] Integration with government systems

---

## 📄 License

This project is developed for **academic and research purposes** using open-source satellite data from the Copernicus Programme and Open-Meteo API.

Licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* **Google Earth Engine** for satellite imagery platform
* **ESA Copernicus Programme** for Sentinel-2 data
* **Open-Meteo** for weather data API
* **Karnataka SDMA** for historical flood records
* **PES University** for project support
* **Prof. Pavitra** for guidance and mentorship

---

## 📞 Contact

* **GitHub Issues:** [Report bugs or request features](https://github.com/yourusername/Flood-Monitoring-Sentinel2-GEE/issues)
* **Email:** vinaykrishna@example.com
* **Project ID:** 138

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

**Project ID: 138 | Team: Vinay Krishna, Sujay M, Karthik P, Sudeep A Biradar**  
**PES University | Department of Computer Science & Engineering | 2024**

</div>
