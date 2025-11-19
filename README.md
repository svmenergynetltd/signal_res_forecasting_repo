# **Solar PV Forecasting System (Machine Learning)**

A complete machine-learning pipeline for forecasting photovoltaic (PV) power generation with 24-hour ahead predictions and 30-minute output resolution.  
This project supports multiple ML models, automatic retraining, and clean JSON outputs suitable for integration into SCADA, EMS, and PV monitoring systems.

---

## 🚀 **Features**

- Multi-model forecasting (LightGBM, XGBoost, RandomForest, GradientBoosting, ExtraTrees, DecisionTree, LinearRegression)
- Multi-step (24-hour) forecasting using supervised learning
- Converts hourly forecasts into **48× 30-minute predictions**
- Auto-cleaning of timestamps & missing data
- Feature engineering with lags & rolling windows
- Nighttime–zero enforcement
- Daily or weekly retraining options
- Outputs structured JSON files ready for API ingestion
- Evaluation metrics (MAE, RMSE, daytime MAE)

---

## 📂 **Folder Structure**

```
project/
│
├── PV_Forecast_ML.py          # Main forecasting script
├── Data/
│   └── Data_updated.xlsx      # Input dataset
└── results_json/
    ├── 2025-08-25_xgb_PVYield_30min.json
    ├── 2025-08-25_lgb_PVYield_30min.json
    ├── metrics_summary.csv
    └── next_day_forecast_final.json
```

---

## 🛠️ **Installation**

### 1️⃣ Create virtual environment
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows
```

### 2️⃣ Install dependencies
```bash
pip install -U pip
pip install numpy pandas scikit-learn lightgbm xgboost joblib
```

---

## 📘 **Input Data Requirements**

The Excel file must contain a timestamp and at least the PV yield column:

| Column | Required | Description |
|--------|----------|-------------|
| Statistical Period | ✔ | Timestamp |
| PV Yield (kWh) | ✔ | Measured PV output |
| Global Irradiation (kWh/㎡) | ✔ | Solar irradiation |
| Average Temperature(°C) | ✔ | Ambient temperature |
| Loss Due to Export Limitation (kWh) | Optional | Curtailment |
| Inverter Yield (kWh) | Optional | Inverter output |
| Theoretical Yield (kWh) | Optional | Ideal PV |

Missing optional columns are automatically created if absent.

---

## ⚙️ **Configuration (inside `PV_Forecast_ML.py`)**

### **Select forecast target**
```python
TARGET_COLUMNS = ["PV Yield (kWh)"]
# or combine:
# TARGET_COLUMNS = ["PV Yield (kWh)", "Loss Due to Export Limitation (kWh)"]
```

### **Retraining frequency**
```python
RETRAIN_FREQUENCY = "daily"    # options: "daily", "weekly", None
```

### **Initial training cutoff**
```python
INITIAL_CUTOFF = pd.to_datetime("2025-08-25 00:00:00")
```

### **Output resolution**
```python
UPSAMPLE_TO_30MIN = True
```

### **Nighttime zero rule**
```python
IRR_DAY_THRESHOLD = 1e-4
```

### **Enable/disable individual ML models**
```python
DO_LGB = True
DO_XGB = True
DO_RF = True
DO_GBR = True
DO_ET  = True
DO_DT  = True
DO_LR  = True
```

---

## ▶️ **Running the Forecast**

Run the script:

```bash
python PV_Forecast_ML.py
```

The pipeline will:

1. Load and clean the dataset  
2. Train initial ML models  
3. Forecast each day sequentially  
4. Create 48× 30-minute predictions  
5. Apply nighttime zero rule  
6. Save all forecasts as JSON  

---

## 📤 **Output Files**

### **1. 30-minute forecast JSON files**

Example:
```
2025-08-25_xgb_PVYield_30min.json
```

Each file contains 48 entries:

```json
[
  {"timestamp": "2025-08-25 00:00:00", "data": 0.0},
  {"timestamp": "2025-08-25 00:30:00", "data": 0.0},
  ...
  {"timestamp": "2025-08-25 23:30:00", "data": 12.34}
]
```

---

### **2. Metrics Summary**
```
results_json/metrics_summary.csv
```
Includes MAE, RMSE, and daytime MAE per model.

---

### **3. Final Next-Day Forecast**
```
results_json/next_day_forecast_final.json
```
Uses the best available model at the end of the pipeline.

---

## 📈 **Validation Checklist**

✔ 48 timestamps per forecast day  
✔ Zero values at night  
✔ Smooth shape during daytime  
✔ No negative values  
✔ Reasonable MAE in metrics_summary.csv  
