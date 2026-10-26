# Explainable Traffic Forecasting with Counterfactual Explanations

An enterprise-grade, AI-powered traffic speed prediction system using multiple advanced architectures (LSTM + Transformer Hybrid, XGBoost, and Temporal Fusion Transformer). This project heavily features **Explainable AI (XAI)**, providing counterfactual what-if scenarios, temporal attention visualization, and permutation-based feature importance.

## Core Architecture & Benchmarking

The system is designed to forecast short-term traffic speeds from multivariate time-series data using a 12-hour sliding window across 8 core features.

*   **Hybrid PyTorch Model (LSTM + Transformer):** Combines LSTM sequential pattern extraction with Transformer Encoders. Uses **Time2Vec** embeddings for rich temporal representations.
*   **Optimization:** Trained using **AdamW**, **OneCycleLR** scheduling for extremely fast convergence, and **Huber Loss** for robustness against outlier anomalies (accidents, sensor errors).
*   **Baselines:** Benchmarked against a blazingly fast **XGBoost** model and a state-of-the-art **Temporal Fusion Transformer (TFT)**. 
*   **Performance Metrics:** Evaluated rigorously using MAE (Mean Absolute Error), RMSE, R² score, and custom threshold accuracies (percentage of predictions falling within a highly actionable ±3 to ±5 mph tolerance).

*(For example, XGBoost baseline achieves an R² of ~0.91 and MAE of ~3.42 mph, while the fully converged LSTM-Transformer typically hits R² of ~0.94 and MAE of ~2.85 mph).*

## Dataset & Feature Engineering

The project is trained on an engineered dataset of **48,000+ hourly records** (simulating nearly 5.5 years of continuous highway telemetry). 

The data pipeline processes **8 core features**:
1. `speed`: Upstream traffic speed (mph)
2. `volume`: Traffic volume (vehicles/hour)
3. `hour`: Hour of the day (0-23)
4. `day_of_week`: Day of the week (0-6)
5. `temp`: Temperature (Kelvin)
6. `rain_1h`: Rainfall in the last hour (mm)
7. `clouds_all`: Cloud cover percentage
8. `is_holiday`: Boolean flag for public holidays (0 or 1)

### The Greenshields Model
When raw lab data is provided lacking direct speed telemetry, the `prepare_data.py` script automatically derives target speeds using the **Greenshields macroscopic traffic flow model**:
`speed = free_flow_speed x (1 - volume / road_capacity)`

## Explainable AI (XAI) Engine

Because black-box neural networks are often rejected by traffic authorities, this project includes a dedicated XAI module:
*   **Dynamic Counterfactuals (What-If Scenarios):** Automatically perturbs features to show actionable alternative scenarios (e.g., *"If traffic volume dropped 70%,"* or *"If there was heavy rain (15mm/h),"* or *"If it was a public holiday"*).
*   **Temporal Attention Extraction:** Maps out exactly which historical time-steps the model weighted most heavily in making its current prediction. 
*   **Permutation Feature Importance:** Replaces variables with their baseline means to calculate the absolute percentage impact of each feature on the prediction.

## Tech Stack
*   **Languages & Frameworks:** Python, PyTorch, PyTorch Lightning, FastAPI, Uvicorn
*   **Machine Learning:** LSTM, Transformers, Time2Vec, XGBoost, Temporal Fusion Transformer, Scikit-Learn
*   **Data Processing:** Pandas, NumPy

## Project Structure

```
traffic-predictor/
├── backend/
│   ├── app.py             # FastAPI Server & Endpoints
│   ├── model.py           # PyTorch LSTM, TFT, & XGBoost Architectures
│   ├── explainer.py       # XAI Engine (Counterfactuals & Attention)
│   └── requirements.txt   # Python Dependencies
├── scripts/
│   ├── prepare_data.py    # Lab CSV parser & Greenshields target derivation
│   └── train_fresh.py     # 48K Data Generator & Training Benchmark Script
└── README.md
```

## Getting Started

### 1. Data Setup (Optional)
If you want to train models from scratch rather than using pre-trained weights:
```bash
# Generate 48,000 realistic records and benchmark models
python scripts/train_fresh.py

# OR: Clean raw lab data
python scripts/prepare_data.py
```

### 2. Run the Backend API
Navigate to the `backend` directory:
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 5000
```

### 3. API Endpoints
The scalable **FastAPI** backend supports real-time inference and asynchronous training:
*   `POST /train`: Asynchronously triggers a training pipeline for the selected architecture (`lstm`, `xgboost`, `tft`, `all`).
*   `POST /predict`: Serves predictions via the selected engine alongside full XAI analytics.
*   `GET /status`: Polls the asynchronous training status/progress.
*   `POST /upload`: Cleans and normalizes raw dataset uploads on the fly.
