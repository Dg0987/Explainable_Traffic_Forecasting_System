"""
Traffic Speed Predictor - FastAPI Backend
Simple API for traffic prediction with explanations
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import threading
from model import TrafficLSTM, XGBoostPredictor, TFTPredictor
from explainer import explain_prediction

app = FastAPI(title="Traffic Speed Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Training state (shared across threads)
training_state = {
    "running": False,
    "progress": 0,
    "result": None,
    "error": None
}

# Ensure data directory exists
os.makedirs('data', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

# Default dataset path
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DEFAULT_DATA = os.path.join(DATA_DIR, 'traffic_final_clean.csv')
WORKING_DATA = 'data/traffic.csv'

# Global model instances
trained_model = None
trained_model_xgb = None
trained_model_tft = None

# Copy default dataset to working location on startup
if os.path.exists(DEFAULT_DATA):
    import shutil
    shutil.copy2(DEFAULT_DATA, WORKING_DATA)

# Auto-load trained model on startup
if os.path.exists('saved_models/traffic_model.pth'):
    try:
        trained_model = TrafficLSTM(input_size=8)
        trained_model.load('saved_models/traffic_model.pth')
        print("Pre-trained LSTM model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load saved LSTM model: {e}")

if os.path.exists('saved_models/traffic_model_xgb.json'):
    try:
        trained_model_xgb = XGBoostPredictor(input_size=8)
        trained_model_xgb.load('saved_models/traffic_model_xgb.json')
        print("Pre-trained XGBoost model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load saved XGBoost model: {e}")

if os.path.exists('saved_models/traffic_model_tft.pth'):
    try:
        trained_model_tft = TFTPredictor(input_size=8)
        trained_model_tft.load('saved_models/traffic_model_tft.pth')
        print("Pre-trained TFT model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load saved TFT model: {e}")


@app.get('/')
def home():
    return {
        "message": "Traffic Speed Predictor API",
        "endpoints": {
            "/upload": "POST - Upload CSV traffic data",
            "/train": "POST - Train the LSTM model",
            "/predict": "POST - Get predictions with explanations",
            "/status": "GET - Check model status",
            "/data-stats": "GET - Get statistics about the loaded dataset",
            "/generate-sample": "POST - Generate realistic traffic data"
        }
    }


def convert_format(df):
    """Auto-convert lab Metro Interstate Traffic Volume format to project format."""
    if 'date_time' in df.columns and 'traffic_volume' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'], format='mixed')
        df = df.drop_duplicates(subset='date_time', keep='first').sort_values('date_time').reset_index(drop=True)
        df['hour'] = df['date_time'].dt.hour
        df['day_of_week'] = df['date_time'].dt.dayofweek

        # Greenshields model: speed = Vf * (1 - V/C)
        FREE_FLOW_SPEED, ROAD_CAPACITY = 70.0, 7200.0
        volume_ratio = df['traffic_volume'].clip(upper=ROAD_CAPACITY) / ROAD_CAPACITY
        raw_speed = FREE_FLOW_SPEED * (1.0 - volume_ratio)
        noise = np.random.default_rng(42).normal(0, 1.5, size=len(df))
        df['speed'] = (raw_speed + noise).clip(lower=10.0, upper=75.0).round(1)
        df['volume'] = (df['traffic_volume'] / 100).round(1)
        df['timestamp'] = df['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Add weather and holiday if they don't exist (when uploading raw kaggle it might, when uploading mock it might not)
        for col in ["temp", "rain_1h", "clouds_all"]:
            if col not in df.columns:
                df[col] = 0.0
        if "holiday" in df.columns:
            df["is_holiday"] = (~df["holiday"].isin(["None", None]) & df["holiday"].notna()).astype(int)
        else:
            df["is_holiday"] = 0

        return df[['timestamp', 'speed', 'volume', 'hour', 'day_of_week', 'temp', 'rain_1h', 'clouds_all', 'is_holiday']], True
    return df, False


@app.post('/upload')
async def upload_data(file: UploadFile = File(...)):
    """Upload traffic data CSV file (supports project format and lab format)"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")

        # Save uploaded file
        filepath = WORKING_DATA
        with open(filepath, "wb") as f:
            f.write(await file.read())

        # Validate the data
        df = pd.read_csv(filepath)

        # Auto-detect and convert lab Metro Interstate Traffic Volume format
        df, was_converted = convert_format(df)
        if was_converted:
            df.to_csv(filepath, index=False)

        required_cols = ['speed', 'volume', 'hour', 'day_of_week', 'temp', 'rain_1h', 'clouds_all', 'is_holiday']

        # Check if required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail={
                "error": f"Missing required columns: {missing_cols}",
                "required": required_cols,
                "found": list(df.columns)
            })

        msg = "Lab data converted & loaded" if was_converted else "Data uploaded successfully"

        return {
            "status": "success",
            "message": msg,
            "rows": len(df),
            "columns": list(df.columns)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _on_progress(info):
    """Callback from model.train_model — receives a dict per epoch."""
    training_state.update({
        "progress": info['percent'],
        "epoch": info['epoch'],
        "total_epochs": info['total_epochs'],
        "train_loss": info['train_loss'],
        "val_loss": info['val_loss'],
        "best_val_loss": info['best_val_loss'],
        "patience": info['patience'],
    })


class TrainRequest(BaseModel):
    model_type: str = "all"

def _train_worker(model_type="all"):
    """Background worker that trains the models."""
    global trained_model, trained_model_xgb, trained_model_tft, training_state
    try:
        df = pd.read_csv(WORKING_DATA)
        
        xgb_metrics = {}
        lstm_metrics = {}
        tft_metrics = {}
        
        # Train XGBoost baseline first (it's fast)
        if model_type in ["all", "xgboost"]:
            try:
                xgb_model = XGBoostPredictor(input_size=8)
                xgb_history = xgb_model.train_model(df)
                xgb_model.save('saved_models/traffic_model_xgb.json')
                trained_model_xgb = xgb_model
                xgb_metrics = xgb_history.get('metrics', {})
            except Exception as xgb_e:
                print(f"XGBoost training failed: {xgb_e}")
                xgb_metrics = {"error": str(xgb_e)}
        
        # Train LSTM+Transformer
        if model_type in ["all", "lstm"]:
            try:
                model = TrafficLSTM(input_size=8)
                history = model.train_model(df, progress_callback=_on_progress)
                model.save('saved_models/traffic_model.pth')
                trained_model = model
                lstm_metrics = history.get('metrics', {})
                lstm_metrics['epochs'] = len(history['losses'])
                lstm_metrics['final_loss'] = history['losses'][-1]
            except Exception as lstm_e:
                print(f"LSTM training failed: {lstm_e}")
                lstm_metrics = {"error": str(lstm_e)}

        # Train TFT
        if model_type in ["all", "tft"]:
            try:
                tft_model = TFTPredictor(input_size=8)
                tft_history = tft_model.train_model(df)
                tft_model.save('saved_models/traffic_model_tft.pth')
                trained_model_tft = tft_model
                tft_metrics = tft_history.get('metrics', {})
            except Exception as tft_e:
                print(f"TFT training failed: {tft_e}")
                tft_metrics = {"error": str(tft_e)}

        training_state.update({
            "running": False,
            "progress": 100,
            "result": {
                "status": "success",
                "message": f"Model(s) ({model_type}) trained successfully",
                "lstm_metrics": lstm_metrics,
                "xgb_metrics": xgb_metrics,
                "tft_metrics": tft_metrics
            },
            "error": None
        })
    except Exception as e:
        training_state.update({
            "running": False,
            "progress": 0,
            "error": str(e)
        })


@app.post('/train')
def train_model(req: TrainRequest = TrainRequest(model_type="all")):
    """Start model training in the background"""
    global training_state

    if training_state["running"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    if not os.path.exists(WORKING_DATA):
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload data first.")

    training_state = {"running": True, "progress": 0, "epoch": 0, "total_epochs": 80,
                      "train_loss": None, "val_loss": None, "best_val_loss": None,
                      "patience": 0, "result": None, "error": None}
    thread = threading.Thread(target=_train_worker, args=(req.model_type,), daemon=True)
    thread.start()

    return {"status": "started", "message": "Training started"}


@app.get('/train/status')
def train_status():
    """Poll training progress with epoch-level detail"""
    if training_state["error"]:
        raise HTTPException(status_code=500, detail=training_state["error"])
    if training_state["result"]:
        return training_state["result"]
    return {
        "status": "training",
        "progress": training_state["progress"],
        "epoch": training_state.get("epoch", 0),
        "total_epochs": training_state.get("total_epochs", 80),
        "train_loss": training_state.get("train_loss"),
        "val_loss": training_state.get("val_loss"),
        "best_val_loss": training_state.get("best_val_loss"),
        "patience": training_state.get("patience", 0),
    }


class PredictRequest(BaseModel):
    data: list
    model_type: str = "lstm"  # "lstm" or "xgboost"

@app.post('/predict')
def predict(req: PredictRequest):
    """Make traffic speed prediction with explanations"""
    global trained_model

    try:
        input_data = np.array(req.data)

        # Validate input shape
        if input_data.shape[1] != 8:
            raise HTTPException(status_code=400, detail="Input must have 8 features: speed, volume, hour, day_of_week, temp, rain_1h, clouds_all, is_holiday")

        # Select model based on toggle
        if req.model_type == "xgboost":
            if trained_model_xgb is None:
                raise HTTPException(status_code=400, detail="XGBoost model not trained yet.")
            active_model = trained_model_xgb
        elif req.model_type == "tft":
            if trained_model_tft is None:
                raise HTTPException(status_code=400, detail="TFT model not trained yet.")
            active_model = trained_model_tft
        else:
            if trained_model is None:
                raise HTTPException(status_code=400, detail="LSTM model not trained yet.")
            active_model = trained_model

        # Make prediction
        prediction = active_model.predict(input_data)

        # Get explanation
        explanation = explain_prediction(active_model, input_data)

        return {
            "prediction": round(float(prediction), 1),
            "unit": "mph",
            "explanation": explanation,
            "temporal_attention": explanation.get("temporal_attention", None)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/status')
def status():
    """Check model and data status"""
    data_exists = os.path.exists(WORKING_DATA)
    model_exists = os.path.exists('saved_models/traffic_model.pth')

    data_info = None
    if data_exists:
        df = pd.read_csv(WORKING_DATA)
        data_info = {
            "rows": len(df),
            "columns": list(df.columns)
        }

    return {
        "data_uploaded": data_exists,
        "model_trained": model_exists,
        "model_loaded": trained_model is not None,
        "data_info": data_info
    }


@app.get('/data-stats')
def data_stats():
    """Get statistics about the loaded dataset"""
    try:
        if not os.path.exists(WORKING_DATA):
            raise HTTPException(status_code=400, detail="No data loaded")

        df = pd.read_csv(WORKING_DATA)
        return {
            "rows": int(len(df)),
            "speed": {"min": float(round(df['speed'].min(), 1)), "max": float(round(df['speed'].max(), 1)), "mean": float(round(df['speed'].mean(), 1))},
            "volume": {"min": float(round(df['volume'].min(), 1)), "max": float(round(df['volume'].max(), 1)), "mean": float(round(df['volume'].mean(), 1))},
            "date_range": {"start": str(df['timestamp'].iloc[0]), "end": str(df['timestamp'].iloc[-1])}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class GenerateSampleRequest(BaseModel):
    hours: int = 168

@app.post('/generate-sample')
def generate_sample(req: GenerateSampleRequest):
    """Generate realistic traffic data modeled after Lab Metro Interstate patterns."""
    try:
        num_hours = req.hours

        rng = np.random.default_rng(42)
        timestamps = pd.date_range(start='2016-01-01', periods=num_hours, freq='h')

        data = []
        prev_speed = 60.0  # for temporal continuity

        for i, ts in enumerate(timestamps):
            hour = ts.hour
            day = ts.dayofweek
            month = ts.month

            # --- Volume model (modeled after metro interstate patterns) ---
            # Base hourly profile (vehicles/hr, scaled /100 for model input)
            hour_volume_profile = {
                0: 3, 1: 2, 2: 1.5, 3: 1.2, 4: 1.5, 5: 3,
                6: 8, 7: 20, 8: 35, 9: 28, 10: 18, 11: 16,
                12: 18, 13: 17, 14: 18, 15: 22, 16: 35, 17: 42,
                18: 32, 19: 20, 20: 14, 21: 10, 22: 7, 23: 5
            }
            base_vol = hour_volume_profile[hour]

            # Weekend: less commuter traffic, more midday
            if day >= 5:
                if 7 <= hour <= 9 or 16 <= hour <= 18:
                    base_vol *= 0.45  # No rush hour on weekends
                elif 10 <= hour <= 15:
                    base_vol *= 1.3   # More midday activity

            # Seasonal variation (more volume in summer)
            seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (month - 1) / 12 - np.pi / 3)
            base_vol *= seasonal_factor

            # Random variation
            volume = max(0.5, base_vol + rng.normal(0, base_vol * 0.15))

            # --- Speed model (Greenshields + noise + temporal continuity) ---
            free_flow = 70.0
            capacity_vol = 50.0  # in scaled units (/100)
            speed_from_volume = free_flow * (1.0 - min(volume / capacity_vol, 0.95))

            # Weather/random events (occasional drops)
            if rng.random() < 0.03:
                speed_from_volume *= rng.uniform(0.6, 0.85)  # Bad weather / incident

            # Temporal continuity — speed doesn't jump wildly hour to hour
            target_speed = speed_from_volume + rng.normal(0, 2.0)
            speed = 0.7 * target_speed + 0.3 * prev_speed  # Smoothing
            speed = max(12.0, min(72.0, speed))
            prev_speed = speed

            data.append({
                'timestamp': ts,
                'speed': round(speed, 1),
                'volume': round(volume, 1),
                'hour': hour,
                'day_of_week': day
            })

        df = pd.DataFrame(data)
        df.to_csv(WORKING_DATA, index=False)

        return {
            "status": "success",
            "message": f"Generated {num_hours} hours of realistic traffic data",
            "rows": len(df)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    print("Starting Traffic Prediction Server...")
    print("API Documentation: http://localhost:5000/docs")
    uvicorn.run("app:app", host='0.0.0.0', port=5000, reload=False)