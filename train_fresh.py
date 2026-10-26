"""
Generate a large realistic dataset (48K+ records) and train the model.
Run from project root:  python scripts/train_fresh.py
"""
import pandas as pd
import numpy as np
import os
import sys

# Resolve paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'backend'))
from model import TrafficLSTM


def generate_large_dataset(num_hours=48000):
    """Generate ~5.5 years of realistic hourly traffic data modeled after I-94 patterns."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range(start='2012-10-01', periods=num_hours, freq='h')

    records = []
    prev_speed = 60.0

    for ts in timestamps:
        hour = ts.hour
        day = ts.dayofweek
        month = ts.month

        hour_vol = {
            0: 3.0, 1: 2.0, 2: 1.5, 3: 1.2, 4: 1.5, 5: 3.5,
            6: 9.0, 7: 22.0, 8: 38.0, 9: 30.0, 10: 19.0, 11: 17.0,
            12: 19.0, 13: 18.0, 14: 19.0, 15: 24.0, 16: 38.0, 17: 45.0,
            18: 34.0, 19: 22.0, 20: 15.0, 21: 10.0, 22: 7.0, 23: 4.5
        }
        base_vol = hour_vol[hour]

        if day >= 5:
            if 7 <= hour <= 9 or 16 <= hour <= 18:
                base_vol *= 0.4
            elif 10 <= hour <= 16:
                base_vol *= 1.25
            else:
                base_vol *= 0.8

        seasonal = 1.0 + 0.18 * np.sin(2 * np.pi * (month - 1) / 12 - np.pi / 3)
        base_vol *= seasonal

        daily_factor = 1.0 + rng.normal(0, 0.08)
        volume = max(0.3, base_vol * daily_factor + rng.normal(0, base_vol * 0.12))

        free_flow = 70.0
        capacity = 50.0
        vol_ratio = min(volume / capacity, 0.95)
        base_speed = free_flow * (1.0 - vol_ratio)

        if rng.random() < 0.05:
            base_speed *= rng.uniform(0.55, 0.85)

        day_of_year = ts.timetuple().tm_yday
        if day_of_year in [1, 20, 51, 145, 186, 247, 282, 315, 329, 359]:
            base_speed *= 1.1
            volume *= 0.6

        target = base_speed + rng.normal(0, 1.8)
        speed = 0.65 * target + 0.35 * prev_speed
        speed = max(10.0, min(74.0, speed))
        prev_speed = speed

        records.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'speed': round(speed, 1),
            'volume': round(volume, 1),
            'hour': hour,
            'day_of_week': day,
            'temp': round(rng.normal(280, 10), 2),
            'rain_1h': round(max(0, rng.normal(-1, 2)), 2),
            'clouds_all': round(rng.uniform(0, 100), 2),
            'is_holiday': 1 if day_of_year in [1, 20, 51, 145, 186, 247, 282, 315, 329, 359] else 0
        })

    return pd.DataFrame(records)


if __name__ == '__main__':
    os.chdir(PROJECT_ROOT)

    print("=" * 60)
    print("  Traffic Speed Predictor - Full Training Pipeline")
    print("=" * 60)

    print("\n[1/3] Generating 48,000 hours of realistic traffic data...")
    df = generate_large_dataset(48000)

    os.makedirs('data', exist_ok=True)
    os.makedirs('backend/data', exist_ok=True)
    df.to_csv('data/traffic_final_clean.csv', index=False)
    df.to_csv('backend/data/traffic.csv', index=False)

    print(f"  Saved: {len(df):,} rows")
    print(f"  Speed range:  {df['speed'].min():.1f} - {df['speed'].max():.1f} mph")
    print(f"  Volume range: {df['volume'].min():.1f} - {df['volume'].max():.1f}")

    print(f"\n[2/3] Training LSTM+Transformer model (80 epochs, 85/15 split)...")
    model = TrafficLSTM()
    history = model.train_model(df)

    print(f"\n[3/3] Saving model...")
    os.makedirs('backend/saved_models', exist_ok=True)
    model.save('backend/saved_models/traffic_model.pth')

    metrics = history.get('metrics', {})
    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS (LSTM)")
    print(f"{'=' * 60}")
    print(f"  Training samples:        {history['samples']:,}")
    print(f"  Final training loss:     {history['losses'][-1]:.6f}")
    print(f"  Final validation loss:   {history['val_losses'][-1]:.6f}")
    if metrics:
        print(f"  MAE:                     {metrics['mae']:.2f} mph")
        print(f"  RMSE:                    {metrics['rmse']:.2f} mph")
        print(f"  R²:                      {metrics['r2']:.4f}")
        print(f"  Accuracy (within 3 mph): {metrics['accuracy_within_3mph']:.1f}%")
        print(f"  Accuracy (within 5 mph): {metrics['accuracy_within_5mph']:.1f}%")
    print(f"{'=' * 60}")

    print(f"\n[4/4] Training XGBoost baseline...")
    from backend.model import XGBoostPredictor, TFTPredictor
    xgb_model = XGBoostPredictor()
    xgb_history = xgb_model.train_model(df)
    xgb_metrics = xgb_history.get('metrics', {})
    
    print(f"\n{'=' * 60}")
    print(f"  XGBoost RESULTS")
    print(f"{'=' * 60}")
    if xgb_metrics:
        print(f"  MAE:                     {xgb_metrics.get('mae', 'N/A')} mph")
        print(f"  RMSE:                    {xgb_metrics.get('rmse', 'N/A')} mph")
        print(f"  R²:                      {xgb_metrics.get('r2', 'N/A')}")
    
    # We will skip TFT as it takes a long time and uses lightning.
    print(f"{'=' * 60}")
