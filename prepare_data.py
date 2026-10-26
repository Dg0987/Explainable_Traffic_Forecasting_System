"""
Lab Data Preprocessor for Traffic Speed Predictor
=================================================

Converts the "Metro Interstate Traffic Volume" dataset from the lab
into the format required by this project.

Download the CSV and place it in the project root, then run:
    python prepare_data.py

Lab columns used:
    - date_time       : hourly timestamp (local CST)
    - traffic_volume   : vehicles per hour on I-94 Westbound

Derived columns (Greenshields traffic flow model):
    - speed           : estimated speed (mph) from volume
    - volume          : normalized traffic volume
    - hour            : 0-23
    - day_of_week     : 0=Monday ... 6=Sunday

The Greenshields model is a standard traffic-engineering formula:
    speed = free_flow_speed * (1 - volume / road_capacity)
For I-94 (multi-lane interstate): free_flow_speed ≈ 70 mph, capacity ≈ 7200 veh/hr
"""

import pandas as pd
import numpy as np
import os
import sys
import glob


def find_csv():
    """Try to locate the Lab CSV in common locations."""
    # Resolve project root (one level up from scripts/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    patterns = [
        os.path.join(project_root, "data", "raw", "Metro_Interstate_Traffic_Volume.csv"),
        os.path.join(project_root, "data", "raw", "metro_interstate_traffic_volume.csv"),
        os.path.join(project_root, "Metro_Interstate_Traffic_Volume.csv"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


def preprocess_data(input_path, output_path="data/traffic_final_clean.csv"):
    """
    Convert Lab Metro Interstate Traffic Volume CSV to project format.

    Parameters
    ----------
    input_path : str
        Path to the downloaded Lab CSV.
    output_path : str
        Where to save the processed CSV.
    """
    print(f"Reading Lab CSV: {input_path}")
    df = pd.read_csv(input_path)

    # --- Validate columns ---
    required = {"date_time", "traffic_volume"}
    found = set(df.columns)
    if not required.issubset(found):
        missing = required - found
        print(f"ERROR: Missing columns: {missing}")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    print(f"  Raw rows: {len(df):,}")

    # --- Parse datetime ---
    df["date_time"] = pd.to_datetime(df["date_time"], format='mixed')

    # Remove duplicate timestamps (keep first)
    df = df.drop_duplicates(subset="date_time", keep="first")
    df = df.sort_values("date_time").reset_index(drop=True)

    print(f"  After dedup: {len(df):,}")
    print(f"  Date range: {df['date_time'].min()} -> {df['date_time'].max()}")

    # --- Extract time features ---
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek  # 0=Monday

    # --- Extract weather & holiday features ---
    # Create is_holiday binary feature (1 if holiday is not 'None' or NaN)
    if "holiday" in df.columns:
        df["is_holiday"] = (~df["holiday"].isin(["None", None]) & df["holiday"].notna()).astype(int)
    else:
        df["is_holiday"] = 0

    # Ensure weather columns exist
    for col in ["temp", "rain_1h", "clouds_all"]:
        if col not in df.columns:
            df[col] = 0.0  # Default if missing

    # Normalize temperature (Kaggle dataset uses Kelvin, let's keep it as is or convert. 
    # For neural nets, normalization happens later, so we just pass the raw value)
    df["temp"] = df["temp"].astype(float).round(2)
    df["rain_1h"] = df["rain_1h"].astype(float).round(2)
    df["clouds_all"] = df["clouds_all"].astype(float).round(2)

    # --- Derive speed using Greenshields model ---
    # I-94 parameters (multi-lane interstate highway)
    FREE_FLOW_SPEED = 70.0   # mph — typical for I-94
    ROAD_CAPACITY = 7200.0   # vehicles/hour — practical capacity
    MIN_SPEED = 10.0         # minimum realistic speed (extreme congestion)

    # Greenshields: speed = Vf * (1 - volume / capacity)
    # Clamp volume so speed never goes below MIN_SPEED
    volume_ratio = df["traffic_volume"].clip(upper=ROAD_CAPACITY) / ROAD_CAPACITY
    raw_speed = FREE_FLOW_SPEED * (1.0 - volume_ratio)

    # Add small realistic noise (sensors aren't perfectly consistent)
    noise = np.random.default_rng(42).normal(0, 1.5, size=len(df))
    df["speed"] = (raw_speed + noise).clip(lower=MIN_SPEED, upper=FREE_FLOW_SPEED + 5)
    df["speed"] = df["speed"].round(1)

    # --- Normalize volume to a smaller scale (divide by 100) ---
    # The original project's volume is in the ~5-40 range (low scale)
    # Lab volumes are 0-7000+, so we scale down for compatibility
    df["volume"] = (df["traffic_volume"] / 100).round(1)

    # --- Build output ---
    df["timestamp"] = df["date_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    output = df[["timestamp", "speed", "volume", "hour", "day_of_week", "temp", "rain_1h", "clouds_all", "is_holiday"]]

    # --- Save ---
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"\n  Output saved to: {output_path}")
    print(f"  Total rows: {len(output):,}")
    print(f"  Speed  range: {output['speed'].min():.1f} - {output['speed'].max():.1f} mph")
    print(f"  Volume range: {output['volume'].min():.1f} - {output['volume'].max():.1f}")
    print(f"  Hours:  {output['hour'].min()} - {output['hour'].max()}")
    print(f"  Days:   {output['day_of_week'].min()} - {output['day_of_week'].max()}")

    # --- Also copy to backend working directory ---
    backend_path = os.path.join("backend", "data", "traffic.csv")
    os.makedirs(os.path.dirname(backend_path), exist_ok=True)
    output.to_csv(backend_path, index=False)
    print(f"  Also copied to: {backend_path}")

    print("\nDone! You can now start the backend server.")
    return output


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = find_csv()
        if csv_path is None:
            print("=" * 60)
            print("  Lab Data Preprocessor")
            print("=" * 60)
            print()
            print("Could not find the Lab CSV automatically.")
            print()
            print("Steps:")
            print("  1. Ensure the Lab data CSV is downloaded.")
            print()
            print("  2. Place the CSV in this folder and run:")
            print("     python prepare_data.py metro_interstate_traffic_volume.csv")
            print()
            sys.exit(0)

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    preprocess_data(csv_path)
