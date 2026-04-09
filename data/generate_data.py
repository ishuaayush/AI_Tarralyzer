"""
Generate sample climate/environment raw data with intentional
missing values, duplicates, and outliers for cleaning demonstration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random

random.seed(42)
np.random.seed(42)

CITIES = [
    "Mumbai", "Delhi", "Chennai", "Kolkata", "Bangalore",
    "New York", "London", "Tokyo", "Sydney", "Berlin",
    "Paris", "Beijing", "Toronto", "Dubai", "São Paulo"
]

def generate_climate_data(n_rows=500):
    '''Generate synthetic climate dataset with intentional issues.
    Args:
        n_rows (int): Number of rows to generate.
    
    Returns:
        None: Saves the dataset to CSV and JSON files.
    '''
    # ── Generate base data ─────────────────────────────────────────────────────

    start_date = datetime(2020, 1, 1)
    # Generate a list of dates starting from the start_date, incrementing by one day for each row
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]

    data = {
        "date": dates,
        "city": [random.choice(CITIES) for _ in range(n_rows)],
        "temperature_c": np.random.normal(25, 10, n_rows).round(2),
        "humidity_pct": np.random.uniform(20, 95, n_rows).round(2),
        "co2_ppm": np.random.normal(415, 15, n_rows).round(2),
        "pm2_5_ugm3": np.abs(np.random.normal(35, 20, n_rows)).round(2),
        "aqi": np.random.randint(20, 300, n_rows),
        "rainfall_mm": np.abs(np.random.normal(5, 10, n_rows)).round(2),
        "wind_speed_kmh": np.abs(np.random.normal(15, 8, n_rows)).round(2),
        "uv_index": np.random.uniform(0, 11, n_rows).round(1),
        "pressure_hpa": np.random.normal(1013, 10, n_rows).round(2),
        "visibility_km": np.random.uniform(1, 20, n_rows).round(1),
        "solar_radiation_wm2": np.abs(np.random.normal(200, 100, n_rows)).round(2),
    }

    df = pd.DataFrame(data)

    # ── Introduce missing values ──────────────────────────────────────────────
    for col in ["temperature_c", "humidity_pct", "co2_ppm", "pm2_5_ugm3",
                "rainfall_mm", "wind_speed_kmh"]:
        mask = np.random.random(n_rows) < 0.07
        df.loc[mask, col] = np.nan

    # ── Introduce outliers ────────────────────────────────────────────────────
    outlier_idx = np.random.choice(n_rows, 15, replace=False)
    df.loc[outlier_idx[:5], "temperature_c"] = np.random.choice([85, -40, 120], 5)
    df.loc[outlier_idx[5:10], "pm2_5_ugm3"] = np.random.choice([500, 800, 999], 5)
    df.loc[outlier_idx[10:], "co2_ppm"] = np.random.choice([900, 1200, 5], 5)

    # ── Introduce duplicates ──────────────────────────────────────────────────
    dup_rows = df.sample(10)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # ── Introduce bad formats ─────────────────────────────────────────────────
    df["date"] = df["date"].astype(str)
    bad_date_idx = np.random.choice(len(df), 5, replace=False)
    for i in bad_date_idx:
        df.loc[i, "date"] = random.choice(["N/A", "unknown", "2020/99/01", ""])

    df.to_csv("C:\\Users\\akhare\\OneDrive - Nice Software Solutions\\AI_Data_Analysis\\data\\climate_raw.csv", index=False)
    print(f"✅ CSV saved: {len(df)} rows × {len(df.columns)} columns")

    # ── Also save a JSON subset ───────────────────────────────────────────────
    json_data = df.head(100).to_dict(orient="records")
    with open("C:\\Users\\akhare\\OneDrive - Nice Software Solutions\\AI_Data_Analysis\\data\\climate_raw.json", "w") as f:
        json.dump(json_data, f, default=str, indent=2)
    print("✅ JSON saved: 100 records")

if __name__ == "__main__":
    generate_climate_data()
