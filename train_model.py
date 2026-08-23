import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = [
    "distance_km",
    "speed_kmh",
    "delay_min",
    "segment_progress",
    "hour_of_day",
    "is_weekend",
    "stop_count",
]

def generate_sih_training_dataset(n=50000, seed=42):
    rng = np.random.default_rng(seed)

    distance = rng.uniform(1, 300, n)
    speed = rng.uniform(15, 130, n)
    delay = rng.uniform(0, 120, n)
    progress = rng.uniform(0, 1, n)
    hour = rng.integers(0, 24, n)
    is_weekend = rng.integers(0, 2, n)
    stop_count = rng.integers(1, 12, n)

    is_peak = ((hour >= 8) & (hour <= 11)) | ((hour >= 17) & (hour <= 20))
    congestion_factor = np.where(is_peak, 1.25, 1.0)

    effective_speed = np.maximum((speed * (1 - 0.15 * (delay > 25))) / congestion_factor, 10)
    
    base_eta = (distance / effective_speed) * 60
    delay_impact = delay * 0.30
    dwell_overhead = stop_count * 2.5
    noise = rng.normal(0, 3.5, n)

    eta = np.maximum(base_eta + delay_impact + dwell_overhead + noise, 1)

    return pd.DataFrame({
        "distance_km": distance,
        "speed_kmh": speed,
        "delay_min": delay,
        "segment_progress": progress,
        "hour_of_day": hour,
        "is_weekend": is_weekend,
        "stop_count": stop_count,
        "eta_min": eta,
    })

def train():
    df = generate_sih_training_dataset()

    model_mean = HistGradientBoostingRegressor(loss="squared_error", random_state=42)
    model_lower = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, random_state=42)
    model_upper = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, random_state=42)

    model_mean.fit(df[FEATURES], df["eta_min"])
    model_lower.fit(df[FEATURES], df["eta_min"])
    model_upper.fit(df[FEATURES], df["eta_min"])

    pipeline = {
        "mean": model_mean,
        "lower": model_lower,
        "upper": model_upper,
        "features": FEATURES,
    }

    joblib.dump(pipeline, "eta_model.pkl")
    print("Model pipeline saved to 'eta_model.pkl'.")

if __name__ == "__main__":
    train()
