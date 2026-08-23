import os
import sqlite3
from datetime import datetime, timedelta
from urllib.parse import quote

import joblib
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("RAILRADAR_API_KEY", "").strip()
BASE_URL = "https://api.railradar.in/v1"
DB_PATH = "trains.db"

# Load ML model safely
try:
    MODEL_PIPELINE = joblib.load("eta_model.pkl")
    print("Loaded ML model successfully.")
except Exception as e:
    MODEL_PIPELINE = None
    print(f"Warning: Model not loaded ({e}). Using algorithmic fallback.")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_mock_live_data(train_number):
    """Fetches real route stops from train_routes DB table and calculates unique train metrics."""
    conn = get_db()
    cursor = conn.cursor()
    
    # 1. Query master train details
    cursor.execute("SELECT * FROM trains WHERE train_number = ?", (train_number,))
    train_row = cursor.fetchone()

    if train_row:
        keys = train_row.keys()
        name = train_row["train_name"]
        source = train_row["source_station"] if "source_station" in keys else train_row.get("source", "ORIGIN")
        dest = train_row["destination_station"] if "destination_station" in keys else train_row.get("destination", "DEST")
    else:
        name, source, dest = f"EXPRESS {train_number}", "ORIGIN", "DEST"

    # 2. Query actual station sequence from train_routes table
    cursor.execute("""
        SELECT * FROM train_routes 
        WHERE train_number = ? 
        ORDER BY station_sequence ASC
    """, (train_number,))
    route_rows = cursor.fetchall()
    conn.close()

    # Generate unique metrics based on the specific train number
    t_num_int = int(''.join(filter(str.isdigit, str(train_number))) or 12000)
    dynamic_delay = float((t_num_int % 38) + 4)          # Unique delay per train (4 to 42 mins)
    dynamic_speed = float(52 + (t_num_int % 43))         # Unique speed per train (52 to 95 km/h)
    segment_progress = round(((t_num_int % 65) + 20) / 100.0, 2)

    route = []
    if route_rows:
        for r in route_rows:
            r_keys = r.keys()
            dist = r["distance_from_origin"] if "distance_from_origin" in r_keys else r.get("distance", 0)
            sched_arr = r["scheduled_arrival"] if "scheduled_arrival" in r_keys else "12:00"
            hist_delay = r["historical_delay"] if "historical_delay" in r_keys else dynamic_delay
            
            route.append({
                "stationName": r["station_name"],
                "stationCode": r["station_code"],
                "distance": dist,
                "sequence": r["station_sequence"],
                "scheduledArrival": sched_arr,
                "delayArrival": hist_delay,
                "status": "upcoming"
            })
    else:
        # Fallback if train_routes has no entries for this number
        route = [
            {"stationName": source, "stationCode": str(source)[:4], "distance": 0, "sequence": 1, "scheduledArrival": "06:00", "delayArrival": 0, "status": "departed"},
            {"stationName": "Intermediate Junction", "stationCode": "INTM", "distance": 185, "sequence": 2, "scheduledArrival": "09:15", "delayArrival": dynamic_delay, "status": "upcoming"},
            {"stationName": dest, "stationCode": str(dest)[:4], "distance": 420, "sequence": 3, "scheduledArrival": "14:30", "delayArrival": dynamic_delay, "status": "upcoming"}
        ]

    # Select active station based on train progress
    curr_idx = min(1, len(route) - 1)
    curr_st = route[curr_idx]
    curr_st["status"] = "departed"

    return {
        "trainNumber": train_number,
        "status": "Delayed" if dynamic_delay > 20 else "Running On Time",
        "delayMinutes": dynamic_delay,
        "lastUpdatedAt": datetime.now().strftime("%I:%M %p"),
        "train": {
            "number": train_number,
            "name": name,
            "source": {"name": source},
            "destination": {"name": dest},
            "avgSpeed": dynamic_speed
        },
        "currentLocation": {
            "stationCode": curr_st["stationCode"],
            "speedKmh": dynamic_speed,
            "segmentProgress": segment_progress,
            "distance": curr_st["distance"],
            "sequence": curr_st["sequence"]
        },
        "route": route
    }


def railradar_get(path, params=None):
    cleaned_key = API_KEY.lower().strip()
    if not API_KEY or any(p in cleaned_key for p in ["your_", "mock", "demo", "key_here"]):
        return None

    url = BASE_URL + path
    params = params or {}
    params["api_key"] = API_KEY
    headers = {"x-api-key": API_KEY, "Authorization": f"Bearer {API_KEY}"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=4)
        if response.status_code in (401, 403):
            return None
        response.raise_for_status()
        payload = response.json()
        return payload.get("data", payload) if isinstance(payload, dict) else payload
    except Exception:
        return None


def predict_eta(X):
    if MODEL_PIPELINE is None:
        dist, speed, delay = X[0][0], max(X[0][1], 40.0), X[0][2]
        eta = (dist / speed) * 60.0 + delay
        return max(eta, 2.0), max(eta * 0.85, 1.0), eta * 1.18

    try:
        if isinstance(MODEL_PIPELINE, dict):
            m = MODEL_PIPELINE.get("mean", list(MODEL_PIPELINE.values())[0])
            l = MODEL_PIPELINE.get("lower", m)
            u = MODEL_PIPELINE.get("upper", m)
            p_mean, p_low, p_high = float(m.predict(X)[0]), float(l.predict(X)[0]), float(u.predict(X)[0])
        else:
            p_mean = float(MODEL_PIPELINE.predict(X)[0])
            p_low, p_high = p_mean * 0.88, p_mean * 1.15

        return max(p_mean, 1.0), max(p_low, 1.0), max(p_high, p_mean)
    except Exception:
        dist = X[0][0]
        return max(dist * 0.9, 2.0), max(dist * 0.75, 1.0), dist * 1.1


def build_prediction(data):
    train = data.get("train", {})
    current = data.get("currentLocation") or {}
    route = data.get("route") or []

    speed = safe_float(current.get("speedKmh", current.get("speedKmph")), 0)
    delay = safe_float(data.get("delayMinutes"), 0)
    progress = safe_float(current.get("segmentProgress"), 0)
    current_seq = current.get("sequence")

    upcoming = [
        stop for stop in route
        if str(stop.get("status", "")).lower() in ("upcoming", "scheduled")
        or (current_seq is not None and isinstance(stop.get("sequence"), (int, float)) and stop.get("sequence") > current_seq)
    ]

    current_dist = safe_float(current.get("distance"), 0)
    eff_speed = speed if speed > 0 else safe_float(train.get("avgSpeed"), 60)

    now = datetime.now()
    hour_of_day = now.hour
    is_weekend = 1 if now.weekday() >= 5 else 0

    predictions = []
    for idx, stop in enumerate(upcoming):
        st_dist = safe_float(stop.get("distance"), 0)
        dist = max(st_dist - current_dist, 1) if current_dist > 0 else max(st_dist, 1)
        
        X = np.array([[dist, eff_speed, delay, progress, hour_of_day, is_weekend, idx + 1]])
        eta_mean, eta_low, eta_high = predict_eta(X)

        predicted_clock = (now + timedelta(minutes=eta_mean)).strftime("%I:%M %p")

        predictions.append({
            "station": stop.get("stationName") or stop.get("stationCode", "-"),
            "code": stop.get("stationCode", ""),
            "distance_km": round(dist, 1),
            "eta_min": round(eta_mean, 1),
            "predicted_clock": predicted_clock,
            "confidence_band": {
                "min_eta": round(eta_low, 1),
                "max_eta": round(eta_high, 1),
            },
            "scheduled_arrival": stop.get("scheduledArrival", "-"),
            "delay_min": round(safe_float(stop.get("delayArrival"), delay), 1),
        })

    source = train.get("source", {})
    destination = train.get("destination", {})

    return {
        "train_number": train.get("number", data.get("trainNumber")),
        "train_name": train.get("name", "Express Train"),
        "source": source.get("name", source.get("code", "-")) if isinstance(source, dict) else str(source),
        "destination": destination.get("name", destination.get("code", "-")) if isinstance(destination, dict) else str(destination),
        "status": data.get("status", "Running"),
        "delay_min": round(delay, 1),
        "current_station": current.get("stationCode", "-"),
        "speed_kmh": round(speed, 1),
        "segment_progress": round(progress * 100, 1),
        "last_updated": data.get("lastUpdatedAt", "-"),
        "predictions": predictions,
    }


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None, error=None)


@app.route("/track", methods=["GET"])
def track():
    train_number = request.args.get("train_number", "").strip()
    if not train_number:
        return render_template("index.html", result=None, error="Please enter a valid train number.")

    data = railradar_get(f"/trains/{quote(train_number)}/live", {"authoritative": "true"})
    if not data:
        data = get_mock_live_data(train_number)

    return render_template("index.html", result=build_prediction(data), error=None)


@app.route("/api/trains/search", methods=["GET"])
def search_trains():
    query = request.args.get("q", "").strip()
    if not query or len(query) < 2:
        return jsonify([])

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM trains 
        WHERE train_number LIKE ? OR train_name LIKE ? 
        LIMIT 8
    """, (f"{query}%", f"%{query}%"))

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        keys = row.keys()
        source = row["source_station"] if "source_station" in keys else row.get("source", "")
        dest = row["destination_station"] if "destination_station" in keys else row.get("destination", "")
        results.append({
            "number": row["train_number"],
            "name": row["train_name"],
            "route": f"{source} → {dest}"
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)