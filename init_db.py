import os
import sqlite3

DB_FILE = "trains.db"

# 1. Master train metadata (10 trains)
TRAINS_DATA = [
    ("12919", "MALWA EXPRESS", "Superfast Express", "MBDD UTPR", "INDB"),
    ("12002", "BHOPAL SHATABDI", "Shatabdi Express", "NDLS", "RKMP"),
    ("12301", "HOWRAH RAJDHANI", "Rajdhani Express", "HWH", "NDLS"),
    ("12626", "KERALA EXPRESS", "Superfast Express", "NDLS", "TVC"),
    ("12951", "MUMBAI RAJDHANI", "Rajdhani Express", "MMCT", "NDLS"),
    ("12296", "SANGHAMITRA EXP", "Express", "DNR", "SMVB"),
    ("12723", "TELANGANA EXP", "Superfast Express", "HYB", "NDLS"),
    ("12839", "HOWRAH MAIL", "Superfast Express", "HWH", "MAS"),
    ("12615", "GRAND TRUNK EXP", "Superfast Express", "MAS", "NDLS"),
    ("12004", "LKO SHATABDI", "Shatabdi Express", "NDLS", "LJN"),
]

# 2. Complete route data containing all 17 schema fields
ROUTES_DATA = [
    ("12002", 1, "NDLS", "New Delhi", "START", "06:00", 0.0, 28.6424, 77.2195, 0.0, 0.0, "START", "06:00"),
    ("12002", 2, "AGC", "Agra Cantt", "07:50", "07:55", 195.0, 27.1577, 78.0076, 4.5, 110.0, "07:53", "07:58"),
    ("12002", 3, "GWL", "Gwalior Junction", "09:23", "09:28", 313.0, 26.2183, 78.1828, 8.0, 95.5, "09:30", "09:34"),
    ("12002", 4, "VGLJ", "VGL Jhansi Junction", "10:45", "10:50", 411.0, 25.4484, 78.5685, 12.0, 88.0, "10:55", "10:59"),
    ("12002", 5, "RKMP", "Rani Kamalapati", "14:40", "END", 702.0, 23.2201, 77.4362, 10.0, 92.4, "14:50", "END"),
]

def init_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print("Removed existing database file.")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Train master table
    cursor.execute("""
    CREATE TABLE trains (
        train_number TEXT PRIMARY KEY,
        train_name TEXT NOT NULL,
        train_type TEXT NOT NULL,
        source_station TEXT NOT NULL,
        destination_station TEXT NOT NULL
    )
    """)

    # Route details & tracking table
    cursor.execute("""
    CREATE TABLE train_routes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        train_number TEXT NOT NULL,
        station_sequence INTEGER NOT NULL,
        station_code TEXT NOT NULL,
        station_name TEXT NOT NULL,
        scheduled_arrival TEXT,
        scheduled_departure TEXT,
        distance_from_origin REAL NOT NULL,
        latitude REAL,
        longitude REAL,
        historical_delay REAL DEFAULT 0,
        historical_speed REAL DEFAULT 0,
        actual_arrival TEXT,
        actual_departure TEXT,
        FOREIGN KEY (train_number) REFERENCES trains(train_number)
    )
    """)

    # Insert data
    cursor.executemany("""
    INSERT INTO trains (train_number, train_name, train_type, source_station, destination_station)
    VALUES (?, ?, ?, ?, ?)
    """, TRAINS_DATA)

    cursor.executemany("""
    INSERT INTO train_routes (
        train_number, station_sequence, station_code, station_name,
        scheduled_arrival, scheduled_departure, distance_from_origin,
        latitude, longitude, historical_delay, historical_speed,
        actual_arrival, actual_departure
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, ROUTES_DATA)

    conn.commit()
    conn.close()
    print("Database initialized successfully with all 17 schema fields!")

if __name__ == "__main__":
    init_db()