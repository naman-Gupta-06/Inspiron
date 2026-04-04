# ============================================================
# database/alert_db.py
# Persistent storage manager (SQLite WAL mode).
# ============================================================

import sqlite3
import threading
from datetime import datetime, timedelta

from config.settings import (
    DB_NAME,
    DB_ALERT_MAX_AGE_HOURS,
    DB_CLEANUP_INTERVAL_SEC,
    ZONE_LOCK_BUFFER_SEC,
)
from dispatch.geo import haversine_km

_db_lock = threading.Lock()

# ── Schema ───────────────────────────────────────────────────────────────────

def init_db() -> None:
    with sqlite3.connect(DB_NAME) as conn:
        # Enable Write-Ahead Logging for high concurrency
        conn.execute('PRAGMA journal_mode=WAL;')

        conn.execute("""
            CREATE TABLE IF NOT EXISTS stations (
                id INTEGER PRIMARY KEY,
                latitude REAL,
                longitude REAL,
                capacity INTEGER
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS drones (
                id INTEGER PRIMARY KEY,
                station_id INTEGER,
                status TEXT DEFAULT 'idle',
                active_missions INTEGER DEFAULT 0,
                FOREIGN KEY(station_id) REFERENCES stations(id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                camera_id TEXT,
                incident_type TEXT,
                latitude REAL,
                longitude REAL,
                severity REAL,
                confidence REAL,
                timestamp TEXT,
                duration REAL,
                dispatched INTEGER DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS active_dispatches (
                id TEXT PRIMARY KEY,
                alert_id TEXT,
                drone_id INTEGER,
                station_id INTEGER,
                latitude REAL,
                longitude REAL,
                dispatched_at TEXT,
                eta_seconds REAL
            )
        """)
        conn.commit()

# ── Station & Drone Database Operations ──────────────────────────────────────

def insert_station(station_id: int, lat: float, lon: float, capacity: int) -> None:
    with _db_lock:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO stations (id, latitude, longitude, capacity) VALUES (?, ?, ?, ?)",
                (station_id, lat, lon, capacity)
            )
            conn.commit()

def insert_drone(drone_id: int, station_id: int) -> None:
    with _db_lock:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO drones (id, station_id, status, active_missions) VALUES (?, ?, 'idle', 0)",
                (drone_id, station_id)
            )
            conn.commit()

def fetch_all_stations() -> list[dict]:
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM stations").fetchall()
        return [dict(r) for r in rows]

def fetch_drones_for_station(station_id: int) -> list[dict]:
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM drones WHERE station_id = ?", (station_id,)).fetchall()
        # Rename 'active_missions' to 'load_count' to match existing priority.py logic
        return [{"id": r["id"], "status": r["status"], "load_count": r["active_missions"]} for r in rows]

def update_drone_status(drone_id: int, status: str) -> None:
    with _db_lock:
        with sqlite3.connect(DB_NAME) as conn:
            if status == 'busy':
                conn.execute("UPDATE drones SET status = ?, active_missions = active_missions + 1 WHERE id = ?", (status, drone_id))
            else:
                conn.execute("UPDATE drones SET status = ? WHERE id = ?", (status, drone_id))
            conn.commit()

# ── Internal helpers ─────────────────────────────────────────────────────────

def _row_to_dict(row: tuple) -> dict:
    return {
        "id":            row[0],
        "camera_id":     row[1],
        "incident_type": row[2],
        "latitude":      row[3],
        "longitude":     row[4],
        "severity":      row[5],
        "confidence":    row[6],
        "timestamp":     row[7],
        "duration":      row[8],
        "dispatched":    row[9],
    }

# ── alerts write operations ──────────────────────────────────────────────────

def insert_alert(alert: dict) -> None:
    with _db_lock:
        try:
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute(
                    """
                    INSERT INTO alerts
                        (id, camera_id, incident_type, latitude, longitude,
                         severity, confidence, timestamp, duration, dispatched)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        alert["id"], alert["camera_id"], alert["incident_type"],
                        alert["latitude"], alert["longitude"], alert["severity"],
                        alert["confidence"], alert["timestamp"], alert["duration"],
                    ),
                )
                conn.commit()
            print(f"✅ INSERTED: {alert['id']}")
        except Exception as exc:
            print(f"❌ INSERT FAILED: {exc}")

def mark_dispatched(alert_id: str) -> None:
    with _db_lock:
        with sqlite3.connect(DB_NAME) as conn:
            conn.execute("UPDATE alerts SET dispatched = 1 WHERE id = ?", (alert_id,))
            conn.commit()

# ── active_dispatches write operations ───────────────────────────────────────

def insert_active_dispatch(
    dispatch_id: str, alert_id: str, drone_id: int, station_id: int,
    latitude: float, longitude: float, eta_seconds: float
) -> None:
    dispatched_at = datetime.utcnow().isoformat()
    with _db_lock:
        try:
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute(
                    """
                    INSERT INTO active_dispatches
                        (id, alert_id, drone_id, station_id,
                         latitude, longitude, dispatched_at, eta_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (dispatch_id, alert_id, drone_id, station_id, latitude, longitude, dispatched_at, eta_seconds)
                )
                conn.commit()
            print(f"🔒 ZONE LOCKED: dispatch={dispatch_id}  drone={drone_id}  eta={eta_seconds:.0f}s")
        except Exception as exc:
            print(f"❌ ACTIVE DISPATCH INSERT FAILED: {exc}")

def delete_expired_dispatches() -> None:
    now = datetime.utcnow()
    with _db_lock:
        with sqlite3.connect(DB_NAME) as conn:
            rows = conn.execute("SELECT id, dispatched_at, eta_seconds FROM active_dispatches").fetchall()
            expired_ids = []
            for row_id, dispatched_at_str, eta_seconds in rows:
                dispatched_at = datetime.fromisoformat(dispatched_at_str)
                if now >= dispatched_at + timedelta(seconds=eta_seconds):
                    expired_ids.append((row_id,))

            if expired_ids:
                conn.executemany("DELETE FROM active_dispatches WHERE id = ?", expired_ids)
                conn.commit()
                print(f"🔓 ZONE UNLOCKED: released {len(expired_ids)} expired dispatch(es)")

# ── Zone-lock query ───────────────────────────────────────────────────────────

def is_zone_active(latitude: float, longitude: float, threshold_km: float = 0.2) -> bool:
    now = datetime.utcnow()
    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute("SELECT latitude, longitude, dispatched_at, eta_seconds FROM active_dispatches").fetchall()

    for lat, lon, dispatched_at_str, eta_seconds in rows:
        dispatched_at = datetime.fromisoformat(dispatched_at_str)
        if now >= dispatched_at + timedelta(seconds=eta_seconds):
            continue
        if haversine_km(latitude, longitude, lat, lon) <= threshold_km:
            return True
    return False

# ── alerts read operations ───────────────────────────────────────────────────

def fetch_pending_alerts() -> list[dict]:
    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute("SELECT * FROM alerts WHERE dispatched = 0").fetchall()
    return [_row_to_dict(r) for r in rows]

def fetch_all_alerts() -> list[dict]:
    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute("SELECT * FROM alerts").fetchall()
    return [_row_to_dict(r) for r in rows]

def count_alerts() -> int:
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

# ── Cleanup ──────────────────────────────────────────────────────────────────

def delete_old_alerts() -> None:
    cutoff = (datetime.utcnow() - timedelta(hours=DB_ALERT_MAX_AGE_HOURS)).isoformat()
    with _db_lock:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.execute("DELETE FROM alerts WHERE timestamp < ?", (cutoff,))
            conn.commit()
    if cursor.rowcount > 0:
        print(f"🧹 Deleted {cursor.rowcount} old alerts")

def clear_all_data() -> None:
    """Utility function to wipe all dynamic state for testing."""
    with _db_lock:
        with sqlite3.connect(DB_NAME) as conn:
            # Delete all incidents and active flights
            conn.execute("DELETE FROM alerts")
            conn.execute("DELETE FROM active_dispatches")
            # Reset all drones back to available
            conn.execute("UPDATE drones SET status = 'idle', active_missions = 0")
            conn.commit()
    print("🗑️ Database dynamic state cleared (alerts, dispatches, and drone statuses reset).")

import time
def cleanup_worker() -> None:
    while True:
        delete_old_alerts()
        delete_expired_dispatches()
        time.sleep(DB_CLEANUP_INTERVAL_SEC)