# ============================================================
# main.py
# Application entry point.
# ============================================================

import threading
import time

from config.settings import DISPATCH_LOOP_INTERVAL_SEC
# ADDED: fetch_drones_for_station for the battery worker
from database.alert_db import init_db, cleanup_worker, insert_station, insert_drone, fetch_all_stations, clear_all_data, fetch_drones_for_station
from state import fleet_state
from detection.detector import run_detection
from dispatch.priority import run_priority_dispatch
from api.server import run_server

# ── Camera definitions ────────────────────────────────────────────────────────

CAMERAS = [
    {"id": "cam_1", "source": "media/input1.mp4", "lat": 18.5204, "lon": 73.8567},
    {"id": "cam_2", "source": "media/input2.mp4", "lat": 18.5210, "lon": 73.8575},
    {"id": "cam_3", "source": "media/input3.mp4", "lat": 18.5195, "lon": 73.8550},
]

# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _seed_database():
    """Ensure stations and drones exist in the DB on startup."""
    if len(fetch_all_stations()) == 0:
        print("🌱 Seeding database with initial stations and drones...")
        
        # Station 1
        insert_station(1, 18.5204, 73.8567, 2)
        insert_drone(1, 1)
        insert_drone(2, 1)
        fleet_state.update_telemetry(1, 18.5204, 73.8567, 100.0)
        fleet_state.update_telemetry(2, 18.5204, 73.8567, 100.0)

        # Station 2
        insert_station(2, 18.5215, 73.8580, 1)
        insert_drone(3, 2)
        fleet_state.update_telemetry(3, 18.5215, 73.8580, 100.0)

def _priority_loop() -> None:
    """Background thread: run dispatch every DISPATCH_LOOP_INTERVAL_SEC seconds."""
    while True:
        run_priority_dispatch()
        time.sleep(DISPATCH_LOOP_INTERVAL_SEC)

# ADDED: Background battery charger
def _battery_charging_worker() -> None:
    """Daemon thread: Slowly charges drones sitting idle at their stations."""
    while True:
        stations = fetch_all_stations()
        for s in stations:
            drones = fetch_drones_for_station(s["id"])
            for d in drones:
                # If the drone is sitting at the station, charge it by 1% every 5 seconds
                if d["status"] == "idle":
                    fleet_state.charge_idle_drone(d["id"], charge_amount=1.0)
        time.sleep(5)

def main() -> None:
    # 1. Initialise the database schema & seed initial infrastructure.
    init_db()
    _seed_database()

    clear_all_data()  # Clear any leftover alerts from previous runs

    # 2. Start the DB cleanup daemon.
    threading.Thread(target=cleanup_worker, daemon=True, name="db-cleanup").start()

    # ADDED: Start the new autonomous battery charger
    threading.Thread(target=_battery_charging_worker, daemon=True, name="battery-charger").start()

    # 3. Start one detection thread per camera.
    for cam in CAMERAS:
        threading.Thread(
            target=run_detection,
            args=(cam["source"], cam["id"], cam["lat"], cam["lon"]),
            daemon=True,
            name=f"detect-{cam['id']}",
        ).start()

    # 4. Start the priority dispatch loop.
    threading.Thread(target=_priority_loop, daemon=True, name="dispatch-loop").start()

    # 5. Start the Flask-SocketIO Server (This blocks the main thread)
    run_server()

if __name__ == "__main__":
    main()