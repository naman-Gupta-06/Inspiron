# ============================================================
# state/fleet_state.py
# High-velocity in-memory storage + Battery Simulation.
# ============================================================

import threading
import time
from dispatch.geo import haversine_km

_telemetry_lock = threading.Lock()
_drone_telemetry = {}

def update_telemetry(drone_id: int, lat: float, lon: float, frontend_battery: float = None) -> dict:
    """
    Updates location. If the drone moved, automatically drain 10% battery per km.
    """
    with _telemetry_lock:
        old_data = _drone_telemetry.get(drone_id)
        current_battery = old_data["battery"] if old_data else 100.0

        # If frontend didn't pass a specific battery, we simulate the drain
        if old_data and frontend_battery is None:
            dist_km = haversine_km(old_data["lat"], old_data["lon"], lat, lon)
            if dist_km > 0.0001:  # Only drain if it actually moved
                drain = dist_km * 10.0  # 10% per 1 kilometer
                current_battery = max(0.0, current_battery - drain)
        elif frontend_battery is not None:
            current_battery = frontend_battery

        _drone_telemetry[drone_id] = {
            "lat": lat,
            "lon": lon,
            "battery": current_battery,
            "last_updated": time.time()
        }
        return _drone_telemetry[drone_id].copy()

def charge_idle_drone(drone_id: int, charge_amount: float = 1.0) -> None:
    """Safely increments the battery of an idle drone."""
    with _telemetry_lock:
        if drone_id in _drone_telemetry:
            current = _drone_telemetry[drone_id]["battery"]
            _drone_telemetry[drone_id]["battery"] = min(100.0, current + charge_amount)

def get_telemetry(drone_id: int, default_battery: float = 100.0) -> dict:
    with _telemetry_lock:
        data = _drone_telemetry.get(drone_id)
        if data:
            return data.copy()
        return {"lat": 0.0, "lon": 0.0, "battery": default_battery, "speed": 0.0}

def get_all_telemetry() -> dict:
    with _telemetry_lock:
        return _drone_telemetry.copy()