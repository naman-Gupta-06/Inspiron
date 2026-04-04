# ============================================================
# dispatch/geo.py
# Geographic helper utilities.
# ============================================================

from __future__ import annotations
import math
import threading
import warnings as _warnings
from typing import Optional

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

_pathfinder_instance: Optional[object] = None
_pathfinder_lock = threading.Lock()

def _get_pathfinder():
    global _pathfinder_instance
    if _pathfinder_instance is not None:
        return _pathfinder_instance

    with _pathfinder_lock:
        if _pathfinder_instance is not None:
            return _pathfinder_instance
        try:
            from path_finder.drone_pathfinder_final import PunePathfinder
            _pathfinder_instance = PunePathfinder()
            print("✅ PunePathfinder initialised (shared singleton)")
        except Exception as exc:
            _warnings.warn(f"⚠️  PunePathfinder unavailable — falling back to haversine. Reason: {exc}")
            _pathfinder_instance = None
    return _pathfinder_instance

def pathfinder_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    pf = _get_pathfinder()
    if pf is None:
        return haversine_km(lat1, lon1, lat2, lon2)

    try:
        result = pf.find_path(start=(lat1, lon1), end=(lat2, lon2))
        if result["success"]:
            return result["metrics"]["distance_km"]
        return haversine_km(lat1, lon1, lat2, lon2)
    except Exception:
        return haversine_km(lat1, lon1, lat2, lon2)

def min_pathfinder_distance_km(centres: list[tuple[float, float]], incident_lat: float, incident_lon: float) -> tuple[float, tuple[float, float]]:
    if not centres:
        raise ValueError("centres list is empty")

    best_dist = float("inf")
    best_centre = centres[0]

    for lat, lon in centres:
        dist = pathfinder_distance_km(lat, lon, incident_lat, incident_lon)
        if dist < best_dist:
            best_dist = dist
            best_centre = (lat, lon)

    return best_dist, best_centre