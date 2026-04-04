# ============================================================
# dispatch/priority.py
# Alert clustering, drone scoring, and priority dispatch.
# ============================================================

import heapq
import uuid
import requests

from config.settings import (
    BATTERY_PER_KM, HOVER_PER_MIN, RECORDING_TIME_MIN, BATTERY_RESERVE,
    STATION_RADIUS_KM, BATTERY_WEIGHT, LOAD_WEIGHT, CLUSTER_THRESHOLD_KM,
    DRONE_SPEED_KMH, ZONE_LOCK_BUFFER_SEC
)
from database.alert_db import (
    fetch_pending_alerts, mark_dispatched, insert_active_dispatch, is_zone_active,
    fetch_all_stations, fetch_drones_for_station, update_drone_status
)
from dispatch.geo import haversine_km, pathfinder_distance_km
from state import fleet_state

def _build_live_station_view() -> list[dict]:
    stations = fetch_all_stations()
    for s in stations:
        s["lat"] = s.pop("latitude")
        s["lon"] = s.pop("longitude")
        db_drones = fetch_drones_for_station(s["id"])
        s["drones"] = []
        for d in db_drones:
            telemetry = fleet_state.get_telemetry(d["id"])
            d["battery"] = telemetry["battery"]
            s["drones"].append(d)
    return stations

def cluster_alerts(alerts: list[dict], threshold_km: float = CLUSTER_THRESHOLD_KM) -> list[dict]:
    clusters: list[list[dict]] = []
    for alert in alerts:
        placed = False
        for cluster in clusters:
            if haversine_km(alert["latitude"], alert["longitude"], cluster[0]["latitude"], cluster[0]["longitude"]) < threshold_km:
                cluster.append(alert)
                placed = True
                break
        if not placed: clusters.append([alert])

    merged = []
    for cluster in clusters:
        best = max(cluster, key=lambda a: a["severity"])
        rep = best.copy()
        rep["cluster_size"] = len(cluster)
        rep["avg_severity"] = sum(a["severity"] for a in cluster) / len(cluster)
        merged.append(rep)
    return merged

def _required_battery(distance_km: float) -> float:
    return (2 * distance_km * BATTERY_PER_KM) + (RECORDING_TIME_MIN * HOVER_PER_MIN) + BATTERY_RESERVE

def select_drone_from_station(station: dict, distance_km: float) -> dict | None:
    best_drone = None
    best_score = float("inf")
    for drone in station["drones"]:
        if drone["status"] != "idle" or drone["battery"] < _required_battery(distance_km):
            continue
        score = (BATTERY_WEIGHT * (100 - drone["battery"])) + (LOAD_WEIGHT * drone["load_count"])
        if score < best_score:
            best_score = score
            best_drone = drone
    return best_drone

def _execute_dispatch(station, drone, distance, alert):
    """Helper to log the dispatch and broadcast to frontend."""
    eta = ((distance / DRONE_SPEED_KMH) * 3600 * 2) + (RECORDING_TIME_MIN * 60) + ZONE_LOCK_BUFFER_SEC
    did = uuid.uuid4().hex
    
    insert_active_dispatch(did, alert["id"], drone["id"], station["id"], alert["latitude"], alert["longitude"], eta)
    result = {
        "dispatch_id": did, "alert_id": alert["id"], "incident_type": alert["incident_type"],
        "severity": alert["severity"], "station_id": station["id"], "drone_id": drone["id"],
        "distance_km": round(distance, 3), "eta_seconds": round(eta, 1)
    }
    
    try:
        requests.post('http://127.0.0.1:5000/internal/dispatch', json=result)
    except requests.exceptions.ConnectionError: pass
    return result

def dispatch_single(stations: list[dict], alert: dict, all_pending_alerts: list[dict]) -> dict | None:
    ilat, ilon = alert["latitude"], alert["longitude"]

    if is_zone_active(ilat, ilon, threshold_km=CLUSTER_THRESHOLD_KM):
        return None

    # ── Phase 1: Try Local Stations ──────────────────────────────────────────
    reachable = [s for s in stations if haversine_km(s["lat"], s["lon"], ilat, ilon) <= STATION_RADIUS_KM]
    station_distances = sorted([(s, pathfinder_distance_km(s["lat"], s["lon"], ilat, ilon)) for s in reachable], key=lambda x: x[1])

    for station, distance in station_distances:
        drone = select_drone_from_station(station, distance)
        if drone: return _execute_dispatch(station, drone, distance, alert)

    # ── Phase 2: Cross-Station Stealing ──────────────────────────────────────
    unreachable = [s for s in stations if s not in reachable]
    global_distances = []
    
    # Pre-filter stations that are impossibly far (> 15km) to save Pathfinder processing
    for station in unreachable:
        if haversine_km(station["lat"], station["lon"], ilat, ilon) <= 15.0:
            global_distances.append((station, pathfinder_distance_km(station["lat"], station["lon"], ilat, ilon)))
            
    global_distances.sort(key=lambda x: x[1])

    for station, distance in global_distances:
        # Prevent stealing if this station has its OWN pending emergencies nearby
        has_local_alerts = any(
            haversine_km(station["lat"], station["lon"], a["latitude"], a["longitude"]) <= STATION_RADIUS_KM
            for a in all_pending_alerts if a["id"] != alert["id"]
        )
        if has_local_alerts:
            continue # Leave this drone here, it's needed locally!

        drone = select_drone_from_station(station, distance)
        if drone:
            # Broadcast a warning log that cross-station dispatch occurred
            msg = f"⚠️ CROSS-STATION DISPATCH: Drone {drone['id']} assigned to distant alert {alert['id']}."
            try: requests.post('http://127.0.0.1:5000/internal/log', json={"message": msg, "level": "WARNING"})
            except: pass
            return _execute_dispatch(station, drone, distance, alert)

    return None

def priority_dispatch(stations: list[dict], alerts: list[dict]) -> list[dict]:
    # We pass the unclustered list of alerts down so Phase 2 knows global conditions
    all_pending = alerts.copy() 
    
    alerts = cluster_alerts(alerts)
    heap = [(-a["severity"], a["id"], a) for a in alerts]
    heapq.heapify(heap)

    results = []
    while heap:
        _, _, alert = heapq.heappop(heap)
        result = dispatch_single(stations, alert, all_pending)
        if result:
            results.append(result)
            update_drone_status(result["drone_id"], "busy")
            mark_dispatched(alert["id"])
    return results

def run_priority_dispatch() -> None:
    alerts = fetch_pending_alerts()
    if not alerts: return
    stations = _build_live_station_view()
    priority_dispatch(stations, alerts)