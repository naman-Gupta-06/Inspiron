# ============================================================
# detection/generate_alert.py
# Pure functions for constructing alert payloads.
# ============================================================

import time
import uuid
from datetime import datetime

from config.settings import GPS_OFFSET_LAT, GPS_OFFSET_LON

def generate_alert_id(camera_id: str) -> str:
    timestamp = int(time.time())
    short_hash = uuid.uuid4().hex[:4]
    return f"{camera_id}_{timestamp}_{short_hash}"

def calculate_severity(incident_type: str, detections: list[dict]) -> float:
    if incident_type == "crowd":
        persons = sum(1 for d in detections if d["class"] == "person")
        return min(1.0, persons / 15)
    if incident_type == "accident":
        return 0.9
    if incident_type == "intrusion":
        return 0.7
    return 0.5

def calculate_confidence(detections: list[dict]) -> float:
    if not detections:
        return 0.0
    return sum(d["confidence"] for d in detections) / len(detections)

def create_alert(
    camera_id: str, incident_type: str, detections: list[dict],
    camera_lat: float, camera_lon: float, duration: float,
) -> dict:
    return {
        "id":            generate_alert_id(camera_id),
        "camera_id":     camera_id,
        "incident_type": incident_type,
        "latitude":      camera_lat + GPS_OFFSET_LAT,
        "longitude":     camera_lon + GPS_OFFSET_LON,
        "severity":      round(calculate_severity(incident_type, detections), 2),
        "confidence":    round(calculate_confidence(detections), 2),
        "timestamp":     datetime.utcnow().isoformat(),
        "duration":      round(duration, 2),
    }