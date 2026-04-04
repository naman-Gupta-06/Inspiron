# ============================================================
# detection/detector.py
# Per-camera YOLO inference loop with WebSocket triggers.
# ============================================================

import time
import requests
from collections import deque

import cv2
from ultralytics import YOLO

from detection.generate_alert import create_alert
from database.alert_db import insert_alert
from config.settings import (
    WINDOW_SIZE,
    ACCIDENT_FRAME_THRESHOLD_LOW,
    ACCIDENT_FRAME_THRESHOLD_HIGH,
    LOW_DENSITY,
    MEDIUM_DENSITY,
    HIGH_DENSITY,
    ALERT_COOLDOWN_SEC,
    MODEL_CONFIDENCE,
)

# ── Severity helpers ─────────────────────────────────────────────────────────

def _accident_severity(accident_count: int) -> float:
    if accident_count < ACCIDENT_FRAME_THRESHOLD_LOW:
        return 0.0
    if accident_count < ACCIDENT_FRAME_THRESHOLD_HIGH:
        return min(1.0, 0.3 + (accident_count / WINDOW_SIZE) * 0.4)
    return min(1.0, 0.7 + (accident_count / WINDOW_SIZE) * 0.3)

def _crowd_severity(avg_density: float) -> float:
    if avg_density < LOW_DENSITY:
        return 0.0
    if avg_density < MEDIUM_DENSITY:
        return min(1.0, 0.3 + (avg_density / HIGH_DENSITY) * 0.4)
    if avg_density < HIGH_DENSITY:
        return min(1.0, 0.6 + (avg_density / HIGH_DENSITY) * 0.3)
    return min(1.0, 0.9 + (avg_density / (2 * HIGH_DENSITY)) * 0.1)

# ── Alert builders ───────────────────────────────────────────────────────────

def _build_accident_alert(
    camera_id: str, cam_lat: float, cam_lon: float,
    acc_detections: list[dict], accident_window: deque, accident_severity: float,
) -> dict:
    alert = create_alert(camera_id, "accident", acc_detections, cam_lat, cam_lon, len(accident_window) / WINDOW_SIZE)
    alert["severity"] = accident_severity
    alert["location_key"] = f"{round(cam_lat, 4)}_{round(cam_lon, 4)}"
    return alert

def _build_crowd_alert(
    camera_id: str, cam_lat: float, cam_lon: float,
    crowd_detections: list[dict], density_window: deque, crowd_severity: float, head_count: int,
) -> dict:
    alert = create_alert(camera_id, "crowd", crowd_detections, cam_lat, cam_lon, len(density_window) / WINDOW_SIZE)
    alert["severity"] = crowd_severity
    alert["head_count"] = head_count
    alert["location_key"] = f"{round(cam_lat, 4)}_{round(cam_lon, 4)}"
    return alert

# ── Main detection loop ──────────────────────────────────────────────────────

def run_detection(video_path: str, camera_id: str, cam_lat: float, cam_lon: float) -> None:
    print(f"🎥 Starting detection for {camera_id}")

    accident_model = YOLO("models/accident.pt")
    crowd_model = YOLO("models/crowd.pt")

    accident_window: deque = deque(maxlen=WINDOW_SIZE)
    density_window: deque = deque(maxlen=WINDOW_SIZE)
    last_alert_time: float = 0.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video for {camera_id}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️  Stream ended for {camera_id}")
            break

        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w

        # ── Accident inference ───────────────────────────────────────────────
        acc_results = accident_model(frame, conf=MODEL_CONFIDENCE)
        accident_detected = False
        acc_detections: list[dict] = []

        for box in acc_results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = accident_model.names[cls_id]
            acc_detections.append({"class": label, "confidence": conf})
            if label.lower().startswith("accident"):
                accident_detected = True

        accident_window.append(1 if accident_detected else 0)
        accident_count = sum(accident_window)
        acc_severity = round(_accident_severity(accident_count), 2)

        # ── Crowd inference ──────────────────────────────────────────────────
        crowd_results = crowd_model(frame, conf=MODEL_CONFIDENCE)
        head_count = 0
        crowd_detections: list[dict] = []

        for box in crowd_results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = crowd_model.names[cls_id]
            crowd_detections.append({"class": label, "confidence": conf})
            if label.lower() == "head":
                head_count += 1

        density = head_count / frame_area
        density_window.append(density)
        avg_density = sum(density_window) / len(density_window)
        crw_severity = round(_crowd_severity(avg_density), 2)

        # ── Alert gating & Webhook ───────────────────────────────────────────
        current_time = time.time()
        if current_time - last_alert_time <= ALERT_COOLDOWN_SEC:
            continue

        alert_payload = None

        if accident_count >= ACCIDENT_FRAME_THRESHOLD_LOW:
            alert_payload = _build_accident_alert(camera_id, cam_lat, cam_lon, acc_detections, accident_window, acc_severity)
        elif crw_severity > 0:
            alert_payload = _build_crowd_alert(camera_id, cam_lat, cam_lon, crowd_detections, density_window, crw_severity, head_count)

        if alert_payload:
            insert_alert(alert_payload)
            print(f"\n🚨 NEW ALERT ({camera_id}):\n{alert_payload}")
            
            # Broadcast to React Frontend via internal WebSocket server
            try:
                requests.post('http://127.0.0.1:5000/internal/alert', json=alert_payload)
            except requests.exceptions.ConnectionError:
                pass # Server might not be ready yet, fail silently
            
            last_alert_time = current_time

    cap.release()