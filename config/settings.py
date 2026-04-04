# ============================================================
# config/settings.py
# Central configuration for all system parameters.
# ============================================================

# ── Database ────────────────────────────────────────────────
DB_NAME = "database/alerts.db"
DB_CLEANUP_INTERVAL_SEC = 3600
DB_ALERT_MAX_AGE_HOURS = 24

# ── Drone energy model ──────────────────────────────────────
BATTERY_PER_KM = 10
HOVER_PER_MIN = 3
RECORDING_TIME_MIN = 5
BATTERY_RESERVE = 15

# ── Dispatch scoring weights ────────────────────────────────
BATTERY_WEIGHT = 0.1
LOAD_WEIGHT = 2.0

# ── Station coverage ────────────────────────────────────────
STATION_RADIUS_KM = 2.5

# ── Alert clustering ────────────────────────────────────────
CLUSTER_THRESHOLD_KM = 0.2

# ── Detection sliding-window ────────────────────────────────
WINDOW_SIZE = 30
ACCIDENT_FRAME_THRESHOLD_LOW = 5
ACCIDENT_FRAME_THRESHOLD_HIGH = 10

# ── Crowd density thresholds (heads / pixel) ────────────────
LOW_DENSITY = 0.00002
MEDIUM_DENSITY = 0.00008
HIGH_DENSITY = 0.00015

# ── Alert generation ────────────────────────────────────────
ALERT_COOLDOWN_SEC = 5

# ── YOLO model confidence ───────────────────────────────────
MODEL_CONFIDENCE = 0.3

# ── Priority dispatch loop ──────────────────────────────────
DISPATCH_LOOP_INTERVAL_SEC = 2

# ── Camera GPS offset ───────────────────────────────────────
GPS_OFFSET_LAT = 0.0001
GPS_OFFSET_LON = 0.0001

# ── Zone locking ────────────────────────────────────────────
DRONE_SPEED_KMH = 60
ZONE_LOCK_BUFFER_SEC = 60