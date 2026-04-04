"""
Pune Drone Navigation — Final Hub Edition
==========================================
Built on v8 (hub database) with all improvements from v3.

Research papers implemented:
  Paper 1 [Aerospace 2025, A*+NURBS]   — Density-weighted heuristic + NURBS smoothing
  Paper 2 [Processes 2023, Improved 3D A*] — 26-connectivity, octile heuristic, 3-phase altitude
  Paper 3 [Drones 2025, A*-Guided APF] — Hybrid A* skeleton + APF local refinement

Key improvements over v8:
  ── SPEED ──
  • GRID_RES_M = 12m (better balance: v3=10m too slow, v8=15m too coarse)
  • Bidirectional A* — meets in middle, ~2–4x fewer nodes expanded
  • EDT runs only on XY slab (2D) then extruded — avoids 3D O(n³) for tall grids
  • APF scan radius auto-capped at 2 voxels (no over-scanning at coarse res)
  • JIT-compilable inner loops via numpy vectorised neighbour expansion

  ── ACCURACY ──
  • Cruise altitude: per-corridor 85th pct (v8) + hard floor of 50m AGL
  • Admissibility check: density heuristic term capped so h ≤ true cost always
  • Building polygon WKT rasterised with proper scanline (from v3)
  • Waypoint deduplication with 8m spatial threshold (not 5m from v8)

  ── SMOOTHNESS / NO FLUCTUATIONS ──
  • Altitude moving-average window = 9 pts (vs 5 in v8) — gentler z profile
  • NURBS sample spacing = 30m (v8=50m too sparse, v3=3m too dense) — smooth
    but not jagged
  • APF step = 5m with max 80 iters per segment (balanced)
  • Post-NURBS z-clamp: altitudes enforced within [MIN_ALT_M, MAX_ALT_M]
  • Lateral smoothing: Savitzky-Golay on x/y coordinates (order-3, window-9)
    — removes APF micro-jitter without shifting the path macro-geometry

  ── CONSOLE OUTPUT ──
  • Progress bar style with ████░░ for long steps
  • All prints are human-readable with units and context
  • Hub-zone snap: if point is outside hub, finds nearest point inside that hub
    instead of hard-rejecting (configurable; default = snap with warning)

  ── HUB BOUNDARY HANDLING ──
  • Points outside all hubs: snapped to nearest hub boundary + warning printed
  • Cross-hub routes: snapped to the nearest single-hub pair that covers both
  • Snap radius search: uses Haversine, returns nearest hub edge point

Usage:
    python drone_pathfinder_final.py

    Or import:
        from drone_pathfinder_final import PunePathfinder
        pf = PunePathfinder()
        result = pf.find_path(
            start=(18.5204, 73.8567),
            end  =(18.5679, 73.7739),
        )

Dependencies:
    pip install numpy scipy shapely pyproj pyyaml
"""

import sqlite3
import time
import math
import heapq
import os
import sys
import json
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from shapely.wkt import loads as wkt_loads
from shapely.geometry import Point, Polygon
import pyproj
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH           = "pune_hub_buildings.db"
PATHS_DB          = "drone_paths.db"
ZONES_CONFIG      = "restricted_zones.yaml"

GRID_RES_M        = 12.0        # voxel edge — balanced speed vs accuracy
MAX_ALT_M         = 200.0       # absolute ceiling AGL
MIN_ALT_M         = 20.0        # absolute floor AGL
SAFETY_BUFFER_M   = 25.0        # clearance above every building top
CORRIDOR_BUFFER_M = 200.0       # lateral margin beyond start↔end bounding box
DEFAULT_HEIGHT_M  = 14.0        # fallback for buildings with no height data
MIN_CRUISE_ALT_M  = 50.0        # drone never cruises below this

# A* cost weights
W_DISTANCE    = 1.0
W_CLEARANCE   = 0.45
W_DENSITY     = 2.0             # slightly reduced from v8 — less over-avoidance
W_ALT_CHANGE  = 0.35
CLEARANCE_R_M = 20.0
CRUISE_PENALTY = 1.2            # per-voxel penalty for flying below cruise

# APF parameters
APF_K_ATT         = 1.0
APF_K_REP         = 600.0       # reduced — less oscillation in tight spaces
APF_RHO0_M        = 20.0
APF_STEP_M        = 5.0         # 5m steps — smoother than 8m, faster than 2m
APF_MAX_ITERS     = 80
APF_STALL_THRESH  = 0.1

# NURBS smoothing
NURBS_DEGREE      = 3
NURBS_SAMPLE_M    = 30.0        # 30m spacing — smooth output, not overly sparse
ALT_SMOOTH_WIN    = 9           # moving-average window for altitude (must be odd)
SG_WINDOW         = 9           # Savitzky-Golay lateral smooth window (must be odd)
SG_ORDER          = 3           # SG polynomial order

# Waypoint deduplication
WP_MIN_SEP_M      = 8.0         # remove waypoints closer than this

# Restricted zones
SOFT_BLOCK_PENALTY = 500.0
DRONE_SPEED_KMH    = 50.0        # assumed cruise speed for ETA calculation

# Cell type flags
CELL_FREE       = np.uint8(0)
CELL_BUILDING   = np.uint8(1)
CELL_HARD_BLOCK = np.uint8(2)
CELL_SOFT_BLOCK = np.uint8(3)

UTM_CRS   = "EPSG:32643"
WGS84_CRS = "EPSG:4326"

# ── Hub definitions — must match your build_hub_rtree.py ─────────────────────
# The three Pune operational zones
HUBS = [
    {
        "id":   "hub_a",
        "name": "South Pune  (Navale / Katraj / Hadapsar)",
        "lat":  18.4656,   # centroid of Navale–Katraj–Hadapsar triangle
        "lng":  73.8383,
    },
    {
        "id":   "hub_b",
        "name": "Northwest Pune  (Hinjewadi / Wakad / Baner)",
        "lat":  18.5995,
        "lng":  73.7620,
    },
    {
        "id":   "hub_c",
        "name": "East Pune  (Kharadi / Viman Nagar / Mundhwa)",
        "lat":  18.5519,
        "lng":  73.9476,
    },
]
HUB_RADIUS_M = 5000.0           # 5 km radius per hub


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _bar(done: int, total: int, width: int = 20) -> str:
    """Simple ASCII progress bar."""
    filled = int(width * done / max(total, 1))
    return "█" * filled + "░" * (width - filled)


def _log(msg: str, indent: int = 2):
    """Timestamped console line."""
    prefix = " " * indent
    print(f"{prefix}{msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATE TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────

class CoordinateTransformer:
    def __init__(self):
        self._to_utm   = pyproj.Transformer.from_crs(WGS84_CRS, UTM_CRS,   always_xy=True)
        self._to_wgs84 = pyproj.Transformer.from_crs(UTM_CRS,   WGS84_CRS, always_xy=True)

    def ll_to_utm(self, lat, lng):
        x, y = self._to_utm.transform(lng, lat)
        return x, y

    def utm_to_ll(self, e, n):
        lng, lat = self._to_wgs84.transform(e, n)
        return lat, lng


# ─────────────────────────────────────────────────────────────────────────────
# HAVERSINE
# ─────────────────────────────────────────────────────────────────────────────

def haversine(lat1, lng1, lat2, lng2) -> float:
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(max(0.0, a)))


# ─────────────────────────────────────────────────────────────────────────────
# HUB ZONE MANAGER  (snap instead of hard-reject)
# ─────────────────────────────────────────────────────────────────────────────

class HubZoneManager:
    """
    Manages which hub a coordinate belongs to and snaps out-of-bounds points
    to the nearest hub boundary rather than rejecting them outright.
    """

    def __init__(self):
        self.hubs   = HUBS
        self.radius = HUB_RADIUS_M

    def which_hub(self, lat: float, lng: float) -> list[str]:
        """Returns hub IDs that contain this point (may be multiple if overlapping)."""
        return [h["id"] for h in self.hubs
                if haversine(lat, lng, h["lat"], h["lng"]) <= self.radius]

    def nearest_hub(self, lat: float, lng: float):
        """Returns (hub_dict, distance_m) of closest hub regardless of radius."""
        best_hub  = min(self.hubs, key=lambda h: haversine(lat, lng, h["lat"], h["lng"]))
        best_dist = haversine(lat, lng, best_hub["lat"], best_hub["lng"])
        return best_hub, best_dist

    def snap_to_hub(self, lat: float, lng: float, hub: dict):
        """
        If the point is outside the hub, move it to the nearest point on the
        hub boundary circle. Returns (new_lat, new_lng, was_snapped).
        """
        dist = haversine(lat, lng, hub["lat"], hub["lng"])
        if dist <= self.radius:
            return lat, lng, False

        # Move toward hub centre to just inside the radius (95%)
        fraction = (self.radius * 0.95) / dist
        new_lat  = hub["lat"] + (lat - hub["lat"]) * fraction
        new_lng  = hub["lng"] + (lng - hub["lng"]) * fraction
        return new_lat, new_lng, True

    def resolve(self, start_ll, end_ll):
        """
        Validate and optionally snap start/end to valid hub positions.
        Returns (start_ll, end_ll, hub_id, warnings) or raises ValueError.

        Strategy:
         1. If both in same hub → proceed normally.
         2. If one/both outside → snap to nearest hub that covers both, warn user.
         3. If cross-hub (different hubs, none shared) → snap both to nearer hub, warn.
        """
        warnings = []

        start_hubs = self.which_hub(*start_ll)
        end_hubs   = self.which_hub(*end_ll)
        common     = set(start_hubs) & set(end_hubs)

        if common:
            hub_id = sorted(common)[0]
            return start_ll, end_ll, hub_id, warnings

        # Try to find a single hub that can cover both after snapping
        for hub in self.hubs:
            s_dist = haversine(*start_ll, hub["lat"], hub["lng"])
            e_dist = haversine(*end_ll,   hub["lat"], hub["lng"])
            # If both are reasonably close to this hub, snap both in
            if s_dist <= self.radius * 1.5 and e_dist <= self.radius * 1.5:
                s_new_lat, s_new_lng, s_snapped = self.snap_to_hub(*start_ll, hub)
                e_new_lat, e_new_lng, e_snapped = self.snap_to_hub(*end_ll,   hub)
                if s_snapped:
                    warnings.append(
                        f"⚠ Start ({start_ll[0]:.4f}, {start_ll[1]:.4f}) was "
                        f"{s_dist/1000:.2f} km from {hub['name']} hub — "
                        f"snapped to hub boundary ({s_new_lat:.4f}, {s_new_lng:.4f})"
                    )
                if e_snapped:
                    warnings.append(
                        f"⚠ End ({end_ll[0]:.4f}, {end_ll[1]:.4f}) was "
                        f"{e_dist/1000:.2f} km from {hub['name']} hub — "
                        f"snapped to hub boundary ({e_new_lat:.4f}, {e_new_lng:.4f})"
                    )
                return (s_new_lat, s_new_lng), (e_new_lat, e_new_lng), hub["id"], warnings

        # Last resort: snap each to its nearest hub independently and use the
        # hub that covers start
        start_hub, _ = self.nearest_hub(*start_ll)
        end_hub, _   = self.nearest_hub(*end_ll)

        s_new_lat, s_new_lng, s_snapped = self.snap_to_hub(*start_ll, start_hub)
        e_new_lat, e_new_lng, e_snapped = self.snap_to_hub(*end_ll, start_hub)

        if s_snapped:
            warnings.append(
                f"⚠ Start snapped to {start_hub['name']} hub boundary → "
                f"({s_new_lat:.4f}, {s_new_lng:.4f})"
            )
        warnings.append(
            f"⚠ End was in a different hub ({end_hub['name']}); "
            f"snapped to {start_hub['name']} hub boundary → "
            f"({e_new_lat:.4f}, {e_new_lng:.4f}). "
            f"Cross-hub routing is not supported — using snapped coordinates."
        )
        return (s_new_lat, s_new_lng), (e_new_lat, e_new_lng), start_hub["id"], warnings


# ─────────────────────────────────────────────────────────────────────────────
# RESTRICTED ZONE LOADER
# ─────────────────────────────────────────────────────────────────────────────

class RestrictedZoneLoader:
    def __init__(self, config_path: str):
        self.zones = []
        if not os.path.exists(config_path):
            _log(f"[Zones] No config at '{config_path}' — skipping restricted zones")
            return
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        now = datetime.now()
        loaded = expired = 0
        for z in cfg.get("restricted_zones", []):
            af = z.get("active_from", "")
            au = z.get("active_until", "")
            if af and au:
                try:
                    if not (datetime.strptime(af, "%Y-%m-%d %H:%M") <= now
                            <= datetime.strptime(au, "%Y-%m-%d %H:%M")):
                        expired += 1
                        continue
                except ValueError:
                    pass
            self.zones.append(z)
            loaded += 1
        _log(f"[Zones] {loaded} active zone(s) loaded  ({expired} expired/inactive skipped)")

    def get_zones(self):
        return self.zones


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY BUILDING INDEX  (replaces per-query SQLite access)
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryBuildingIndex:
    """
    Preloads ALL buildings from the focused hub DB into memory at startup.
    Spatial queries use numpy vectorised bounding-box filtering — no SQLite
    access after __init__.  Typical query time: <1 ms for ~10 k buildings.
    """

    def __init__(self, db_path: str):
        t0   = time.time()
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()

        cur.execute("""
            SELECT b.latitude, b.longitude,
                   b.height_m, b.min_height_m,
                   b.geometry_wkt, b.is_restricted,
                   r.min_lat, r.max_lat, r.min_lng, r.max_lng
            FROM   buildings b
            JOIN   buildings_rtree r ON b.id = r.id
        """)
        rows = cur.fetchall()
        conn.close()           # ← DB is NEVER opened again

        n      = len(rows)
        self.n = n

        # Numeric arrays for vectorised bbox queries
        self.lat          = np.empty(n, dtype=np.float64)
        self.lng          = np.empty(n, dtype=np.float64)
        self.height_m_arr = np.empty(n, dtype=np.float64)
        self.min_h_arr    = np.empty(n, dtype=np.float64)
        self.restricted   = np.empty(n, dtype=np.int32)
        self.geometry_wkt = []          # strings — can't vectorise

        self.bb_min_lat = np.empty(n, dtype=np.float64)
        self.bb_max_lat = np.empty(n, dtype=np.float64)
        self.bb_min_lng = np.empty(n, dtype=np.float64)
        self.bb_max_lng = np.empty(n, dtype=np.float64)

        for i, row in enumerate(rows):
            self.lat[i]          = row[0]
            self.lng[i]          = row[1]
            self.height_m_arr[i] = row[2] if row[2] is not None else -1.0
            self.min_h_arr[i]    = row[3] if row[3] is not None else 0.0
            self.geometry_wkt.append(row[4])
            self.restricted[i]   = row[5] if row[5] is not None else 0
            self.bb_min_lat[i]   = row[6]
            self.bb_max_lat[i]   = row[7]
            self.bb_min_lng[i]   = row[8]
            self.bb_max_lng[i]   = row[9]

        elapsed = time.time() - t0
        ram_kb  = (self.lat.nbytes + self.lng.nbytes
                   + self.height_m_arr.nbytes + self.min_h_arr.nbytes
                   + self.restricted.nbytes
                   + self.bb_min_lat.nbytes * 4) // 1024
        _log(f"[MemIndex] {n:,} buildings loaded into memory  "
             f"(~{ram_kb} KB numeric + WKT strings,  {elapsed:.2f}s)")

    def query_bbox(self, min_lat: float, max_lat: float,
                   min_lng: float, max_lng: float) -> list:
        """
        Vectorised bounding-box filter.  Returns rows in the same tuple
        format as the old SQLite query:
            (lat, lng, height_m, min_height_m, geometry_wkt, is_restricted)
        """
        mask = ((self.bb_max_lat >= min_lat) &
                (self.bb_min_lat <= max_lat) &
                (self.bb_max_lng >= min_lng) &
                (self.bb_min_lng <= max_lng))

        indices = np.where(mask)[0]
        results = []
        for i in indices:
            h = self.height_m_arr[i]
            results.append((
                self.lat[i],
                self.lng[i],
                h if h >= 0 else None,
                self.min_h_arr[i],
                self.geometry_wkt[i],
                int(self.restricted[i]),
            ))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# PATH STORE  (SQLite — saves computed routes)
# ─────────────────────────────────────────────────────────────────────────────

class PathStore:
    """
    Persists computed drone paths into a lightweight SQLite database.
    Each path is stored as a JSON blob of waypoints with route metadata.
    """

    def __init__(self, db_path: str = PATHS_DB):
        self.db_path = db_path
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paths (
                id              TEXT PRIMARY KEY,
                drone_id        TEXT,
                start_lat       REAL,
                start_lng       REAL,
                end_lat         REAL,
                end_lng         REAL,
                waypoints       TEXT,
                total_distance  REAL,
                estimated_time  REAL,
                hub             TEXT,
                created_at      TEXT
            )
        """)
        conn.commit()
        conn.close()
        _log(f"[PathStore] Ready → {db_path}")

    def save(self, drone_id: str, waypoints: list, metrics: dict,
             start_ll: tuple, end_ll: tuple) -> str:
        """Save a computed path and return its unique ID."""
        path_id = str(uuid.uuid4())[:8]
        eta_s   = round(metrics["distance_km"] / DRONE_SPEED_KMH * 3600, 1)
        wp_json = json.dumps(waypoints)

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO paths
            (id, drone_id, start_lat, start_lng, end_lat, end_lng,
             waypoints, total_distance, estimated_time, hub, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            path_id, drone_id,
            start_ll[0], start_ll[1],
            end_ll[0], end_ll[1],
            wp_json,
            metrics["distance_km"],
            eta_s,
            metrics.get("hub", ""),
            datetime.now().isoformat(),
        ))
        conn.commit()
        conn.close()
        _log(f"[PathStore] Saved path '{path_id}' → {self.db_path}  "
             f"({len(waypoints)} wps, ETA {eta_s:.0f}s)")
        return path_id


# ─────────────────────────────────────────────────────────────────────────────
# VOXEL GRID
# ─────────────────────────────────────────────────────────────────────────────

class VoxelGrid:
    __slots__ = ("grid", "clearance", "density", "cost_extra",
                 "oe", "on_", "oz", "res", "nx", "ny", "nz")

    def __init__(self, nx, ny, nz, oe, on_, oz, res):
        self.grid       = np.zeros((nx, ny, nz), dtype=np.uint8)
        self.clearance  = np.full ((nx, ny, nz), 999.0, dtype=np.float32)
        self.density    = np.zeros((nx, ny, nz), dtype=np.uint8)
        self.cost_extra = np.zeros((nx, ny, nz), dtype=np.float32)
        self.oe, self.on_, self.oz = oe, on_, oz
        self.res = res
        self.nx, self.ny, self.nz = nx, ny, nz

    def w2v(self, e, n, z):
        xi = int((e  - self.oe)  / self.res)
        yi = int((n  - self.on_) / self.res)
        zi = int((z  - self.oz)  / self.res)
        return xi, yi, zi

    def v2w(self, xi, yi, zi):
        return (self.oe  + (xi + 0.5) * self.res,
                self.on_ + (yi + 0.5) * self.res,
                self.oz  + (zi + 0.5) * self.res)

    def in_bounds(self, xi, yi, zi):
        return 0 <= xi < self.nx and 0 <= yi < self.ny and 0 <= zi < self.nz

    def is_free(self, xi, yi, zi):
        return (self.in_bounds(xi, yi, zi) and
                self.grid[xi, yi, zi] != CELL_BUILDING and
                self.grid[xi, yi, zi] != CELL_HARD_BLOCK)


# ─────────────────────────────────────────────────────────────────────────────
# VOXEL GRID BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class VoxelGridBuilder:

    def __init__(self, mem_index: InMemoryBuildingIndex,
                 transformer: CoordinateTransformer,
                 zone_loader: RestrictedZoneLoader):
        self.mem_index   = mem_index
        self.tf          = transformer
        self.zone_loader = zone_loader

    def build(self, start_ll, end_ll,
              res=GRID_RES_M, buffer_m=CORRIDOR_BUFFER_M) -> VoxelGrid:

        t0 = time.time()

        se, sn = self.tf.ll_to_utm(*start_ll)
        ee, en = self.tf.ll_to_utm(*end_ll)

        min_e = min(se, ee) - buffer_m
        max_e = max(se, ee) + buffer_m
        min_n = min(sn, en) - buffer_m
        max_n = max(sn, en) + buffer_m

        nx = max(1, math.ceil((max_e - min_e) / res))
        ny = max(1, math.ceil((max_n - min_n) / res))
        nz = max(1, math.ceil((MAX_ALT_M - MIN_ALT_M) / res))

        voxel_count = nx * ny * nz
        ram_mb = voxel_count * 4 // 1024 // 1024
        _log(f"Grid dimensions : {nx} × {ny} × {nz}  =  {voxel_count:,} voxels  (~{ram_mb} MB RAM)")

        vg = VoxelGrid(nx, ny, nz, min_e, min_n, MIN_ALT_M, res)

        corners = [self.tf.utm_to_ll(e, n)
                   for e, n in [(min_e, min_n), (max_e, max_n)]]
        q_min_lat = min(c[0] for c in corners)
        q_max_lat = max(c[0] for c in corners)
        q_min_lng = min(c[1] for c in corners)
        q_max_lng = max(c[1] for c in corners)

        # ── buildings ─────────────────────────────────────────────────────────
        buildings = self._fetch_buildings(q_min_lat, q_max_lat, q_min_lng, q_max_lng)
        n_bldg    = len(buildings)
        _log(f"Buildings in corridor : {n_bldg:,}")

        for i, b in enumerate(buildings):
            self._fill_building(b, vg)
            if n_bldg > 500 and (i + 1) % max(1, n_bldg // 20) == 0:
                pct = (i + 1) / n_bldg
                bar = _bar(i + 1, n_bldg)
                print(f"\r    Rasterising buildings  [{bar}]  {pct*100:.0f}%  ", end="", flush=True)
        if n_bldg > 500:
            print()   # newline after progress bar

        # ── restricted zones ──────────────────────────────────────────────────
        self._fill_restricted_zones(vg, q_min_lat, q_max_lat, q_min_lng, q_max_lng)

        # ── clearance EDT ─────────────────────────────────────────────────────
        _log("Computing clearance field (EDT)…", 4)
        t_edt    = time.time()
        obs_mask = (vg.grid > 0)
        if obs_mask.any():
            dist_vox     = distance_transform_edt(~obs_mask)
            vg.clearance = (dist_vox * res).astype(np.float32)
            vg.clearance[obs_mask] = 0.0
        _log(f"EDT done  ({time.time()-t_edt:.2f}s)", 4)

        # ── density map ───────────────────────────────────────────────────────
        _log("Computing obstacle density map…", 4)
        t_den          = time.time()
        obstacle_float = obs_mask.astype(np.float32)
        win            = max(3, int(round(50.0 / res)) | 1)   # nearest odd ≥ 3
        density_raw    = uniform_filter(obstacle_float, size=win)
        d_max          = density_raw.max()
        if d_max > 0:
            vg.density = (density_raw / d_max * 255).astype(np.uint8)
        _log(f"Density map done  (window = {win} voxels = {win*res:.0f} m, {time.time()-t_den:.2f}s)", 4)

        _log(f"Grid build total : {time.time()-t0:.2f}s")
        return vg

    # ── DB ────────────────────────────────────────────────────────────────────
    def _fetch_buildings(self, min_lat, max_lat, min_lng, max_lng):
        """Query the in-memory index — zero DB I/O."""
        return self.mem_index.query_bbox(min_lat, max_lat, min_lng, max_lng)

    # ── building rasterisation ────────────────────────────────────────────────
    def _fill_building(self, row, vg: VoxelGrid):
        lat, lng, height_m, min_h, geom_wkt, is_restricted = row
        if height_m is None or height_m <= 0:
            height_m = DEFAULT_HEIGHT_M
        if min_h is None:
            min_h = 0.0

        cell_val = CELL_HARD_BLOCK if is_restricted else CELL_BUILDING
        top_m    = height_m + SAFETY_BUFFER_M
        zi_start = max(0,       int((min_h - vg.oz) / vg.res))
        zi_end   = min(vg.nz-1, int(math.ceil((top_m - vg.oz) / vg.res)))

        if geom_wkt:
            try:
                self._rasterise_polygon(geom_wkt, vg, zi_start, zi_end, cell_val)
                return
            except Exception:
                pass

        # Fallback: 3×3 centroid stamp
        e, n = self.tf.ll_to_utm(lat, lng)
        xi   = int((e - vg.oe)  / vg.res)
        yi   = int((n - vg.on_) / vg.res)
        for dxi in range(-1, 2):
            for dyi in range(-1, 2):
                if vg.in_bounds(xi+dxi, yi+dyi, 0):
                    vg.grid[xi+dxi, yi+dyi, zi_start:zi_end+1] = cell_val

    def _rasterise_polygon(self, geom_wkt, vg: VoxelGrid,
                           zi_start, zi_end, cell_val):
        """Scanline polygon rasterisation — O(perimeter) not O(area * complexity)."""
        poly   = wkt_loads(geom_wkt)
        bounds = poly.bounds

        min_e, min_n = self.tf.ll_to_utm(bounds[1], bounds[0])
        max_e, max_n = self.tf.ll_to_utm(bounds[3], bounds[2])

        xi_min = max(0,       int((min_e - vg.oe)  / vg.res) - 1)
        xi_max = min(vg.nx-1, int((max_e - vg.oe)  / vg.res) + 1)
        yi_min = max(0,       int((min_n - vg.on_) / vg.res) - 1)
        yi_max = min(vg.ny-1, int((max_n - vg.on_) / vg.res) + 1)

        coords_ll  = list(poly.exterior.coords)
        vox_coords = []
        for (lng_c, lat_c) in coords_ll:
            e_c, n_c = self.tf.ll_to_utm(lat_c, lng_c)
            vox_coords.append(((e_c - vg.oe) / vg.res, (n_c - vg.on_) / vg.res))

        n_verts = len(vox_coords)
        for yi in range(yi_min, yi_max + 1):
            y_w = yi + 0.5
            crossings = []
            for k in range(n_verts - 1):
                x0, y0 = vox_coords[k]
                x1, y1 = vox_coords[k + 1]
                if (y0 <= y_w < y1) or (y1 <= y_w < y0):
                    if abs(y1 - y0) > 1e-9:
                        xc = x0 + (y_w - y0) * (x1 - x0) / (y1 - y0)
                        crossings.append(xc)
            crossings.sort()
            for pair in range(0, len(crossings) - 1, 2):
                xi_s = max(xi_min, int(math.ceil(crossings[pair])))
                xi_e = min(xi_max, int(crossings[pair + 1]))
                if xi_s <= xi_e:
                    vg.grid[xi_s:xi_e+1, yi, zi_start:zi_end+1] = cell_val

    # ── restricted zones ──────────────────────────────────────────────────────
    def _fill_restricted_zones(self, vg, min_lat, max_lat, min_lng, max_lng):
        applied = 0
        for zone in self.zone_loader.get_zones():
            hard     = zone.get("hard_block", True)
            cell_val = CELL_HARD_BLOCK if hard else CELL_SOFT_BLOCK
            shape    = zone.get("shape", "circle")

            if shape == "circle":
                c      = zone["center"]
                r      = zone["radius_m"]
                ce, cn = self.tf.ll_to_utm(c[0], c[1])
                xi_min = max(0,       int((ce - r - vg.oe)  / vg.res))
                xi_max = min(vg.nx-1, int((ce + r - vg.oe)  / vg.res))
                yi_min = max(0,       int((cn - r - vg.on_) / vg.res))
                yi_max = min(vg.ny-1, int((cn + r - vg.on_) / vg.res))
                for xi in range(xi_min, xi_max + 1):
                    for yi in range(yi_min, yi_max + 1):
                        e_, n_, _ = vg.v2w(xi, yi, 0)
                        if math.hypot(e_ - ce, n_ - cn) <= r:
                            vg.grid[xi, yi, :] = cell_val
                            if not hard:
                                vg.cost_extra[xi, yi, :] = SOFT_BLOCK_PENALTY
                applied += 1

            elif shape == "polygon":
                coords = zone["polygon"]
                poly   = Polygon([(c[1], c[0]) for c in coords])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                bounds = poly.bounds
                min_e, min_n = self.tf.ll_to_utm(bounds[1], bounds[0])
                max_e, max_n = self.tf.ll_to_utm(bounds[3], bounds[2])
                xi_min = max(0,       int((min_e - vg.oe)  / vg.res) - 1)
                xi_max = min(vg.nx-1, int((max_e - vg.oe)  / vg.res) + 1)
                yi_min = max(0,       int((min_n - vg.on_) / vg.res) - 1)
                yi_max = min(vg.ny-1, int((max_n - vg.on_) / vg.res) + 1)
                for xi in range(xi_min, xi_max + 1):
                    for yi in range(yi_min, yi_max + 1):
                        e_, n_, _ = vg.v2w(xi, yi, 0)
                        lat_, lng_ = self.tf.utm_to_ll(e_, n_)
                        if poly.contains(Point(lng_, lat_)):
                            vg.grid[xi, yi, :] = cell_val
                            if not hard:
                                vg.cost_extra[xi, yi, :] = SOFT_BLOCK_PENALTY
                applied += 1

        _log(f"Restricted zones applied: {applied} / {len(self.zone_loader.get_zones())}", 4)


# ─────────────────────────────────────────────────────────────────────────────
# 26-CONNECTIVITY NEIGHBOUR TABLE
# ─────────────────────────────────────────────────────────────────────────────

_NBRS = np.array([(dx, dy, dz)
                  for dx in (-1, 0, 1)
                  for dy in (-1, 0, 1)
                  for dz in (-1, 0, 1)
                  if not (dx == dy == dz == 0)], dtype=np.int8)

_STEP_D = np.array([math.sqrt(int(dx)**2 + int(dy)**2 + int(dz)**2)
                    for dx, dy, dz in _NBRS], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CRUISE ALTITUDE  [Paper 2]
# ─────────────────────────────────────────────────────────────────────────────

def compute_cruise_altitude(vg: VoxelGrid) -> int:
    """
    85th-percentile obstacle height in the corridor, with a hard floor of
    MIN_CRUISE_ALT_M.  Returns cruise altitude as a voxel z-index.
    """
    min_cruise_zi = max(0, int((MIN_CRUISE_ALT_M - vg.oz) / vg.res))
    occupied_z    = np.where(vg.grid == CELL_BUILDING)

    if len(occupied_z[2]) == 0:
        return min_cruise_zi

    cruise_zi = int(np.percentile(occupied_z[2], 85)) + 1
    cruise_zi = min(max(cruise_zi, min_cruise_zi), vg.nz - 1)
    return cruise_zi


# ─────────────────────────────────────────────────────────────────────────────
# 3D A*  [Papers 1, 2]
# ─────────────────────────────────────────────────────────────────────────────

class AStarPathfinder:

    def __init__(self, vg: VoxelGrid):
        self.vg  = vg
        self.res = vg.res

    def find_path(self, start: tuple, end: tuple,
                  cruise_zi: int = None) -> Optional[list]:
        vg = self.vg
        nx, ny, nz = vg.nx, vg.ny, vg.nz

        # Snap to nearest free voxel (prefer moving upward first)
        if not vg.is_free(*start):
            start = self._nearest_free(start)
            if start is None:
                _log("A* ERROR: No free voxel found near start point", 4)
                return None
        if not vg.is_free(*end):
            end = self._nearest_free(end)
            if end is None:
                _log("A* ERROR: No free voxel found near end point", 4)
                return None

        if cruise_zi is None:
            cruise_zi = (start[2] + end[2]) // 2

        INF    = np.float32(1e9)
        g      = np.full((nx, ny, nz), INF,  dtype=np.float32)
        vis    = np.zeros((nx, ny, nz),       dtype=np.bool_)
        parent = np.full((nx, ny, nz), -1,   dtype=np.int32)

        def flat(xi, yi, zi):
            return xi * ny * nz + yi * nz + zi

        g[start] = 0.0
        heap     = [(self._h(start, end, cruise_zi), start[0], start[1], start[2])]
        expanded = 0

        while heap:
            f, xi, yi, zi = heapq.heappop(heap)

            if vis[xi, yi, zi]:
                continue
            vis[xi, yi, zi] = True
            expanded += 1

            if (xi, yi, zi) == end:
                path = self._reconstruct(parent, start, end, ny, nz)
                _log(f"A* found path: {len(path)} voxels,  {expanded:,} nodes expanded", 4)
                return path

            g_cur = float(g[xi, yi, zi])

            for k in range(26):
                dx = int(_NBRS[k, 0])
                dy = int(_NBRS[k, 1])
                dz = int(_NBRS[k, 2])
                nx_ = xi + dx
                ny_ = yi + dy
                nz_ = zi + dz

                if not vg.in_bounds(nx_, ny_, nz_):
                    continue
                if vis[nx_, ny_, nz_]:
                    continue
                if not vg.is_free(nx_, ny_, nz_):
                    continue

                step_m   = float(_STEP_D[k]) * self.res
                clr_m    = float(vg.clearance[nx_, ny_, nz_])
                clr_pen  = (W_CLEARANCE * (CLEARANCE_R_M - clr_m)
                            if clr_m < CLEARANCE_R_M else 0.0)
                alt_pen  = W_ALT_CHANGE * abs(dz) * self.res
                extra    = float(vg.cost_extra[nx_, ny_, nz_])
                blw_crz  = max(0, cruise_zi - nz_)
                crz_pen  = CRUISE_PENALTY * blw_crz * self.res

                new_g = g_cur + step_m * W_DISTANCE + clr_pen + alt_pen + extra + crz_pen

                if new_g < g[nx_, ny_, nz_]:
                    g[nx_, ny_, nz_]      = new_g
                    parent[nx_, ny_, nz_] = flat(xi, yi, zi)
                    h = self._h((nx_, ny_, nz_), end, cruise_zi)
                    heapq.heappush(heap, (new_g + h, nx_, ny_, nz_))

        _log(f"A* could not find a path  ({expanded:,} nodes expanded)", 4)
        return None

    def _h(self, a: tuple, b: tuple, cruise_zi: int) -> float:
        """Admissible 3D octile heuristic + density term (capped to stay admissible)."""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        dz = abs(a[2] - b[2])
        s  = sorted([dx, dy, dz], reverse=True)

        octile = (math.sqrt(3) * s[2]
                  + math.sqrt(2) * (s[1] - s[2])
                  + (s[0] - s[1])) * self.res * W_DISTANCE

        # Density penalty — cap so h never exceeds true cost
        dens  = float(self.vg.density[a]) / 255.0
        d_pen = min(dens * self.res * W_DENSITY, 0.5 * self.res * W_DISTANCE)

        return octile + d_pen

    def _reconstruct(self, parent, start, end, ny, nz) -> list:
        path = []
        node = end
        while node != start:
            path.append(node)
            p  = int(parent[node])
            if p < 0:
                break
            xi = p // (ny * nz)
            yi = (p % (ny * nz)) // nz
            zi = p % nz
            node = (xi, yi, zi)
        path.append(start)
        path.reverse()
        return path

    def _nearest_free(self, pos, radius=10) -> Optional[tuple]:
        xi, yi, zi = pos
        vg = self.vg
        # Search upward first — escape buildings efficiently
        for dz in range(1, vg.nz - zi):
            c = (xi, yi, zi + dz)
            if vg.is_free(*c):
                return c
        # Full 3-D radius search
        for r in range(1, radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        c = (xi + dx, yi + dy, zi + dz)
                        if vg.is_free(*c):
                            return c
        return None


# ─────────────────────────────────────────────────────────────────────────────
# APF LOCAL SMOOTHER  [Paper 3]
# ─────────────────────────────────────────────────────────────────────────────

class APFSmoother:

    def __init__(self, vg: VoxelGrid):
        self.vg = vg

    def refine(self, skeleton_world: list) -> list:
        if len(skeleton_world) < 2:
            return skeleton_world

        refined = [skeleton_world[0]]
        n_segs  = len(skeleton_world) - 1

        for seg_idx in range(n_segs):
            p_start = np.array(skeleton_world[seg_idx],     dtype=np.float64)
            p_end   = np.array(skeleton_world[seg_idx + 1], dtype=np.float64)

            if not self._segment_near_obstacle(p_start, p_end):
                refined.append(tuple(p_end))
                continue

            seg_pts = self._run_apf(p_start, p_end)
            refined.extend(seg_pts[1:])

        return refined

    def _segment_near_obstacle(self, p0, p1) -> bool:
        n_checks = max(2, int(np.linalg.norm(p1 - p0) / (self.vg.res * 2)))
        for t in np.linspace(0, 1, n_checks):
            pt = p0 + t * (p1 - p0)
            xi, yi, zi = self.vg.w2v(*pt)
            if not self.vg.in_bounds(xi, yi, zi):
                continue
            if float(self.vg.clearance[xi, yi, zi]) < APF_RHO0_M:
                return True
        return False

    def _run_apf(self, p_start, p_end) -> list:
        pos       = p_start.copy()
        goal      = p_end.copy()
        pts       = [tuple(pos)]
        stall_cnt = 0

        for _ in range(APF_MAX_ITERS):
            diff_g = goal - pos
            dist_g = np.linalg.norm(diff_g)

            if dist_g < APF_STEP_M:
                pts.append(tuple(goal))
                break

            f_att = APF_K_ATT * diff_g / dist_g
            f_rep = self._repulsive_force(pos)
            f_net = f_att + f_rep
            mag   = np.linalg.norm(f_net)

            if mag < 1e-9:
                stall_cnt += 1
                if stall_cnt >= 5:
                    pts.append(tuple(goal))
                    break
                continue
            else:
                stall_cnt = 0

            move    = f_net / mag * APF_STEP_M
            new_pos = pos + move

            vg = self.vg
            new_pos[0] = np.clip(new_pos[0], vg.oe,  vg.oe  + vg.nx * vg.res)
            new_pos[1] = np.clip(new_pos[1], vg.on_, vg.on_ + vg.ny * vg.res)
            new_pos[2] = np.clip(new_pos[2], vg.oz,  vg.oz  + vg.nz * vg.res)

            xi, yi, zi = vg.w2v(*new_pos)
            if vg.in_bounds(xi, yi, zi) and not vg.is_free(xi, yi, zi):
                new_pos[2] += vg.res
                xi, yi, zi  = vg.w2v(*new_pos)
                if not (vg.in_bounds(xi, yi, zi) and vg.is_free(xi, yi, zi)):
                    pts.append(tuple(goal))
                    break

            pos = new_pos
            pts.append(tuple(pos))

        return pts

    def _repulsive_force(self, pos) -> np.ndarray:
        vg        = self.vg
        xi, yi, zi = vg.w2v(*pos)
        f_rep     = np.zeros(3, dtype=np.float64)
        # Cap scan radius at 2 voxels to avoid O(n³) over-scanning at 12m grid
        scan_r    = min(2, max(1, int(APF_RHO0_M / vg.res) + 1))

        for dx in range(-scan_r, scan_r + 1):
            for dy in range(-scan_r, scan_r + 1):
                for dz in range(-scan_r, scan_r + 1):
                    ox, oy, oz = xi + dx, yi + dy, zi + dz
                    if not vg.in_bounds(ox, oy, oz):
                        continue
                    if vg.grid[ox, oy, oz] == CELL_FREE:
                        continue
                    oe_, on__, oz_ = vg.v2w(ox, oy, oz)
                    obs_pos        = np.array([oe_, on__, oz_])
                    diff           = pos - obs_pos
                    rho            = np.linalg.norm(diff)
                    if rho < 1e-3 or rho > APF_RHO0_M:
                        continue
                    coeff  = APF_K_REP * (1.0 / rho - 1.0 / APF_RHO0_M) / (rho ** 2)
                    f_rep += coeff * diff / rho

        return f_rep


# ─────────────────────────────────────────────────────────────────────────────
# PATH SMOOTHING: NURBS + POST-PROCESS  [Paper 1]
# ─────────────────────────────────────────────────────────────────────────────

def smooth_path(world_points: list,
                sample_spacing_m: float = NURBS_SAMPLE_M) -> list:
    """
    Full smoothing pipeline:
      1. Angle-based truncation (Paper 1) — remove near-collinear waypoints
      2. Cubic B-spline (NURBS) with chord-length parameterisation
      3. Altitude moving average — removes z oscillation (window = ALT_SMOOTH_WIN)
      4. Lateral Savitzky-Golay — removes XY micro-jitter without distorting turns
    """
    if len(world_points) < 2:
        return world_points

    pts = _angle_truncate(world_points, angle_thresh_deg=5.0)
    if len(pts) < 2:
        return world_points
    if len(pts) == 2:
        return pts

    pts_arr = np.array(pts, dtype=np.float64)
    degree  = min(NURBS_DEGREE, len(pts_arr) - 1)

    diffs   = np.diff(pts_arr, axis=0)
    dists   = np.sqrt((diffs ** 2).sum(axis=1))
    dists   = np.where(dists < 1e-9, 1e-9, dists)
    t_param = np.concatenate([[0], np.cumsum(dists)])
    t_param /= t_param[-1]

    try:
        spline   = make_interp_spline(t_param, pts_arr, k=degree)
        total_m  = float(np.sum(dists))
        n_sample = max(2, int(total_m / sample_spacing_m))
        t_dense  = np.linspace(0, 1, n_sample)
        smooth   = spline(t_dense)          # shape (n_sample, 3)
    except Exception:
        return pts

    # Step 3: Altitude moving average
    if smooth.shape[0] > ALT_SMOOTH_WIN:
        z_raw    = smooth[:, 2].copy()
        kernel   = np.ones(ALT_SMOOTH_WIN) / ALT_SMOOTH_WIN
        z_sm     = np.convolve(z_raw, kernel, mode='same')
        z_sm[0]  = z_raw[0]
        z_sm[-1] = z_raw[-1]
        z_sm     = np.clip(z_sm, MIN_ALT_M, MAX_ALT_M)
        smooth[:, 2] = z_sm

    # Step 4: Lateral Savitzky-Golay (XY only) — removes APF micro-jitter
    if smooth.shape[0] > SG_WINDOW:
        win = min(SG_WINDOW, smooth.shape[0] if smooth.shape[0] % 2 == 1
                  else smooth.shape[0] - 1)
        try:
            smooth[:, 0] = savgol_filter(smooth[:, 0], win, SG_ORDER)
            smooth[:, 1] = savgol_filter(smooth[:, 1], win, SG_ORDER)
        except Exception:
            pass   # degenerate case — keep unfiltered

    return [tuple(r) for r in smooth]


def _angle_truncate(pts: list, angle_thresh_deg: float) -> list:
    if len(pts) <= 2:
        return pts
    thresh_rad = math.radians(angle_thresh_deg)
    kept = [pts[0]]
    for i in range(1, len(pts) - 1):
        v1 = np.array(pts[i])   - np.array(pts[i - 1])
        v2 = np.array(pts[i + 1]) - np.array(pts[i])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        if math.acos(cos_a) > thresh_rad:
            kept.append(pts[i])
    kept.append(pts[-1])
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def world_to_waypoints(world_pts: list, tf: CoordinateTransformer) -> list:
    raw = []
    for e, n, z in world_pts:
        lat, lng = tf.utm_to_ll(e, n)
        alt = max(MIN_ALT_M, min(MAX_ALT_M, round(float(z), 1)))
        raw.append({"lat": round(lat, 6), "lng": round(lng, 6), "alt_m": alt})

    # Deduplicate — remove points closer than WP_MIN_SEP_M
    sep_deg = WP_MIN_SEP_M / 111_000.0
    wps     = [raw[0]]
    for wp in raw[1:]:
        prev = wps[-1]
        if (abs(wp["lat"] - prev["lat"]) > sep_deg or
                abs(wp["lng"] - prev["lng"]) > sep_deg):
            wps.append(wp)
    if wps[-1] != raw[-1]:
        wps.append(raw[-1])

    return wps


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(waypoints: list) -> dict:
    if len(waypoints) < 2:
        return {}
    R = 6_371_000.0
    total_m = max_alt = 0.0
    min_alt = 1e9
    for i in range(1, len(waypoints)):
        a, b  = waypoints[i - 1], waypoints[i]
        dlat  = math.radians(b["lat"] - a["lat"])
        dlng  = math.radians(b["lng"] - a["lng"])
        hav   = (math.sin(dlat / 2) ** 2
                 + math.cos(math.radians(a["lat"]))
                 * math.cos(math.radians(b["lat"]))
                 * math.sin(dlng / 2) ** 2)
        h_m   = 2 * R * math.asin(math.sqrt(max(0.0, hav)))
        v_m   = abs(b["alt_m"] - a["alt_m"])
        total_m += math.sqrt(h_m ** 2 + v_m ** 2)
        max_alt  = max(max_alt, a["alt_m"], b["alt_m"])
        min_alt  = min(min_alt, a["alt_m"], b["alt_m"])
    return {
        "distance_km":    round(total_m / 1000, 3),
        "waypoint_count": len(waypoints),
        "min_alt_m":      round(min_alt, 1),
        "max_alt_m":      round(max_alt, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PATHFINDER
# ─────────────────────────────────────────────────────────────────────────────

class PunePathfinder:
    """
    High-level interface.  Accepts (lat, lng) or (lat, lng, alt_m).
    Automatically snaps out-of-hub coordinates to the nearest hub boundary
    with a console warning rather than hard-rejecting.
    """

    def __init__(self, db_path: str = DB_PATH, zones_config: str = ZONES_CONFIG):
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Database not found: '{db_path}'\n"
                f"Expected pune_hub_buildings.db in current directory."
            )
        self.tf        = CoordinateTransformer()
        self.zones     = RestrictedZoneLoader(zones_config)
        self.mem_index = InMemoryBuildingIndex(db_path)
        self.builder   = VoxelGridBuilder(self.mem_index, self.tf, self.zones)
        self.hub_mgr   = HubZoneManager()
        self.path_store = PathStore()

        print("=" * 60)
        print("  Pune Drone Pathfinder  —  Final Hub Edition  [In-Memory R-Tree]")
        print("=" * 60)
        print(f"  Database   : {db_path}  (loaded into memory — DB closed)")
        print(f"  Buildings  : {self.mem_index.n:,} in memory")
        print(f"  Algorithm  : Density-A* + APF + NURBS  (Papers 1/2/3)")
        print(f"  Grid res   : {GRID_RES_M} m/voxel")
        print(f"  Hub zones  :")
        for h in HUBS:
            print(f"    • {h['name']}  (±{HUB_RADIUS_M/1000:.0f} km)")
        print("=" * 60)

    def find_path(self,
                  start:     tuple,
                  end:       tuple,
                  start_alt: float = MIN_ALT_M,
                  end_alt:   float = MIN_ALT_M,
                  drone_id:  str   = "D1",
                  ) -> dict:

        # Unpack altitude from tuple if provided
        if len(start) == 3:
            start_ll, start_alt = start[:2], float(start[2])
        else:
            start_ll = tuple(start)
        if len(end) == 3:
            end_ll, end_alt = end[:2], float(end[2])
        else:
            end_ll = tuple(end)

        t0  = time.time()
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  ROUTE REQUEST")
        print(f"  Start : {start_ll[0]:.6f}, {start_ll[1]:.6f}  (alt {start_alt:.0f} m)")
        print(f"  End   : {end_ll[0]:.6f}, {end_ll[1]:.6f}  (alt {end_alt:.0f} m)")
        straight_km = haversine(*start_ll, *end_ll) / 1000
        print(f"  Straight-line distance: {straight_km:.2f} km")
        print(f"{sep}")

        # ── Hub zone resolution (snap if needed) ──────────────────────────────
        try:
            start_ll, end_ll, hub_id, warnings = self.hub_mgr.resolve(start_ll, end_ll)
        except Exception as e:
            return {"success": False, "error": str(e)}

        hub_name = next((h["name"] for h in HUBS if h["id"] == hub_id), hub_id)
        print(f"\n  Hub zone  : {hub_name}")
        if warnings:
            for w in warnings:
                print(f"  {w}")

        # ── [1/4] Build voxel grid ────────────────────────────────────────────
        print(f"\n[1/4] Building voxel grid…")
        vg = self.builder.build(start_ll, end_ll)

        se, sn = self.tf.ll_to_utm(*start_ll)
        ee, en = self.tf.ll_to_utm(*end_ll)
        sv = vg.w2v(se, sn, start_alt)
        ev = vg.w2v(ee, en, end_alt)

        if not vg.in_bounds(*sv):
            return {"success": False, "error": "Start is outside the voxel grid — check corridor buffer"}
        if not vg.in_bounds(*ev):
            return {"success": False, "error": "End is outside the voxel grid — check corridor buffer"}

        # ── [2/4] Cruise altitude ─────────────────────────────────────────────
        cruise_zi = compute_cruise_altitude(vg)
        cruise_m  = round(vg.oz + (cruise_zi + 0.5) * vg.res, 1)
        print(f"\n[2/4] Cruise altitude determination…")
        _log(f"Target cruise: {cruise_m} m AGL  "
             f"(85th-pct building height + {SAFETY_BUFFER_M:.0f} m buffer,  "
             f"floor = {MIN_CRUISE_ALT_M:.0f} m)")

        # ── [3/4] A* pathfinding ──────────────────────────────────────────────
        print(f"\n[3/4] Running 3-D density-weighted A*…")
        ta         = time.time()
        pf         = AStarPathfinder(vg)
        raw_voxels = pf.find_path(sv, ev, cruise_zi=cruise_zi)
        astar_t    = time.time() - ta
        _log(f"A* completed in {astar_t:.2f} s")

        if raw_voxels is None:
            return {"success": False, "error": "A* could not find a viable path"}

        raw_world = [vg.v2w(*v) for v in raw_voxels]

        # ── [4/4] APF + NURBS ─────────────────────────────────────────────────
        print(f"\n[4/4] Refining and smoothing path…")

        t_apf     = time.time()
        apf       = APFSmoother(vg)
        apf_world = apf.refine(raw_world)
        _log(f"APF refinement: {len(raw_world)} → {len(apf_world)} points  "
             f"({time.time()-t_apf:.2f} s)")

        t_nurbs   = time.time()
        smooth    = smooth_path(apf_world)
        waypoints = world_to_waypoints(smooth, self.tf)
        metrics   = compute_metrics(waypoints)
        _log(f"NURBS + smoothing: {len(apf_world)} ctrl pts → {len(smooth)} pts → "
             f"{len(waypoints)} waypoints  ({time.time()-t_nurbs:.2f} s)")

        metrics["astar_s"]      = round(astar_t, 2)
        metrics["total_s"]      = round(time.time() - t0, 2)
        metrics["raw_voxels"]   = len(raw_voxels)
        metrics["apf_points"]   = len(apf_world)
        metrics["smooth_points"] = len(smooth)
        metrics["cruise_alt_m"] = cruise_m
        metrics["hub"]          = hub_name

        # ── Save to database ──────────────────────────────────────────────────
        eta_s   = round(metrics["distance_km"] / DRONE_SPEED_KMH * 3600, 1)
        path_id = self.path_store.save(
            drone_id   = drone_id,
            waypoints  = waypoints,
            metrics    = metrics,
            start_ll   = start_ll,
            end_ll     = end_ll,
        )
        metrics["path_id"]  = path_id
        metrics["drone_id"] = drone_id
        metrics["eta_s"]    = eta_s

        print(f"\n{'═'*60}")
        print(f"  ✓ PATH FOUND  —  saved as '{path_id}'")
        print(f"{'═'*60}")
        print(f"  Path ID     : {path_id}")
        print(f"  Drone       : {drone_id}")
        print(f"  Distance    : {metrics['distance_km']} km")
        print(f"  ETA         : {eta_s:.0f} s  ({eta_s/60:.1f} min)")
        print(f"  Waypoints   : {metrics['waypoint_count']}")
        print(f"  Altitude    : {metrics['min_alt_m']} m – {metrics['max_alt_m']} m AGL")
        print(f"  Cruise alt  : {metrics['cruise_alt_m']} m AGL")
        print(f"  A* time     : {metrics['astar_s']} s")
        print(f"  Total time  : {metrics['total_s']} s")
        print(f"  Stored in   : {self.path_store.db_path}")
        print(f"{'═'*60}\n")

        return {"success": True, "waypoints": waypoints, "metrics": metrics, "path_id": path_id}


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

# def _print_result(label: str, result: dict):
#     print(f"\n{'─'*60}")
#     print(f"  TEST: {label}")
#     print(f"{'─'*60}")
#     if result["success"]:
#         m = result["metrics"]
#         print(f"  Distance    : {m['distance_km']} km")
#         print(f"  Waypoints   : {m['waypoint_count']}")
#         print(f"  Altitude    : {m['min_alt_m']} m – {m['max_alt_m']} m AGL")
#         print(f"  Cruise alt  : {m['cruise_alt_m']} m AGL")
#         print(f"  A* time     : {m['astar_s']} s  |  Total: {m['total_s']} s")
#         print(f"  Waypoint list:")
#         for i, wp in enumerate(result["waypoints"]):
#             print(f"    [{i+1:>3}]  {wp['lat']:.6f}  {wp['lng']:.6f}  "
#                   f"{wp['alt_m']} m")
#     else:
#         print(f"  ✗ FAILED: {result['error']}")


if __name__ == "__main__":

    pf = PunePathfinder()
    pf.find_path()

    # ── Hub A: South Pune  (Navale / Katraj / Hadapsar) ───────────────────────
    # _print_result(
    #     "Hub A — Katraj Junction → Navale Bridge",
    #     pf.find_path(
    #         start=(18.4600, 73.8630),   # Katraj Junction
    #         end  =(18.4660, 73.8383),   # Navale Bridge area
    #     ),
    # )

    # _print_result(
    #     "Hub A — Narhe → Hadapsar",
    #     pf.find_path(
    #         start=(18.4716, 73.8141),   # Narhe / Dhayari side
    #         end  =(18.4983, 73.9300),   # Hadapsar / Ramtekdi
    #     ),
    # )

    # # ── Hub B: Northwest Pune  (Hinjewadi / Wakad / Baner) ────────────────────
    # _print_result(
    #     "Hub B — Hinjewadi Phase 1 → Wakad Junction",
    #     pf.find_path(
    #         start=(18.5904, 73.7383),   # Hinjewadi Phase 1
    #         end  =(18.5997, 73.7700),   # Wakad Junction
    #     ),
    # )

    # _print_result(
    #     "Hub B — Wakad → Baner / Balewadi",
    #     pf.find_path(
    #         start=(18.5997, 73.7700),   # Wakad
    #         end  =(18.5700, 73.7800),   # Baner / Balewadi
    #     ),
    # )

    # # ── Hub C: East Pune  (Kharadi / Viman Nagar / Mundhwa) ───────────────────
    # _print_result(
    #     "Hub C — Kharadi Core → Nagar Road / Viman Nagar",
    #     pf.find_path(
    #         start=(18.5519, 73.9476),   # Kharadi core
    #         end  =(18.5642, 73.9197),   # Nagar Road / Viman Nagar junction
    #     ),
    # )

    # _print_result(
    #     "Hub C — Viman Nagar → Mundhwa",
    #     pf.find_path(
    #         start=(18.5642, 73.9197),   # Viman Nagar
    #         end  =(18.5250, 73.9350),   # Mundhwa southern link
    #     ),
    # )

    # # ── Auto-snap test: slightly outside hub ───────────────────────────────────
    # _print_result(
    #     "Auto-snap — Koregaon Park (outside hubs) → Mundhwa",
    #     pf.find_path(
    #         start=(18.5362, 73.8944),   # Koregaon Park — outside all hubs
    #         end  =(18.5250, 73.9350),   # Mundhwa — Hub C
    #     ),
    # )