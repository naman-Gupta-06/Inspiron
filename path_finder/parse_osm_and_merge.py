"""
OSM Western Zone PBF Parser + Pune Buildings Merger
=====================================================
1. Streams through western-zone-latest.osm.pbf
2. Filters to Pune bounding box only
3. Extracts building footprint polygons + height tags
4. Estimates missing heights from floor count (flagged separately)
5. Merges with existing pune_google_buildings.db
6. Produces one unified pune_buildings_merged.db with R-Tree index

Usage:
    python parse_osm_and_merge.py

Dependencies:
    pip install osmium shapely pyproj
"""

import os
import sys
import time
import sqlite3
import osmium
from shapely.geometry import Polygon, mapping
from shapely.wkt import dumps as wkt_dumps
from shapely.ops import transform
import pyproj

# ─────────────────────────────────────────────────────────────
# CONFIG — edit these paths if needed
# ─────────────────────────────────────────────────────────────

OSM_FILE          = "western-zone-260401.osm.pbf"
GOOGLE_BUILDINGS_DB = "pune_google_buildings.db"   # your existing DB
OUTPUT_DB         = "pune_buildings_merged.db"      # final unified DB

PUNE_BBOX = {
    "min_lat": 18.40,
    "max_lat": 18.65,
    "min_lng": 73.72,
    "max_lng": 73.98,
}

FLOOR_HEIGHT_M    = 3.5    # metres per floor for estimation
SAFETY_BUFFER_M   = 10.0   # added on top of building height for drone clearance
BATCH_SIZE        = 5_000

# ─────────────────────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────────────────────

def setup_merged_database(db_path: str) -> sqlite3.Connection:
    """
    Creates the unified merged database schema.
    One buildings table combining OSM + Google Open Buildings.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS buildings (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            source              TEXT    NOT NULL,   -- 'osm' or 'google'
            osm_id              INTEGER,            -- OSM way ID if from OSM
            latitude            REAL    NOT NULL,   -- centroid lat
            longitude           REAL    NOT NULL,   -- centroid lng
            height_m            REAL,               -- building height in metres (NULL if unknown)
            min_height_m        REAL    DEFAULT 0,  -- where building starts (for stilt buildings)
            floor_count         INTEGER,            -- number of floors (if known)
            height_source       TEXT,               -- 'osm_height' | 'osm_levels' | 'estimated' | NULL
            drone_min_alt_m     REAL,               -- height_m + safety buffer (NULL if height unknown)
            confidence          REAL,               -- Google confidence score (NULL for OSM)
            area_m2             REAL,               -- footprint area
            geometry_wkt        TEXT,               -- polygon in WKT format
            building_type       TEXT,               -- OSM building tag value
            is_restricted       INTEGER DEFAULT 0   -- 0=free, 1=no-fly zone
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS buildings_rtree
        USING rtree(
            id,
            min_lat, max_lat,
            min_lng, max_lng
        );

        CREATE TABLE IF NOT EXISTS merge_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────
# HEIGHT RESOLUTION HELPERS
# ─────────────────────────────────────────────────────────────

def resolve_height(tags: dict) -> tuple:
    """
    Resolves building height from OSM tags.
    Returns (height_m, floor_count, height_source)

    Priority:
      1. Explicit 'height' tag              -> most accurate
      2. 'building:levels' × 3.5m          -> estimated, flagged
      3. Generic 'levels' tag × 3.5m       -> estimated, flagged
      4. None                               -> unknown, filled later by DSM
    """
    # 1. Explicit height
    raw_height = tags.get("height") or tags.get("building:height")
    if raw_height:
        try:
            # Handle values like "45m", "45", "45.5 m"
            h = float(raw_height.replace("m", "").replace(" ", "").strip())
            if h > 0:
                return h, None, "osm_height"
        except ValueError:
            pass

    # 2. Building levels
    raw_levels = tags.get("building:levels") or tags.get("levels")
    if raw_levels:
        try:
            levels = float(raw_levels.strip())
            if levels > 0:
                height_m = levels * FLOOR_HEIGHT_M
                return height_m, int(levels), "osm_levels_estimated"
        except ValueError:
            pass

    return None, None, None


def resolve_min_height(tags: dict) -> float:
    """
    Returns the height at which the building STARTS.
    Useful for buildings on stilts or with open ground floors.
    Default is 0.0 (building starts at ground level).
    """
    raw = tags.get("building:min_height") or tags.get("min_height")
    if raw:
        try:
            return float(raw.replace("m", "").replace(" ", "").strip())
        except ValueError:
            pass
    return 0.0


# ─────────────────────────────────────────────────────────────
# OSM HANDLER
# ─────────────────────────────────────────────────────────────

class PuneBuildingHandler(osmium.SimpleHandler):
    """
    Streams through the OSM PBF file.
    Collects only building ways inside Pune's bounding box.
    """

    def __init__(self, bbox: dict):
        super().__init__()
        self.bbox     = bbox
        self.buildings = []
        self.count_read    = 0
        self.count_kept    = 0
        self.count_skipped = 0
        self._last_log     = time.time()

    def way(self, w):
        """Called for every OSM way (line/polygon) in the file."""
        self.count_read += 1

        # Progress log every 10 seconds
        now = time.time()
        if now - self._last_log > 10:
            print(f"  Ways read: {self.count_read:,} | "
                  f"Buildings kept: {self.count_kept:,}")
            self._last_log = now

        # Only process buildings
        if "building" not in w.tags:
            self.count_skipped += 1
            return

        # Build polygon from nodes
        try:
            coords = [(n.lon, n.lat) for n in w.nodes]
        except osmium.InvalidLocationError:
            # Node locations not available — need location index (handled below)
            self.count_skipped += 1
            return

        if len(coords) < 3:
            self.count_skipped += 1
            return

        # Quick centroid estimate for bbox check (average of coords)
        avg_lat = sum(c[1] for c in coords) / len(coords)
        avg_lng = sum(c[0] for c in coords) / len(coords)

        bbox = self.bbox
        if not (bbox["min_lat"] <= avg_lat <= bbox["max_lat"] and
                bbox["min_lng"] <= avg_lng <= bbox["max_lng"]):
            self.count_skipped += 1
            return

        # Build Shapely polygon
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                self.count_skipped += 1
                return
        except Exception:
            self.count_skipped += 1
            return

        # Get centroid
        centroid  = poly.centroid
        cent_lat  = centroid.y
        cent_lng  = centroid.x

        # Resolve height
        tags_dict = {t.k: t.v for t in w.tags}
        height_m, floor_count, height_source = resolve_height(tags_dict)
        min_height_m = resolve_min_height(tags_dict)

        # Drone minimum altitude (height + safety buffer, or None if unknown)
        drone_min_alt = (height_m + SAFETY_BUFFER_M) if height_m else None

        # Area in m² (approximate, using degree-based area — good enough for indexing)
        # Proper UTM area computed separately if needed
        area_deg2 = poly.area
        # Rough conversion: at Pune's latitude, 1 degree ≈ 111km
        area_m2   = area_deg2 * (111_000 ** 2)

        # Polygon bounding box for R-Tree
        bounds    = poly.bounds  # (min_lng, min_lat, max_lng, max_lat)

        self.buildings.append({
            "osm_id":        w.id,
            "latitude":      cent_lat,
            "longitude":     cent_lng,
            "height_m":      height_m,
            "min_height_m":  min_height_m,
            "floor_count":   floor_count,
            "height_source": height_source,
            "drone_min_alt": drone_min_alt,
            "area_m2":       area_m2,
            "geometry_wkt":  wkt_dumps(poly),
            "building_type": tags_dict.get("building", "yes"),
            "bbox": {
                "min_lat": bounds[1], "max_lat": bounds[3],
                "min_lng": bounds[0], "max_lng": bounds[2],
            },
        })
        self.count_kept += 1


# ─────────────────────────────────────────────────────────────
# BATCH FLUSH
# ─────────────────────────────────────────────────────────────

def flush_batch(cur: sqlite3.Cursor, rows: list):
    """Flush a batch of building rows into buildings + rtree tables."""
    if not rows:
        return

    cur.executemany(
        """INSERT INTO buildings
           (source, osm_id, latitude, longitude,
            height_m, min_height_m, floor_count, height_source,
            drone_min_alt_m, confidence, area_m2,
            geometry_wkt, building_type, is_restricted)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                r["source"],
                r.get("osm_id"),
                r["latitude"],
                r["longitude"],
                r.get("height_m"),
                r.get("min_height_m", 0.0),
                r.get("floor_count"),
                r.get("height_source"),
                r.get("drone_min_alt"),
                r.get("confidence"),
                r.get("area_m2"),
                r.get("geometry_wkt"),
                r.get("building_type"),
                0,
            )
            for r in rows
        ],
    )

    last_id  = cur.execute("SELECT last_insert_rowid()").fetchone()[0]
    n        = len(rows)
    start_id = last_id - n + 1

    cur.executemany(
        """INSERT INTO buildings_rtree
           (id, min_lat, max_lat, min_lng, max_lng)
           VALUES (?,?,?,?,?)""",
        [
            (
                start_id + i,
                rows[i]["bbox"]["min_lat"],
                rows[i]["bbox"]["max_lat"],
                rows[i]["bbox"]["min_lng"],
                rows[i]["bbox"]["max_lng"],
            )
            for i in range(n)
        ],
    )


# ─────────────────────────────────────────────────────────────
# STEP 1 — PARSE OSM
# ─────────────────────────────────────────────────────────────

def parse_osm_buildings(osm_file: str, bbox: dict) -> list:
    """
    Streams through the OSM PBF file and extracts Pune buildings.
    Uses osmium's location index to resolve node coordinates.
    """
    print(f"\n{'='*55}")
    print(f"  STEP 1 — Parsing OSM PBF file")
    print(f"{'='*55}")
    print(f"  File: {osm_file} "
          f"({os.path.getsize(osm_file)/1e6:.0f} MB)")
    print(f"  Filtering to Pune bbox...")
    print()

    handler = PuneBuildingHandler(bbox)

    # apply_buffer=True loads node locations into memory for polygon building
    # This is required to get actual coordinates from way nodes
    handler.apply_file(osm_file, locations=True)

    print(f"\n  OSM parse complete.")
    print(f"  Ways read    : {handler.count_read:,}")
    print(f"  Buildings kept: {handler.count_kept:,}")
    print(f"  Skipped      : {handler.count_skipped:,}")

    # Height stats
    with_height    = sum(1 for b in handler.buildings if b["height_m"] is not None)
    with_estimated = sum(1 for b in handler.buildings
                         if b.get("height_source") == "osm_levels_estimated")
    print(f"\n  Height stats:")
    print(f"    Confirmed heights  : {with_height - with_estimated:,}")
    print(f"    Estimated (floors) : {with_estimated:,}")
    print(f"    No height data     : {len(handler.buildings) - with_height:,}")

    return handler.buildings


# ─────────────────────────────────────────────────────────────
# STEP 2 — LOAD GOOGLE BUILDINGS
# ─────────────────────────────────────────────────────────────

def load_google_buildings(db_path: str) -> list:
    """
    Loads buildings from the existing Google Open Buildings SQLite DB.
    Converts to the unified format.
    """
    print(f"\n{'='*55}")
    print(f"  STEP 2 — Loading Google Open Buildings")
    print(f"{'='*55}")

    if not os.path.exists(db_path):
        print(f"  WARNING: {db_path} not found. Skipping Google buildings.")
        return []

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.execute("""
        SELECT b.latitude, b.longitude, b.confidence, b.area_m2,
               b.geometry_wkt, r.min_lat, r.max_lat, r.min_lng, r.max_lng
        FROM buildings b
        JOIN buildings_rtree r ON b.id = r.id
    """)
    rows = cur.fetchall()
    conn.close()

    buildings = []
    for lat, lng, conf, area, geom_wkt, min_lat, max_lat, min_lng, max_lng in rows:
        buildings.append({
            "source":        "google",
            "osm_id":        None,
            "latitude":      lat,
            "longitude":     lng,
            "height_m":      None,   # Google OB has no height data
            "min_height_m":  0.0,
            "floor_count":   None,
            "height_source": None,
            "drone_min_alt": None,
            "confidence":    conf,
            "area_m2":       area,
            "geometry_wkt":  geom_wkt,
            "building_type": None,
            "bbox": {
                "min_lat": min_lat, "max_lat": max_lat,
                "min_lng": min_lng, "max_lng": max_lng,
            },
        })

    print(f"  Loaded {len(buildings):,} Google buildings.")
    return buildings


# ─────────────────────────────────────────────────────────────
# STEP 3 — MERGE & DEDUPLICATE
# ─────────────────────────────────────────────────────────────

def merge_buildings(osm_buildings: list, google_buildings: list) -> list:
    """
    Merges OSM and Google buildings.

    Deduplication strategy:
    - If an OSM building centroid falls within 20m of a Google building
      centroid, prefer the OSM record (it has height data) and discard
      the Google duplicate.
    - Google buildings with no nearby OSM match are kept as-is
      (they fill footprint gaps in areas OSM hasn't mapped).

    Simple spatial grid approach for dedup — groups buildings into
    0.001-degree cells (~110m) and checks only within-cell pairs.
    This avoids O(n²) comparison across all 500K+ buildings.
    """
    print(f"\n{'='*55}")
    print(f"  STEP 3 — Merging & deduplicating")
    print(f"{'='*55}")
    print(f"  OSM buildings    : {len(osm_buildings):,}")
    print(f"  Google buildings : {len(google_buildings):,}")

    # Build spatial grid of OSM building centroids
    # Key: (int(lat/0.001), int(lng/0.001)) → list of osm building indices
    CELL_SIZE = 0.001  # ~110m grid cells
    osm_grid  = {}
    for i, b in enumerate(osm_buildings):
        cell = (int(b["latitude"] / CELL_SIZE),
                int(b["longitude"] / CELL_SIZE))
        osm_grid.setdefault(cell, []).append(i)

    # For each Google building, check if an OSM building is nearby
    DEDUP_THRESHOLD_DEG = 0.0002  # ~22m
    kept_google  = 0
    duped_google = 0

    merged = []

    # Add all OSM buildings first (they have height data — always keep)
    for b in osm_buildings:
        b["source"] = "osm"
        merged.append(b)

    # Add Google buildings only if no nearby OSM building exists
    for b in google_buildings:
        lat  = b["latitude"]
        lng  = b["longitude"]
        cell = (int(lat / CELL_SIZE), int(lng / CELL_SIZE))

        # Check this cell and 8 neighbours
        is_duplicate = False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                neighbour = (cell[0] + dr, cell[1] + dc)
                for osm_idx in osm_grid.get(neighbour, []):
                    osm_b   = osm_buildings[osm_idx]
                    dlat    = abs(lat - osm_b["latitude"])
                    dlng    = abs(lng - osm_b["longitude"])
                    if dlat < DEDUP_THRESHOLD_DEG and dlng < DEDUP_THRESHOLD_DEG:
                        is_duplicate = True
                        break
                if is_duplicate:
                    break
            if is_duplicate:
                break

        if is_duplicate:
            duped_google += 1
        else:
            merged.append(b)
            kept_google += 1

    print(f"\n  Google buildings kept    : {kept_google:,}  "
          f"(no nearby OSM match)")
    print(f"  Google buildings dropped : {duped_google:,}  "
          f"(OSM duplicate)")
    print(f"  Total merged buildings   : {len(merged):,}")

    return merged


# ─────────────────────────────────────────────────────────────
# STEP 4 — WRITE TO UNIFIED DB
# ─────────────────────────────────────────────────────────────

def write_merged_database(
    buildings: list,
    output_db: str,
    batch_size: int = BATCH_SIZE,
):
    print(f"\n{'='*55}")
    print(f"  STEP 4 — Writing unified database")
    print(f"{'='*55}")

    conn = setup_merged_database(output_db)
    cur  = conn.cursor()

    t_start = time.time()
    batch   = []

    for i, b in enumerate(buildings):
        batch.append(b)

        if len(batch) >= batch_size:
            flush_batch(cur, batch)
            conn.commit()
            batch = []

            if (i + 1) % 50_000 == 0:
                elapsed = time.time() - t_start
                print(f"  Written {i+1:,} / {len(buildings):,} "
                      f"({elapsed:.0f}s)")

    if batch:
        flush_batch(cur, batch)
        conn.commit()

    elapsed = time.time() - t_start

    # Save metadata
    osm_count    = sum(1 for b in buildings if b.get("source") == "osm")
    google_count = sum(1 for b in buildings if b.get("source") == "google")
    height_count = sum(1 for b in buildings if b.get("height_m") is not None)

    cur.executemany(
        "INSERT OR REPLACE INTO merge_meta (key, value) VALUES (?,?)",
        [
            ("total_buildings",    str(len(buildings))),
            ("osm_count",          str(osm_count)),
            ("google_count",       str(google_count)),
            ("buildings_with_height", str(height_count)),
            ("safety_buffer_m",    str(SAFETY_BUFFER_M)),
            ("write_time_s",       f"{elapsed:.1f}"),
        ]
    )
    conn.commit()
    conn.close()

    db_size = os.path.getsize(output_db) / 1e6
    print(f"\n  Write complete in {elapsed:.1f}s")
    print(f"  Output size: {db_size:.1f} MB")
    print(f"  File: {output_db}")


# ─────────────────────────────────────────────────────────────
# STEP 5 — VERIFY
# ─────────────────────────────────────────────────────────────

def verify_merged_database(db_path: str):
    print(f"\n{'='*55}")
    print(f"  VERIFICATION: {db_path}")
    print(f"{'='*55}")

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM buildings")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM buildings_rtree")
    rtree = cur.fetchone()[0]

    cur.execute("""
        SELECT source, COUNT(*), 
               SUM(CASE WHEN height_m IS NOT NULL THEN 1 ELSE 0 END)
        FROM buildings GROUP BY source
    """)
    by_source = cur.fetchall()

    cur.execute("""
        SELECT height_source, COUNT(*) 
        FROM buildings 
        WHERE height_source IS NOT NULL 
        GROUP BY height_source
    """)
    by_height_src = cur.fetchall()

    # Spot check Koregaon Park
    cur.execute("""
        SELECT COUNT(*) FROM buildings b
        JOIN buildings_rtree r ON b.id = r.id
        WHERE r.min_lat <= 18.545 AND r.max_lat >= 18.535
          AND r.min_lng <= 73.900 AND r.max_lng >= 73.890
    """)
    kp_count = cur.fetchone()[0]

    cur.execute("SELECT key, value FROM merge_meta")
    meta = dict(cur.fetchall())
    conn.close()

    print(f"  Total buildings  : {total:,}")
    print(f"  R-Tree entries   : {rtree:,}")
    print(f"\n  By source:")
    for source, count, with_height in by_source:
        print(f"    {source:<10} : {count:>8,} total | "
              f"{with_height:>8,} with height")

    print(f"\n  Height sources:")
    for src, count in by_height_src:
        print(f"    {src:<30} : {count:,}")

    print(f"\n  Koregaon Park test : {kp_count} buildings")

    if rtree == total:
        print(f"\n  R-Tree healthy — counts match.")
    else:
        print(f"\n  WARNING: R-Tree count ({rtree}) != buildings ({total})")

    if kp_count > 0:
        print(f"  Spatial queries working correctly.")
    else:
        print(f"  WARNING: No buildings found in Koregaon Park test area.")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Check files exist
    if not os.path.exists(OSM_FILE):
        print(f"ERROR: OSM file not found: {OSM_FILE}")
        print(f"Download from: https://download.geofabrik.de/asia/india/western-zone.html")
        sys.exit(1)

    t_total = time.time()

    print(f"\n{'='*55}")
    print(f"  Pune Buildings — OSM + Google Merge Pipeline")
    print(f"{'='*55}")

    # Step 1 — Parse OSM
    osm_buildings = parse_osm_buildings(OSM_FILE, PUNE_BBOX)

    # Step 2 — Load Google Buildings
    google_buildings = load_google_buildings(GOOGLE_BUILDINGS_DB)

    # Step 3 — Merge & deduplicate
    merged = merge_buildings(osm_buildings, google_buildings)

    # Step 4 — Write unified DB
    write_merged_database(merged, OUTPUT_DB)

    # Step 5 — Verify
    verify_merged_database(OUTPUT_DB)

    total_time = time.time() - t_total
    print(f"\n{'='*55}")
    print(f"  TOTAL TIME: {total_time/60:.1f} minutes")
    print(f"  Output: {OUTPUT_DB}")
    print(f"{'='*55}\n")
