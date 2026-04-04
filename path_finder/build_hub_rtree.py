"""
Pune Drone Hubs — Focused R-Tree Builder
==========================================
Takes the 3 drone hub coordinates and builds a new focused SQLite
database containing ONLY buildings within 5km of each hub.

This replaces the city-wide pune_buildings_merged.db R-Tree with a
much smaller, faster index covering only the operational zones.

Hub centers:
  Hub A — 18.4656°N, 73.8383°E  (South Pune / Katraj area)
  Hub B — 18.5995°N, 73.7620°E  (Northwest Pune / Pimpri area)
  Hub C — 18.5519°N, 73.9476°E  (East Pune / Viman Nagar area)

Usage:
    python build_hub_rtree.py

Output:
    pune_hub_buildings.db  — focused building DB with R-Tree
    pune_hubs.db           — hub metadata table

Dependencies:
    pip install pyproj
"""

import sqlite3
import math
import time
import os

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

SOURCE_DB  = "pune_buildings_merged.db"   # your existing full DB
OUTPUT_DB  = "pune_hub_buildings.db"      # new focused DB

HUB_RADIUS_M = 5000   # 5km radius around each hub

HUBS = [
    {
        "id":   "hub_a",
        "name": "South Pune Hub",
        "lat":  18.4656,
        "lng":  73.8383,
    },
    {
        "id":   "hub_b",
        "name": "Northwest Pune Hub",
        "lat":  18.5995,
        "lng":  73.7620,
    },
    {
        "id":   "hub_c",
        "name": "East Pune Hub",
        "lat":  18.5519,
        "lng":  73.9476,
    },
]

BATCH_SIZE = 5_000


# ─────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────

def haversine_m(lat1, lng1, lat2, lng2) -> float:
    """Exact great-circle distance in metres between two lat/lng points."""
    R = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def bbox_for_circle(lat, lng, radius_m) -> dict:
    """
    Returns a bounding box that fully contains the circle.
    Used for the initial R-Tree pre-filter before exact distance check.
    """
    deg_lat = radius_m / 111_000
    deg_lng = radius_m / (111_000 * math.cos(math.radians(lat)))
    return {
        "min_lat": lat - deg_lat,
        "max_lat": lat + deg_lat,
        "min_lng": lng - deg_lng,
        "max_lng": lng + deg_lng,
    }


# ─────────────────────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────────────────────

def setup_output_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS buildings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id       INTEGER,        -- original id from source DB
            source          TEXT,           -- 'osm' or 'google'
            hub_ids         TEXT,           -- comma-separated hub IDs this building belongs to
            latitude        REAL NOT NULL,
            longitude       REAL NOT NULL,
            height_m        REAL,
            min_height_m    REAL DEFAULT 0,
            floor_count     INTEGER,
            height_source   TEXT,
            drone_min_alt_m REAL,
            confidence      REAL,
            area_m2         REAL,
            geometry_wkt    TEXT,
            building_type   TEXT,
            is_restricted   INTEGER DEFAULT 0
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS buildings_rtree
        USING rtree(
            id,
            min_lat, max_lat,
            min_lng, max_lng
        );

        CREATE TABLE IF NOT EXISTS hubs (
            id          TEXT PRIMARY KEY,
            name        TEXT,
            latitude    REAL,
            longitude   REAL,
            radius_m    REAL,
            bbox_min_lat REAL,
            bbox_max_lat REAL,
            bbox_min_lng REAL,
            bbox_max_lng REAL,
            building_count INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS build_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────
# FLUSH BATCH
# ─────────────────────────────────────────────────────────────

def flush_batch(cur, rows):
    if not rows:
        return

    cur.executemany("""
        INSERT INTO buildings
        (source_id, source, hub_ids, latitude, longitude,
         height_m, min_height_m, floor_count, height_source,
         drone_min_alt_m, confidence, area_m2,
         geometry_wkt, building_type, is_restricted)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        (r["source_id"], r["source"], r["hub_ids"],
         r["latitude"], r["longitude"],
         r["height_m"], r["min_height_m"], r["floor_count"],
         r["height_source"], r["drone_min_alt_m"],
         r["confidence"], r["area_m2"],
         r["geometry_wkt"], r["building_type"], r["is_restricted"])
        for r in rows
    ])

    last_id  = cur.execute("SELECT last_insert_rowid()").fetchone()[0]
    n        = len(rows)
    start_id = last_id - n + 1

    cur.executemany("""
        INSERT INTO buildings_rtree
        (id, min_lat, max_lat, min_lng, max_lng)
        VALUES (?,?,?,?,?)
    """, [
        (start_id + i,
         rows[i]["min_lat"], rows[i]["max_lat"],
         rows[i]["min_lng"], rows[i]["max_lng"])
        for i in range(n)
    ])


# ─────────────────────────────────────────────────────────────
# MAIN BUILD FUNCTION
# ─────────────────────────────────────────────────────────────

def build_hub_rtree(
    source_db:  str = SOURCE_DB,
    output_db:  str = OUTPUT_DB,
    hubs:       list = HUBS,
    radius_m:   float = HUB_RADIUS_M,
):
    if not os.path.exists(source_db):
        raise FileNotFoundError(
            f"Source DB not found: {source_db}\n"
            f"Run parse_osm_and_merge.py first."
        )

    print(f"\n{'='*58}")
    print(f"  Pune Drone Hubs — Focused R-Tree Builder")
    print(f"{'='*58}")
    print(f"  Source : {source_db}")
    print(f"  Output : {output_db}")
    print(f"  Hubs   : {len(hubs)} centers, {radius_m/1000:.1f}km radius each")
    print()

    t_total  = time.time()
    src_conn = sqlite3.connect(source_db)
    src_cur  = src_conn.cursor()
    out_conn = setup_output_db(output_db)
    out_cur  = out_conn.cursor()

    hub_counts = {h["id"]: 0 for h in hubs}

    # ── Step 1: compute bboxes for all hubs ──────────────────
    hub_bboxes = {}
    for hub in hubs:
        bb = bbox_for_circle(hub["lat"], hub["lng"], radius_m)
        hub_bboxes[hub["id"]] = bb
        print(f"  {hub['name']} ({hub['id']})")
        print(f"    Center : {hub['lat']:.4f}°N, {hub['lng']:.4f}°E")
        print(f"    BBox   : lat [{bb['min_lat']:.4f}, {bb['max_lat']:.4f}]"
              f"  lng [{bb['min_lng']:.4f}, {bb['max_lng']:.4f}]")
        print()

    # ── Step 2: combined bbox covering all 3 circles ─────────
    # Use this for a single efficient DB query, then filter per-circle
    all_min_lat = min(bb["min_lat"] for bb in hub_bboxes.values())
    all_max_lat = max(bb["max_lat"] for bb in hub_bboxes.values())
    all_min_lng = min(bb["min_lng"] for bb in hub_bboxes.values())
    all_max_lng = max(bb["max_lng"] for bb in hub_bboxes.values())

    print(f"  Combined query bbox:")
    print(f"    lat [{all_min_lat:.4f}, {all_max_lat:.4f}]"
          f"  lng [{all_min_lng:.4f}, {all_max_lng:.4f}]")
    print()

    # ── Step 3: fetch all candidate buildings from source ────
    print(f"  Fetching candidate buildings from source DB...")
    t_fetch = time.time()

    src_cur.execute("""
        SELECT b.id, b.source, b.latitude, b.longitude,
               b.height_m, b.min_height_m, b.floor_count,
               b.height_source, b.drone_min_alt_m,
               b.confidence, b.area_m2, b.geometry_wkt,
               b.building_type, b.is_restricted,
               r.min_lat, r.max_lat, r.min_lng, r.max_lng
        FROM   buildings b
        JOIN   buildings_rtree r ON b.id = r.id
        WHERE  r.max_lat >= ? AND r.min_lat <= ?
          AND  r.max_lng >= ? AND r.min_lng <= ?
    """, (all_min_lat, all_max_lat, all_min_lng, all_max_lng))

    candidates = src_cur.fetchall()
    print(f"  Candidates in combined bbox: {len(candidates):,}"
          f"  ({time.time()-t_fetch:.1f}s)")

    # ── Step 4: filter each building per circle ───────────────
    # A building belongs to a hub if its centroid is within radius_m
    # of that hub's center. A building can belong to multiple hubs.
    print(f"\n  Filtering to exact circles...")
    t_filter = time.time()

    # Use a dict to deduplicate — a building near 2 hubs is stored once
    # with both hub_ids listed
    building_map = {}   # source_id -> row dict with hub_ids set

    for row in candidates:
        (src_id, source, lat, lng,
         height_m, min_height_m, floor_count,
         height_source, drone_min_alt_m,
         confidence, area_m2, geom_wkt,
         building_type, is_restricted,
         min_lat, max_lat, min_lng, max_lng) = row

        # Check which hubs this building belongs to
        belonging_hubs = []
        for hub in hubs:
            dist = haversine_m(lat, lng, hub["lat"], hub["lng"])
            if dist <= radius_m:
                belonging_hubs.append(hub["id"])
                hub_counts[hub["id"]] += 1

        if not belonging_hubs:
            continue  # outside all circles

        if src_id in building_map:
            # Already added — just add new hub_ids
            existing_hubs = set(building_map[src_id]["hub_ids"].split(","))
            existing_hubs.update(belonging_hubs)
            building_map[src_id]["hub_ids"] = ",".join(sorted(existing_hubs))
        else:
            building_map[src_id] = {
                "source_id":       src_id,
                "source":          source,
                "hub_ids":         ",".join(belonging_hubs),
                "latitude":        lat,
                "longitude":       lng,
                "height_m":        height_m,
                "min_height_m":    min_height_m or 0.0,
                "floor_count":     floor_count,
                "height_source":   height_source,
                "drone_min_alt_m": drone_min_alt_m,
                "confidence":      confidence,
                "area_m2":         area_m2,
                "geometry_wkt":    geom_wkt,
                "building_type":   building_type,
                "is_restricted":   is_restricted or 0,
                "min_lat":         min_lat,
                "max_lat":         max_lat,
                "min_lng":         min_lng,
                "max_lng":         max_lng,
            }

    kept = list(building_map.values())
    print(f"  Buildings in circles  : {len(kept):,}"
          f"  ({time.time()-t_filter:.1f}s)")
    print()
    for hub in hubs:
        print(f"    {hub['name']:<25} : {hub_counts[hub['id']]:>7,} buildings")
    print()

    # ── Step 5: write to output DB ────────────────────────────
    print(f"  Writing to {output_db}...")
    t_write = time.time()
    batch   = []

    for i, row in enumerate(kept):
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            flush_batch(out_cur, batch)
            out_conn.commit()
            batch = []

    if batch:
        flush_batch(out_cur, batch)
        out_conn.commit()

    print(f"  Write complete  ({time.time()-t_write:.1f}s)")

    # ── Step 6: write hub metadata ────────────────────────────
    for hub in hubs:
        bb = hub_bboxes[hub["id"]]
        out_cur.execute("""
            INSERT OR REPLACE INTO hubs
            (id, name, latitude, longitude, radius_m,
             bbox_min_lat, bbox_max_lat, bbox_min_lng, bbox_max_lng,
             building_count)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            hub["id"], hub["name"], hub["lat"], hub["lng"], radius_m,
            bb["min_lat"], bb["max_lat"], bb["min_lng"], bb["max_lng"],
            hub_counts[hub["id"]],
        ))

    # ── Step 7: save build metadata ───────────────────────────
    elapsed = time.time() - t_total
    out_cur.executemany(
        "INSERT OR REPLACE INTO build_meta (key,value) VALUES (?,?)",
        [
            ("source_db",       source_db),
            ("total_buildings", str(len(kept))),
            ("hub_count",       str(len(hubs))),
            ("radius_m",        str(radius_m)),
            ("build_time_s",    f"{elapsed:.1f}"),
        ]
    )
    out_conn.commit()

    src_conn.close()
    out_conn.close()

    db_size = os.path.getsize(output_db) / 1e6
    print(f"\n{'='*58}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"{'='*58}")
    print(f"  Total buildings  : {len(kept):,}")
    print(f"  Output size      : {db_size:.1f} MB")
    print(f"  Output file      : {output_db}")
    print(f"{'='*58}\n")


# ─────────────────────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────────────────────

def verify_hub_db(db_path: str):
    if not os.path.exists(db_path):
        print("DB not found.")
        return

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    print(f"\n{'='*58}")
    print(f"  Verification: {db_path}")
    print(f"{'='*58}")

    cur.execute("SELECT COUNT(*) FROM buildings")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM buildings_rtree")
    rtree = cur.fetchone()[0]

    print(f"  Total buildings : {total:,}")
    print(f"  R-Tree entries  : {rtree:,}")

    cur.execute("SELECT id, name, latitude, longitude, building_count FROM hubs")
    print(f"\n  Hubs:")
    for row in cur.fetchall():
        print(f"    {row[0]:<8} {row[1]:<25} "
              f"({row[2]:.4f}, {row[3]:.4f})  "
              f"{row[4]:,} buildings")

    cur.execute("""
        SELECT source, COUNT(*),
               SUM(CASE WHEN height_m IS NOT NULL THEN 1 ELSE 0 END)
        FROM buildings GROUP BY source
    """)
    print(f"\n  By source:")
    for source, count, with_h in cur.fetchall():
        print(f"    {source:<8} : {count:>7,} total  |  {with_h:>6,} with height")

    # Spot check — nearest buildings to each hub center
    print(f"\n  Nearest buildings to each hub center:")
    cur.execute("SELECT id, name, latitude, longitude FROM hubs")
    for hub_id, name, hlat, hlng in cur.fetchall():
        cur2 = conn.cursor()
        cur2.execute("""
            SELECT b.latitude, b.longitude, b.height_m, b.building_type
            FROM buildings b
            JOIN buildings_rtree r ON b.id = r.id
            WHERE r.min_lat <= ? AND r.max_lat >= ?
              AND r.min_lng <= ? AND r.max_lng >= ?
            LIMIT 1
        """, (hlat + 0.001, hlat - 0.001,
              hlng + 0.001, hlng - 0.001))
        row = cur2.fetchone()
        if row:
            print(f"    {name}: found building at "
                  f"({row[0]:.4f}, {row[1]:.4f}) "
                  f"h={row[2]}m type={row[3]}")
        else:
            print(f"    {name}: no building found near center")

    if rtree == total:
        print(f"\n  R-Tree healthy — counts match.")
    else:
        print(f"\n  WARNING: R-Tree mismatch ({rtree} vs {total})")

    conn.close()
    print(f"{'='*58}\n")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_hub_rtree()
    verify_hub_db(OUTPUT_DB)
