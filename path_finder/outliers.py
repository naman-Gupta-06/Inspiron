import sqlite3
conn = sqlite3.connect("pune_buildings_merged.db")
cur = conn.cursor()

# Find suspiciously tall buildings
cur.execute("""
    SELECT id, height_m, height_source, latitude, longitude, building_type
    FROM buildings
    WHERE height_m > 150
    ORDER BY height_m DESC
    LIMIT 20
""")
for row in cur.fetchall():
    print(row)
conn.close()
