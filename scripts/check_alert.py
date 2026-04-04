# ============================================================
# scripts/check_alerts.py
# Quick CLI utility to inspect the alerts database.
# ============================================================

import sys
import os

# Allow running from repo root: python scripts/check_alerts.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.alert_db import fetch_all_alerts, count_alerts

def main() -> None:
    total = count_alerts()
    print(f"\n📊 TOTAL ALERTS: {total}\n")

    alerts = fetch_all_alerts()
    for alert in alerts:
        dispatched = "✅ dispatched" if alert["dispatched"] else "⏳ pending"
        print(
            f"[{alert['id']}] {alert['incident_type'].upper():10s} "
            f"sev={alert['severity']:.2f}  {dispatched}"
        )

if __name__ == "__main__":
    main()