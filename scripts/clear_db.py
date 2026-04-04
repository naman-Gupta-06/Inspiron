# ============================================================
# scripts/clear_db.py
# Safely resets testing data without destroying infrastructure.
# ============================================================

import sys
import os

# Allow running from the root or scripts directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.alert_db import clear_all_data

def main() -> None:
    print("Executing database reset...")
    clear_all_data()

if __name__ == "__main__":
    main()