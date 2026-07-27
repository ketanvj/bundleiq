"""
data/seed.py
------------
Seeds the BundleIQ SQLite database with TeleConnect India data.

Run this before ingest.py:
    python data/seed.py

What this script does:
  1. Creates data/teleconnect_data.db
  2. Drops and recreates all tables (idempotent -- safe to run multiple times)
  3. Inserts sample data for mobile plans, broadband plans, devices, bundles,
     and promotions

Why are prices NOT in the documents?
  Plan prices, device prices, and promotional discounts live exclusively in this
  database. Putting "Daily 2GB: Rs. 299" in a document would create two sources
  of truth. When prices change (re-run seed.py), the document would still show
  the old figure while the database has the new one. The compliance node checks
  that BundleIQ never quotes prices from documents -- prices must come from a
  tool call to this database.
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / "teleconnect_data.db"


def seed() -> None:
    print(f"Seeding BundleIQ database at {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys = ON")

    # ------------------------------------------------------------------
    # mobile_plans
    # ------------------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS mobile_plans")
    conn.execute("""
        CREATE TABLE mobile_plans (
            plan_id          TEXT PRIMARY KEY,
            name             TEXT NOT NULL,
            data_gb_per_day  REAL,
            call_type        TEXT,
            validity_days    INTEGER NOT NULL,
            price            INTEGER NOT NULL,
            is_5g            INTEGER NOT NULL DEFAULT 0
        )
    """)

    mobile_plans = [
        ("mob_001", "Daily 1GB",                1.0, "unlimited",      28, 179, 0),
        ("mob_002", "Daily 2GB",                2.0, "unlimited",      28, 299, 0),
        ("mob_003", "Daily 3GB 5G",             3.0, "unlimited",      28, 399, 1),
        ("mob_004", "Unlimited 5G",             2.5, "unlimited",      56, 599, 1),
        ("mob_005", "Data Top-up 10GB",        10.0, None,             30, 149, 0),
        ("mob_006", "International Roaming Add-on", 0.5, "unlimited_intl", 30, 499, 0),
    ]
    conn.executemany(
        "INSERT INTO mobile_plans VALUES (?, ?, ?, ?, ?, ?, ?)",
        mobile_plans,
    )
    print(f"  Inserted {len(mobile_plans)} mobile plans")

    # ------------------------------------------------------------------
    # broadband_plans
    # ------------------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS broadband_plans")
    conn.execute("""
        CREATE TABLE broadband_plans (
            plan_id           TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            speed_mbps        INTEGER NOT NULL,
            data_type         TEXT NOT NULL,
            monthly_price     INTEGER NOT NULL,
            installation_fee  INTEGER NOT NULL
        )
    """)

    broadband_plans = [
        ("bb_001", "Basic 50Mbps",    50,  "unlimited",  499,  500),
        ("bb_002", "Standard 100Mbps",100, "unlimited",  799,  500),
        ("bb_003", "Fast 200Mbps",    200, "unlimited", 1099,  500),
        ("bb_004", "Ultra 500Mbps",   500, "unlimited", 1599,    0),
    ]
    conn.executemany(
        "INSERT INTO broadband_plans VALUES (?, ?, ?, ?, ?, ?)",
        broadband_plans,
    )
    print(f"  Inserted {len(broadband_plans)} broadband plans")

    # ------------------------------------------------------------------
    # devices
    # ------------------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS devices")
    conn.execute("""
        CREATE TABLE devices (
            device_id         TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            brand             TEXT NOT NULL,
            category          TEXT NOT NULL,
            price             INTEGER NOT NULL,
            compatible_plans  TEXT NOT NULL
        )
    """)

    devices = [
        ("dev_001", "TeleConnect 5G WiFi Router", "TeleConnect", "router", 2999, "mob_003,mob_004,bb_003,bb_004"),
        ("dev_002", "TeleConnect 4G MiFi",         "TeleConnect", "mifi",  1499, "mob_001,mob_002"),
        ("dev_003", "TeleConnect ONU Modem",        "TeleConnect", "onu",      0, "bb_001,bb_002,bb_003,bb_004"),
        ("dev_004", "TeleConnect SIM (4G)",         "TeleConnect", "sim",     99, "mob_001,mob_002,mob_003,mob_004"),
    ]
    conn.executemany(
        "INSERT INTO devices VALUES (?, ?, ?, ?, ?, ?)",
        devices,
    )
    print(f"  Inserted {len(devices)} devices")

    # ------------------------------------------------------------------
    # bundles
    # ------------------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS bundles")
    conn.execute("""
        CREATE TABLE bundles (
            bundle_id         TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            mobile_plan_id    TEXT NOT NULL,
            broadband_plan_id TEXT NOT NULL,
            device_id         TEXT,
            bundle_price      INTEGER NOT NULL,
            savings           INTEGER NOT NULL
        )
    """)

    bundles = [
        ("bndl_001", "Work From Home Bundle", "mob_003", "bb_002", "dev_001", 1099, 99),
        ("bndl_002", "Family Connect",         "mob_004", "bb_003", "dev_001", 1499, 199),
        ("bndl_003", "Basic Starter",          "mob_001", "bb_001", "dev_004",  599, 79),
    ]
    conn.executemany(
        "INSERT INTO bundles VALUES (?, ?, ?, ?, ?, ?, ?)",
        bundles,
    )
    print(f"  Inserted {len(bundles)} bundles")

    # ------------------------------------------------------------------
    # promotions
    # ------------------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS promotions")
    conn.execute("""
        CREATE TABLE promotions (
            promo_id          TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            eligible_plan_ids TEXT NOT NULL,
            discount_type     TEXT NOT NULL,
            discount_value    TEXT NOT NULL,
            valid_until       TEXT NOT NULL
        )
    """)

    promotions = [
        ("promo_001", "Monsoon Data Offer",   "mob_001,mob_002", "extra_data", "5GB",  "2026-08-31"),
        ("promo_002", "Broadband Cashback",   "bb_003,bb_004",   "cashback",   "500",  "2026-07-31"),
    ]
    conn.executemany(
        "INSERT INTO promotions VALUES (?, ?, ?, ?, ?, ?)",
        promotions,
    )
    print(f"  Inserted {len(promotions)} promotions")

    conn.commit()
    conn.close()
    print(f"\nDone. Database seeded at {DB_PATH}")
    print("Run 'python data/ingest.py' next to build the vector store.")


if __name__ == "__main__":
    seed()
