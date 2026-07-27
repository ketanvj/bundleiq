"""
s05/tests/conftest.py
---------------------
Pytest configuration for Session 5 tests.

Sets dummy API keys and clears any stale bundleiq modules at collection time.
The seeded_db fixture creates a temporary SQLite database with real schema and
sample data so tool tests can run actual SQL without touching the production db.
"""
import os
import sqlite3
import sys

import pytest

for _key in list(sys.modules):
    if _key == "bundleiq" or _key.startswith("bundleiq."):
        sys.modules.pop(_key)

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY", "test-langsmith-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")


@pytest.fixture
def seeded_db(tmp_path):
    """Temporary SQLite database with TeleConnect schema and sample rows.

    Used by tool tests via monkeypatch.setattr("bundleiq.tools.DB_PATH", seeded_db).
    The schema matches data/seed.py exactly.
    """
    db_path = tmp_path / "test_teleconnect.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE mobile_plans (
            plan_id          TEXT PRIMARY KEY,
            name             TEXT NOT NULL,
            data_gb_per_day  REAL,
            call_type        TEXT,
            validity_days    INTEGER NOT NULL,
            price            INTEGER NOT NULL,
            is_5g            INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE broadband_plans (
            plan_id           TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            speed_mbps        INTEGER NOT NULL,
            data_type         TEXT NOT NULL,
            monthly_price     INTEGER NOT NULL,
            installation_fee  INTEGER NOT NULL
        );
        CREATE TABLE promotions (
            promo_id          TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            eligible_plan_ids TEXT NOT NULL,
            discount_type     TEXT NOT NULL,
            discount_value    TEXT NOT NULL,
            valid_until       TEXT NOT NULL
        );
        INSERT INTO mobile_plans VALUES
            ('mob_001', 'Daily 1GB',       1.0, 'unlimited', 28, 179, 0),
            ('mob_002', 'Daily 2GB',       2.0, 'unlimited', 28, 299, 0),
            ('mob_003', 'Daily 3GB 5G',    3.0, 'unlimited', 28, 399, 1),
            ('mob_004', 'Unlimited 5G',    2.5, 'unlimited', 56, 599, 1);
        INSERT INTO broadband_plans VALUES
            ('bb_001', 'Basic 50Mbps',    50,  'unlimited',  499,  500),
            ('bb_002', 'Standard 100Mbps',100, 'unlimited',  799,  500),
            ('bb_003', 'Fast 200Mbps',    200, 'unlimited', 1099,  500),
            ('bb_004', 'Ultra 500Mbps',   500, 'unlimited', 1599,    0);
        INSERT INTO promotions VALUES
            ('promo_001', 'Monsoon Data Offer', 'mob_001,mob_002', 'extra_data', '5GB',  '2026-08-31'),
            ('promo_002', 'Broadband Cashback', 'bb_003,bb_004',   'cashback',   '500',  '2026-07-31');
    """)
    conn.commit()
    conn.close()
    return db_path
