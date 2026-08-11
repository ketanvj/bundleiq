"""
s07/tests/test_s07.py
----------------------
Tests for Session 7: MCP Server (US-06 Part 1).

Run with:
    pytest s07/tests/ -v

These tests call the tool functions directly. FastMCP's @mcp.tool()
decorator registers the function with the server but returns the original
callable unchanged -- so query_plans("mobile") works just like calling
any Python function.

DB_PATH is patched to a test database created in conftest.py. Tests do not
require data/seed.py to have been run.

Test groups:
  TestServerStructure  -- server name, tool count, tool names
  TestQueryPlans       -- return format, plan_type filtering, empty result
  TestQueryPromotions  -- return format, plan_id filtering, empty result
  TestSQLInjection     -- parameterised query protects against injection
"""

import sys
from pathlib import Path

import pytest

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))

import mcp_server
from mcp_server import mcp, query_plans, query_promotions


# ---------------------------------------------------------------------------
# TestServerStructure
# ---------------------------------------------------------------------------

class TestServerStructure:
    def test_server_name(self):
        assert mcp.name == "bundleiq-tools"

    def test_server_has_two_tools(self):
        tools = mcp._tool_manager.list_tools()
        assert len(tools) == 2, f"Expected 2 tools, found {len(tools)}"

    def test_server_has_query_plans_tool(self):
        tools = mcp._tool_manager.list_tools()
        names = [t.name for t in tools]
        assert "query_plans" in names

    def test_server_has_query_promotions_tool(self):
        tools = mcp._tool_manager.list_tools()
        names = [t.name for t in tools]
        assert "query_promotions" in names

    def test_query_plans_has_description(self):
        tools = mcp._tool_manager.list_tools()
        tool = next(t for t in tools if t.name == "query_plans")
        assert tool.description and len(tool.description) > 10

    def test_query_promotions_has_description(self):
        tools = mcp._tool_manager.list_tools()
        tool = next(t for t in tools if t.name == "query_promotions")
        assert tool.description and len(tool.description) > 10


# ---------------------------------------------------------------------------
# TestQueryPlans
# ---------------------------------------------------------------------------

class TestQueryPlans:
    def test_all_returns_string(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_plans("all")
        assert isinstance(result, str)

    def test_all_contains_mobile_and_broadband(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_plans("all")
        assert "Daily 1GB" in result
        assert "Basic 50Mbps" in result

    def test_mobile_filter_excludes_broadband(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_plans("mobile")
        assert "Daily 1GB" in result
        assert "Basic 50Mbps" not in result

    def test_broadband_filter_excludes_mobile(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_plans("broadband")
        assert "Basic 50Mbps" in result
        assert "Daily 1GB" not in result

    def test_5g_flag_appears_for_5g_plans(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_plans("mobile")
        assert "Daily 3GB 5G (5G)" in result

    def test_non_5g_plan_has_no_flag(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_plans("mobile")
        assert "Daily 1GB: Rs. 179" in result

    def test_broadband_includes_installation_fee(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_plans("broadband")
        assert "installation Rs. 500" in result

    def test_broadband_free_installation(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_plans("broadband")
        assert "free installation" in result

    def test_no_data_returns_not_found(self, tmp_path, monkeypatch):
        empty_db = tmp_path / "empty.db"
        import sqlite3
        conn = sqlite3.connect(str(empty_db))
        conn.executescript("""
            CREATE TABLE mobile_plans (
                plan_id TEXT PRIMARY KEY, name TEXT, data_gb_per_day REAL,
                call_type TEXT, validity_days INTEGER, price INTEGER, is_5g INTEGER
            );
            CREATE TABLE broadband_plans (
                plan_id TEXT PRIMARY KEY, name TEXT, speed_mbps INTEGER,
                data_type TEXT, monthly_price INTEGER, installation_fee INTEGER
            );
        """)
        conn.commit()
        conn.close()
        monkeypatch.setattr(mcp_server, "DB_PATH", empty_db)
        result = query_plans("all")
        assert result == "No plan data found."

    def test_default_argument_is_all(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result_default = query_plans()
        result_all = query_plans("all")
        assert result_default == result_all


# ---------------------------------------------------------------------------
# TestQueryPromotions
# ---------------------------------------------------------------------------

class TestQueryPromotions:
    def test_all_returns_string(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_promotions("all")
        assert isinstance(result, str)

    def test_all_returns_multiple_promotions(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_promotions("all")
        assert "Monsoon Data Offer" in result
        assert "Broadband Cashback" in result

    def test_filter_by_plan_id_returns_matching_promo(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_promotions("mob_001")
        assert "Monsoon Data Offer" in result
        assert "Broadband Cashback" not in result

    def test_unknown_plan_returns_not_found_message(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_promotions("mob_999")
        assert "No active promotions found" in result

    def test_result_includes_discount_type(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_promotions("mob_001")
        assert "extra_data" in result

    def test_result_includes_valid_until(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_promotions("bb_003")
        assert "2026-07-31" in result

    def test_default_argument_is_all(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result_default = query_promotions()
        result_all = query_promotions("all")
        assert result_default == result_all

    def test_no_promotions_returns_generic_message(self, tmp_path, monkeypatch):
        empty_db = tmp_path / "empty.db"
        import sqlite3
        conn = sqlite3.connect(str(empty_db))
        conn.executescript("""
            CREATE TABLE promotions (
                promo_id TEXT PRIMARY KEY, name TEXT, eligible_plan_ids TEXT,
                discount_type TEXT, discount_value TEXT, valid_until TEXT
            );
        """)
        conn.commit()
        conn.close()
        monkeypatch.setattr(mcp_server, "DB_PATH", empty_db)
        result = query_promotions("all")
        assert result == "No active promotions."


# ---------------------------------------------------------------------------
# TestSQLInjection
# ---------------------------------------------------------------------------

class TestSQLInjection:
    def test_promotions_injection_does_not_crash(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        result = query_promotions("'; DROP TABLE promotions; --")
        assert isinstance(result, str)
        assert "No active promotions found" in result

    def test_promotions_injection_does_not_drop_table(self, test_db, monkeypatch):
        monkeypatch.setattr(mcp_server, "DB_PATH", test_db)
        query_promotions("'; DROP TABLE promotions; --")
        result = query_promotions("mob_001")
        assert "Monsoon Data Offer" in result
