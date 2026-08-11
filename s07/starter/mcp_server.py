"""
BundleIQ -- Session 7: MCP Server (US-06 Part 1)
=================================================
STARTER FILE -- your task is to implement the two TODO sections below.

Goal
  Build a standalone MCP server that exposes BundleIQ's two database
  tools -- query_plans and query_promotions -- over the MCP protocol.
  When finished, MCP Inspector should be able to discover both tools and
  call them without touching any agent code.

What is already done for you
  - FastMCP server created: mcp = FastMCP("bundleiq-tools")
  - Both @mcp.tool() decorators and function signatures are in place
  - DB_PATH points to the same teleconnect_data.db used in Session 5
  - mcp.run() at the bottom starts the STDIO server

Your task
  Implement the SQL queries inside TODO 1 (query_plans) and TODO 2
  (query_promotions). The logic is identical to s05/solution/bundleiq/tools.py --
  open that file, find the two @tool functions, and adapt them here.
  The only change: replace @tool with @mcp.tool() (already done).

Run when done
  python s07/starter/mcp_server.py

Inspect with MCP Inspector
  npx @modelcontextprotocol/inspector python s07/starter/mcp_server.py
  Open http://localhost:5173 -- both tools should appear.
"""

import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server instantiation -- already done for you
# ---------------------------------------------------------------------------

mcp = FastMCP("bundleiq-tools")

# ---------------------------------------------------------------------------
# Configuration -- already done for you
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH  = DATA_DIR / "teleconnect_data.db"

# ---------------------------------------------------------------------------
# TODO 1: Implement query_plans
# Hint: copy the query_plans() function from s05/solution/bundleiq/tools.py.
#       The SQL queries and return format are identical.
#       The only difference: @tool becomes @mcp.tool() (already in place).
# ---------------------------------------------------------------------------

@mcp.tool()
def query_plans(plan_type: str = "all") -> str:
    """Fetch current TeleConnect plan prices and details from the database.

    Args:
        plan_type: Which plans to return. Options:
            "mobile"    -- all mobile plans (daily data, price, validity, 5G flag)
            "broadband" -- all broadband plans (speed, monthly price, installation fee)
            "all"       -- both mobile and broadband plans (default)

    Returns formatted plan information as a plain-text string.
    """
    # TODO 1: Connect to DB_PATH with sqlite3.connect()
    # If plan_type is "mobile" or "all":
    #   SELECT name, data_gb_per_day, validity_days, price, is_5g
    #   FROM mobile_plans ORDER BY price
    #   For each row: data_str = f"{data_gb}GB/day" if data_gb else "unlimited data"
    #                 flag = " (5G)" if is_5g else ""
    #   Append: f"{name}{flag}: Rs. {price} -- {data_str}, valid {validity} days"
    # If plan_type is "broadband" or "all":
    #   SELECT name, speed_mbps, monthly_price, installation_fee
    #   FROM broadband_plans ORDER BY monthly_price
    #   For each row: install_str = f"installation Rs. {install_fee}" if install_fee else "free installation"
    #   Append: f"{name}: Rs. {price}/mo -- {speed} Mbps, {install_str}"
    # Close the connection and return "\n".join(lines) or "No plan data found."
    raise NotImplementedError("TODO 1: implement the SQL queries for query_plans()")


# ---------------------------------------------------------------------------
# TODO 2: Implement query_promotions
# Hint: copy the query_promotions() function from s05/solution/bundleiq/tools.py.
#       Same SQL, same return format, same @mcp.tool() decorator.
# ---------------------------------------------------------------------------

@mcp.tool()
def query_promotions(plan_id: str = "all") -> str:
    """Fetch current TeleConnect promotions from the database.

    Args:
        plan_id: Filter promotions for a specific plan ID (e.g. "mob_001").
                 Use "all" to return every active promotion.

    Returns formatted promotion information as a plain-text string.
    """
    # TODO 2: Connect to DB_PATH with sqlite3.connect()
    # If plan_id.lower() == "all":
    #   SELECT name, eligible_plan_ids, discount_type, discount_value, valid_until
    #   FROM promotions ORDER BY valid_until
    # Else (filter by plan_id):
    #   Same SELECT with "WHERE eligible_plan_ids LIKE ? ORDER BY valid_until"
    #   Pass (f"%{plan_id}%",) as the parameter -- never interpolate into the SQL string
    # If no rows found: return f"No active promotions found for plan: '{plan_id}'."
    #   (or "No active promotions." if plan_id == "all")
    # Format each row as:
    #   f"{name}\n  Eligible plans: {eligible}\n  Discount: {discount_type} -- {discount_value}\n"
    #   f"  Valid until: {valid_until}"
    # Return entries joined by "\n\n"
    raise NotImplementedError("TODO 2: implement the SQL queries for query_promotions()")


# ---------------------------------------------------------------------------
# Entry point -- already done for you
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()  # STDIO transport by default
