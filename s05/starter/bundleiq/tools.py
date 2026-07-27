"""
bundleiq/tools.py
-----------------
LLM clients and database tool functions for BundleIQ.

Session 5: adds query_plans() and query_promotions() so the LLM can
look up live data instead of relying on hardcoded prices.
"""
import os
import sqlite3

from langchain_core.tools import tool
from langchain_groq import ChatGroq

from .config import DB_PATH, MODEL_NAME, TEMPERATURE, MAX_TOKENS

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Did you copy .env.example to .env and fill in your key?\n"
        "  Windows:  copy .env.example .env\n"
        "  Mac/Linux: cp .env.example .env"
    )

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

classifier_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0.0,
    max_tokens=10,
)


# ---------------------------------------------------------------------------
# TODO 1 of 4 -- Implement query_plans()
# ---------------------------------------------------------------------------
# Steps:
#   1. Open: conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#   2. Build lines = []
#   3. If plan_type in ("mobile", "all"), query mobile_plans:
#        rows = conn.execute(
#            "SELECT name, data_gb_per_day, validity_days, price, is_5g "
#            "FROM mobile_plans ORDER BY price"
#        ).fetchall()
#      For each row (name, data_gb, validity, price, is_5g) append:
#        data_str = f"{data_gb}GB/day" if data_gb else "unlimited data"
#        flag     = " (5G)" if is_5g else ""
#        f"{name}{flag}: Rs. {price} -- {data_str}, valid {validity} days"
#   4. If plan_type in ("broadband", "all"), query broadband_plans:
#        rows = conn.execute(
#            "SELECT name, speed_mbps, monthly_price, installation_fee "
#            "FROM broadband_plans ORDER BY monthly_price"
#        ).fetchall()
#      For each row (name, speed, price, install_fee) append:
#        install_str = f"installation Rs. {install_fee}" if install_fee else "free installation"
#        f"{name}: Rs. {price}/mo -- {speed} Mbps, {install_str}"
#   5. conn.close()
#   6. Return "\n".join(lines) if lines else "No plan data found."
# ---------------------------------------------------------------------------
@tool
def query_plans(plan_type: str = "all") -> str:
    """Fetch current TeleConnect plan prices and details from the database.

    Args:
        plan_type: Which plans to return. Options:
            "mobile"    -- all mobile plans (daily data, price, validity, 5G flag)
            "broadband" -- all broadband plans (speed, monthly price, installation fee)
            "all"       -- both mobile and broadband plans (default)

    Returns formatted plan information as a plain-text string.
    """
    # TODO: implement this tool
    pass


# ---------------------------------------------------------------------------
# TODO 2 of 4 -- Implement query_promotions()
# ---------------------------------------------------------------------------
# Steps:
#   1. Open: conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#   2. If plan_id.lower() == "all":
#        rows = conn.execute(
#            "SELECT name, eligible_plan_ids, discount_type, discount_value, valid_until "
#            "FROM promotions ORDER BY valid_until"
#        ).fetchall()
#      Otherwise use a parameterised query (CRITICAL -- prevents SQL injection):
#        rows = conn.execute(
#            "SELECT name, eligible_plan_ids, discount_type, discount_value, valid_until "
#            "FROM promotions WHERE eligible_plan_ids LIKE ? ORDER BY valid_until",
#            (f"%{plan_id}%",),
#        ).fetchall()
#   3. conn.close()
#   4. If not rows: return appropriate message.
#   5. Build parts = [] and for each row append:
#        f"{name}\n  Eligible plans: {eligible}\n  Discount: {discount_type} -- {discount_value}\n  Valid until: {valid_until}"
#      Return "\n\n".join(parts)
# ---------------------------------------------------------------------------
@tool
def query_promotions(plan_id: str = "all") -> str:
    """Fetch current TeleConnect promotions from the database.

    Args:
        plan_id: Filter promotions for a specific plan ID (e.g. "mob_001").
                 Use "all" to return every active promotion.

    Returns formatted promotion information as a plain-text string.
    """
    # TODO: implement this tool
    pass


# ---------------------------------------------------------------------------
# TODO 3 of 4 -- Bind tools to the LLM
# ---------------------------------------------------------------------------
# Create llm_with_tools by binding both tools to llm:
#   llm_with_tools = llm.bind_tools([query_plans, query_promotions])
#
# This tells the LLM what tools are available so it can decide when to call them.
# llm_with_tools is used for the FIRST call in respond(). The second call
# (after tools have run) uses plain llm.
# ---------------------------------------------------------------------------
# TODO: add llm_with_tools = llm.bind_tools([query_plans, query_promotions])


def _run_tool(tool_name: str, tool_args: dict) -> str:
    """Dispatch a tool call by name. Provided -- no changes needed."""
    _registry = {
        "query_plans":      query_plans,
        "query_promotions": query_promotions,
    }
    if tool_name not in _registry:
        return f"Unknown tool: {tool_name}"
    try:
        return _registry[tool_name].invoke(tool_args)
    except Exception as e:
        return f"Tool error ({tool_name}): {e}"
