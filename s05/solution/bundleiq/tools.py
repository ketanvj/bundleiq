import os
import sqlite3

from langchain_core.tools import tool
from langchain_groq import ChatGroq

from .config import CLASSIFIER_MAX_TOKENS, CLASSIFIER_MODEL, DB_PATH, MODEL_NAME, TEMPERATURE, MAX_TOKENS

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
    model=CLASSIFIER_MODEL,
    temperature=0.0,
    max_tokens=CLASSIFIER_MAX_TOKENS,
)


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
    conn  = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    lines = []

    if plan_type in ("mobile", "all"):
        rows = conn.execute(
            "SELECT name, data_gb_per_day, validity_days, price, is_5g "
            "FROM mobile_plans ORDER BY price"
        ).fetchall()
        for name, data_gb, validity, price, is_5g in rows:
            data_str = f"{data_gb}GB/day" if data_gb else "unlimited data"
            flag     = " (5G)" if is_5g else ""
            lines.append(f"{name}{flag}: Rs. {price} -- {data_str}, valid {validity} days")

    if plan_type in ("broadband", "all"):
        rows = conn.execute(
            "SELECT name, speed_mbps, monthly_price, installation_fee "
            "FROM broadband_plans ORDER BY monthly_price"
        ).fetchall()
        for name, speed, price, install_fee in rows:
            install_str = f"installation Rs. {install_fee}" if install_fee else "free installation"
            lines.append(f"{name}: Rs. {price}/mo -- {speed} Mbps, {install_str}")

    conn.close()
    return "\n".join(lines) if lines else "No plan data found."


@tool
def query_promotions(plan_id: str = "all") -> str:
    """Fetch current TeleConnect promotions from the database.

    Args:
        plan_id: Filter promotions for a specific plan ID (e.g. "mob_001").
                 Use "all" to return every active promotion.

    Returns formatted promotion information as a plain-text string.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)

    if plan_id.lower() == "all":
        rows = conn.execute(
            "SELECT name, eligible_plan_ids, discount_type, discount_value, valid_until "
            "FROM promotions ORDER BY valid_until"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT name, eligible_plan_ids, discount_type, discount_value, valid_until "
            "FROM promotions WHERE eligible_plan_ids LIKE ? ORDER BY valid_until",
            (f"%{plan_id}%",),
        ).fetchall()

    conn.close()

    if not rows:
        return f"No active promotions found for plan: '{plan_id}'." if plan_id != "all" else "No active promotions."

    parts = []
    for name, eligible, discount_type, discount_value, valid_until in rows:
        parts.append(
            f"{name}\n"
            f"  Eligible plans: {eligible}\n"
            f"  Discount: {discount_type} -- {discount_value}\n"
            f"  Valid until: {valid_until}"
        )
    return "\n\n".join(parts)


llm_with_tools = llm.bind_tools([query_plans, query_promotions])


def _run_tool(tool_name: str, tool_args: dict) -> str:
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
