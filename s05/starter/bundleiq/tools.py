"""
bundleiq/tools.py
-----------------
Tool definitions and LLM setup for BundleIQ.

Session 5 adds two @tool functions that give the LLM live access to the
TeleConnect database. Instead of relying only on RAG policy chunks (Session 4),
the LLM can now call query_plans() and query_promotions() to get exact data.

New this session:
  query_plans()     -- fetches current mobile and broadband plan details from the DB
  query_promotions()-- fetches current promotions and offers
  TOOLS             -- list of available tools (used by both bind_tools and _run_tool)
  llm_with_tools    -- the respond LLM with tools bound
  _run_tool()       -- helper that executes whichever tool the LLM requested

Carries forward from Session 4:
  llm             -- still used internally; llm_with_tools wraps it
  classifier_llm  -- unchanged: classifier does not use tools
"""
import os
import sqlite3

# langchain_core.tools is a submodule that ships several tool-related utilities:
#   tool            -- decorator used here: wraps a plain function into a LangChain Tool object
#   BaseTool        -- base class all tools inherit from (used internally by @tool)
#   StructuredTool  -- explicit alternative to @tool when you need more control
#   InjectedToolArg -- marks arguments the system fills in, not the LLM
# We only import `tool` because the @tool decorator handles everything automatically.
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from .config import (
    CLASSIFIER_MAX_TOKENS, CLASSIFIER_MODEL, CLASSIFIER_TEMPERATURE,
    DB_PATH, MAX_TOKENS, MODEL_NAME, TEMPERATURE,
)

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
    temperature=CLASSIFIER_TEMPERATURE,
    max_tokens=CLASSIFIER_MAX_TOKENS,
    reasoning_effort="low",
)


# ---------------------------------------------------------------------------
# TODO 1 of 4 -- implement query_plans()
# ---------------------------------------------------------------------------
# query_plans() fetches current TeleConnect plan details from the SQLite database.
#
# Decorate with @tool and give it a clear docstring -- the docstring becomes
# the tool's description in the JSON schema sent to the LLM. The LLM reads
# it to decide when and how to call the tool. Poor docstrings = wrong calls.
#
# Steps:
#
#   1. Decorate with @tool:
#        @tool
#        def query_plans(plan_type: str = "all") -> str:
#
#   2. Write a descriptive docstring explaining:
#        - what the function does (fetch TeleConnect plan details)
#        - what plan_type accepts: "mobile", "broadband", or "all"
#        - what the return value looks like
#
#   3. Open a SQLite connection (use check_same_thread=False):
#        conn  = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#        lines = []
#
#   4. Query mobile plans when plan_type is "mobile" or "all":
#        if plan_type in ("mobile", "all"):
#            rows = conn.execute(
#                "SELECT name, data_gb_per_day, call_type, validity_days, price, is_5g "
#                "FROM mobile_plans ORDER BY price"
#            ).fetchall()
#            if rows:
#                lines.append("Mobile Plans:")
#                for name, data_gb, call_type, validity, price, is_5g in rows:
#                    data_str = f"{data_gb} GB/day" if data_gb else "N/A"
#                    call_str = call_type if call_type else "No calls"
#                    g5_str   = "5G" if is_5g else "non-5G"
#                    lines.append(
#                        f"  {name}: Rs.{price} for {validity} days | {data_str} | {call_str} | {g5_str}"
#                    )
#
#   5. Query broadband plans when plan_type is "broadband" or "all":
#        if plan_type in ("broadband", "all"):
#            rows = conn.execute(
#                "SELECT name, speed_mbps, data_type, monthly_price, installation_fee "
#                "FROM broadband_plans ORDER BY monthly_price"
#            ).fetchall()
#            if rows:
#                lines.append("Broadband Plans:")
#                for name, speed, data_type, price, install_fee in rows:
#                    lines.append(
#                        f"  {name}: Rs.{price}/month | {speed} Mbps {data_type} | Installation: Rs.{install_fee}"
#                    )
#
#   6. Close the connection and return the result:
#        conn.close()
#        return "\n".join(lines) if lines else "No plan data found."
#
# WHY check_same_thread=False?
# SQLite raises ProgrammingError if a connection is used from a different thread
# than the one that created it. LangGraph runs tool calls in a thread-pool
# executor, so the tool may run on a different thread than the one that opened
# the connection. This flag disables that check.
# ---------------------------------------------------------------------------

# TODO 1: implement query_plans() below -- keep the @tool decorator and docstring
raise NotImplementedError("TODO 1: implement query_plans() in bundleiq/tools.py")


# ---------------------------------------------------------------------------
# TODO 2 of 4 -- implement query_promotions()
# ---------------------------------------------------------------------------
# query_promotions() fetches current promotions from the SQLite database.
#
# Steps:
#
#   1. Decorate with @tool:
#        @tool
#        def query_promotions() -> str:
#
#   2. Write a descriptive docstring explaining:
#        - what the function does (fetch current promotions and offers)
#        - no parameters needed
#        - what the return value looks like
#
#   3. Open connection (check_same_thread=False):
#        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
#
#   4. Fetch promotions:
#        rows = conn.execute(
#            "SELECT name, eligible_plan_ids, discount_type, discount_value, valid_until "
#            "FROM promotions ORDER BY valid_until"
#        ).fetchall()
#        conn.close()
#
#   5. Format each promotion as:
#        f"  {name}: {discount_type} of {discount_value} on plans {plan_ids} "
#        f"(valid until {valid_until})"
#      Prepend "Current Promotions:" as a header.
#
#   6. If no rows: return "No active promotions at this time."
#
# ---------------------------------------------------------------------------

# TODO 2: implement query_promotions() below -- keep the @tool decorator and docstring
raise NotImplementedError("TODO 2: implement query_promotions() in bundleiq/tools.py")


# ---------------------------------------------------------------------------
# TODO 3 of 4 -- TOOLS list + llm_with_tools + _run_tool()
# ---------------------------------------------------------------------------
# Three steps:
#
#   Step A: build the TOOLS list
#     Declaring TOOLS as a named list makes it easy to add or remove tools
#     in one place. Both bind_tools() and _run_tool's registry are built from
#     this single list, so you never have to update two places when adding
#     a new tool.
#
#        TOOLS = [
#            query_plans,
#            query_promotions,
#        ]
#
#   Step B: create llm_with_tools
#     bind_tools() reads each function's type hints and docstring, converts them
#     to a JSON schema, and includes that schema in every request sent to the LLM.
#     The LLM was fine-tuned on examples with this schema format, so it knows to
#     respond with a structured tool call instead of guessing the answer from memory.
#
#        llm_with_tools = llm.bind_tools(TOOLS)
#
#   Step C: implement _run_tool()
#
#        def _run_tool(tool_name: str, tool_args: dict) -> str:
#
#     Build a registry dict from TOOLS:
#        _registry = {t.name: t for t in TOOLS}
#     Return an error string if the tool name is unknown.
#     Otherwise call the tool's .invoke() method:
#        return _registry[tool_name].invoke(tool_args)
#     Wrap in try/except and return an error string on failure.
#
# WHY .invoke() and not a plain function call?
#   .invoke() validates arguments against the tool's schema before executing,
#   and it integrates with LangSmith — the tool call appears as a separate trace span.
# ---------------------------------------------------------------------------

# TODO 3: add TOOLS list, llm_with_tools, and _run_tool() below
raise NotImplementedError("TODO 3: add TOOLS, llm_with_tools, and _run_tool() in bundleiq/tools.py")
