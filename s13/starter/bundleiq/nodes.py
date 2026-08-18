"""
bundleiq/nodes.py
-----------------
STARTER FILE -- your task is to implement the two TODO sections below.

Session 12 adds a Compliance Agent sub-graph that post-processes every
specialist response before it reaches the customer.

What is already done for you
  - _plans_respond / _promotions_respond and their factory functions (from S10)
  - _load_valid_prices, _extract_prices, _check_compliance_logic, check_trai,
    revise_response, route_compliance (compliance helpers)
  - call_plans_agent, call_promotions_agent, classify, escalate, decline (supervisor nodes)

Your task
  TODO 1 of 2: Implement create_compliance_agent()
    - Build a StateGraph(BundleIQState)
    - Nodes: "check_trai" (check_trai), "revise" (revise_response)
    - Entry point: "check_trai"
    - add_conditional_edges from "check_trai" using route_compliance:
        {"revise": "revise", END: END}
    - Edge: revise -> END
    - Return builder.compile()

  TODO 2 of 2: Implement call_compliance_agent() supervisor node
    - Invoke _compliance_agent with the full state dict
    - Return {"response": ..., "compliance_status": ...}
"""
import re
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from langgraph.graph import END, StateGraph

from .config import (
    BUNDLEIQ_BANNED_PHRASES,
    CLASSIFY_SYSTEM,
    DB_PATH,
    DECLINE_RESPONSE,
    ESCALATE_RESPONSE,
    PLANS_SYSTEM_PROMPT,
    PROMOTIONS_SYSTEM_PROMPT,
    SAFE_COMPLIANCE_RESPONSE,
)
from .state import BundleIQState
from .tools import _run_tool, classifier_llm, llm, llm_with_tools


def _agent_respond(state: BundleIQState, system_prompt: str, label: str) -> dict:
    """Shared respond logic for both specialist agents."""
    history  = state.get("history", [])
    messages = [SystemMessage(content=system_prompt)]
    for turn in history:
        messages.append(
            HumanMessage(content=turn["content"]) if turn["role"] == "user"
            else AIMessage(content=turn["content"])
        )
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result = llm_with_tools.invoke(messages)

        if result.tool_calls:
            messages.append(result)
            for tc in result.tool_calls:
                tool_output = _run_tool(tc["name"], tc["args"])
                print(
                    f"[BundleIQ] {label} MCP: {tc['name']}({tc['args']}) "
                    f"-> {str(tool_output)[:80]}"
                )
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))
            result = llm.invoke(messages)

        response_text = result.content

    except Exception as e:
        print(f"[BundleIQ] {label} LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    return {
        "response": response_text,
        "history":  history + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": response_text},
        ],
    }


def _plans_respond(state: BundleIQState) -> dict:
    return _agent_respond(state, PLANS_SYSTEM_PROMPT, "Plans Agent")


def _promotions_respond(state: BundleIQState) -> dict:
    return _agent_respond(state, PROMOTIONS_SYSTEM_PROMPT, "Promotions Agent")


# ---------------------------------------------------------------------------
# Agent factory functions (specialist agents)
# ---------------------------------------------------------------------------

def create_plans_agent():
    builder = StateGraph(BundleIQState)
    builder.add_node("respond", _plans_respond)
    builder.set_entry_point("respond")
    builder.add_edge("respond", END)
    return builder.compile()


def create_promotions_agent():
    builder = StateGraph(BundleIQState)
    builder.add_node("respond", _promotions_respond)
    builder.set_entry_point("respond")
    builder.add_edge("respond", END)
    return builder.compile()


_plans_agent      = create_plans_agent()
_promotions_agent = create_promotions_agent()


# ---------------------------------------------------------------------------
# Compliance helpers (provided -- no changes needed)
# ---------------------------------------------------------------------------

def _load_valid_prices() -> set:
    try:
        conn         = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        mob_prices   = {row[0] for row in conn.execute("SELECT price FROM mobile_plans").fetchall()}
        bb_prices    = {row[0] for row in conn.execute("SELECT monthly_price FROM broadband_plans").fetchall()}
        dev_prices   = {row[0] for row in conn.execute("SELECT price FROM devices").fetchall()}
        bndl_prices  = {row[0] for row in conn.execute("SELECT bundle_price FROM bundles").fetchall()}
        promo_values = set()
        for row in conn.execute("SELECT discount_value FROM promotions").fetchall():
            try:
                promo_values.add(int(row[0]))
            except (ValueError, TypeError):
                pass
        conn.close()
        return mob_prices | bb_prices | dev_prices | bndl_prices | promo_values
    except Exception:
        return set()


def _extract_prices(text: str) -> list:
    matches = re.findall(r"(?:Rs\.|₹)\s*(\d+(?:,\d+)*)", text, re.IGNORECASE)
    result = []
    for m in matches:
        try:
            result.append(int(m.replace(",", "")))
        except ValueError:
            pass
    return result


@traceable(name="trai_compliance_check")
def _check_compliance_logic(draft: str) -> tuple:
    lower = draft.lower()

    for phrase in BUNDLEIQ_BANNED_PHRASES:
        if phrase in lower:
            return False, f"banned phrase: '{phrase}'"

    mentioned_prices = _extract_prices(draft)
    if mentioned_prices:
        valid_prices = _load_valid_prices()
        if valid_prices:
            for price in mentioned_prices:
                if price not in valid_prices:
                    return False, f"incorrect price: Rs. {price} not in product catalogue"

    return True, "PASS"


def check_trai(state: BundleIQState) -> dict:
    draft          = state["response"]
    passed, reason = _check_compliance_logic(draft)

    if not passed:
        print(f"[BundleIQ] Compliance FAIL: {reason}")
        return {"compliance_status": f"FAIL: {reason}"}

    print("[BundleIQ] Compliance PASS")
    return {"compliance_status": "PASS"}


def revise_response(state: BundleIQState) -> dict:
    draft  = state["response"]
    reason = state.get("compliance_status", "violation").replace("FAIL: ", "")

    prompt = (
        "You are a TeleConnect India compliance officer reviewing an AI assistant response.\n\n"
        f"The response was flagged for: {reason}\n\n"
        "Rewrite it to fix the violation while keeping the response helpful.\n\n"
        "Rules:\n"
        "  1. Never guarantee coverage, signal quality, or network availability.\n"
        "  2. Only state prices that appeared in the original response -- do not change them.\n"
        "  3. Keep the rewritten response under 150 words.\n"
        "  4. End with 'BundleIQ | TeleConnect India'\n\n"
        f"Original response:\n{draft}\n\n"
        "Compliant rewrite:"
    )

    try:
        result       = llm.invoke([HumanMessage(content=prompt)])
        revised_text = result.content.strip() or SAFE_COMPLIANCE_RESPONSE
    except Exception as e:
        print(f"[BundleIQ] Compliance Agent revision error: {e}")
        revised_text = SAFE_COMPLIANCE_RESPONSE

    print("[BundleIQ] Compliance Agent: response revised")
    return {
        "response":          revised_text,
        "compliance_status": "REVISED",
    }


def route_compliance(state: BundleIQState) -> str:
    return "revise" if state.get("compliance_status", "").startswith("FAIL") else END


# ---------------------------------------------------------------------------
# TODO 1 of 2 -- Implement create_compliance_agent()
# ---------------------------------------------------------------------------
#   Steps:
#     builder = StateGraph(BundleIQState)
#     Add node "check_trai" -> check_trai
#     Add node "revise"     -> revise_response
#     Set entry point to "check_trai"
#     add_conditional_edges("check_trai", route_compliance, {"revise": "revise", END: END})
#     Add edge "revise" -> END
#     Return builder.compile()
# ---------------------------------------------------------------------------
def create_compliance_agent():
    raise NotImplementedError("TODO 1: implement create_compliance_agent()")


_compliance_agent = None  # TODO 1: set to create_compliance_agent()


# ---------------------------------------------------------------------------
# Supervisor nodes
# ---------------------------------------------------------------------------

def classify(state: BundleIQState) -> dict:
    messages = [SystemMessage(content=CLASSIFY_SYSTEM)]
    for turn in state.get("history", [])[-2:]:
        messages.append(
            HumanMessage(content=turn["content"]) if turn["role"] == "user"
            else AIMessage(content=turn["content"])
        )
    messages.append(HumanMessage(content=state["customer_message"]))
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in {"PLANS", "PROMOTIONS", "COMPLEX", "OUT_OF_SCOPE"}:
            query_type = "PLANS"
    except Exception as e:
        print(f"[BundleIQ] Supervisor classification error: {e}")
        query_type = "PLANS"
    return {"query_type": query_type}


def call_plans_agent(state: BundleIQState) -> dict:
    print("[BundleIQ] Supervisor -> Plans Agent")
    result = _plans_agent.invoke({
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          "",
        "query_type":        state.get("query_type", "PLANS"),
        "retrieved_docs":    [],
        "specialist":        "",
        "compliance_status": "",
    })
    return {
        "response":   result["response"],
        "history":    result.get("history", state.get("history", [])),
        "specialist": "plans_agent",
    }


def call_promotions_agent(state: BundleIQState) -> dict:
    print("[BundleIQ] Supervisor -> Promotions Agent")
    result = _promotions_agent.invoke({
        "customer_message":  state["customer_message"],
        "history":           state.get("history", []),
        "response":          "",
        "query_type":        state.get("query_type", "PROMOTIONS"),
        "retrieved_docs":    [],
        "specialist":        "",
        "compliance_status": "",
    })
    return {
        "response":   result["response"],
        "history":    result.get("history", state.get("history", [])),
        "specialist": "promotions_agent",
    }


# ---------------------------------------------------------------------------
# TODO 2 of 2 -- Implement call_compliance_agent() supervisor node
# ---------------------------------------------------------------------------
#   Steps:
#     Print "[BundleIQ] Supervisor -> Compliance Agent"
#     Invoke _compliance_agent with a dict containing all BundleIQState fields
#     Return {"response": result["response"], "compliance_status": result.get("compliance_status", "PASS")}
# ---------------------------------------------------------------------------
def call_compliance_agent(state: BundleIQState) -> dict:
    raise NotImplementedError("TODO 2: implement call_compliance_agent()")


def escalate(state: BundleIQState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history, "specialist": "escalated"}


def decline(state: BundleIQState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history, "specialist": "declined"}


def route_supervisor(state: BundleIQState) -> str:
    qt = state.get("query_type", "PLANS")
    if qt == "PROMOTIONS":
        return "call_promotions_agent"
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "call_plans_agent"
