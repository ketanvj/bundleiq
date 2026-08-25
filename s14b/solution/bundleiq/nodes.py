"""
bundleiq/nodes.py  (Session 14b)
---------------------------------
S14b replaces Llama Prompt Guard 2 (S14) with LlamaGuard 3 8B.

The two-layer guard architecture is unchanged from S14:
  Layer 1 — regex: PII (DPDP Act 2023) + injection keywords (< 1 ms, zero cost)
  Layer 2 — LlamaGuard 3 8B: 13 safety categories, catches rephrased attacks

Key change in _llamaguard_safe():
  S14:  model returns a float string → threshold comparison → (bool, float)
  S14b: model returns "safe" or "unsafe\\nS<n>" → parse categories → bool

S6 exclusion — BundleIQ passes S6-only findings through to the router:
  LlamaGuard S6 (Specialized Advice) catches "Which plan is best for me?"
  BundleIQ's COMPLEX routing already handles this correctly: COMPLEX →
  escalate → TeleConnect advisor. Blocking at the guard gives worse UX.
  We block on all other categories (S1–S5, S7–S13).
"""
import re
import sqlite3
import unicodedata
from typing import Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from langgraph.graph import END, StateGraph

from .config import (
    BUNDLEIQ_BANNED_PHRASES,
    CLASSIFY_SYSTEM,
    DB_PATH,
    DECLINE_RESPONSE,
    ESCALATE_RESPONSE,
    GUARD_BLOCKED_RESPONSE,
    GUARD_PII_RESPONSE,
    GUARD_UNSAFE_RESPONSE,
    INJECTION_PATTERNS,
    PLANS_SYSTEM_PROMPT,
    PROMOTIONS_SYSTEM_PROMPT,
    PII_PATTERNS,
    SAFE_COMPLIANCE_RESPONSE,
)
from .state import BundleIQState
from .tools import _run_tool, classifier_llm, llamaguard_llm, llm, llm_with_tools

_stream_callback: Optional[Callable[[str], None]] = None

_pii_compiled       = [re.compile(p)               for p in PII_PATTERNS]
_injection_compiled = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

_INVISIBLE_UNICODE_RE = re.compile(
    "[\U000E0000-\U000E007F︀-️​-‍⁠]"
)


# ---------------------------------------------------------------------------
# S14b: Input Guard
# ---------------------------------------------------------------------------

def _llamaguard_safe(message: str) -> bool:
    """Call LlamaGuard 3 8B and return True if the message is safe.

    LlamaGuard 3 8B response format:
      "safe"            — message passes all 13 safety checks
      "unsafe\\nS6"    — flagged for one category (Specialized Advice)
      "unsafe\\nS1,S13" — flagged for multiple categories

    S6 exclusion — BundleIQ intentionally handles personalised plan queries:
      LlamaGuard S6 (Specialized Advice) catches "Which plan is best for me?"
      BundleIQ's COMPLEX → escalate routing already handles this correctly.
      If we block at the guard, the customer gets a generic blocked response
      instead of the more helpful escalation path to a TeleConnect advisor.
      So we pass S6-only findings through — the router handles them properly.

    Fail-open design: if LlamaGuard is unreachable, log a warning and let
    the message through. Service availability beats one missed safety check.
    """
    try:
        result  = llamaguard_llm.invoke([HumanMessage(content=message)])
        verdict = result.content.strip().lower()

        if verdict.startswith("safe"):
            print(f"[BundleIQ] LlamaGuard: {verdict!r} → safe")
            return True

        categories: set[str] = set()
        if "\n" in verdict:
            raw_cats = verdict.split("\n", 1)[1]
            categories = {c.strip() for c in raw_cats.split(",")}

        # S6 (Specialized Advice) is handled by COMPLEX routing — don't block
        non_s6 = categories - {"s6"}
        safe = len(non_s6) == 0
        print(f"[BundleIQ] LlamaGuard: {verdict!r} → {'safe (S6 handled by router)' if safe else 'UNSAFE'}")
        return safe
    except Exception as e:
        print(f"[BundleIQ] LlamaGuard unavailable — defaulting to safe: {e}")
        return True


@traceable(name="input_guard")
def guard(state: BundleIQState) -> dict:
    """Inspect customer_message for PII, injection patterns, and unsafe content.

    Returns {"blocked_reason": ""} for a clean message, or
    {"blocked_reason": "pii"|"injection"|"llamaguard"} when blocked.
    """
    raw = state["customer_message"]
    msg = unicodedata.normalize("NFKD", raw)

    for rx in _pii_compiled:
        if rx.search(msg):
            print("[BundleIQ] Guard: PII detected — blocked")
            return {"blocked_reason": "pii"}

    for rx in _injection_compiled:
        if rx.search(msg):
            print("[BundleIQ] Guard: injection detected — blocked")
            return {"blocked_reason": "injection"}

    if not _llamaguard_safe(msg):
        print("[BundleIQ] Guard: LlamaGuard flagged message — blocked")
        return {"blocked_reason": "llamaguard"}

    return {"blocked_reason": ""}


def blocked(state: BundleIQState) -> dict:
    reason = state.get("blocked_reason", "injection")
    if reason == "pii":
        response = GUARD_PII_RESPONSE
    elif reason == "llamaguard":
        response = GUARD_UNSAFE_RESPONSE
    else:
        response = GUARD_BLOCKED_RESPONSE
    return {
        "response":   response,
        "specialist": "guard",
        "history": state.get("history", []) + [
            {"role": "user",      "content": state["customer_message"]},
            {"role": "assistant", "content": response},
        ],
    }


def route_guard(state: BundleIQState) -> str:
    return "blocked" if state.get("blocked_reason") else "classify"


# ---------------------------------------------------------------------------
# Specialist agent helpers (unchanged from S14)
# ---------------------------------------------------------------------------

def _agent_respond(state: BundleIQState, system_prompt: str, label: str) -> dict:
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
            if _stream_callback is not None:
                response_text = ""
                for chunk in llm.stream(messages):
                    if chunk.content:
                        response_text += chunk.content
                        _stream_callback(chunk.content)
            else:
                response_text = llm.invoke(messages).content
        else:
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
# Compliance helpers (unchanged from S14)
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
    return {"response": revised_text, "compliance_status": "REVISED"}


def route_compliance(state: BundleIQState) -> str:
    return "revise" if state.get("compliance_status", "").startswith("FAIL") else END


def create_compliance_agent():
    builder = StateGraph(BundleIQState)
    builder.add_node("check_trai", check_trai)
    builder.add_node("revise",     revise_response)
    builder.set_entry_point("check_trai")
    builder.add_conditional_edges(
        "check_trai", route_compliance, {"revise": "revise", END: END},
    )
    builder.add_edge("revise", END)
    return builder.compile()


_compliance_agent = create_compliance_agent()


# ---------------------------------------------------------------------------
# Supervisor nodes (unchanged from S14)
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
        "blocked_reason":    "",
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
        "blocked_reason":    "",
    })
    return {
        "response":   result["response"],
        "history":    result.get("history", state.get("history", [])),
        "specialist": "promotions_agent",
    }


def call_compliance_agent(state: BundleIQState) -> dict:
    print("[BundleIQ] Supervisor -> Compliance Agent")
    result = _compliance_agent.invoke({
        "customer_message":  state["customer_message"],
        "response":          state["response"],
        "history":           state.get("history", []),
        "query_type":        state.get("query_type", ""),
        "retrieved_docs":    state.get("retrieved_docs", []),
        "specialist":        state.get("specialist", ""),
        "compliance_status": "",
        "blocked_reason":    "",
    })
    return {
        "response":          result["response"],
        "compliance_status": result.get("compliance_status", "PASS"),
    }


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
