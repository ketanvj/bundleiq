"""
bundleiq/nodes.py
-----------------
Graph nodes for BundleIQ multi-agent architecture (Session 14).

Session 14 adds an Input Guard node (guard) as the new entry point.
The guard runs before the classifier and blocks:
  - PII (Aadhaar/PAN) — DPDP Act 2023
  - Prompt injection / jailbreak — regex Layer 1
  - Semantic injection — Llama Prompt Guard 2 Layer 2

Supervisor routes to:
  - Plans Agent      (query_plans MCP tool)
  - Promotions Agent (query_promotions MCP tool)
  → then always through Compliance Agent (TRAI phrase + price check, revision if needed)
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
    LLAMAGUARD_THRESHOLD,
    PLANS_SYSTEM_PROMPT,
    PROMOTIONS_SYSTEM_PROMPT,
    PII_PATTERNS,
    SAFE_COMPLIANCE_RESPONSE,
)
from .state import BundleIQState
from .tools import _run_tool, classifier_llm, llamaguard_llm, llm, llm_with_tools

# ---------------------------------------------------------------------------
# S13: Token streaming hook
# ---------------------------------------------------------------------------
_stream_callback: Optional[Callable[[str], None]] = None

# Pre-compile guard patterns once at module load
_pii_compiled       = [re.compile(p)               for p in PII_PATTERNS]
_injection_compiled = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# OWASP LLM01:2026 — strip invisible Unicode used to smuggle injection payloads
_INVISIBLE_UNICODE_RE = re.compile(
    "[\U000E0000-\U000E007F︀-️​-‍⁠]"
)


# ---------------------------------------------------------------------------
# S14: Input Guard — two-layer defence
# ---------------------------------------------------------------------------

def _llamaguard_safe(message: str) -> tuple[bool, float]:
    """Call Llama Prompt Guard 2 via Groq and return (is_safe, score).

    Returns a probability (0.0–1.0) that the message is a prompt injection.
    Scores above LLAMAGUARD_THRESHOLD (0.5) are treated as injection.
    Fail-open on any API error (returns True, -1.0).
    """
    try:
        result = llamaguard_llm.invoke([HumanMessage(content=message)])
        score  = float(result.content.strip())
        safe   = score < LLAMAGUARD_THRESHOLD
        print(f"[BundleIQ] LlamaPromptGuard: score={score:.4f} → {'safe' if safe else 'INJECTION'}")
        return safe, score
    except Exception as e:
        print(f"[BundleIQ] LlamaPromptGuard unavailable — defaulting to safe: {e}")
        return True, -1.0


@traceable(name="input_guard")
def guard(state: BundleIQState) -> dict:
    """Inspect customer_message for PII, injection patterns, and unsafe content.

    Returns {"blocked_reason": "", "llamaguard_score": float} always.
    blocked_reason is "pii"|"injection"|"llamaguard" when blocked, "" when clean.
    llamaguard_score is the raw probability (0.0–1.0); -1.0 if Layer 2 was not reached.
    """
    raw = state["customer_message"]

    # Strip invisible Unicode then NFKD-normalize to collapse compatibility chars
    msg = unicodedata.normalize("NFKD", _INVISIBLE_UNICODE_RE.sub("", raw))

    # Layer 1a: PII — identifier must not reach the LLM.
    for rx in _pii_compiled:
        if rx.search(msg):
            print("[BundleIQ] Guard: PII detected — blocked")
            return {"blocked_reason": "pii", "llamaguard_score": -1.0}

    # Layer 1b: Injection / jailbreak / persona-hijack — always case-insensitive.
    for rx in _injection_compiled:
        if rx.search(msg):
            print("[BundleIQ] Guard: injection (regex) detected — blocked")
            return {"blocked_reason": "injection", "llamaguard_score": -1.0}

    # Layer 2: Llama Prompt Guard 2 — semantic injection detection.
    safe, score = _llamaguard_safe(msg)
    if not safe:
        print("[BundleIQ] Guard: jailbreak (LlamaPromptGuard) detected — blocked")
        return {"blocked_reason": "llamaguard", "llamaguard_score": score}

    return {"blocked_reason": "", "llamaguard_score": score}


def blocked(state: BundleIQState) -> dict:
    """Return the appropriate canned response for a blocked message."""
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
# Specialist agent helpers (unchanged from S13)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Agent factory functions
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
# Compliance helpers (unchanged from S13)
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


# ---------------------------------------------------------------------------
# Compliance Agent node functions
# ---------------------------------------------------------------------------

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


def create_compliance_agent():
    builder = StateGraph(BundleIQState)

    builder.add_node("check_trai", check_trai)
    builder.add_node("revise",     revise_response)

    builder.set_entry_point("check_trai")
    builder.add_conditional_edges(
        "check_trai",
        route_compliance,
        {"revise": "revise", END: END},
    )
    builder.add_edge("revise", END)

    return builder.compile()


_compliance_agent = create_compliance_agent()


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
        "blocked_reason":    "",
        "llamaguard_score":  -1.0,
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
        "llamaguard_score":  -1.0,
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
        "llamaguard_score":  -1.0,
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
