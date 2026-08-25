"""
bundleiq/nodes.py  (Session 14 — starter)
------------------------------------------
The guard functions below have stub skeletons — complete the TODOs.
Everything else (compliance, specialist agents, supervisor) is unchanged from S13.
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

_stream_callback: Optional[Callable[[str], None]] = None

_pii_compiled       = [re.compile(p)               for p in PII_PATTERNS]
_injection_compiled = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

_INVISIBLE_UNICODE_RE = re.compile(
    "[\U000E0000-\U000E007F︀-️​-‍⁠]"
)


# ---------------------------------------------------------------------------
# S14 TODO: implement the four guard functions below
# ---------------------------------------------------------------------------

def _llamaguard_safe(message: str) -> tuple[bool, float]:
    """Call Llama Prompt Guard 2 via Groq and return (is_safe, score).

    TODO:
      - Invoke llamaguard_llm with [HumanMessage(content=message)]
      - result.content is a float string — the injection probability (e.g. "0.9996")
      - Parse it: score = float(result.content.strip())
      - Return (True, score) if score < LLAMAGUARD_THRESHOLD (0.5), else (False, score)
      - Wrap in try/except; on any error print a warning and return (True, -1.0) — fail-open
      - Print: f"[BundleIQ] LlamaPromptGuard: score={score:.4f}"
    """
    raise NotImplementedError("TODO: implement _llamaguard_safe()")


@traceable(name="input_guard")
def guard(state: BundleIQState) -> dict:
    """Inspect customer_message for PII, injection patterns, and unsafe content.

    Two-layer defence:
      Layer 1 (regex, < 1 ms):
        1a. Strip invisible Unicode and NFKD-normalize the raw message.
        1b. Loop through _pii_compiled. If any matches, return
            {"blocked_reason": "pii", "llamaguard_score": -1.0}.
        1c. Loop through _injection_compiled. If any matches, return
            {"blocked_reason": "injection", "llamaguard_score": -1.0}.
      Layer 2 (Llama Prompt Guard 2, semantic):
        2.  Call _llamaguard_safe(msg) -> (safe, score).
            If not safe, return {"blocked_reason": "llamaguard", "llamaguard_score": score}.

    Return {"blocked_reason": "", "llamaguard_score": score} if all layers pass.
    Print a short log line when something is blocked.

    TODO: implement this function.
    """
    raise NotImplementedError("TODO: implement guard()")


def blocked(state: BundleIQState) -> dict:
    """Return the appropriate canned response for a blocked message.

    TODO:
      - If blocked_reason is "pii",        set response = GUARD_PII_RESPONSE.
      - If blocked_reason is "llamaguard", set response = GUARD_UNSAFE_RESPONSE.
      - Otherwise (injection),             set response = GUARD_BLOCKED_RESPONSE.
      - Return response, specialist="guard", and updated history (append user + assistant turn).
    """
    raise NotImplementedError("TODO: implement blocked()")


def route_guard(state: BundleIQState) -> str:
    """Return "blocked" if blocked_reason is set, else "classify".

    TODO: check state.get("blocked_reason") and return the correct string.
    """
    raise NotImplementedError("TODO: implement route_guard()")


# ---------------------------------------------------------------------------
# Specialist agent helpers (unchanged from S13 — do not edit)
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
# Compliance helpers (unchanged from S13 — do not edit)
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
# Supervisor nodes (unchanged from S13 — do not edit)
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
