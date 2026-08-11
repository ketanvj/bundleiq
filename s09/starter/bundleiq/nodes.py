"""
bundleiq/nodes.py
-----------------
STARTER FILE -- implement the two TODO sections to complete the compliance layer.

Session 9 adds a TRAI compliance filter to BundleIQ:
  - _check_compliance() scans every LLM response for banned phrases and incorrect prices
  - check_compliance() is a LangGraph node that calls _check_compliance() and replaces
    non-compliant responses with a safe fallback

What is already provided
  - _BANNED_PATTERN     : compiled regex of all banned phrases (ready to use)
  - _normalize_for_check(): Unicode normalization + dash unification
  - _load_valid_prices()  : reads all valid prices from teleconnect_data.db
  - _extract_prices()     : extracts Rs./₹ prices from a response string
  - All S08 nodes unchanged (classify, retrieve_docs, respond, escalate, decline)

Your task
  TODO 1: Implement _check_compliance(draft) -> tuple[bool, str]
  TODO 2: Implement check_compliance(state) -> dict  (the LangGraph node)
"""
import re
import sqlite3
import unicodedata

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable

from .config import (
    BUNDLEIQ_BANNED_PHRASES,
    CLASSIFY_SYSTEM,
    DB_PATH,
    DECLINE_RESPONSE,
    EMBED_MODEL,
    ESCALATE_RESPONSE,
    RETRIEVAL_K,
    SAFE_COMPLIANCE_RESPONSE,
    SYSTEM_PROMPT,
    VECTORSTORE_DIR,
)
from .state import BundleIQState
from .tools import _run_tool, classifier_llm, llm, llm_with_tools

vectorstore = None

_BANNED_PATTERN: re.Pattern = re.compile(
    "|".join(re.escape(p) for p in BUNDLEIQ_BANNED_PHRASES),
    re.IGNORECASE,
)


def _init_vectorstore() -> None:
    global vectorstore
    if vectorstore is not None:
        return
    try:
        embeddings  = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=embeddings,
        )
    except Exception as e:
        print(f"[BundleIQ] Could not load vectorstore: {e}")
        print("  Run 'python data/ingest.py' to create it.")


def _normalize_for_check(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for ch in "‐‑‒–—―−":
        text = text.replace(ch, "-")
    return text.lower()


def _load_valid_prices() -> set:
    try:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
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


# ---------------------------------------------------------------------------
# TODO 1 of 2 -- Implement _check_compliance()
# ---------------------------------------------------------------------------
# 1. Normalize: normalized = _normalize_for_check(draft)
# 2. Banned phrase check: match = _BANNED_PATTERN.search(normalized)
#    if match: return False, f"banned phrase: '{match.group()}'"
# 3. Price check: prices = _extract_prices(draft)  ← use original draft, not normalized
#    if prices:
#        valid = _load_valid_prices()
#        if valid:
#            for price in prices:
#                if price not in valid:
#                    return False, f"incorrect price: Rs. {price} not in product catalogue"
# 4. return True, "PASS"
# ---------------------------------------------------------------------------
@traceable(name="trai_compliance_check")
def _check_compliance(draft: str) -> tuple:
    raise NotImplementedError("TODO 1: implement _check_compliance()")


# ---------------------------------------------------------------------------
# TODO 2 of 2 -- Implement check_compliance() node
# ---------------------------------------------------------------------------
# Call _check_compliance(state["response"]).
# On FAIL: print the reason, return SAFE_COMPLIANCE_RESPONSE and
#          compliance_status = f"FAIL: {reason}"
# On PASS: print "Compliance PASS", return compliance_status = "PASS"
# ---------------------------------------------------------------------------
def check_compliance(state: BundleIQState) -> dict:
    raise NotImplementedError("TODO 2: implement check_compliance() node")


def classify(state: BundleIQState) -> dict:
    messages = [
        SystemMessage(content=CLASSIFY_SYSTEM),
        HumanMessage(content=state["customer_message"]),
    ]
    try:
        result     = classifier_llm.invoke(messages)
        query_type = result.content.strip().upper()
        if query_type not in {"SIMPLE", "COMPLEX", "OUT_OF_SCOPE"}:
            query_type = "SIMPLE"
    except Exception as e:
        print(f"[BundleIQ] Classification error: {e}")
        query_type = "SIMPLE"
    return {"query_type": query_type}


def retrieve_docs(state: BundleIQState) -> dict:
    _init_vectorstore()
    if vectorstore is None:
        return {"retrieved_docs": []}
    try:
        docs      = vectorstore.similarity_search(state["customer_message"], k=RETRIEVAL_K)
        retrieved = [
            f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ]
    except Exception as e:
        print(f"[BundleIQ] Retrieval error: {e}")
        retrieved = []
    return {"retrieved_docs": retrieved}


def respond(state: BundleIQState) -> dict:
    history   = state.get("history", [])
    retrieved = state.get("retrieved_docs", [])

    if retrieved:
        context_block  = "\n\n---\n\n".join(retrieved)
        system_content = (
            SYSTEM_PROMPT
            + "\n\nThe following sections from TeleConnect's policy documents are relevant "
              "to the customer's question. Use this information in your answer:\n\n"
            + context_block
        )
    else:
        system_content = SYSTEM_PROMPT

    messages = [SystemMessage(content=system_content)]
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=state["customer_message"]))

    try:
        result = llm_with_tools.invoke(messages)

        if result.tool_calls:
            messages.append(result)
            for tc in result.tool_calls:
                tool_output = _run_tool(tc["name"], tc["args"])
                print(
                    f"[BundleIQ] MCP tool: {tc['name']}({tc['args']}) "
                    f"-> {str(tool_output)[:80]}"
                )
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc["id"]))
            result = llm.invoke(messages)

        response_text = result.content

    except Exception as e:
        print(f"[BundleIQ] LLM error: {e}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    new_history = history + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, "history": new_history}


def escalate(state: BundleIQState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history}


def decline(state: BundleIQState) -> dict:
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history}


def route_query(state: BundleIQState) -> str:
    qt = state.get("query_type", "SIMPLE")
    if qt == "COMPLEX":
        return "escalate"
    if qt == "OUT_OF_SCOPE":
        return "decline"
    return "retrieve_docs"
