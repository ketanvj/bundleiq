"""
app.py
------
STARTER FILE -- Session 13: Streamlit Frontend + Human-in-the-Loop.

What is already provided (no changes needed):
  - compliance_badge()    returns a display badge string for compliance_status
  - format_route_label()  formats the routing info into one line
  - _init_session()       initialises graph + session state on first load
  - _sidebar()            renders the sidebar with agent descriptions
  - _render_history()     renders previous chat turns
  - _handle_hitl()        shows the HITL approval form when compliance revision needed
  - main()                wires everything together

Your task (3 TODOs):
  TODO 1: Implement build_input_state(message)
          Return the dict that graph.invoke() expects as its first argument.
          Fields: customer_message, response, specialist, retrieved_docs, compliance_status

  TODO 2: Implement get_thread_config(thread_id)
          Return {"configurable": {"thread_id": thread_id}}

  TODO 3: Implement needs_human_review(result)
          Return True when result["compliance_status"] == "REVISED"
          (this triggers the HITL approval step)

Run when done:
    streamlit run app.py   (from inside s13/starter/)
"""
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from bundleiq.agent import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# TODO 1 of 3 -- Implement build_input_state()
# ---------------------------------------------------------------------------
# Return a dict with these keys (the graph's initial state):
#   "customer_message":  message   (the user's text)
#   "response":          ""        (empty -- agent fills this in)
#   "specialist":        ""        (empty -- supervisor fills this)
#   "retrieved_docs":    []        (empty list)
#   "compliance_status": ""        (empty -- compliance agent fills this)
#
# Template:
#   def build_input_state(message: str) -> dict:
#       return {
#           "customer_message":  message,
#           "response":          "",
#           "specialist":        "",
#           "retrieved_docs":    [],
#           "compliance_status": "",
#       }
# ---------------------------------------------------------------------------
def build_input_state(message: str) -> dict:
    raise NotImplementedError("TODO 1: implement build_input_state()")


# ---------------------------------------------------------------------------
# TODO 2 of 3 -- Implement get_thread_config()
# ---------------------------------------------------------------------------
# LangGraph needs a thread ID to keep memory across turns in the same session.
# Return {"configurable": {"thread_id": thread_id}}
#
# Template:
#   def get_thread_config(thread_id: str) -> dict:
#       return {"configurable": {"thread_id": thread_id}}
# ---------------------------------------------------------------------------
def get_thread_config(thread_id: str) -> dict:
    raise NotImplementedError("TODO 2: implement get_thread_config()")


# ---------------------------------------------------------------------------
# TODO 3 of 3 -- Implement needs_human_review()
# ---------------------------------------------------------------------------
# Return True when the Compliance Agent revised the response.
# The human operator must approve before the response is shown to the customer.
#
# Hint: check result.get("compliance_status", "") == "REVISED"
#
# Template:
#   def needs_human_review(result: dict) -> bool:
#       return result.get("compliance_status", "") == "REVISED"
# ---------------------------------------------------------------------------
def needs_human_review(result: dict) -> bool:
    raise NotImplementedError("TODO 3: implement needs_human_review()")


# ---------------------------------------------------------------------------
# Already implemented -- no changes needed below this line
# ---------------------------------------------------------------------------

def compliance_badge(status: str) -> str:
    if status == "PASS":
        return "✅ Compliant"
    if status == "REVISED":
        return "⚠️ Revised"
    if status.startswith("FAIL"):
        return "❌ Violation"
    return ""


def format_route_label(result: dict) -> str:
    qt    = result.get("query_type", "—")
    sp    = result.get("specialist", "—")
    cs    = result.get("compliance_status", "")
    badge = compliance_badge(cs)
    label = f"Route: {qt} → {sp}"
    if badge:
        label += f" | {badge}"
    return label


def _init_session() -> None:
    if "graph" not in st.session_state:
        from langgraph.checkpoint.memory import MemorySaver
        st.session_state.graph     = build_graph(checkpointer=MemorySaver())
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages  = []
        st.session_state.routes    = []


def _sidebar() -> None:
    with st.sidebar:
        st.header("📱 BundleIQ")
        st.caption("TeleConnect Customer Assistant")
        st.divider()
        if st.button("🔄 New Conversation", use_container_width=True):
            for key in ["graph", "thread_id", "messages", "routes", "pending_hitl"]:
                st.session_state.pop(key, None)
            st.rerun()
        if "thread_id" in st.session_state:
            st.caption(f"Session: {st.session_state.thread_id[:8]}…")
        st.divider()
        st.subheader("Agents")
        st.markdown(
            "- **Supervisor** — classifies query\n"
            "- **Plans Agent** — handles plan queries\n"
            "- **Promotions Agent** — handles offers\n"
            "- **Compliance Agent** — TRAI rules check\n"
            "- **Human-in-the-Loop** — reviews revisions"
        )


def _render_history() -> None:
    messages = st.session_state.get("messages", [])
    routes   = st.session_state.get("routes",   [])
    assistant_idx = 0
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if assistant_idx < len(routes):
                st.caption(routes[assistant_idx])
            assistant_idx += 1


def _handle_hitl() -> bool:
    if "pending_hitl" not in st.session_state:
        return False
    pending = st.session_state.pending_hitl
    st.warning(
        "⚠️ **Compliance Review Required** — The Compliance Agent revised this response. "
        "Please review and approve before sending to the customer."
    )
    with st.form("hitl_approval"):
        edited    = st.text_area("Review and edit the response if needed:", value=pending["response"], height=220)
        col1, col2 = st.columns(2)
        approved  = col1.form_submit_button("✅ Approve & Send", use_container_width=True)
        discarded = col2.form_submit_button("❌ Discard",         use_container_width=True)
    if approved:
        st.session_state.messages.append({"role": "assistant", "content": edited})
        st.session_state.routes.append(pending["route_label"])
        del st.session_state.pending_hitl
        st.rerun()
    elif discarded:
        del st.session_state.pending_hitl
        st.rerun()
    return True


def main() -> None:
    st.set_page_config(page_title="BundleIQ | TeleConnect", page_icon="📱", layout="wide")
    st.title("📱 BundleIQ | TeleConnect")
    st.caption("AI-powered plan assistant — Session 13: Streamlit UI + Human-in-the-Loop")

    _init_session()
    _sidebar()
    _render_history()

    hitl_active = _handle_hitl()

    if not hitl_active:
        prompt = st.chat_input("Ask about your TeleConnect plan, offers, or services…")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("BundleIQ is thinking…"):
                result = st.session_state.graph.invoke(
                    build_input_state(prompt),
                    config=get_thread_config(st.session_state.thread_id),
                )

            route_label = format_route_label(result)

            if needs_human_review(result):
                st.session_state.pending_hitl = {
                    "response":    result["response"],
                    "route_label": route_label,
                }
                st.rerun()
            else:
                response = result["response"]
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.caption(route_label)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.routes.append(route_label)


if __name__ == "__main__":
    main()
