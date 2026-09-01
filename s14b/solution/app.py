"""
app.py
------
Streamlit chat UI for BundleIQ — TeleConnect's AI plan assistant.

Session 14b: Security and Guardrails with LlamaGuard 3 8B.

Architecture:
  - Guard node runs before every query — regex (Layer 1) + LlamaGuard 3 8B (Layer 2)
  - Supervisor classifies clean queries
  - Plans / Promotions specialists handle them
  - Compliance Agent checks every draft response
  - Human-in-the-Loop: operator must approve compliance-revised responses

Run:
    streamlit run app.py   (from inside s14b/solution/)
"""
import hashlib
import re as _re
import sys
import time
from pathlib import Path
from uuid import uuid4

_SCRIPT_RE  = _re.compile(r"<script[^>]*>.*?</script>", _re.IGNORECASE | _re.DOTALL)
_STYLE_RE   = _re.compile(r"<style[^>]*>.*?</style>",  _re.IGNORECASE | _re.DOTALL)
_HTML_TAG_RE = _re.compile(r"<[^>]+>")


def _sanitise(text: str) -> str:
    """Strip script/style blocks and all HTML tags from LLM response before rendering."""
    text = _SCRIPT_RE.sub("", text)
    text = _STYLE_RE.sub("", text)
    return _HTML_TAG_RE.sub("", text)


def _pseudonymise(raw: str) -> str:
    """SHA-256 one-way hash of raw session UUID. Raw UUID never stored or traced."""
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from bundleiq.agent import build_graph  # noqa: E402
import bundleiq.nodes as _nodes         # noqa: E402


# ---------------------------------------------------------------------------
# Token streaming (carried forward from S13)
# ---------------------------------------------------------------------------

class _StreamingState:
    def __init__(self, placeholder, token_delay: float = 0.0) -> None:
        self._placeholder = placeholder
        self._text = ""
        self._delay = token_delay

    def __call__(self, token: str) -> None:
        self._text += token
        self._placeholder.markdown(self._text + "▌")
        if self._delay > 0:
            time.sleep(self._delay)

    @property
    def text(self) -> str:
        return self._text


# ---------------------------------------------------------------------------
# Helper functions (pure, testable -- no Streamlit calls)
# ---------------------------------------------------------------------------

def build_input_state(message: str) -> dict:
    """Return the initial state dict for graph.invoke()."""
    return {
        "customer_message":  message,
        "response":          "",
        "specialist":        "",
        "retrieved_docs":    [],
        "compliance_status": "",
        "blocked_reason":    "",
    }


def get_thread_config(thread_id: str) -> dict:
    """Return the LangGraph thread config dict."""
    return {"configurable": {"thread_id": thread_id}}


def compliance_badge(status: str) -> str:
    if status == "PASS":
        return "✅ Compliant"
    if status == "REVISED":
        return "⚠️ Revised"
    if status.startswith("FAIL"):
        return "❌ Violation"
    return ""


def guard_badge(blocked_reason: str) -> str:
    """Return a guard status badge. S14b: no score displayed (LlamaGuard 3 doesn't return a float)."""
    if blocked_reason == "pii":
        return "🔒 Blocked (PII)"
    if blocked_reason == "llamaguard":
        return "🤖 Blocked (LlamaGuard 3)"
    if blocked_reason:
        return "🛡️ Blocked (injection — regex)"
    return ""


def needs_human_review(result: dict) -> bool:
    return result.get("compliance_status", "") == "REVISED"


def format_route_label(result: dict) -> str:
    blocked_r = result.get("blocked_reason", "")
    if blocked_r:
        return f"Guard: {guard_badge(blocked_r)}"

    sp    = result.get("specialist", "—")
    cs    = result.get("compliance_status", "")
    badge = compliance_badge(cs)
    label = f"Route: {sp}"
    if badge:
        label += f" | {badge}"
    return label


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def _init_session() -> None:
    if "graph" not in st.session_state:
        from langgraph.checkpoint.memory import MemorySaver
        st.session_state.graph     = build_graph(checkpointer=MemorySaver())
        st.session_state.thread_id = _pseudonymise(str(uuid4()))
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
            "- **Guard** — regex + LlamaGuard 3 8B *(upgraded in S14b)*\n"
            "- **Supervisor** — classifies clean queries\n"
            "- **Plans Agent** — handles plan queries\n"
            "- **Promotions Agent** — handles offers\n"
            "- **Compliance Agent** — TRAI rules check\n"
            "- **Human-in-the-Loop** — reviews revisions"
        )

        st.divider()
        st.subheader("Demo settings")
        st.session_state["token_delay"] = st.slider(
            "Token delay (ms)",
            min_value=0, max_value=100, value=st.session_state.get("token_delay", 0),
            step=5,
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
        edited = st.text_area(
            "Review and edit the response if needed:",
            value=pending["response"],
            height=220,
        )
        col1, col2 = st.columns(2)
        approved  = col1.form_submit_button("✅ Approve & Send", use_container_width=True)
        discarded = col2.form_submit_button("❌ Discard",         use_container_width=True)

    if approved:
        st.session_state.messages.append({"role": "assistant", "content": _sanitise(edited)})
        st.session_state.routes.append(pending["route_label"])
        del st.session_state.pending_hitl
        st.rerun()
    elif discarded:
        del st.session_state.pending_hitl
        st.rerun()

    return True


def main() -> None:
    st.set_page_config(
        page_title="BundleIQ | TeleConnect",
        page_icon="📱",
        layout="wide",
    )
    st.title("📱 BundleIQ | TeleConnect")
    st.caption("AI-powered plan assistant — Session 14b: LlamaGuard 3 8B")

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

            with st.chat_message("assistant"):
                placeholder = st.empty()

            delay_ms = st.session_state.get("token_delay", 0)
            streamer = _StreamingState(placeholder, token_delay=delay_ms / 1000)
            _nodes._stream_callback = streamer
            try:
                result = st.session_state.graph.invoke(
                    build_input_state(prompt),
                    config=get_thread_config(st.session_state.thread_id),
                )
            finally:
                _nodes._stream_callback = None

            route_label = format_route_label(result)

            if needs_human_review(result):
                placeholder.empty()
                st.session_state.pending_hitl = {
                    "response":    result["response"],
                    "route_label": route_label,
                }
                st.rerun()
            else:
                safe_response = _sanitise(result["response"])
                placeholder.markdown(safe_response)
                st.caption(route_label)
                st.session_state.messages.append({"role": "assistant", "content": safe_response})
                st.session_state.routes.append(route_label)


if __name__ == "__main__":
    main()
