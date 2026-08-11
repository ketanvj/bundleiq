"""
bundleiq/agent.py
-----------------
STARTER FILE -- implement TODO 3 to wire the compliance node into the graph.

Session 9: The check_compliance node is already implemented in nodes.py (once you
complete TODOs 1 and 2). Your remaining task here is to add it to the graph so
every SIMPLE response passes through the compliance filter before reaching the user.

What is already done for you
  - All imports (including check_compliance from nodes)
  - build_graph() with all existing nodes and edges
  - The run() loop

Your task
  TODO 3: Add the check_compliance node and reroute respond → check_compliance → END

Run when done
  python -m bundleiq.agent   (from inside s09/starter/)
"""
import os
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .config import CHECKPOINT_DB, MCP_SERVER_PATH
from .nodes import check_compliance, classify, decline, escalate, respond, retrieve_docs, route_query
from .state import BundleIQState


def build_graph(checkpointer=None):
    builder = StateGraph(BundleIQState)

    builder.add_node("classify",      classify)
    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("respond",       respond)
    builder.add_node("escalate",      escalate)
    builder.add_node("decline",       decline)

    # ---------------------------------------------------------------------------
    # TODO 3 of 3 -- Wire check_compliance into the graph
    # ---------------------------------------------------------------------------
    # 1. Add node:  builder.add_node("check_compliance", check_compliance)
    # 2. Replace the direct respond → END edge with:
    #      builder.add_edge("respond",          "check_compliance")
    #      builder.add_edge("check_compliance", END)
    # ---------------------------------------------------------------------------

    builder.set_entry_point("classify")
    builder.add_conditional_edges("classify", route_query, {
        "retrieve_docs": "retrieve_docs",
        "escalate":      "escalate",
        "decline":       "decline",
    })

    builder.add_edge("retrieve_docs", "respond")
    builder.add_edge("respond",       END)  # TODO 3: route through check_compliance first
    builder.add_edge("escalate",      END)
    builder.add_edge("decline",       END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


def run() -> None:
    conn      = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    _graph    = build_graph(checkpointer=SqliteSaver(conn))
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    if not MCP_SERVER_PATH.exists():
        print(f"[BundleIQ] WARNING: MCP server not found at {MCP_SERVER_PATH}")
        print("  Complete Session 7 first.")

    tracing_on = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project    = os.environ.get("LANGCHAIN_PROJECT", "batch1-bundleiq")

    print("=" * 60)
    print("  BundleIQ | TeleConnect India")
    print("  Compliance: TRAI banned-phrase + price verification")
    print(f"  Tracing   : {'LangSmith (' + project + ')' if tracing_on else 'off (set LANGSMITH_API_KEY to enable)'}")
    print("  Type 'quit' to exit")
    print("=" * 60)
    print(f"  Session: {thread_id[:8]}...")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nBundleIQ: Session ended. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nBundleIQ: Thank you for choosing TeleConnect India. Goodbye!")
            break

        result = _graph.invoke(
            {"customer_message": user_input, "response": "", "compliance_status": ""},
            config=config,
        )
        route      = result.get("query_type", "?")
        compliance = result.get("compliance_status", "")
        docs       = result.get("retrieved_docs", [])

        print(f"\n[Routed: {route}]", end="")
        if docs:
            sources = {d.split("]\n")[0].lstrip("[") for d in docs if "]\n" in d}
            print(f"  [RAG: {len(docs)} chunk(s) from {', '.join(sorted(sources))}]", end="")
        if compliance:
            print(f"  [Compliance: {compliance}]", end="")
        print()
        print(f"\nBundleIQ: {result['response']}")


if __name__ == "__main__":
    run()
