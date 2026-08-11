"""
bundleiq/agent.py
-----------------
Builds and runs the BundleIQ multi-agent LangGraph supervisor.

Session 10: Supervisor + Specialist Agent architecture.
  - Supervisor classifies into PLANS / PROMOTIONS / COMPLEX / OUT_OF_SCOPE
  - Plans Agent      uses MCP query_plans tool
  - Promotions Agent uses MCP query_promotions tool

The graph wiring below is complete -- once you implement the TODOs in
nodes.py (create_plans_agent, create_promotions_agent, call_plans_agent,
call_promotions_agent, route_supervisor), this file will work as-is.

Run when done
  python -m bundleiq.agent   (from inside s10/starter/)
"""
import os
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from .config import CHECKPOINT_DB, MCP_SERVER_PATH
from .nodes import (
    call_plans_agent,
    call_promotions_agent,
    classify,
    decline,
    escalate,
    route_supervisor,
)
from .state import BundleIQState


def build_graph(checkpointer=None):
    builder = StateGraph(BundleIQState)

    builder.add_node("classify",              classify)
    builder.add_node("call_plans_agent",      call_plans_agent)
    builder.add_node("call_promotions_agent", call_promotions_agent)
    builder.add_node("escalate",              escalate)
    builder.add_node("decline",               decline)

    builder.set_entry_point("classify")
    builder.add_conditional_edges("classify", route_supervisor, {
        "call_plans_agent":      "call_plans_agent",
        "call_promotions_agent": "call_promotions_agent",
        "escalate":              "escalate",
        "decline":               "decline",
    })

    builder.add_edge("call_plans_agent",      END)
    builder.add_edge("call_promotions_agent", END)
    builder.add_edge("escalate",              END)
    builder.add_edge("decline",               END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


def run() -> None:
    conn      = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    g         = build_graph(checkpointer=SqliteSaver(conn))
    thread_id = str(uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    if not MCP_SERVER_PATH.exists():
        print(f"[BundleIQ] WARNING: MCP server not found at {MCP_SERVER_PATH}")
        print("  Complete Session 7 first.")

    tracing_on = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project    = os.environ.get("LANGCHAIN_PROJECT", "batch1-bundleiq")

    print("=" * 60)
    print("  BundleIQ | TeleConnect India")
    print("  Architecture: Supervisor + Plans Agent + Promotions Agent")
    print(f"  Tracing: {'LangSmith (' + project + ')' if tracing_on else 'off'}")
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

        result = g.invoke(
            {"customer_message": user_input, "response": "",
             "specialist": "", "retrieved_docs": []},
            config=config,
        )
        specialist = result.get("specialist", "?")
        print(f"\n[Route: {result.get('query_type','?')} -> {specialist}]")
        print(f"\nBundleIQ: {result['response']}")


if __name__ == "__main__":
    run()
