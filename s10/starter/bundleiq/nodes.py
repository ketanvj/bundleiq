"""
bundleiq/nodes.py
-----------------
STARTER FILE -- complete the three TODO sections below.

Goal
  Refactor BundleIQ into a Supervisor + Specialist Agent architecture.
  The supervisor classifies customer queries and routes them to the
  correct specialist sub-agent (Plans Agent or Promotions Agent).

What is already done for you
  - _agent_respond()    -- shared LLM + tool-call logic for both agents
  - _plans_respond()    -- wrapper that uses PLANS_SYSTEM_PROMPT
  - _promotions_respond() -- wrapper that uses PROMOTIONS_SYSTEM_PROMPT
  - classify()          -- supervisor classifier node
  - escalate()          -- escalation node
  - decline()           -- decline node

Your task
  TODO 1: Implement create_plans_agent() and create_promotions_agent()
          (single-node StateGraphs: respond -> END)
  TODO 2: Implement call_plans_agent() and call_promotions_agent()
          (supervisor caller nodes that invoke the sub-agents)
  TODO 3: Implement route_supervisor()
          (maps query_type to the correct node name)

Run when done
  python -m bundleiq.agent   (from inside s10/starter/)
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from .config import (
    CLASSIFY_SYSTEM,
    DECLINE_RESPONSE,
    ESCALATE_RESPONSE,
    PLANS_SYSTEM_PROMPT,
    PROMOTIONS_SYSTEM_PROMPT,
)
from .state import BundleIQState
from .tools import _run_tool, classifier_llm, llm, llm_with_tools


def _agent_respond(state: BundleIQState, system_prompt: str, label: str) -> dict:
    """Shared respond logic for both specialist agents. Already implemented."""
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
# TODO 1 of 3 -- Implement the agent factory functions
# ---------------------------------------------------------------------------
# Each factory creates a small sub-graph (StateGraph) that represents one
# specialist agent. Both have just a single "respond" node -> END.
#
# Template for create_plans_agent():
#   def create_plans_agent():
#       builder = StateGraph(BundleIQState)
#       builder.add_node("respond", _plans_respond)
#       builder.set_entry_point("respond")
#       builder.add_edge("respond", END)
#       return builder.compile()
#
# create_promotions_agent() follows the same pattern with _promotions_respond.
# ---------------------------------------------------------------------------
def create_plans_agent():
    raise NotImplementedError("TODO 1: implement create_plans_agent()")


def create_promotions_agent():
    raise NotImplementedError("TODO 1: implement create_promotions_agent()")


_plans_agent      = None  # TODO 1: replace with create_plans_agent()
_promotions_agent = None  # TODO 1: replace with create_promotions_agent()


# ---------------------------------------------------------------------------
# Supervisor nodes
# ---------------------------------------------------------------------------

def classify(state: BundleIQState) -> dict:
    """Already implemented -- classifies into PLANS/PROMOTIONS/COMPLEX/OUT_OF_SCOPE."""
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


# ---------------------------------------------------------------------------
# TODO 2 of 3 -- Implement the supervisor caller nodes
# ---------------------------------------------------------------------------
# call_plans_agent() invokes _plans_agent with the current state and returns
# the specialist's response with specialist="plans_agent".
#
# Template:
#   def call_plans_agent(state: BundleIQState) -> dict:
#       print("[BundleIQ] Supervisor -> Plans Agent")
#       result = _plans_agent.invoke({
#           "customer_message": state["customer_message"],
#           "history":          state.get("history", []),
#           "response":         "",
#           "query_type":       state.get("query_type", "PLANS"),
#           "retrieved_docs":   [],
#           "specialist":       "",
#       })
#       return {
#           "response":   result["response"],
#           "history":    result.get("history", state.get("history", [])),
#           "specialist": "plans_agent",
#       }
#
# call_promotions_agent() follows the same pattern with _promotions_agent
# and specialist="promotions_agent".
# ---------------------------------------------------------------------------
def call_plans_agent(state: BundleIQState) -> dict:
    raise NotImplementedError("TODO 2: implement call_plans_agent()")


def call_promotions_agent(state: BundleIQState) -> dict:
    raise NotImplementedError("TODO 2: implement call_promotions_agent()")


def escalate(state: BundleIQState) -> dict:
    """Already implemented."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": ESCALATE_RESPONSE},
    ]
    return {"response": ESCALATE_RESPONSE, "history": new_history, "specialist": "escalated"}


def decline(state: BundleIQState) -> dict:
    """Already implemented."""
    new_history = state.get("history", []) + [
        {"role": "user",      "content": state["customer_message"]},
        {"role": "assistant", "content": DECLINE_RESPONSE},
    ]
    return {"response": DECLINE_RESPONSE, "history": new_history, "specialist": "declined"}


# ---------------------------------------------------------------------------
# TODO 3 of 3 -- Implement route_supervisor()
# ---------------------------------------------------------------------------
# Map query_type to a node name:
#   PROMOTIONS    -> "call_promotions_agent"
#   COMPLEX       -> "escalate"
#   OUT_OF_SCOPE  -> "decline"
#   default       -> "call_plans_agent"
# ---------------------------------------------------------------------------
def route_supervisor(state: BundleIQState) -> str:
    raise NotImplementedError("TODO 3: implement route_supervisor()")
