"""
s10/tests/test_s10.py
---------------------
Tests for Session 10: Multi-Agent Architecture Part 1.

Run with:
    pytest s10/tests/ -v

Test groups:
  TestState              -- BundleIQState has specialist field; no compliance_status
  TestClassifyNode       -- returns PLANS/PROMOTIONS/COMPLEX/OUT_OF_SCOPE; safe default
  TestPlansAgent         -- factory returns compiled graph; invocable; updates history
  TestPromotionsAgent    -- factory returns compiled graph; invocable; updates history
  TestSupervisorNodes    -- call_plans_agent/call_promotions_agent return correct state
  TestRouting            -- route_supervisor maps all 4 categories correctly
  TestSupervisorGraph    -- graph compiles; routing integration tests
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "bundleiq" or _k.startswith("bundleiq."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

from bundleiq.state import BundleIQState          # noqa: E402
import bundleiq.nodes as _nodes                   # noqa: E402
from bundleiq.nodes import (                      # noqa: E402
    call_plans_agent, call_promotions_agent, classify,
    create_plans_agent, create_promotions_agent,
    decline, escalate, route_supervisor,
)
from bundleiq.agent import build_graph            # noqa: E402


# ---------------------------------------------------------------------------
# TestState
# ---------------------------------------------------------------------------

class TestState:
    def _make_state(self, specialist: str = "") -> BundleIQState:
        return {
            "customer_message": "test",
            "response":         "",
            "history":          [],
            "query_type":       "PLANS",
            "retrieved_docs":   [],
            "specialist":       specialist,
        }

    def test_state_has_specialist_field(self):
        state = self._make_state("plans_agent")
        assert "specialist" in state

    def test_specialist_accepts_plans_agent(self):
        state = self._make_state("plans_agent")
        assert state["specialist"] == "plans_agent"

    def test_specialist_accepts_promotions_agent(self):
        state = self._make_state("promotions_agent")
        assert state["specialist"] == "promotions_agent"

    def test_state_has_no_compliance_status(self):
        assert "compliance_status" not in BundleIQState.__annotations__


# ---------------------------------------------------------------------------
# TestClassifyNode
# ---------------------------------------------------------------------------

class TestClassifyNode:
    def _state(self, message: str) -> BundleIQState:
        return {
            "customer_message": message, "response": "", "history": [],
            "query_type": "", "retrieved_docs": [], "specialist": "",
        }

    def test_plans_query_classified(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="PLANS")
            result = classify(self._state("What are the 5G plans?"))
        assert result["query_type"] == "PLANS"

    def test_promotions_query_classified(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="PROMOTIONS")
            result = classify(self._state("Are there any cashback offers?"))
        assert result["query_type"] == "PROMOTIONS"

    def test_complex_query_classified(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="COMPLEX")
            result = classify(self._state("Which plan is best for me?"))
        assert result["query_type"] == "COMPLEX"

    def test_oos_query_classified(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            result = classify(self._state("What is the weather today?"))
        assert result["query_type"] == "OUT_OF_SCOPE"

    def test_invalid_response_defaults_to_plans(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="BANANA")
            result = classify(self._state("test"))
        assert result["query_type"] == "PLANS"

    def test_exception_defaults_to_plans(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.side_effect = Exception("API error")
            result = classify(self._state("test"))
        assert result["query_type"] == "PLANS"

    def test_classify_strips_whitespace(self):
        with patch.object(_nodes, "classifier_llm") as mock:
            mock.invoke.return_value = MagicMock(content="  PROMOTIONS  ")
            result = classify(self._state("test"))
        assert result["query_type"] == "PROMOTIONS"


# ---------------------------------------------------------------------------
# TestPlansAgent
# ---------------------------------------------------------------------------

class TestPlansAgent:
    def test_factory_returns_compiled_graph(self):
        agent = create_plans_agent()
        assert agent is not None

    def test_factory_returns_different_instances(self):
        a1 = create_plans_agent()
        a2 = create_plans_agent()
        assert a1 is not a2

    def test_agent_has_respond_node(self):
        agent = create_plans_agent()
        assert "respond" in agent.get_graph().nodes

    def test_agent_is_invocable(self):
        agent = create_plans_agent()
        with patch.object(_nodes, "llm_with_tools") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Plans answer.", tool_calls=[])
            result = agent.invoke({
                "customer_message": "What are the mobile plans?",
                "history": [], "response": "",
                "query_type": "PLANS", "retrieved_docs": [], "specialist": "",
            })
        assert "response" in result
        assert isinstance(result["response"], str)

    def test_agent_updates_history(self):
        agent = create_plans_agent()
        with patch.object(_nodes, "llm_with_tools") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Plans answer.", tool_calls=[])
            result = agent.invoke({
                "customer_message": "test", "history": [],
                "response": "", "query_type": "PLANS",
                "retrieved_docs": [], "specialist": "",
            })
        assert len(result.get("history", [])) == 2


# ---------------------------------------------------------------------------
# TestPromotionsAgent
# ---------------------------------------------------------------------------

class TestPromotionsAgent:
    def test_factory_returns_compiled_graph(self):
        agent = create_promotions_agent()
        assert agent is not None

    def test_factory_returns_different_instances(self):
        a1 = create_promotions_agent()
        a2 = create_promotions_agent()
        assert a1 is not a2

    def test_agent_has_respond_node(self):
        agent = create_promotions_agent()
        assert "respond" in agent.get_graph().nodes

    def test_agent_is_invocable(self):
        agent = create_promotions_agent()
        with patch.object(_nodes, "llm_with_tools") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Promo answer.", tool_calls=[])
            result = agent.invoke({
                "customer_message": "Any cashback offers?",
                "history": [], "response": "",
                "query_type": "PROMOTIONS", "retrieved_docs": [], "specialist": "",
            })
        assert "response" in result
        assert isinstance(result["response"], str)

    def test_agent_updates_history(self):
        agent = create_promotions_agent()
        with patch.object(_nodes, "llm_with_tools") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Promo answer.", tool_calls=[])
            result = agent.invoke({
                "customer_message": "test", "history": [],
                "response": "", "query_type": "PROMOTIONS",
                "retrieved_docs": [], "specialist": "",
            })
        assert len(result.get("history", [])) == 2


# ---------------------------------------------------------------------------
# TestSupervisorNodes
# ---------------------------------------------------------------------------

class TestSupervisorNodes:
    def _state(self, message: str = "test", qt: str = "PLANS") -> BundleIQState:
        return {
            "customer_message": message, "response": "", "history": [],
            "query_type": qt, "retrieved_docs": [], "specialist": "",
        }

    def test_call_plans_agent_sets_specialist(self):
        with patch.object(_nodes, "_plans_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Plan info.", "history": [],
            }
            result = call_plans_agent(self._state())
        assert result["specialist"] == "plans_agent"

    def test_call_plans_agent_returns_response(self):
        with patch.object(_nodes, "_plans_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Plan info.", "history": [],
            }
            result = call_plans_agent(self._state())
        assert result["response"] == "Plan info."

    def test_call_promotions_agent_sets_specialist(self):
        with patch.object(_nodes, "_promotions_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Promo info.", "history": [],
            }
            result = call_promotions_agent(self._state(qt="PROMOTIONS"))
        assert result["specialist"] == "promotions_agent"

    def test_call_promotions_agent_returns_response(self):
        with patch.object(_nodes, "_promotions_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Promo info.", "history": [],
            }
            result = call_promotions_agent(self._state(qt="PROMOTIONS"))
        assert result["response"] == "Promo info."

    def test_escalate_sets_specialist(self):
        result = escalate(self._state())
        assert result["specialist"] == "escalated"

    def test_decline_sets_specialist(self):
        result = decline(self._state())
        assert result["specialist"] == "declined"


# ---------------------------------------------------------------------------
# TestRouting
# ---------------------------------------------------------------------------

class TestRouting:
    def _state(self, qt: str) -> BundleIQState:
        return {
            "customer_message": "test", "response": "", "history": [],
            "query_type": qt, "retrieved_docs": [], "specialist": "",
        }

    def test_plans_routes_to_plans_agent(self):
        assert route_supervisor(self._state("PLANS")) == "call_plans_agent"

    def test_promotions_routes_to_promotions_agent(self):
        assert route_supervisor(self._state("PROMOTIONS")) == "call_promotions_agent"

    def test_complex_routes_to_escalate(self):
        assert route_supervisor(self._state("COMPLEX")) == "escalate"

    def test_oos_routes_to_decline(self):
        assert route_supervisor(self._state("OUT_OF_SCOPE")) == "decline"

    def test_unknown_defaults_to_plans_agent(self):
        assert route_supervisor(self._state("UNKNOWN")) == "call_plans_agent"


# ---------------------------------------------------------------------------
# TestSupervisorGraph
# ---------------------------------------------------------------------------

class TestSupervisorGraph:
    def test_build_graph_returns_compiled_graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert graph is not None

    def test_graph_has_classify_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "classify" in graph.get_graph().nodes

    def test_graph_has_call_plans_agent_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "call_plans_agent" in graph.get_graph().nodes

    def test_graph_has_call_promotions_agent_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "call_promotions_agent" in graph.get_graph().nodes

    def test_plans_query_routes_to_plans_agent(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_plans_agent") as mock_plans:
            mock_clf.invoke.return_value = MagicMock(content="PLANS")
            mock_plans.invoke.return_value = {"response": "Plan answer.", "history": []}
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "What are the 5G plans?",
                 "response": "", "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-plans"}},
            )
        assert result["specialist"] == "plans_agent"

    def test_promotions_query_routes_to_promotions_agent(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_promotions_agent") as mock_promo:
            mock_clf.invoke.return_value = MagicMock(content="PROMOTIONS")
            mock_promo.invoke.return_value = {"response": "Promo answer.", "history": []}
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Any cashback offers?",
                 "response": "", "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-promotions"}},
            )
        assert result["specialist"] == "promotions_agent"

    def test_complex_query_escalates(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="COMPLEX")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Which plan is best for me?",
                 "response": "", "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-complex"}},
            )
        assert result["specialist"] == "escalated"
        assert "TeleConnect advisor" in result["response"]

    def test_oos_query_declines(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "What is the weather?",
                 "response": "", "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-oos"}},
            )
        assert result["specialist"] == "declined"
        assert "TeleConnect" in result["response"]

    def test_graph_result_has_specialist_field(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="COMPLEX")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "test", "response": "",
                 "specialist": "", "retrieved_docs": []},
                config={"configurable": {"thread_id": "test-field"}},
            )
        assert "specialist" in result
