"""
s12/tests/test_s12.py
---------------------
Tests for Session 12: Multi-Agent Architecture Part 2 (adds Compliance Agent).

Run with:
    pytest s12/tests/ -v

Test groups:
  TestState                -- BundleIQState has both specialist + compliance_status fields
  TestClassifyNode         -- PLANS/PROMOTIONS/COMPLEX/OUT_OF_SCOPE; safe default
  TestPlansAgent           -- factory returns compiled graph; invocable; updates history
  TestPromotionsAgent      -- factory returns compiled graph; invocable; updates history
  TestComplianceHelpers    -- _check_compliance_logic detects banned phrases + bad prices
  TestCheckTraiNode        -- check_trai sets compliance_status PASS/FAIL
  TestRouteCompliance      -- route_compliance returns "revise" on FAIL, END on PASS
  TestReviseResponse       -- revise_response calls LLM and returns revised text
  TestComplianceAgent      -- create_compliance_agent() builds and invokes correctly
  TestCallComplianceAgent  -- call_compliance_agent supervisor node works end-to-end
  TestSupervisorNodes      -- call_plans_agent/call_promotions_agent return correct state
  TestRouting              -- route_supervisor maps all 4 categories correctly
  TestSupervisorGraph      -- graph compiles with compliance; routing integration tests
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from langgraph.graph import END

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "bundleiq" or _k.startswith("bundleiq."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

from bundleiq.state import BundleIQState          # noqa: E402
import bundleiq.nodes as _nodes                   # noqa: E402
from bundleiq.nodes import (                      # noqa: E402
    call_compliance_agent,
    call_plans_agent, call_promotions_agent, classify,
    check_trai,
    create_compliance_agent,
    create_plans_agent, create_promotions_agent,
    decline, escalate,
    revise_response, route_compliance, route_supervisor,
)
from bundleiq.agent import build_graph            # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    message: str = "test",
    response: str = "",
    qt: str = "PLANS",
    specialist: str = "",
    compliance_status: str = "",
) -> BundleIQState:
    return {
        "customer_message":  message,
        "response":          response,
        "history":           [],
        "query_type":        qt,
        "retrieved_docs":    [],
        "specialist":        specialist,
        "compliance_status": compliance_status,
    }


# ---------------------------------------------------------------------------
# TestState
# ---------------------------------------------------------------------------

class TestState:
    def test_state_has_specialist_field(self):
        state = _make_state(specialist="plans_agent")
        assert "specialist" in state

    def test_state_has_compliance_status_field(self):
        state = _make_state(compliance_status="PASS")
        assert "compliance_status" in state
        assert state["compliance_status"] == "PASS"

    def test_specialist_accepts_plans_agent(self):
        assert _make_state(specialist="plans_agent")["specialist"] == "plans_agent"

    def test_specialist_accepts_promotions_agent(self):
        assert _make_state(specialist="promotions_agent")["specialist"] == "promotions_agent"

    def test_compliance_status_accepts_pass(self):
        assert _make_state(compliance_status="PASS")["compliance_status"] == "PASS"

    def test_compliance_status_accepts_fail(self):
        state = _make_state(compliance_status="FAIL: banned phrase: 'guaranteed coverage'")
        assert state["compliance_status"].startswith("FAIL")


# ---------------------------------------------------------------------------
# TestClassifyNode
# ---------------------------------------------------------------------------

class TestClassifyNode:
    def _state(self, message: str) -> BundleIQState:
        return _make_state(message=message)

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
        assert create_plans_agent() is not None

    def test_factory_returns_different_instances(self):
        assert create_plans_agent() is not create_plans_agent()

    def test_agent_has_respond_node(self):
        assert "respond" in create_plans_agent().get_graph().nodes

    def test_agent_is_invocable(self):
        agent = create_plans_agent()
        with patch.object(_nodes, "llm_with_tools") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Plans answer.", tool_calls=[])
            result = agent.invoke(_make_state(message="What are the mobile plans?"))
        assert isinstance(result["response"], str)

    def test_agent_updates_history(self):
        agent = create_plans_agent()
        with patch.object(_nodes, "llm_with_tools") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Plans answer.", tool_calls=[])
            result = agent.invoke(_make_state())
        assert len(result.get("history", [])) == 2


# ---------------------------------------------------------------------------
# TestPromotionsAgent
# ---------------------------------------------------------------------------

class TestPromotionsAgent:
    def test_factory_returns_compiled_graph(self):
        assert create_promotions_agent() is not None

    def test_factory_returns_different_instances(self):
        assert create_promotions_agent() is not create_promotions_agent()

    def test_agent_has_respond_node(self):
        assert "respond" in create_promotions_agent().get_graph().nodes

    def test_agent_is_invocable(self):
        agent = create_promotions_agent()
        with patch.object(_nodes, "llm_with_tools") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Promo answer.", tool_calls=[])
            result = agent.invoke(_make_state(message="Any cashback offers?", qt="PROMOTIONS"))
        assert isinstance(result["response"], str)

    def test_agent_updates_history(self):
        agent = create_promotions_agent()
        with patch.object(_nodes, "llm_with_tools") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Promo answer.", tool_calls=[])
            result = agent.invoke(_make_state(qt="PROMOTIONS"))
        assert len(result.get("history", [])) == 2


# ---------------------------------------------------------------------------
# TestComplianceHelpers
# ---------------------------------------------------------------------------

class TestComplianceHelpers:
    def test_banned_phrase_detected(self):
        draft  = "TeleConnect provides guaranteed coverage across all areas."
        passed, reason = _nodes._check_compliance_logic(draft)
        assert not passed
        assert "banned phrase" in reason

    def test_no_banned_phrase_passes(self):
        draft  = "TeleConnect offers wide coverage across most regions."
        with patch.object(_nodes, "_load_valid_prices", return_value=set()):
            passed, reason = _nodes._check_compliance_logic(draft)
        assert passed
        assert reason == "PASS"

    def test_invalid_price_detected(self):
        draft = "Our plan costs Rs. 99999 per month."
        with patch.object(_nodes, "_load_valid_prices", return_value={299, 499, 799}):
            passed, reason = _nodes._check_compliance_logic(draft)
        assert not passed
        assert "incorrect price" in reason

    def test_valid_price_passes(self):
        draft = "Our plan costs Rs. 299 per month."
        with patch.object(_nodes, "_load_valid_prices", return_value={299, 499, 799}):
            passed, reason = _nodes._check_compliance_logic(draft)
        assert passed

    def test_no_prices_in_draft_skips_price_check(self):
        draft = "We have a wide range of plans available."
        with patch.object(_nodes, "_load_valid_prices", return_value={299}):
            passed, reason = _nodes._check_compliance_logic(draft)
        assert passed

    def test_load_valid_prices_returns_set_on_error(self):
        with patch("bundleiq.nodes.sqlite3.connect", side_effect=Exception("db error")):
            result = _nodes._load_valid_prices()
        assert isinstance(result, set)

    def test_extract_prices_parses_rs_prefix(self):
        prices = _nodes._extract_prices("This plan is Rs. 299 per month.")
        assert 299 in prices

    def test_extract_prices_parses_rupee_symbol(self):
        prices = _nodes._extract_prices("Special offer at ₹499.")
        assert 499 in prices

    def test_extract_prices_handles_comma_separator(self):
        prices = _nodes._extract_prices("Premium plan at Rs. 1,499.")
        assert 1499 in prices

    def test_extract_prices_returns_empty_for_no_match(self):
        prices = _nodes._extract_prices("No prices here.")
        assert prices == []


# ---------------------------------------------------------------------------
# TestCheckTraiNode
# ---------------------------------------------------------------------------

class TestCheckTraiNode:
    def test_check_trai_pass(self):
        state = _make_state(response="TeleConnect has wide coverage.")
        with patch.object(_nodes, "_check_compliance_logic", return_value=(True, "PASS")):
            result = check_trai(state)
        assert result["compliance_status"] == "PASS"

    def test_check_trai_fail_banned_phrase(self):
        state = _make_state(response="We guarantee 100% coverage.")
        with patch.object(_nodes, "_check_compliance_logic",
                          return_value=(False, "banned phrase: '100% coverage'")):
            result = check_trai(state)
        assert result["compliance_status"].startswith("FAIL")
        assert "100% coverage" in result["compliance_status"]

    def test_check_trai_fail_bad_price(self):
        state = _make_state(response="Plan costs Rs. 99999.")
        with patch.object(_nodes, "_check_compliance_logic",
                          return_value=(False, "incorrect price: Rs. 99999 not in product catalogue")):
            result = check_trai(state)
        assert result["compliance_status"].startswith("FAIL")

    def test_check_trai_returns_dict(self):
        state = _make_state(response="Normal response.")
        with patch.object(_nodes, "_check_compliance_logic", return_value=(True, "PASS")):
            result = check_trai(state)
        assert isinstance(result, dict)
        assert "compliance_status" in result


# ---------------------------------------------------------------------------
# TestRouteCompliance
# ---------------------------------------------------------------------------

class TestRouteCompliance:
    def test_pass_routes_to_end(self):
        state = _make_state(compliance_status="PASS")
        assert route_compliance(state) is END

    def test_fail_routes_to_revise(self):
        state = _make_state(compliance_status="FAIL: banned phrase")
        assert route_compliance(state) == "revise"

    def test_revised_does_not_route_to_revise(self):
        state = _make_state(compliance_status="REVISED")
        assert route_compliance(state) is END

    def test_empty_compliance_status_routes_to_end(self):
        state = _make_state(compliance_status="")
        assert route_compliance(state) is END


# ---------------------------------------------------------------------------
# TestReviseResponse
# ---------------------------------------------------------------------------

class TestReviseResponse:
    def test_revise_returns_response(self):
        state = _make_state(
            response="We guarantee 100% coverage.",
            compliance_status="FAIL: banned phrase: '100% coverage'",
        )
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(
                content="TeleConnect coverage is available in most areas.\n\nBundleIQ | TeleConnect India"
            )
            result = revise_response(state)
        assert "response" in result
        assert isinstance(result["response"], str)

    def test_revise_sets_status_to_revised(self):
        state = _make_state(
            response="We guarantee coverage.",
            compliance_status="FAIL: banned phrase",
        )
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Revised response.")
            result = revise_response(state)
        assert result["compliance_status"] == "REVISED"

    def test_revise_falls_back_on_llm_error(self):
        state = _make_state(
            response="We guarantee coverage.",
            compliance_status="FAIL: banned phrase",
        )
        with patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.side_effect = Exception("LLM down")
            result = revise_response(state)
        assert "response" in result
        assert result["response"]


# ---------------------------------------------------------------------------
# TestComplianceAgent
# ---------------------------------------------------------------------------

class TestComplianceAgent:
    def test_factory_returns_compiled_graph(self):
        agent = create_compliance_agent()
        assert agent is not None

    def test_factory_returns_different_instances(self):
        assert create_compliance_agent() is not create_compliance_agent()

    def test_agent_has_check_trai_node(self):
        agent = create_compliance_agent()
        assert "check_trai" in agent.get_graph().nodes

    def test_agent_has_revise_node(self):
        agent = create_compliance_agent()
        assert "revise" in agent.get_graph().nodes

    def test_agent_pass_path(self):
        agent = create_compliance_agent()
        with patch.object(_nodes, "_check_compliance_logic", return_value=(True, "PASS")):
            result = agent.invoke(_make_state(response="Good clean response."))
        assert result["compliance_status"] == "PASS"
        assert result["response"] == "Good clean response."

    def test_agent_fail_then_revise_path(self):
        agent = create_compliance_agent()
        with patch.object(_nodes, "_check_compliance_logic",
                          return_value=(False, "banned phrase: 'guaranteed coverage'")), \
             patch.object(_nodes, "llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Coverage may vary.\n\nBundleIQ | TeleConnect India")
            result = agent.invoke(_make_state(response="We guarantee coverage."))
        assert result["compliance_status"] == "REVISED"
        assert "guarantee" not in result["response"].lower() or "Coverage may vary" in result["response"]


# ---------------------------------------------------------------------------
# TestCallComplianceAgent
# ---------------------------------------------------------------------------

class TestCallComplianceAgent:
    def _state(self) -> BundleIQState:
        return _make_state(
            message="What are the 5G plans?",
            response="Here are the plans.",
            qt="PLANS",
            specialist="plans_agent",
        )

    def test_call_compliance_agent_returns_response(self):
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response":          "Clean response.",
                "compliance_status": "PASS",
            }
            result = call_compliance_agent(self._state())
        assert result["response"] == "Clean response."

    def test_call_compliance_agent_returns_compliance_status(self):
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response":          "Clean response.",
                "compliance_status": "PASS",
            }
            result = call_compliance_agent(self._state())
        assert result["compliance_status"] == "PASS"

    def test_call_compliance_agent_defaults_status_to_pass(self):
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {"response": "Ok."}
            result = call_compliance_agent(self._state())
        assert result["compliance_status"] == "PASS"

    def test_call_compliance_agent_invokes_sub_graph(self):
        with patch.object(_nodes, "_compliance_agent") as mock_agent:
            mock_agent.invoke.return_value = {
                "response": "Ok.", "compliance_status": "PASS",
            }
            call_compliance_agent(self._state())
        mock_agent.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# TestSupervisorNodes
# ---------------------------------------------------------------------------

class TestSupervisorNodes:
    def test_call_plans_agent_sets_specialist(self):
        with patch.object(_nodes, "_plans_agent") as mock_agent:
            mock_agent.invoke.return_value = {"response": "Plan info.", "history": []}
            result = call_plans_agent(_make_state())
        assert result["specialist"] == "plans_agent"

    def test_call_plans_agent_returns_response(self):
        with patch.object(_nodes, "_plans_agent") as mock_agent:
            mock_agent.invoke.return_value = {"response": "Plan info.", "history": []}
            result = call_plans_agent(_make_state())
        assert result["response"] == "Plan info."

    def test_call_promotions_agent_sets_specialist(self):
        with patch.object(_nodes, "_promotions_agent") as mock_agent:
            mock_agent.invoke.return_value = {"response": "Promo info.", "history": []}
            result = call_promotions_agent(_make_state(qt="PROMOTIONS"))
        assert result["specialist"] == "promotions_agent"

    def test_call_promotions_agent_returns_response(self):
        with patch.object(_nodes, "_promotions_agent") as mock_agent:
            mock_agent.invoke.return_value = {"response": "Promo info.", "history": []}
            result = call_promotions_agent(_make_state(qt="PROMOTIONS"))
        assert result["response"] == "Promo info."

    def test_escalate_sets_specialist(self):
        result = escalate(_make_state())
        assert result["specialist"] == "escalated"

    def test_decline_sets_specialist(self):
        result = decline(_make_state())
        assert result["specialist"] == "declined"


# ---------------------------------------------------------------------------
# TestRouting
# ---------------------------------------------------------------------------

class TestRouting:
    def test_plans_routes_to_plans_agent(self):
        assert route_supervisor(_make_state(qt="PLANS")) == "call_plans_agent"

    def test_promotions_routes_to_promotions_agent(self):
        assert route_supervisor(_make_state(qt="PROMOTIONS")) == "call_promotions_agent"

    def test_complex_routes_to_escalate(self):
        assert route_supervisor(_make_state(qt="COMPLEX")) == "escalate"

    def test_oos_routes_to_decline(self):
        assert route_supervisor(_make_state(qt="OUT_OF_SCOPE")) == "decline"

    def test_unknown_defaults_to_plans_agent(self):
        assert route_supervisor(_make_state(qt="UNKNOWN")) == "call_plans_agent"


# ---------------------------------------------------------------------------
# TestSupervisorGraph
# ---------------------------------------------------------------------------

class TestSupervisorGraph:
    def test_build_graph_returns_compiled_graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert build_graph(checkpointer=MemorySaver()) is not None

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

    def test_graph_has_call_compliance_agent_node(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert "call_compliance_agent" in graph.get_graph().nodes

    def test_plans_query_routes_through_compliance(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_plans_agent") as mock_plans, \
             patch.object(_nodes, "_compliance_agent") as mock_comp:
            mock_clf.invoke.return_value = MagicMock(content="PLANS")
            mock_plans.invoke.return_value = {"response": "Plan answer.", "history": []}
            mock_comp.invoke.return_value = {
                "response": "Plan answer.", "compliance_status": "PASS",
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "What are the 5G plans?",
                 "response": "", "specialist": "", "retrieved_docs": [],
                 "compliance_status": ""},
                config={"configurable": {"thread_id": "test-plans-s12"}},
            )
        assert result["specialist"] == "plans_agent"
        assert "compliance_status" in result

    def test_promotions_query_routes_through_compliance(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_promotions_agent") as mock_promo, \
             patch.object(_nodes, "_compliance_agent") as mock_comp:
            mock_clf.invoke.return_value = MagicMock(content="PROMOTIONS")
            mock_promo.invoke.return_value = {"response": "Promo answer.", "history": []}
            mock_comp.invoke.return_value = {
                "response": "Promo answer.", "compliance_status": "PASS",
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Any cashback offers?",
                 "response": "", "specialist": "", "retrieved_docs": [],
                 "compliance_status": ""},
                config={"configurable": {"thread_id": "test-promos-s12"}},
            )
        assert result["specialist"] == "promotions_agent"
        assert "compliance_status" in result

    def test_complex_query_escalates_without_compliance(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="COMPLEX")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Which plan is best for me?",
                 "response": "", "specialist": "", "retrieved_docs": [],
                 "compliance_status": ""},
                config={"configurable": {"thread_id": "test-complex-s12"}},
            )
        assert result["specialist"] == "escalated"
        assert "TeleConnect advisor" in result["response"]

    def test_oos_query_declines_without_compliance(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "What is the weather?",
                 "response": "", "specialist": "", "retrieved_docs": [],
                 "compliance_status": ""},
                config={"configurable": {"thread_id": "test-oos-s12"}},
            )
        assert result["specialist"] == "declined"
        assert "TeleConnect" in result["response"]

    def test_graph_result_has_compliance_status(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_clf, \
             patch.object(_nodes, "_plans_agent") as mock_plans, \
             patch.object(_nodes, "_compliance_agent") as mock_comp:
            mock_clf.invoke.return_value = MagicMock(content="PLANS")
            mock_plans.invoke.return_value = {"response": "Plans.", "history": []}
            mock_comp.invoke.return_value = {
                "response": "Plans.", "compliance_status": "PASS",
            }
            graph  = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "test", "response": "",
                 "specialist": "", "retrieved_docs": [], "compliance_status": ""},
                config={"configurable": {"thread_id": "test-compliance-field"}},
            )
        assert "compliance_status" in result
