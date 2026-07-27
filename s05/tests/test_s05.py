"""
s05/tests/test_s05.py
---------------------
Tests for Session 5: SQLite tool calls.

Run with:
    pytest s05/tests/ -v

Test groups:
  TestBundleIQState       -- state TypedDict has all five fields (unchanged from S04)
  TestQueryPlansTool      -- query_plans() SQL correctness, filtering, output format
  TestQueryPromotionsTool -- query_promotions() SQL correctness, filtering
  TestToolSQLSafety       -- SQL injection protection and parameterised-query enforcement
  TestToolsBinding        -- llm_with_tools exists; tools are @tool decorated; prompt updated
  TestRunToolDispatch     -- _run_tool dispatches correctly; handles unknown names
  TestRespondWithTools    -- respond() calls llm_with_tools; executes tool calls; calls llm again
  TestGraphRouting        -- SIMPLE goes through retrieve_docs -> respond; COMPLEX/OOS skip tools
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "bundleiq" or _k.startswith("bundleiq."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

import bundleiq  # noqa: E402
import bundleiq.nodes as _nodes  # noqa: E402
import bundleiq.tools as _tools  # noqa: E402
from bundleiq.config import SYSTEM_PROMPT  # noqa: E402
from bundleiq.state import BundleIQState  # noqa: E402
from bundleiq.tools import _run_tool, query_plans, query_promotions  # noqa: E402
from bundleiq.nodes import respond  # noqa: E402
from bundleiq.agent import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# TestBundleIQState
# ---------------------------------------------------------------------------

class TestBundleIQState:
    def test_state_has_customer_message_field(self):
        assert "customer_message" in BundleIQState.__annotations__

    def test_state_has_response_field(self):
        assert "response" in BundleIQState.__annotations__

    def test_state_has_history_field(self):
        assert "history" in BundleIQState.__annotations__

    def test_state_has_query_type_field(self):
        assert "query_type" in BundleIQState.__annotations__

    def test_state_has_retrieved_docs_field(self):
        assert "retrieved_docs" in BundleIQState.__annotations__

    def test_state_has_exactly_five_fields(self):
        assert len(BundleIQState.__annotations__) == 5


# ---------------------------------------------------------------------------
# TestQueryPlansTool
# ---------------------------------------------------------------------------

class TestQueryPlansTool:
    def test_query_plans_mobile_returns_daily_plans(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_plans.invoke({"plan_type": "mobile"})
        assert "Daily 1GB" in result or "Daily 2GB" in result

    def test_query_plans_mobile_includes_price(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_plans.invoke({"plan_type": "mobile"})
        assert "179" in result or "299" in result

    def test_query_plans_mobile_shows_5g_flag(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_plans.invoke({"plan_type": "mobile"})
        assert "5G" in result

    def test_query_plans_broadband_returns_speed(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_plans.invoke({"plan_type": "broadband"})
        assert "Mbps" in result or "100" in result

    def test_query_plans_broadband_includes_price(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_plans.invoke({"plan_type": "broadband"})
        assert "499" in result or "799" in result

    def test_query_plans_all_contains_mobile_and_broadband(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_plans.invoke({"plan_type": "all"})
        assert "Daily" in result
        assert "Mbps" in result

    def test_query_plans_default_is_all(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_plans.invoke({})
        assert "Daily" in result
        assert "Mbps" in result

    def test_query_plans_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        assert isinstance(query_plans.invoke({"plan_type": "mobile"}), str)

    def test_query_plans_mobile_does_not_include_broadband(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_plans.invoke({"plan_type": "mobile"})
        assert "Mbps" not in result or "Daily" in result

    def test_query_plans_broadband_does_not_include_mobile_names(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_plans.invoke({"plan_type": "broadband"})
        assert "Daily 1GB" not in result


# ---------------------------------------------------------------------------
# TestQueryPromotionsTool
# ---------------------------------------------------------------------------

class TestQueryPromotionsTool:
    def test_query_promotions_all_returns_multiple(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_promotions.invoke({"plan_id": "all"})
        assert "Monsoon" in result
        assert "Broadband" in result

    def test_query_promotions_filter_by_plan_id(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_promotions.invoke({"plan_id": "mob_001"})
        assert "Monsoon" in result

    def test_query_promotions_filter_excludes_unrelated(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_promotions.invoke({"plan_id": "mob_001"})
        assert "Broadband Cashback" not in result

    def test_query_promotions_includes_discount_info(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_promotions.invoke({"plan_id": "all"})
        assert "extra_data" in result or "cashback" in result or "5GB" in result

    def test_query_promotions_no_match_returns_message(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_promotions.invoke({"plan_id": "nonexistent_plan_xyz"})
        assert "No active promotions" in result or "nonexistent_plan_xyz" in result

    def test_query_promotions_default_is_all(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_promotions.invoke({})
        assert "Monsoon" in result

    def test_query_promotions_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        assert isinstance(query_promotions.invoke({"plan_id": "all"}), str)


# ---------------------------------------------------------------------------
# TestToolSQLSafety
# ---------------------------------------------------------------------------

class TestToolSQLSafety:
    def test_query_promotions_sql_injection_safe(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = query_promotions.invoke({"plan_id": "'; DROP TABLE promotions; --"})
        assert isinstance(result, str)
        normal = query_promotions.invoke({"plan_id": "mob_001"})
        assert "Monsoon" in normal

    def test_query_promotions_uses_question_mark_placeholder(self):
        import inspect
        source = inspect.getsource(query_promotions.func)
        assert "LIKE ?" in source, (
            "query_promotions must use a ? placeholder for the plan_id parameter. "
            "Never interpolate user input directly into a SQL string."
        )


# ---------------------------------------------------------------------------
# TestToolsBinding
# ---------------------------------------------------------------------------

class TestToolsBinding:
    def test_llm_with_tools_exists(self):
        assert hasattr(_tools, "llm_with_tools"), (
            "llm_with_tools not found. Create it with llm.bind_tools([query_plans, query_promotions])."
        )

    def test_query_plans_is_tool_decorated(self):
        assert hasattr(query_plans, "name"), (
            "query_plans does not appear to be decorated with @tool."
        )

    def test_query_promotions_is_tool_decorated(self):
        assert hasattr(query_promotions, "name"), (
            "query_promotions does not appear to be decorated with @tool."
        )

    def test_query_plans_tool_name(self):
        assert query_plans.name == "query_plans"

    def test_query_promotions_tool_name(self):
        assert query_promotions.name == "query_promotions"

    def test_system_prompt_has_no_hardcoded_prices(self):
        assert "Rs. 299" not in SYSTEM_PROMPT and "Rs. 499" not in SYSTEM_PROMPT, (
            "Session 5 removes the hardcoded price table from SYSTEM_PROMPT. "
            "Prices now come from query_plans(). Remove the 'Products and services' price block."
        )

    def test_system_prompt_mentions_tools_or_database(self):
        assert "tool" in SYSTEM_PROMPT.lower() or "database" in SYSTEM_PROMPT.lower(), (
            "SYSTEM_PROMPT should instruct the LLM to use database tools for prices."
        )


# ---------------------------------------------------------------------------
# TestRunToolDispatch
# ---------------------------------------------------------------------------

class TestRunToolDispatch:
    def test_run_tool_dispatches_query_plans(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = _run_tool("query_plans", {"plan_type": "mobile"})
        assert "Daily" in result

    def test_run_tool_dispatches_query_promotions(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = _run_tool("query_promotions", {"plan_id": "all"})
        assert "Monsoon" in result

    def test_run_tool_unknown_name_returns_error_string(self):
        result = _run_tool("nonexistent_tool", {})
        assert "Unknown tool" in result
        assert "nonexistent_tool" in result

    def test_run_tool_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr(_tools, "DB_PATH", seeded_db)
        result = _run_tool("query_plans", {"plan_type": "broadband"})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestRespondWithTools
# ---------------------------------------------------------------------------

class TestRespondWithTools:
    def _make_tool_call_result(self, tool_name, args, call_id="call_abc123"):
        result = MagicMock()
        result.content = ""
        result.tool_calls = [{"id": call_id, "name": tool_name, "args": args}]
        return result

    def _make_text_result(self, content):
        result = MagicMock()
        result.content = content
        result.tool_calls = []
        return result

    def _base_state(self):
        return {
            "customer_message": "What is the price of the Daily 2GB plan?",
            "response": "", "history": [], "query_type": "SIMPLE", "retrieved_docs": [],
        }

    def test_respond_calls_llm_with_tools_first(self):
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm:
            mock_wt.invoke.return_value = self._make_text_result("The Daily 2GB plan is Rs. 299.")
            respond(self._base_state())
        mock_wt.invoke.assert_called_once()
        mock_llm.invoke.assert_not_called()

    def test_respond_no_tool_calls_returns_first_result(self):
        expected = "TeleConnect offers mobile plans starting from Rs. 179."
        state    = {**self._base_state(), "customer_message": "What mobile plans do you offer?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm"):
            mock_wt.invoke.return_value = self._make_text_result(expected)
            result = respond(state)
        assert result["response"] == expected

    def test_respond_makes_second_call_when_tool_requested(self):
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="Daily 2GB: Rs. 299"):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_plans", {"plan_type": "mobile"}
            )
            mock_llm.invoke.return_value = self._make_text_result(
                "The Daily 2GB plan is Rs. 299. BundleIQ | TeleConnect India"
            )
            respond(self._base_state())
        mock_llm.invoke.assert_called_once()

    def test_respond_executes_tool_via_run_tool(self):
        state = {**self._base_state(), "customer_message": "Are there any promotions?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="Monsoon Data Offer...") as mock_rt:
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_promotions", {"plan_id": "all"}
            )
            mock_llm.invoke.return_value = self._make_text_result("We have a Monsoon Data Offer.")
            respond(state)
        mock_rt.assert_called_once_with("query_promotions", {"plan_id": "all"})

    def test_respond_uses_second_call_content_as_response(self):
        final_answer = "The Ultra 500Mbps broadband plan is Rs. 1,599/mo. BundleIQ | TeleConnect India"
        state = {**self._base_state(), "customer_message": "What is the fastest broadband plan?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="Ultra 500Mbps: Rs. 1599/mo"):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_plans", {"plan_type": "broadband"}
            )
            mock_llm.invoke.return_value = self._make_text_result(final_answer)
            result = respond(state)
        assert result["response"] == final_answer

    def test_respond_history_grows_by_two(self):
        state = {**self._base_state(), "customer_message": "What promotions are available?"}
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm"):
            mock_wt.invoke.return_value = self._make_text_result("We have a Monsoon Data Offer.")
            result = respond(state)
        assert len(result["history"]) == 2
        assert result["history"][0]["role"] == "user"
        assert result["history"][1]["role"] == "assistant"

    def test_respond_appends_tool_message_to_conversation(self):
        captured_messages = []

        def capture_invoke(msgs):
            captured_messages.extend(msgs)
            return MagicMock(content="Daily 2GB is Rs. 299. BundleIQ | TeleConnect India", tool_calls=[])

        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "llm") as mock_llm, \
             patch.object(_nodes, "_run_tool", return_value="Daily 2GB: Rs. 299"):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_plans", {"plan_type": "mobile"}
            )
            mock_llm.invoke.side_effect = capture_invoke
            respond(self._base_state())

        from langchain_core.messages import ToolMessage as TM
        tool_messages = [m for m in captured_messages if isinstance(m, TM)]
        assert len(tool_messages) == 1
        assert "Daily 2GB" in tool_messages[0].content or "299" in tool_messages[0].content


# ---------------------------------------------------------------------------
# TestGraphRouting
# ---------------------------------------------------------------------------

class TestGraphRouting:
    def _mock_vectorstore(self):
        vs = MagicMock()
        vs.similarity_search.return_value = []
        return vs

    def test_simple_path_calls_llm_with_tools(self):
        from langgraph.checkpoint.memory import MemorySaver

        mock_vs = self._mock_vectorstore()
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "classifier_llm") as mock_cl, \
             patch.object(_nodes, "llm"), \
             patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            mock_cl.invoke.return_value = MagicMock(content="SIMPLE")
            mock_wt.invoke.return_value = MagicMock(
                content="The Daily 2GB plan is Rs. 299.", tool_calls=[]
            )
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-simple"}}
            graph.invoke(
                {"customer_message": "What is the price of the Daily 2GB plan?", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_called_once()

    def test_complex_path_skips_llm_with_tools(self):
        from langgraph.checkpoint.memory import MemorySaver

        mock_vs = self._mock_vectorstore()
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "classifier_llm") as mock_cl, \
             patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            mock_cl.invoke.return_value = MagicMock(content="COMPLEX")
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-complex"}}
            result = graph.invoke(
                {"customer_message": "Which plan is best for heavy streaming?", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_not_called()
        assert "TeleConnect advisor" in result["response"] or "1800-123-4567" in result["response"]

    def test_out_of_scope_path_skips_llm_with_tools(self):
        from langgraph.checkpoint.memory import MemorySaver

        mock_vs = self._mock_vectorstore()
        with patch.object(_nodes, "llm_with_tools") as mock_wt, \
             patch.object(_nodes, "classifier_llm") as mock_cl, \
             patch.object(_nodes, "vectorstore", mock_vs), \
             patch.object(_nodes, "_init_vectorstore"):
            mock_cl.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-oos"}}
            result = graph.invoke(
                {"customer_message": "What is the weather today?", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_not_called()
        assert "only help with TeleConnect" in result["response"]
