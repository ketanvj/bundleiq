"""
s09/tests/test_s09.py
---------------------
Tests for Session 9: TRAI Compliance Filter + LangSmith tracing.

Run with:
    pytest s09/tests/ -v
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

from bundleiq.config import (         # noqa: E402
    BUNDLEIQ_BANNED_PHRASES,
    DB_PATH,
    SAFE_COMPLIANCE_RESPONSE,
)
from bundleiq.state import BundleIQState   # noqa: E402
import bundleiq.nodes as _nodes            # noqa: E402
from bundleiq.nodes import (               # noqa: E402
    _BANNED_PATTERN,
    _check_compliance,
    _extract_prices,
    _load_valid_prices,
    _normalize_for_check,
    check_compliance,
    classify,
    decline,
    escalate,
)
from bundleiq.agent import build_graph     # noqa: E402


# ---------------------------------------------------------------------------
# TestBannedPattern
# ---------------------------------------------------------------------------

class TestBannedPattern:
    def test_detects_guaranteed_coverage(self):
        assert _BANNED_PATTERN.search("guaranteed coverage in your area") is not None

    def test_detects_coverage_guaranteed(self):
        assert _BANNED_PATTERN.search("coverage guaranteed for 5G areas") is not None

    def test_does_not_flag_safe_text(self):
        assert _BANNED_PATTERN.search("TeleConnect has great network in most cities") is None

    def test_case_insensitive(self):
        assert _BANNED_PATTERN.search("GUARANTEED COVERAGE") is not None

    def test_detects_no_dead_zones(self):
        assert _BANNED_PATTERN.search("experience no dead zones with TeleConnect") is not None

    def test_detects_guaranteed_signal(self):
        assert _BANNED_PATTERN.search("guaranteed signal strength everywhere") is not None


# ---------------------------------------------------------------------------
# TestNormalizeForCheck
# ---------------------------------------------------------------------------

class TestNormalizeForCheck:
    def test_lowercases_text(self):
        assert _normalize_for_check("TeleConnect INDIA") == "teleconnect india"

    def test_replaces_unicode_hyphen(self):
        text_with_unicode = "high‑speed broadband"   # U+2011 non-breaking hyphen
        result = _normalize_for_check(text_with_unicode)
        assert "-" in result
        assert "‑" not in result

    def test_nfkc_normalization(self):
        result = _normalize_for_check("Ｔeleconnect")   # full-width T
        assert result[0] == "t"


# ---------------------------------------------------------------------------
# TestExtractPrices
# ---------------------------------------------------------------------------

class TestExtractPrices:
    def test_extracts_rs_with_space(self):
        assert _extract_prices("Rs. 179 per month") == [179]

    def test_extracts_rs_without_space_and_rupee_symbol(self):
        result = _extract_prices("Rs.299 or ₹399")
        assert 299 in result
        assert 399 in result

    def test_extracts_comma_formatted_price(self):
        assert _extract_prices("Rs. 1,499 bundle price") == [1499]

    def test_no_prices_returns_empty(self):
        assert _extract_prices("no prices here") == []

    def test_multiple_prices(self):
        result = _extract_prices("plans from Rs. 179 to Rs. 599")
        assert result == [179, 599]


# ---------------------------------------------------------------------------
# TestLoadValidPrices
# ---------------------------------------------------------------------------

DB_AVAILABLE = DB_PATH.exists()

class TestLoadValidPrices:
    @pytest.mark.skipif(not DB_AVAILABLE, reason="teleconnect_data.db not found")
    def test_returns_nonempty_set(self):
        prices = _load_valid_prices()
        assert len(prices) > 0

    @pytest.mark.skipif(not DB_AVAILABLE, reason="teleconnect_data.db not found")
    def test_contains_known_mobile_price(self):
        prices = _load_valid_prices()
        assert 179 in prices  # mob_001 Daily 1GB

    @pytest.mark.skipif(not DB_AVAILABLE, reason="teleconnect_data.db not found")
    def test_contains_known_broadband_price(self):
        prices = _load_valid_prices()
        assert 499 in prices  # bb_001 Basic 50Mbps

    def test_returns_set_on_missing_db(self):
        with patch("bundleiq.nodes.DB_PATH", Path("/nonexistent/path.db")):
            result = _load_valid_prices()
        assert isinstance(result, set)


# ---------------------------------------------------------------------------
# TestCheckCompliance
# ---------------------------------------------------------------------------

MOCK_VALID_PRICES = {179, 299, 399, 499, 599, 799, 1099, 1499, 2999, 500}

class TestCheckCompliance:
    def test_detects_guaranteed_coverage(self):
        ok, reason = _check_compliance("We offer guaranteed coverage in all cities.")
        assert ok is False
        assert "banned phrase" in reason

    def test_detects_no_dead_zones(self):
        ok, reason = _check_compliance("There are no dead zones with our network.")
        assert ok is False
        assert "banned phrase" in reason

    def test_passes_safe_response(self):
        ok, reason = _check_compliance(
            "TeleConnect's Daily 1GB plan costs Rs. 179 per month. BundleIQ | TeleConnect India"
        )
        # Valid price, no banned phrase
        assert ok is True
        assert reason == "PASS"

    def test_passes_valid_price(self):
        with patch("bundleiq.nodes._load_valid_prices", return_value=MOCK_VALID_PRICES):
            ok, reason = _check_compliance("The plan is available at Rs. 179 per month.")
        assert ok is True
        assert reason == "PASS"

    def test_detects_wrong_price(self):
        with patch("bundleiq.nodes._load_valid_prices", return_value=MOCK_VALID_PRICES):
            ok, reason = _check_compliance("Get our plan for just Rs. 999 per month.")
        assert ok is False
        assert "incorrect price" in reason


# ---------------------------------------------------------------------------
# TestCheckComplianceNode
# ---------------------------------------------------------------------------

class TestCheckComplianceNode:
    def _make_state(self, response: str) -> BundleIQState:
        return {
            "customer_message": "test",
            "response":         response,
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   [],
            "compliance_status": "",
        }

    def test_compliant_response_passes(self):
        result = check_compliance(self._make_state("TeleConnect offers great plans."))
        assert result.get("compliance_status") == "PASS"
        assert "response" not in result  # response unchanged

    def test_banned_phrase_triggers_safe_response(self):
        result = check_compliance(self._make_state("We have guaranteed coverage everywhere."))
        assert result["response"] == SAFE_COMPLIANCE_RESPONSE
        assert result["compliance_status"].startswith("FAIL")

    def test_wrong_price_triggers_safe_response(self):
        with patch("bundleiq.nodes._load_valid_prices", return_value=MOCK_VALID_PRICES):
            result = check_compliance(self._make_state("Our plan is only Rs. 888 per month."))
        assert result["response"] == SAFE_COMPLIANCE_RESPONSE
        assert result["compliance_status"].startswith("FAIL")

    def test_check_compliance_importable(self):
        from bundleiq.nodes import check_compliance as cc
        assert callable(cc)


# ---------------------------------------------------------------------------
# TestGraphNodes
# ---------------------------------------------------------------------------

class TestGraphNodes:
    def _make_state(self, message="test") -> BundleIQState:
        return {
            "customer_message": message,
            "response": "",
            "history": [],
            "query_type": "SIMPLE",
            "retrieved_docs": [],
            "compliance_status": "",
        }

    def test_escalate_response_mentions_advisor(self):
        result = escalate(self._make_state())
        assert "TeleConnect advisor" in result["response"]

    def test_escalate_response_includes_phone(self):
        result = escalate(self._make_state())
        assert "1800-123-4567" in result["response"]

    def test_escalate_updates_history(self):
        result = escalate(self._make_state("which plan?"))
        assert len(result["history"]) == 2

    def test_decline_response_mentions_teleconnect(self):
        result = decline(self._make_state())
        assert "TeleConnect" in result["response"]

    def test_decline_updates_history(self):
        result = decline(self._make_state("off-topic"))
        assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# TestBuildGraph
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_build_graph_returns_compiled_graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        assert build_graph(checkpointer=MemorySaver()) is not None

    def test_graph_invoke_complex_escalates(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="COMPLEX")
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Which plan is best for me?", "response": "", "compliance_status": ""},
                config={"configurable": {"thread_id": "test-complex"}},
            )
        assert "TeleConnect advisor" in result["response"]
        assert result["query_type"] == "COMPLEX"

    def test_graph_invoke_oos_declines(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Write me a poem", "response": "", "compliance_status": ""},
                config={"configurable": {"thread_id": "test-oos"}},
            )
        assert "TeleConnect" in result["response"]
        assert result["query_type"] == "OUT_OF_SCOPE"

    def test_graph_simple_result_has_compliance_status(self):
        from langgraph.checkpoint.memory import MemorySaver
        with patch.object(_nodes, "classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="SIMPLE")
            with patch.object(_nodes, "llm_with_tools") as mock_lm:
                mock_lm.invoke.return_value = MagicMock(
                    content="TeleConnect offers great plans.",
                    tool_calls=[],
                )
                with patch.object(_nodes, "_init_vectorstore"):
                    _nodes.vectorstore = MagicMock()
                    _nodes.vectorstore.similarity_search.return_value = []
                    graph = build_graph(checkpointer=MemorySaver())
                    result = graph.invoke(
                        {"customer_message": "Tell me about plans", "response": "", "compliance_status": ""},
                        config={"configurable": {"thread_id": "test-simple"}},
                    )
        assert "compliance_status" in result
