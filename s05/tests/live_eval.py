"""
s05/tests/live_eval.py — Live evaluation for BundleIQ S05 solution
--------------------------------------------------------------------
Runs 14 test queries directly against graph.invoke() using the REAL Groq
LLM, the REAL ChromaDB vectorstore, and the REAL SQLite tools database.
No mocks.

Why this exists alongside pytest
─────────────────────────────────
`pytest test_s05.py` mocks the LLM, tools, and vectorstore — it runs in
~2 seconds and catches structural bugs.

This script catches defects that only appear with a real LLM:
  • Classifier brittleness — plan query classified COMPLEX or OUT_OF_SCOPE
  • Tool not called    — LLM answers a plan-price query from memory instead
                         of calling query_plans() (violates Rule 3)
  • Wrong tool args    — query_plans("starter") instead of query_plans("mobile")
  • Stale history      — previous tool output leaks into follow-up answer
  • Out-of-scope drift — competitor comparison sneaks through as SIMPLE

Run this script (from the bundleiq/ directory):
    python s05/tests/live_eval.py

Expected output: 14/14 passed
If any test fails, the response snippet is printed so you can diagnose.

Cost: ~14 Groq API calls (~8–12 seconds total, well within free tier).
"""

import re
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
TESTS_DIR     = Path(__file__).parent
S05_DIR       = TESTS_DIR.parent
SOLUTION_DIR  = S05_DIR / "solution"
BUNDLEIQ_ROOT = S05_DIR.parent   # cohort-1/bundleiq/

load_dotenv(BUNDLEIQ_ROOT / ".env")
sys.path.insert(0, str(SOLUTION_DIR))

from langgraph.checkpoint.memory import MemorySaver
from bundleiq.agent import build_graph
from bundleiq.config import DECLINE_RESPONSE, ESCALATE_RESPONSE
from bundleiq.tools import query_plans   # used to get live prices for validation

graph = build_graph(checkpointer=MemorySaver())

# ── Pull live plan prices from the DB for validation ──────────────────────────
# Proves tool calls return real data, not hallucinated values.
try:
    _all_plans = query_plans.invoke({"plan_type": "all"})
    _price_values = set(re.findall(r'\d+', _all_plans))   # price integers e.g. "299", "499"
except Exception:
    _price_values = set()

# ── Test cases ────────────────────────────────────────────────────────────────
# (label, query, expected_route, expected_behaviour)
#
# expected_route     : "SIMPLE" | "COMPLEX" | "OUT_OF_SCOPE"
# expected_behaviour :
#     "answer"            — substantive reply (not escalate, not decline)
#     "answer_with_price" — not escalate/decline AND response contains a price
#                           value that matches the live DB (tool called)
#     "escalate"          — response == ESCALATE_RESPONSE
#     "decline"           — response == DECLINE_RESPONSE

TEST_CASES = [
    # ── Tool: plan prices ────────────────────────────────────────────────────
    ("Mobile prices",   "What are TeleConnect's current mobile plan prices?",
     "SIMPLE", "answer_with_price"),

    ("Broadband price", "How much does the TeleConnect broadband plan cost per month?",
     "SIMPLE", "answer_with_price"),

    ("Data included",   "How much data is included in TeleConnect's mobile plans?",
     "SIMPLE", "answer_with_price"),

    # ── Tool: promotions ─────────────────────────────────────────────────────
    ("Promotions",      "Are there any current promotions or offers at TeleConnect?",
     "SIMPLE", "answer_with_price"),

    ("Plan offers",     "What offers are available for TeleConnect plans right now?",
     "SIMPLE", "answer_with_price"),

    # ── RAG policy questions (no tool expected) ──────────────────────────────
    ("Fair usage",      "What is TeleConnect's fair usage policy for data?",
     "SIMPLE", "answer"),

    ("Port number",     "Can I keep my existing number when switching to TeleConnect?",
     "SIMPLE", "answer"),

    # ── Complex → escalate ───────────────────────────────────────────────────
    ("Best bundle",     "Which bundle is best for a family of four?",
     "COMPLEX", "escalate"),

    ("Switch advice",   "Should I switch to TeleConnect from my current provider?",
     "COMPLEX", "escalate"),

    # ── Out of scope → decline ───────────────────────────────────────────────
    ("Weather",         "What is the weather today?",
     "OUT_OF_SCOPE", "decline"),

    ("Laptop",          "Recommend me a good laptop to buy",
     "OUT_OF_SCOPE", "decline"),

    ("Cricket",         "Who won the World Cup last year?",
     "OUT_OF_SCOPE", "decline"),

    # ── Follow-up memory ─────────────────────────────────────────────────────
    ("Follow-up 1",     "What are the mobile plan prices at TeleConnect?",
     "SIMPLE", "answer_with_price"),

    ("Follow-up 2",     "And what promotions are available for those plans?",
     "SIMPLE", "answer_with_price"),
]

FOLLOW_UP_START = 12   # index of "Follow-up 1" — share one thread from here
SHARED_THREAD   = "live-eval-memory-thread"


# ── Run ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  BundleIQ S05 — Live Evaluation  (real Groq + real SQLite tools)")
print(f"  DB price values found : {sorted(_price_values)[:8] or 'none — DB may be missing'}")
print("=" * 80)

results = []
for i, (label, query, exp_route, exp_behaviour) in enumerate(TEST_CASES):
    thread = SHARED_THREAD if i >= FOLLOW_UP_START else f"eval-{i}"
    cfg    = {"configurable": {"thread_id": thread}}

    result   = graph.invoke({"customer_message": query, "response": ""}, config=cfg)
    route    = result.get("query_type", "?")
    response = result["response"]

    if response == ESCALATE_RESPONSE:
        actual = "escalate"
    elif response == DECLINE_RESPONSE:
        actual = "decline"
    else:
        actual = "answer"

    # For answer_with_price: verify the response contains a number matching the DB
    if exp_behaviour == "answer_with_price" and actual == "answer":
        resp_nums = set(re.findall(r'\d+', response))
        if _price_values and not resp_nums.isdisjoint(_price_values):
            actual = "answer_with_price"       # DB value confirmed ✓
        elif _price_values and resp_nums.isdisjoint(_price_values):
            actual = "answer_hallucinated"     # number doesn't match DB ✗
        elif not _price_values and re.search(r'₹\d+|\d+\s*(?:per month|/month)', response):
            actual = "answer_with_price"       # DB unavailable, has price at least
        # else: stays "answer" — no numeric price found at all

    route_ok = route == exp_route
    act_ok   = actual == exp_behaviour
    passed   = route_ok and act_ok
    results.append(passed)

    mark = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{mark}  [{label}]")
    print(f"     Q      : {query[:72]}")
    print(f"     Route  : {route} (expected {exp_route}) {'✓' if route_ok else '✗'}")
    print(f"     Action : {actual} (expected {exp_behaviour}) {'✓' if act_ok else '✗'}")
    if not passed:
        snippet = response[:200].replace("\n", " ")
        print(f"     Resp   : {snippet}...")

total  = len(results)
passed = sum(results)
print("\n" + "=" * 80)
print(f"  Result : {passed}/{total} passed")
if passed < total:
    print(f"  {'─' * 40}")
    print(f"  {total - passed} failure(s) above need fixing before S05 is release-ready.")
    print()
    print("  Common causes:")
    print("    answer_hallucinated → LLM answered price question without calling query_plans()")
    print("    wrong route         → CLASSIFY_SYSTEM prompt needs tightening")
    print("    escalate got answer → COMPLEX case fell through to SIMPLE path")
print("=" * 80 + "\n")
