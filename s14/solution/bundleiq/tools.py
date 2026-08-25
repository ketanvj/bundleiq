"""
bundleiq/tools.py
-----------------
LLM clients and MCP-backed tool loading for BundleIQ.

Session 14 adds a separate classifier_llm (compound-mini) and
llamaguard_llm (Llama Prompt Guard 2) for the input guard.
"""
import asyncio
import sys

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import (
    CLASSIFIER_MAX_TOKENS,
    CLASSIFIER_MODEL,
    GROQ_API_KEY,
    LLAMAGUARD_MAX_TOKENS,
    LLAMAGUARD_MODEL,
    LLAMAGUARD_THRESHOLD,
    MAX_TOKENS,
    MCP_SERVER_PATH,
    MODEL_NAME,
    TEMPERATURE,
)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

classifier_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=CLASSIFIER_MODEL,
    temperature=0.0,
    max_tokens=CLASSIFIER_MAX_TOKENS,
)

# S14: Llama Prompt Guard 2 — Layer 2 of the input guard.
# Separate client so its settings don't bleed into the main LLM.
llamaguard_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=LLAMAGUARD_MODEL,
    temperature=0.0,
    max_tokens=LLAMAGUARD_MAX_TOKENS,
)

# ---------------------------------------------------------------------------
# MCP tool loading -- langchain-mcp-adapters
# ---------------------------------------------------------------------------

_mcp_client = MultiServerMCPClient({
    "bundleiq": {
        "transport": "stdio",
        "command":   sys.executable,
        "args":      [str(MCP_SERVER_PATH)],
    }
})

mcp_tools      = asyncio.run(_mcp_client.get_tools())   # [query_plans, query_promotions]
_tool_registry = {t.name: t for t in mcp_tools}

llm_with_tools = llm.bind_tools(mcp_tools)


def _extract_text(result) -> str:
    """MCP tool results come back as a list of content blocks. Join text blocks."""
    if isinstance(result, list):
        return "\n".join(
            block.get("text", "") for block in result if isinstance(block, dict)
        )
    return str(result)


def _run_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name not in _tool_registry:
        return f"Unknown tool: {tool_name}"
    try:
        result = asyncio.run(_tool_registry[tool_name].ainvoke(tool_args))
        return _extract_text(result)
    except Exception as e:
        return f"Tool error ({tool_name}): {e}"
