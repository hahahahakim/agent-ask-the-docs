"""
Smoke tests for the 0G Labs documentation agent.

Tests are grouped into three levels:
  1. Imports       — every module loads without error
  2. Wiring        — skill loads, graph compiles, tools have correct schemas
  3. Integration   — a real query is answered end-to-end (requires .env)

Run all:
    python3 -m pytest tests/ -v

Run only fast tests (skip the live API call):
    python3 -m pytest tests/ -v -m "not integration"
"""

import asyncio
import contextlib
import io
import os
import sys
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

# Skills and agent.py live one level up from this file
ROOT = Path(__file__).parent.parent
SKILL_DIR = ROOT / "skills" / "0g-docs-search"


# ---------------------------------------------------------------------------
# 1. Import tests
# ---------------------------------------------------------------------------

def test_import_graph():
    with contextlib.redirect_stderr(io.StringIO()):
        from core import graph  # noqa: F401


def test_import_persistence():
    with contextlib.redirect_stderr(io.StringIO()):
        from core import persistence  # noqa: F401


def test_import_skill_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("skill_script", SKILL_DIR / "script.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "fetch_page")
    assert hasattr(module, "fetch_pages_parallel")


# ---------------------------------------------------------------------------
# 2. Wiring tests
# ---------------------------------------------------------------------------

def test_load_skill_returns_prompt_and_tools():
    """_load_skill() must return a non-empty system prompt, 2 tools, and callables."""
    with contextlib.redirect_stderr(io.StringIO()):
        import agent
    prompt, tools, clear_cache, warm, prefetch = agent._load_skill()
    assert isinstance(prompt, str) and len(prompt) > 100, "System prompt looks empty"
    assert len(tools) == 2, f"Expected 2 tools, got {len(tools)}"
    assert callable(clear_cache), "clear_page_cache must be callable"
    assert callable(warm), "warm_cache must be callable"
    assert callable(prefetch), "prefetch_urls must be callable"


def test_tool_schemas():
    """fetch_page and fetch_pages_parallel must have correct input schemas."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("skill_script", SKILL_DIR / "script.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # fetch_page must accept a single 'url' string field
    fp_schema = module.fetch_page.args_schema.model_json_schema()
    assert "url" in fp_schema["properties"], "fetch_page missing 'url' field"

    # fetch_pages_parallel must accept a single 'urls' string field (not array)
    fpp_schema = module.fetch_pages_parallel.args_schema.model_json_schema()
    assert "urls" in fpp_schema["properties"], "fetch_pages_parallel missing 'urls' field"
    assert fpp_schema["properties"]["urls"]["type"] == "string", \
        "fetch_pages_parallel 'urls' must be a plain string (not array)"


def test_build_graph_compiles():
    """build_graph() must return a compiled LangGraph graph with a mock LLM."""
    from unittest.mock import MagicMock
    with contextlib.redirect_stderr(io.StringIO()):
        from core.graph import build_graph
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_core.tools import tool

    @tool
    def dummy_tool(query: str) -> str:
        """A dummy tool for testing."""
        return "dummy"

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm

    compiled = build_graph(mock_llm, [dummy_tool], "test prompt", MemorySaver())
    assert compiled is not None
    assert hasattr(compiled, "astream_events"), "Compiled graph missing astream_events"


def test_get_checkpointer_returns_memory_saver():
    with contextlib.redirect_stderr(io.StringIO()):
        from core.persistence import get_checkpointer
        from langgraph.checkpoint.memory import MemorySaver
    checkpointer = get_checkpointer()
    assert isinstance(checkpointer, MemorySaver)



def test_skill_md_has_all_sources():
    """SKILL.md must reference all five 0G documentation sources."""
    skill_md = (SKILL_DIR / "SKILL.md").read_text()
    for url in ["docs.0g.ai", "build.0g.ai", "pc.0g.ai", "app.0g.ai", "0g.ai/blog"]:
        assert url in skill_md, f"SKILL.md missing reference to {url}"


# ---------------------------------------------------------------------------
# 3. Integration tests  (skipped when OPENAI_API_KEY is missing)
# ---------------------------------------------------------------------------

SKIP_INTEGRATION = not os.getenv("OPENAI_API_KEY")
integration = pytest.mark.skipif(SKIP_INTEGRATION, reason="OPENAI_API_KEY not set")


@pytest.mark.integration
@integration
def test_build_agent_does_not_raise():
    """build_agent() must construct the full graph without error."""
    with contextlib.redirect_stderr(io.StringIO()):
        import agent
    graph_obj = agent.build_agent(verbose=False)
    assert graph_obj is not None


@pytest.mark.integration
@integration
def test_fetch_page_returns_content():
    """fetch_page must return non-empty content from docs.0g.ai."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("skill_script", SKILL_DIR / "script.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = asyncio.run(module.fetch_page.ainvoke({"url": "https://docs.0g.ai/"}))
    assert isinstance(result, str) and len(result) > 50, \
        f"fetch_page returned too little content: {repr(result[:200])}"


@pytest.mark.integration
@integration
def test_full_query_returns_answer():
    """A real query must produce a non-empty answer string."""
    with contextlib.redirect_stderr(io.StringIO()):
        import agent

    graph_obj = agent.build_agent(verbose=False)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}

    answer = asyncio.run(agent.run_query(graph_obj, "What is 0G?", config))
    assert isinstance(answer, str) and len(answer) > 50, \
        f"Expected a real answer, got: {repr(answer[:200])}"
