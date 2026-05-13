"""
Golden test suite for 0G Labs documentation agent.

Two test classes:
  1. TestRouting      — pure unit tests on _route_query(); no API key needed
  2. TestAnswerQuality — end-to-end answer quality tests (requires OPENAI_API_KEY)

Run routing-only (fast CI):
    python3 -m pytest tests/test_answers.py::TestRouting -v

Run everything (requires .env with OPENAI_API_KEY):
    python3 -m pytest tests/test_answers.py -v -m integration
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import sys
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
FIXTURES_PATH = Path(__file__).parent / "fixtures" / "golden_queries.json"

# Ensure fixtures directory exists (idempotent)
os.makedirs(FIXTURES_PATH.parent, exist_ok=True)

# ---------------------------------------------------------------------------
# Load golden queries
# ---------------------------------------------------------------------------

with FIXTURES_PATH.open() as _f:
    _GOLDEN_QUERIES: list[dict] = json.load(_f)

# Non-blog queries are used for routing tests (blog uses a sitemap URL which
# does not contain a predictable path substring from _ROUTE_MAP)
_ROUTING_QUERIES = [q for q in _GOLDEN_QUERIES if q["topic"] != "blog"]


# ---------------------------------------------------------------------------
# Integration marker — mirrors tests/test_agent.py exactly
# ---------------------------------------------------------------------------

SKIP_INTEGRATION = not os.getenv("OPENAI_API_KEY")
integration = pytest.mark.skipif(SKIP_INTEGRATION, reason="OPENAI_API_KEY not set")


# ---------------------------------------------------------------------------
# Class 1: TestRouting — pure unit tests, no API key required
# ---------------------------------------------------------------------------

class TestRouting:
    """Verify _route_query() returns URLs matching expected patterns."""

    @pytest.mark.parametrize(
        "case",
        _ROUTING_QUERIES,
        ids=[q["id"] for q in _ROUTING_QUERIES],
    )
    def test_route_returns_expected_urls(self, case: dict) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            import agent  # noqa: PLC0415

        urls = agent._route_query(case["query"])

        # Must return at least one URL
        assert urls, (
            f"[{case['id']}] _route_query returned an empty list for: {case['query']!r}"
        )

        # At least one returned URL must contain at least one expected pattern
        patterns = case["expected_url_patterns"]
        matched = any(
            pattern in url
            for url in urls
            for pattern in patterns
        )
        assert matched, (
            f"[{case['id']}] None of the returned URLs match any of {patterns}.\n"
            f"  Query: {case['query']!r}\n"
            f"  URLs returned: {urls}"
        )


# ---------------------------------------------------------------------------
# Class 2: TestAnswerQuality — integration tests, requires OPENAI_API_KEY
# ---------------------------------------------------------------------------

class TestAnswerQuality:
    """End-to-end answer quality tests against the live API."""

    # Module-scoped agent built once for all tests in this class
    _agent = None

    @classmethod
    def _get_agent(cls):
        if cls._agent is None:
            with contextlib.redirect_stderr(io.StringIO()):
                import agent as _agent_module  # noqa: PLC0415
            cls._agent = _agent_module.build_agent(verbose=False)
        return cls._agent

    @pytest.mark.integration
    @integration
    @pytest.mark.parametrize(
        "case",
        _GOLDEN_QUERIES,
        ids=[q["id"] for q in _GOLDEN_QUERIES],
    )
    def test_answer_contains_keywords(self, case: dict) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            import agent  # noqa: PLC0415

        graph = self._get_agent()
        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": 50,
        }

        answer = asyncio.run(agent.run_query(graph, case["query"], config))

        assert isinstance(answer, str) and len(answer) > 0, (
            f"[{case['id']}] run_query returned an empty answer for: {case['query']!r}"
        )

        answer_lower = answer.lower()
        keywords = case["expected_answer_keywords"]
        matched_count = sum(1 for kw in keywords if kw.lower() in answer_lower)

        assert matched_count >= 2, (
            f"[{case['id']}] Answer only matched {matched_count}/{len(keywords)} keywords.\n"
            f"  Query: {case['query']!r}\n"
            f"  Expected keywords: {keywords}\n"
            f"  Answer preview: {answer[:300]!r}"
        )

    @pytest.mark.integration
    @integration
    def test_baseline_score(self) -> None:
        """Run all golden queries and report baseline pass rate.

        This test always passes — its purpose is to print the baseline score
        that future phases (semantic router, query decomposition) must improve.
        """
        with contextlib.redirect_stderr(io.StringIO()):
            import agent  # noqa: PLC0415

        graph = self._get_agent()
        pass_count = 0
        total = len(_GOLDEN_QUERIES)

        for case in _GOLDEN_QUERIES:
            try:
                config = {
                    "configurable": {"thread_id": str(uuid.uuid4())},
                    "recursion_limit": 50,
                }
                answer = asyncio.run(agent.run_query(graph, case["query"], config))
                if not answer:
                    continue
                answer_lower = answer.lower()
                keywords = case["expected_answer_keywords"]
                matched = sum(1 for kw in keywords if kw.lower() in answer_lower)
                if matched >= 2:
                    pass_count += 1
            except Exception:  # noqa: BLE001
                pass  # count as fail, but don't abort the baseline run

        print(f"\nBASELINE: {pass_count}/{total} pass")
        # Always passes — exists only to establish a measurable baseline
        assert pass_count >= 0
