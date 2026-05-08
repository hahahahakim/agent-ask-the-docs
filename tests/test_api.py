"""
API tests for the 0G Docs Agent FastAPI server.

These tests verify security controls without making real LLM or HTTP calls.
The agent and LLM cache are mocked at the fixture level.

Run:
    python -m pytest tests/test_api.py -v
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

VALID_KEY = "test-api-key-valid"   # matches conftest.py API_KEYS value
INVALID_KEY = "wrong-key"
BASE_URL = "http://testserver"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_agent():
    """A mock LangGraph agent that returns a canned answer."""
    agent = MagicMock()

    async def _fake_astream_events(*args, **kwargs):
        # Yield nothing — run_query falls back to aget_state
        return
        yield  # make it an async generator

    agent.astream_events = _fake_astream_events

    state = MagicMock()
    state.values = {
        "messages": [
            MagicMock(
                content="0G is a decentralised AI infrastructure network.",
                tool_calls=None,
            )
        ]
    }
    agent.aget_state = AsyncMock(return_value=state)
    return agent


@pytest.fixture(scope="module")
def app(mock_agent):
    """Create the FastAPI app with the agent and LLM cache mocked out.

    The lifespan (which calls build_agent) does not run in test mode, so we
    set app.state.agent directly after creation.
    """
    with patch("agent.build_agent", return_value=mock_agent):
        from api.main import create_app
        test_app = create_app()
    # Inject state directly — bypasses the lifespan for unit tests
    test_app.state.agent = mock_agent
    from core.persistence import get_thread_tracker
    test_app.state.thread_tracker = get_thread_tracker(ttl_hours=24)
    return test_app


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url=BASE_URL
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_no_auth(client):
    """Health endpoint must return 200 without any API key."""
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_missing_key_returns_401(client):
    r = await client.post("/chat", json={"query": "What is 0G?"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_chat_invalid_key_returns_401(client):
    r = await client.post(
        "/chat",
        json={"query": "What is 0G?"},
        headers={"X-API-Key": INVALID_KEY},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_chat_valid_key_returns_200(client):
    r = await client.post(
        "/chat",
        json={"query": "What is 0G?"},
        headers={"X-API-Key": VALID_KEY},
    )
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "thread_id" in body


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_query_returns_422(client):
    r = await client.post(
        "/chat",
        json={"query": ""},
        headers={"X-API-Key": VALID_KEY},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_whitespace_only_query_returns_422(client):
    r = await client.post(
        "/chat",
        json={"query": "   "},
        headers={"X-API-Key": VALID_KEY},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_oversized_query_returns_422(client):
    r = await client.post(
        "/chat",
        json={"query": "x" * 2001},
        headers={"X-API-Key": VALID_KEY},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_invalid_thread_id_returns_422(client):
    """thread_id with path traversal chars must be rejected."""
    r = await client.post(
        "/chat",
        json={"query": "What is 0G?", "thread_id": "../../etc/passwd"},
        headers={"X-API-Key": VALID_KEY},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_telegram_style_thread_id_accepted(client):
    """tg-<chat_id> format used by the Telegram bot must be valid."""
    r = await client.post(
        "/chat",
        json={"query": "What is 0G?", "thread_id": "tg-123456789"},
        headers={"X-API-Key": VALID_KEY},
    )
    assert r.status_code == 200
    assert r.json()["thread_id"] == "tg-123456789"


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_security_headers_present(client):
    r = await client.get("/health")
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert "strict-origin-when-cross-origin" in r.headers.get("referrer-policy", "")


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------

def test_ssrf_guard_blocks_external_urls():
    from importlib.util import spec_from_file_location, module_from_spec
    from pathlib import Path
    spec = spec_from_file_location(
        "skill_script",
        Path(__file__).parent.parent / "skills" / "0g-docs-search" / "script.py",
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._is_allowed_url("https://docs.0g.ai/") is True
    assert module._is_allowed_url("https://pc.0g.ai/") is True
    assert module._is_allowed_url("https://0g.ai/blog") is True
    assert module._is_allowed_url("https://evil.com") is False
    assert module._is_allowed_url("http://169.254.169.254") is False   # cloud metadata
    assert module._is_allowed_url("javascript:alert(1)") is False
    assert module._is_allowed_url("file:///etc/passwd") is False
    assert module._is_allowed_url("ftp://0g.ai/file") is False
