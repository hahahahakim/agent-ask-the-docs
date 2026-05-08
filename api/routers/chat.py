"""
Chat endpoints.

POST /chat         — single request / response (Telegram bot, server-side calls)
POST /chat/stream  — SSE token stream (website widget)

Both endpoints require a valid X-API-Key header and share the same rate limit
bucket per key. The streaming endpoint uses POST (not GET) so that query
content and thread_id are never written to server access logs or browser
history as URL parameters.
"""

import json
import logging
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, ToolMessage

from api.config import settings
from api.models import ChatRequest, ChatResponse
from api.security.auth import verify_api_key
from api.security.rate_limit import limiter

# agent.py lives at the project root; importable when uvicorn is started from there.
from agent import RECURSION_LIMIT, _build_initial_input, _clean_text, _extract_text, run_query

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

RATE_LIMIT = "20/minute"


def _build_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMIT)
async def chat(
    request: Request,           # must be first for SlowAPI to read the key
    body: ChatRequest,
    _: str = Depends(verify_api_key),
) -> ChatResponse:
    """Send a query and receive a complete answer."""
    agent = request.app.state.agent
    tracker = request.app.state.thread_tracker

    tracker.maybe_expire(body.thread_id, agent.checkpointer)

    logger.info("chat request  thread_id=%s", body.thread_id)
    t0 = time.perf_counter()
    answer = await run_query(agent, body.query, _build_config(body.thread_id))
    if not answer:
        answer = "I was unable to find an answer. Please try rephrasing your question."

    duration = time.perf_counter() - t0
    tracker.touch(body.thread_id)
    logger.info("chat response thread_id=%s [%.2fs]", body.thread_id, duration)
    return ChatResponse(answer=answer, thread_id=body.thread_id, model=settings.model_name, duration_s=round(duration, 2))


async def _token_generator(agent, query: str, config: dict, thread_id: str):
    """Yield SSE frames: ``data: {"token": "..."}`` then ``data: [DONE]``.

    SSE frame types:
      data: {"token": "..."}   — one text token
      data: [DONE]             — stream finished successfully
      event: error             — stream failed; client should retry via POST /chat
      data: {"message": "..."}
    """
    t0 = time.perf_counter()
    try:
        initial_input = await _build_initial_input(query)
        had_tokens = False
        final_ai_content = None

        async for event in agent.astream_events(initial_input, config=config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                    continue
                text = _extract_text(chunk.content)
                if text and "<tool_call>" not in text:
                    had_tokens = True
                    yield f"data: {json.dumps({'token': text})}\n\n"
            elif event["event"] == "on_chain_end" and event.get("name") == "LangGraph":
                output = event.get("data", {}).get("output") or {}
                for msg in reversed(output.get("messages", [])):
                    if isinstance(msg, (HumanMessage, ToolMessage)):
                        break
                    text = _clean_text(_extract_text(getattr(msg, "content", "")))
                    if text and not getattr(msg, "tool_calls", None):
                        final_ai_content = text
                        break

        # Fallback: endpoint didn't stream tokens — emit the final answer as one chunk
        if not had_tokens and final_ai_content:
            yield f"data: {json.dumps({'token': final_ai_content})}\n\n"

        duration = time.perf_counter() - t0
        logger.info("chat/stream response thread_id=%s [%.2fs]", thread_id, duration)
        yield f"data: {json.dumps({'model': settings.model_name, 'duration_s': round(duration, 2)})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception:
        logger.exception("chat/stream error thread_id=%s [%.2fs]", thread_id, time.perf_counter() - t0)
        yield "event: error\n"
        yield f"data: {json.dumps({'message': 'Stream interrupted. Retry via POST /chat with the same query and thread_id.'})}\n\n"


@router.post("/chat/stream")
@limiter.limit(RATE_LIMIT)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    _: str = Depends(verify_api_key),
) -> StreamingResponse:
    """
    SSE endpoint — streams tokens as they are generated.

    Clients should use ``fetch()`` (not ``EventSource``) so they can send the
    ``X-API-Key`` header and a JSON body.

    Example JavaScript:
        const res = await fetch("/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json", "X-API-Key": key },
            body: JSON.stringify({ query, thread_id }),
        });
        const reader = res.body.getReader();
        // read chunks and parse SSE frames
    """
    agent = request.app.state.agent
    tracker = request.app.state.thread_tracker

    tracker.maybe_expire(body.thread_id, agent.checkpointer)
    tracker.touch(body.thread_id)

    logger.info("chat/stream request thread_id=%s", body.thread_id)
    return StreamingResponse(
        _token_generator(agent, body.query, _build_config(body.thread_id), body.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx proxy buffering
            "Connection": "keep-alive",
        },
    )
