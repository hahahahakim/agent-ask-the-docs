"""
FastAPI application factory for the 0G Docs Agent API.

Security controls are layered in this order (outermost → innermost):
  CORS → SecurityHeaders → SlowAPI rate limit → auth dependency → handler

Starlette applies middlewares in reverse registration order, so CORS is
registered last here (making it run first on every request). This ensures
browser preflight OPTIONS requests are answered before rate limiting or
authentication runs — preventing opaque CORS errors on 429/401 responses.
"""

import asyncio
import contextlib
import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from api.config import settings
from api.log import configure_logging
from api.routers import chat, health
from api.security.rate_limit import limiter, rate_limit_exceeded_handler


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add defensive HTTP headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'none'"
        response.headers["Cache-Control"] = "no-store"
        # HSTS — only meaningful when TLS is terminated at this process or a
        # reverse proxy that forwards the header; safe to include always.
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        return response


# ---------------------------------------------------------------------------
# Periodic cache watch — re-run warm_cache every N hours to detect doc changes
# ---------------------------------------------------------------------------

async def _periodic_cache_watch() -> None:
    """Re-run warm_cache daily at midnight GMT+8 (16:00 UTC) to detect doc changes."""
    _UTC_HOUR = 16  # midnight GMT+8 = 16:00 UTC
    while True:
        now = datetime.now(timezone.utc)
        next_run = now.replace(hour=_UTC_HOUR, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        await asyncio.sleep((next_run - now).total_seconds())
        logging.getLogger("api").info("Scheduled cache watch started", extra={"trigger": "daily", "utc_hour": _UTC_HOUR})
        try:
            from agent import warm_cache  # noqa: PLC0415
            await warm_cache()
        except Exception:
            logging.getLogger("api").exception("Periodic cache watch failed")


# ---------------------------------------------------------------------------
# Application lifespan — build the agent once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    with contextlib.redirect_stderr(io.StringIO()):
        from agent import build_agent, warm_cache  # noqa: PLC0415
        from core.persistence import get_thread_tracker

    logging.getLogger("api").info("Agent starting", extra={"model": settings.model_name})
    app.state.agent = build_agent(verbose=settings.verbose)
    app.state.thread_tracker = get_thread_tracker(ttl_hours=settings.thread_ttl_hours)

    # Pre-fetch known URLs into the page cache in the background so the first
    # query hits cache instead of making live HTTP requests.
    asyncio.create_task(warm_cache())
    asyncio.create_task(_periodic_cache_watch())
    yield
    # MemorySaver and ThreadTracker are in-process; nothing to clean up on shutdown


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="0G Labs Documentation Agent API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # --- Exception handlers -------------------------------------------------
    # Generic handler: never return stack traces or internal error details.
    @app.exception_handler(Exception)
    async def _generic_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Please try again."},
        )

    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    # --- Middlewares (registered inner → outer; Starlette runs outer first) -

    # 1. Rate limiting (innermost — runs after CORS approves the request)
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    # 2. Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # 3. CORS (outermost — answers preflight OPTIONS before anything else fires)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_allowed_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
        allow_credentials=False,
    )

    # --- Routers ------------------------------------------------------------
    app.include_router(health.router)
    app.include_router(chat.router)

    return app


app = create_app()
