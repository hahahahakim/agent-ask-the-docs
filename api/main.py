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
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from api.config import settings
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
        # HSTS — only meaningful when TLS is terminated at this process or a
        # reverse proxy that forwards the header; safe to include always.
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        return response


# ---------------------------------------------------------------------------
# Application lifespan — build the agent once at startup
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    """Inject UTC timestamps into uvicorn's coloured handlers and the api logger."""
    from uvicorn.logging import DefaultFormatter

    fmt = DefaultFormatter(
        fmt="%(asctime)s %(levelprefix)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        use_colors=True,
    )
    fmt.converter = time.gmtime

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        for handler in logging.getLogger(name).handlers:
            handler.setFormatter(fmt)

    api_handler = logging.StreamHandler()
    api_handler.setFormatter(fmt)
    api_logger = logging.getLogger("api")
    api_logger.setLevel(logging.INFO)
    api_logger.addHandler(api_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()

    with contextlib.redirect_stderr(io.StringIO()):
        from agent import build_agent, warm_cache  # noqa: PLC0415
        from core.persistence import get_thread_tracker

    app.state.agent = build_agent(verbose=settings.verbose)
    app.state.thread_tracker = get_thread_tracker(ttl_hours=settings.thread_ttl_hours)

    # Pre-fetch known URLs into the page cache in the background so the first
    # query hits cache instead of making live HTTP requests.
    asyncio.create_task(warm_cache())
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
