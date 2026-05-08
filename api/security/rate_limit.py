"""
Per-API-key rate limiting via SlowAPI.

The bucket key is the X-API-Key header value so that each client (Telegram
bot, website widget, etc.) gets its own independent quota regardless of
shared IP addresses or NAT. Falls back to client IP if no key is present
(which means auth will also reject the request, but the limit still records).
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded


def _key_from_api_key(request: Request) -> str:
    return request.headers.get("X-API-Key") or (request.client.host if request.client else "unknown")


limiter = Limiter(key_func=_key_from_api_key)


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return a clean 429 — never expose internal rate-limit details."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please slow down and try again."},
    )
