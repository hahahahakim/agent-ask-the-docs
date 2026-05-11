"""
API key authentication dependency.

Keys are loaded once at import time from API_KEYS. Rotating a key requires
a process restart. auto_error=False on the header scheme means a missing
header returns a clean 401 instead of FastAPI's default 403 with schema detail.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from api.config import settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Parse once at startup — avoids splitting the env string on every request.
_VALID_API_KEYS: frozenset = frozenset(settings.get_api_keys())


async def verify_api_key(api_key: str = Security(_API_KEY_HEADER)) -> str:
    """FastAPI dependency — raises 401 if the X-API-Key header is missing or invalid."""
    if not api_key or api_key not in _VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key
