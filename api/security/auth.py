"""
API key authentication dependency.

Keys are read from the API_KEYS environment variable on every call so that
rotating a key requires only an env-var update + process restart, not a code
deploy. auto_error=False on the header scheme means a missing header returns a
clean 401 instead of FastAPI's default 403 with schema detail.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from api.config import settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_API_KEY_HEADER)) -> str:
    """FastAPI dependency — raises 401 if the X-API-Key header is missing or invalid."""
    if not api_key or api_key not in settings.get_api_keys():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key
