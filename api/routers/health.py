import logging

from fastapi import APIRouter, Request

from api.log import log_request

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health")
async def health(request: Request) -> dict:
    """Liveness check — no authentication required (used by Docker / uptime monitors)."""
    log_request(logger, request, level=logging.DEBUG)
    return {"status": "ok"}
