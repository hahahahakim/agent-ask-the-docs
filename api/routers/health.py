from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness check — no authentication required (used by Docker / uptime monitors)."""
    return {"status": "ok"}
