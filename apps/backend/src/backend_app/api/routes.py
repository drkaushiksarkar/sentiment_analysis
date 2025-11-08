"""API router definitions."""

from __future__ import annotations

from fastapi import APIRouter

from backend_app.core.config import get_settings

router = APIRouter()


@router.get("/health/live", tags=["health"])
async def live() -> dict:
    """Liveness probe."""

    settings = get_settings()
    return {"status": "ok", "service": settings.app_name, "environment": settings.environment}


@router.get("/health/ready", tags=["health"])
async def ready() -> dict:
    """Readiness probe placeholder."""

    return {"status": "ready"}
