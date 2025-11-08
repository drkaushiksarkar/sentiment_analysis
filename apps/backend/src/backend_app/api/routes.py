"""API router definitions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends

from backend_app.core.config import get_settings
from backend_app.schemas import SentimentRequest, SentimentResponse
from backend_app.services.inference import SentimentService

router = APIRouter()
inference_router = APIRouter(prefix="/api/v1", tags=["inference"])


@lru_cache(maxsize=1)
def get_sentiment_service() -> SentimentService:
    settings = get_settings()
    return SentimentService(weights_path=Path(settings.imdb_weights_path), max_length=settings.imdb_max_length)


@router.get("/health/live", tags=["health"])
async def live() -> dict:
    """Liveness probe."""

    settings = get_settings()
    return {"status": "ok", "service": settings.app_name, "environment": settings.environment}


@router.get("/health/ready", tags=["health"])
async def ready() -> dict:
    """Readiness probe placeholder."""

    return {"status": "ready"}


@inference_router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(
    payload: SentimentRequest, service: SentimentService = Depends(get_sentiment_service)
) -> SentimentResponse:
    """Run a lightweight sentiment analysis heuristic."""

    return service.predict(payload.text)
