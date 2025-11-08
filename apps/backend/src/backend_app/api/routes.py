"""API router definitions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends

from backend_app.core.config import get_settings
from backend_app.schemas import (
    SentimentBatchRequest,
    SentimentBatchResponse,
    SentimentMetrics,
    SentimentRequest,
    SentimentResponse,
)
from backend_app.services.analytics import StatsTracker
from backend_app.services.inference import SentimentService

router = APIRouter()
inference_router = APIRouter(prefix="/api/v1", tags=["inference"])


@lru_cache(maxsize=1)
def get_sentiment_service() -> SentimentService:
    settings = get_settings()
    word_index_path = Path(settings.imdb_word_index_path) if settings.imdb_word_index_path else None
    return SentimentService(
        weights_path=Path(settings.imdb_weights_path),
        max_length=settings.imdb_max_length,
        word_index_path=word_index_path,
    )


@lru_cache(maxsize=1)
def get_stats_tracker() -> StatsTracker:
    return StatsTracker()


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
    payload: SentimentRequest,
    service: SentimentService = Depends(get_sentiment_service),
    tracker: StatsTracker = Depends(get_stats_tracker),
) -> SentimentResponse:
    """Run a lightweight sentiment analysis heuristic."""

    result = service.predict(payload.text)
    tracker.record(result)
    return result


@inference_router.post("/sentiment/batch", response_model=SentimentBatchResponse)
async def analyze_batch(
    payload: SentimentBatchRequest,
    service: SentimentService = Depends(get_sentiment_service),
    tracker: StatsTracker = Depends(get_stats_tracker),
) -> SentimentBatchResponse:
    predictions = [service.predict(text) for text in payload.texts]
    for prediction in predictions:
        tracker.record(prediction)
    return SentimentBatchResponse(predictions=predictions)


@inference_router.get("/metrics/sentiment", response_model=SentimentMetrics)
async def sentiment_metrics(tracker: StatsTracker = Depends(get_stats_tracker)) -> SentimentMetrics:
    return tracker.snapshot()
