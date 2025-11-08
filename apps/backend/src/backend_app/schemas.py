"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Natural language snippet to score.")


class SentimentResponse(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    score: float = Field(..., description="Signed score where positive is favorable sentiment.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    tokens_analyzed: int = Field(..., ge=0)


class SentimentBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Collection of texts to score.")


class SentimentBatchResponse(BaseModel):
    predictions: list[SentimentResponse]


class PredictionSummary(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float
    timestamp: datetime


class TimelinePoint(BaseModel):
    timestamp: datetime
    confidence: float


class SentimentMetrics(BaseModel):
    total_requests: int
    label_counts: dict[Literal["positive", "negative", "neutral"], int]
    average_confidence: float
    recent_predictions: list[PredictionSummary]
    timeline: list[TimelinePoint]
