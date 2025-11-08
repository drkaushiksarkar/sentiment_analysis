"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Natural language snippet to score.")


class SentimentResponse(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    score: float = Field(..., description="Signed score where positive is favorable sentiment.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    tokens_analyzed: int = Field(..., ge=0)
