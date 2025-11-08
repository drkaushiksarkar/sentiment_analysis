"""Track recent sentiment predictions for metrics endpoints."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Deque, Dict, Literal

from backend_app.schemas import PredictionSummary, SentimentMetrics, SentimentResponse, TimelinePoint

Labels = Literal["positive", "negative", "neutral"]


class StatsTracker:
    """Maintains rolling statistics for sentiment predictions."""

    def __init__(self, max_points: int = 50) -> None:
        self.total_requests = 0
        self.label_counts: Dict[Labels, int] = {"positive": 0, "negative": 0, "neutral": 0}
        self.confidence_sum = 0.0
        self.records: Deque[PredictionSummary] = deque(maxlen=max_points)

    def record(self, response: SentimentResponse) -> None:
        self.total_requests += 1
        self.label_counts[response.label] += 1
        self.confidence_sum += response.confidence
        self.records.append(
            PredictionSummary(
                label=response.label,
                confidence=response.confidence,
                timestamp=datetime.utcnow(),
            )
        )

    def snapshot(self) -> SentimentMetrics:
        average_confidence = (
            self.confidence_sum / self.total_requests if self.total_requests else 0.0
        )
        timeline = [
            TimelinePoint(timestamp=record.timestamp, confidence=record.confidence)
            for record in self.records
        ]
        return SentimentMetrics(
            total_requests=self.total_requests,
            label_counts=self.label_counts.copy(),
            average_confidence=round(average_confidence, 3),
            recent_predictions=list(self.records),
            timeline=timeline,
        )
