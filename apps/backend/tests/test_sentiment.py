from fastapi.testclient import TestClient

from backend_app.main import app


client = TestClient(app)


def test_sentiment_endpoint_returns_scores() -> None:
    response = client.post("/api/v1/sentiment", json={"text": "I love the amazing product"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in {"positive", "negative", "neutral"}
    assert 0 <= data["confidence"] <= 1
    assert isinstance(data["tokens_analyzed"], int)


def test_sentiment_batch_endpoint() -> None:
    response = client.post(
        "/api/v1/sentiment/batch",
        json={"texts": ["Great launch", "This is terrible"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["predictions"]) == 2


def test_sentiment_metrics_endpoint_after_calls() -> None:
    client.post("/api/v1/sentiment", json={"text": "Great job"})
    metrics_response = client.get("/api/v1/metrics/sentiment")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert metrics["total_requests"] >= 1
    assert "positive" in metrics["label_counts"]
