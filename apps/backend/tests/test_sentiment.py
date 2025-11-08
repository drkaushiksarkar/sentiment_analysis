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


def test_sentiment_handles_negative_text() -> None:
    response = client.post("/api/v1/sentiment", json={"text": "This is terrible and awful"})
    assert response.status_code == 200
    data = response.json()
    assert data["tokens_analyzed"] > 0
