from fastapi.testclient import TestClient

from backend_app.main import app


def test_live_health() -> None:
    client = TestClient(app)
    response = client.get("/api/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_health() -> None:
    client = TestClient(app)
    response = client.get("/api/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
