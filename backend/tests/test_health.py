import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("app.main.OllamaClient") as mock_ollama_cls, patch("app.main.VectorStoreService") as mock_vs_cls:
        mock_ollama = AsyncMock()
        mock_ollama.check_health.return_value = {"alive": True, "version": "0.5.0"}
        mock_ollama.list_models.return_value = [{"name": "qwen2.5:14b"}]
        mock_ollama.close = AsyncMock()
        mock_ollama_cls.return_value = mock_ollama

        mock_vs = MagicMock()
        mock_vs.check_health.return_value = True
        mock_vs_cls.return_value = mock_vs

        from app.main import app

        with TestClient(app) as tc:
            yield tc


def test_health_readiness(client):
    res = client.get("/health/readiness")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["ollama"] is True
    assert data["chromadb"] is True


def test_list_models(client):
    res = client.get("/api/models/local")
    assert res.status_code == 200
    data = res.json()
    assert "chat" in data
