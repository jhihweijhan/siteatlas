from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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
        mock_vs.find_document_by_source_url.return_value = None
        mock_vs_cls.return_value = mock_vs

        from app.main import app

        with TestClient(app) as tc:
            yield tc


def test_upload_txt_file_returns_queued_task_item(client):
    response = client.post(
        "/api/ingest/files",
        files=[("files", ("notes.txt", b"hello world", "text/plain"))],
    )

    assert response.status_code == 202
    data = response.json()
    assert data["total"] == 1
    assert data["items"][0]["file_name"] == "notes.txt"
    assert data["items"][0]["status"] == "queued"
    assert data["items"][0]["overwritten"] is False
    assert data["items"][0]["task_id"]


def test_upload_rejects_unsupported_extension(client):
    response = client.post(
        "/api/ingest/files",
        files=[("files", ("malware.exe", b"boom", "application/octet-stream"))],
    )

    assert response.status_code == 400
    assert "不支援" in response.json()["detail"]


def test_upload_accepts_multiple_files(client):
    response = client.post(
        "/api/ingest/files",
        files=[
            ("files", ("notes.txt", b"hello world", "text/plain")),
            ("files", ("summary.md", b"# title\nbody", "text/markdown")),
        ],
    )

    assert response.status_code == 202
    data = response.json()
    assert data["total"] == 2
    assert [item["file_name"] for item in data["items"]] == ["notes.txt", "summary.md"]


def test_upload_duplicate_filename_reports_overwrite(client):
    response = client.post(
        "/api/ingest/files",
        files=[("files", ("notes.txt", b"updated content", "text/plain"))],
        data={"overwrite_mode": "replace"},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["items"][0]["file_name"] == "notes.txt"
    assert "overwritten" in data["items"][0]


def test_upload_empty_content_is_handled_gracefully(client):
    response = client.post(
        "/api/ingest/files",
        files=[("files", ("empty.txt", b"", "text/plain"))],
    )

    assert response.status_code == 400
    assert "空" in response.json()["detail"]
