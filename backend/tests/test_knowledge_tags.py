from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.vector_store import VectorStoreService


@pytest.fixture
def mock_vs():
    with patch("chromadb.HttpClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client
        svc = VectorStoreService(host="localhost", port=8100)
        yield svc, mock_collection


@pytest.fixture
def client(mock_vs):
    svc, _ = mock_vs
    app.dependency_overrides = {}
    from app.main import get_vector_store
    app.dependency_overrides[get_vector_store] = lambda: svc
    yield TestClient(app)
    app.dependency_overrides = {}


def test_put_tags_success(client, mock_vs):
    _, mock_collection = mock_vs
    mock_collection.get.return_value = {
        "ids": ["doc_chunk_0", "doc_chunk_1"],
        "metadatas": [
            {"doc_id": "abc123", "tag_1": "", "tag_2": "", "tag_3": ""},
            {"doc_id": "abc123", "tag_1": "", "tag_2": "", "tag_3": ""},
        ],
    }
    resp = client.put("/api/knowledge/abc123/tags", json={"tags": ["Python 開發", "API 設計"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["doc_id"] == "abc123"
    assert data["tags"] == ["Python 開發", "API 設計"]


def test_put_tags_not_found(client, mock_vs):
    _, mock_collection = mock_vs
    mock_collection.get.return_value = {"ids": [], "metadatas": []}
    resp = client.put("/api/knowledge/nonexistent/tags", json={"tags": ["tag"]})
    assert resp.status_code == 404


def test_put_tags_empty_list(client, mock_vs):
    _, mock_collection = mock_vs
    mock_collection.get.return_value = {
        "ids": ["doc_chunk_0"],
        "metadatas": [{"doc_id": "abc123", "tag_1": "old", "tag_2": "", "tag_3": ""}],
    }
    resp = client.put("/api/knowledge/abc123/tags", json={"tags": []})
    assert resp.status_code == 200
