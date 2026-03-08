import pytest
from unittest.mock import MagicMock, patch

from app.services.vector_store import VectorStoreService


@pytest.fixture
def mock_collection():
    collection = MagicMock()
    collection.count.return_value = 0
    collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"title": "Test"}, {"title": "Test"}]],
        "distances": [[0.1, 0.2]],
    }
    return collection


@pytest.fixture
def store(mock_collection):
    with patch("chromadb.HttpClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.heartbeat.return_value = True
        mock_client_cls.return_value = mock_client
        svc = VectorStoreService(host="localhost", port=8100)
        yield svc


def test_add_documents(store, mock_collection):
    store.add_documents(
        doc_id="test-doc",
        chunks=["chunk1", "chunk2"],
        embeddings=[[0.1] * 1024, [0.2] * 1024],
        metadatas=[{"title": "Test"}, {"title": "Test"}],
    )
    mock_collection.upsert.assert_called_once()


def test_search(store):
    results = store.search(query_embedding=[0.1] * 1024, top_k=5)
    assert len(results) == 2
    assert results[0]["document"] == "doc1"


def test_health_check(store):
    assert store.check_health() is True


def test_update_document_relation(store, mock_collection):
    mock_collection.get.return_value = {
        "ids": ["doc_chunk_1", "doc_chunk_2"],
        "metadatas": [{"doc_id": "doc1"}, {"doc_id": "doc1", "domain": "example.com"}],
    }
    ok = store.update_document_relation("doc1", "project-a")
    assert ok is True
    mock_collection.update.assert_called_once()


def test_get_chunks_returns_rows(store, mock_collection):
    mock_collection.get.return_value = {
        "ids": ["doc1_chunk_0", "doc1_img_0"],
        "documents": ["text", "[圖片描述] lobby"],
        "metadatas": [
            {"doc_id": "doc1", "type": "text", "source_url": "https://a"},
            {"doc_id": "doc1", "type": "image_caption", "source_url": "https://a", "image_index": 0},
        ],
    }
    rows = store.get_chunks(where={"doc_id": "doc1"})
    assert len(rows) == 2
    assert rows[1]["metadata"]["type"] == "image_caption"


def test_get_document_detail_contains_dynamic_schema_and_counts(store, mock_collection):
    mock_collection.get.return_value = {
        "ids": ["doc1_chunk_0", "doc1_img_0"],
        "documents": ["Long article paragraph", "[圖片描述] hotel street view"],
        "metadatas": [
            {
                "doc_id": "doc1",
                "title": "Demo",
                "source_url": "https://a",
                "created_at": "2026-03-05T00:00:00+00:00",
                "relation_group": "edition.cnn.com",
                "type": "text",
                "chunk_index": 0,
            },
            {
                "doc_id": "doc1",
                "title": "Demo",
                "source_url": "https://a",
                "created_at": "2026-03-05T00:00:00+00:00",
                "relation_group": "edition.cnn.com",
                "type": "image_caption",
                "image_index": 0,
                "image_url": "https://img.example/1.jpg",
                "image_alt": "hotel",
            },
        ],
    }
    detail = store.get_document_detail("doc1")
    assert detail is not None
    assert detail["item"]["text_chunks_count"] == 1
    assert detail["item"]["image_chunks_count"] == 1
    assert detail["type_counts"]["text"] == 1
    assert detail["type_counts"]["image_caption"] == 1
    assert "source_url" in detail["metadata_schema"]
    assert detail["image_items"][0]["image_url"] == "https://img.example/1.jpg"


def test_collect_existing_tags_empty(store, mock_collection):
    mock_collection.get.return_value = {"metadatas": []}
    result = store.collect_existing_tags()
    assert result == set()


def test_collect_existing_tags_with_data(store, mock_collection):
    mock_collection.get.return_value = {
        "metadatas": [
            {"tag_1": "Python 開發", "tag_2": "API 設計", "tag_3": ""},
            {"tag_1": "Python 開發", "tag_2": "Docker/容器", "tag_3": ""},
            {"tag_1": "前端框架", "tag_2": "", "tag_3": ""},
        ]
    }
    result = store.collect_existing_tags()
    assert result == {"Python 開發", "API 設計", "Docker/容器", "前端框架"}


def test_update_document_tags(store, mock_collection):
    mock_collection.get.return_value = {
        "ids": ["abc_chunk_0", "abc_chunk_1"],
        "metadatas": [
            {"doc_id": "abc", "tag_1": "", "tag_2": "", "tag_3": ""},
            {"doc_id": "abc", "tag_1": "", "tag_2": "", "tag_3": ""},
        ],
    }
    result = store.update_document_tags("abc", ["Python 開發", "API 設計"])
    assert result is True
    call_args = mock_collection.update.call_args
    metadatas = call_args[1]["metadatas"]
    assert metadatas[0]["tag_1"] == "Python 開發"
    assert metadatas[0]["tag_2"] == "API 設計"
    assert metadatas[0]["tag_3"] == ""


def test_update_document_tags_not_found(store, mock_collection):
    mock_collection.get.return_value = {"ids": [], "metadatas": []}
    result = store.update_document_tags("nonexistent", ["tag"])
    assert result is False
    mock_collection.update.assert_not_called()


def test_list_documents_includes_tags(store, mock_collection):
    mock_collection.get.return_value = {
        "metadatas": [
            {
                "doc_id": "doc1",
                "title": "Test Page",
                "source_url": "https://example.com",
                "created_at": "2026-03-05",
                "type": "text",
                "tag_1": "Python 開發",
                "tag_2": "API 設計",
                "tag_3": "",
            },
            {
                "doc_id": "doc1",
                "title": "Test Page",
                "source_url": "https://example.com",
                "created_at": "2026-03-05",
                "type": "text",
                "tag_1": "Python 開發",
                "tag_2": "API 設計",
                "tag_3": "Docker",
            },
        ],
    }
    docs = store.list_documents()
    assert len(docs) == 1
    assert "tags" in docs[0]
    assert sorted(docs[0]["tags"]) == ["API 設計", "Docker", "Python 開發"]


def test_search_by_embedding_excludes_doc_id(store, mock_collection):
    """search_by_embedding should pass where filter to exclude given doc_id."""
    mock_collection.query.return_value = {
        "ids": [["other_chunk_0"]],
        "documents": [["Other doc text"]],
        "metadatas": [[{
            "doc_id": "other123",
            "source_url": "https://other.com",
            "title": "Other",
            "relation_group": "general",
            "tag_1": "python",
            "tag_2": "",
            "tag_3": "",
        }]],
        "distances": [[0.2]],
    }

    results = store.search_by_embedding(
        query_embedding=[0.1] * 10,
        exclude_doc_id="self123",
        top_k=5,
    )

    call_kwargs = mock_collection.query.call_args
    assert call_kwargs.kwargs.get("where") == {"doc_id": {"$ne": "self123"}}
    assert len(results) == 1
    assert results[0]["doc_id"] == "other123"
    assert results[0]["title"] == "Other"
    assert results[0]["url"] == "https://other.com"
    assert results[0]["distance"] == 0.2
    assert results[0]["relation_group"] == "general"
    assert results[0]["tags"] == ["python"]


def test_search_by_embedding_empty_results(store, mock_collection):
    """search_by_embedding returns empty list when no results."""
    mock_collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    results = store.search_by_embedding(
        query_embedding=[0.1] * 10,
        exclude_doc_id="self123",
    )

    assert results == []


def test_list_documents_no_tags(store, mock_collection):
    mock_collection.get.return_value = {
        "metadatas": [
            {
                "doc_id": "doc1",
                "title": "Test Page",
                "source_url": "https://example.com",
                "created_at": "2026-03-05",
                "type": "text",
            },
        ],
    }
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0]["tags"] == []
