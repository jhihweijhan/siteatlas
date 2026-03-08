import pytest
from pydantic import ValidationError

from app.models.schemas import (
    ChatMessage,
    ChatRequest,
    IngestRequest,
    IngestTaskStatus,
    KnowledgeTagsUpdateRequest,
    RagChatRequest,
)


def test_chat_request_valid():
    req = ChatRequest(messages=[ChatMessage(role="user", content="Hello")])
    assert len(req.messages) == 1


def test_chat_request_empty_messages():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])


def test_chat_message_invalid_role():
    with pytest.raises(ValidationError):
        ChatMessage(role="invalid", content="Hello")


def test_ingest_request_valid():
    req = IngestRequest(url="http://example.com", title="Test", content="Some content")
    assert req.language == "zh-TW"
    assert req.embedding_model is None
    assert req.relation_group is None


def test_rag_chat_request_defaults():
    req = RagChatRequest(query="What is this?")
    assert req.top_k == 5
    assert req.model is None
    assert req.embedding_model is None
    assert req.relation_groups is None
    assert req.source_urls is None


def test_ingest_request_with_html_only():
    req = IngestRequest(url="http://example.com", title="Test", html="<html><body>Hi</body></html>")
    assert req.html is not None
    assert req.content is None


def test_ingest_request_with_content_only():
    req = IngestRequest(url="http://example.com", title="Test", content="Some text")
    assert req.content == "Some text"
    assert req.html is None


def test_ingest_request_no_content_no_html_fails():
    with pytest.raises(ValidationError):
        IngestRequest(url="http://example.com", title="Test")


def test_ingest_task_status_has_image_failure_fields():
    task = IngestTaskStatus(
        task_id="abc123",
        status="running",
        stage="captioning",
        message="描述中",
        url="https://example.com",
        title="title",
        created_at="2026-03-05T00:00:00+00:00",
        updated_at="2026-03-05T00:00:00+00:00",
    )
    assert task.image_total == 0
    assert task.image_caption_success == 0
    assert task.image_caption_failed == 0
    assert task.image_caption_failures == []


def test_rag_chat_request_with_tags():
    req = RagChatRequest(query="test", tags=["Python 開發", "API 設計"])
    assert req.tags == ["Python 開發", "API 設計"]


def test_rag_chat_request_without_tags():
    req = RagChatRequest(query="test")
    assert req.tags is None


def test_knowledge_tags_update_request_valid():
    req = KnowledgeTagsUpdateRequest(tags=["Python 開發", "API 設計"])
    assert len(req.tags) == 2


def test_knowledge_tags_update_request_max_3():
    with pytest.raises(ValidationError):
        KnowledgeTagsUpdateRequest(tags=["a", "b", "c", "d"])


def test_knowledge_tags_update_request_empty_list():
    req = KnowledgeTagsUpdateRequest(tags=[])
    assert req.tags == []


def test_ingest_task_status_tagging_fields_defaults():
    task = IngestTaskStatus(
        task_id="abc123",
        status="running",
        stage="tagging",
        message="標籤中",
        url="https://example.com",
        title="title",
        created_at="2026-03-05T00:00:00+00:00",
        updated_at="2026-03-05T00:00:00+00:00",
    )
    assert task.tagging_status is None
    assert task.tagging_tags == []
    assert task.tagging_error is None


def test_ingest_task_status_surfacing_results_default():
    status = IngestTaskStatus(
        task_id="t1",
        status="done",
        stage="done",
        message="ok",
        url="https://example.com",
        title="Test",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    assert status.surfacing_results == []


def test_ingest_task_status_surfacing_results_with_data():
    results = [
        {
            "title": "Related Doc",
            "url": "https://example.com/related",
            "relation_group": "general",
            "tags": ["python"],
            "distance": 0.25,
        }
    ]
    status = IngestTaskStatus(
        task_id="t1",
        status="done",
        stage="done",
        message="ok",
        url="https://example.com",
        title="Test",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        surfacing_results=results,
    )
    assert len(status.surfacing_results) == 1
    assert status.surfacing_results[0]["title"] == "Related Doc"
