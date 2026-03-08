import pytest
from unittest.mock import AsyncMock, MagicMock

import app.main  # noqa: F401

from app.routers.ingest import _run_tagging_stage


@pytest.mark.asyncio
async def test_run_tagging_stage_success():
    ollama = AsyncMock()
    ollama.generate_tags.return_value = ["Python 開發", "API 設計"]
    embedding_svc = AsyncMock()
    vs = MagicMock()
    vs.collect_existing_tags.return_value = {"Python 開發", "Docker/容器"}

    result = await _run_tagging_stage(
        ollama=ollama, embedding_svc=embedding_svc, vs=vs,
        title="FastAPI 教學", content_preview="如何用 FastAPI 建立 REST API...",
        model="qwen2.5:14b",
    )
    assert result["status"] == "success"
    assert "Python 開發" in result["tags"]


@pytest.mark.asyncio
async def test_run_tagging_stage_llm_fails():
    ollama = AsyncMock()
    ollama.generate_tags.return_value = None
    embedding_svc = AsyncMock()
    vs = MagicMock()
    vs.collect_existing_tags.return_value = set()

    result = await _run_tagging_stage(
        ollama=ollama, embedding_svc=embedding_svc, vs=vs,
        title="Test", content_preview="content", model="qwen2.5:14b",
    )
    assert result["status"] == "degraded"
    assert result["tags"] == []


@pytest.mark.asyncio
async def test_run_tagging_stage_exception():
    ollama = AsyncMock()
    ollama.generate_tags.side_effect = Exception("connection error")
    embedding_svc = AsyncMock()
    vs = MagicMock()
    vs.collect_existing_tags.return_value = set()

    result = await _run_tagging_stage(
        ollama=ollama, embedding_svc=embedding_svc, vs=vs,
        title="Test", content_preview="content", model="qwen2.5:14b",
    )
    assert result["status"] == "degraded"
    assert "connection error" in result["error"]
