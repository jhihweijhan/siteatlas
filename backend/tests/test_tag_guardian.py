from unittest.mock import AsyncMock

import pytest

from app.services.tag_guardian import TagGuardian


@pytest.fixture
def mock_embedding_svc():
    return AsyncMock()


@pytest.fixture
def guardian(mock_embedding_svc):
    return TagGuardian(embedding_svc=mock_embedding_svc, threshold=0.3)


@pytest.mark.asyncio
async def test_existing_tag_passes_through(guardian, mock_embedding_svc):
    """既有標籤直接通過，不呼叫 embedding"""
    existing = {"Python", "JavaScript"}
    result = await guardian.resolve_tags(["python", "javascript"], existing)

    assert set(result) == {"Python", "JavaScript"}
    mock_embedding_svc.embed_texts.assert_not_called()
    mock_embedding_svc.embed_query.assert_not_called()


@pytest.mark.asyncio
async def test_new_tag_merged_to_similar(guardian, mock_embedding_svc):
    """新標籤與既有標籤相似 -> 歸併"""
    existing = {"機器學習"}
    # Pre-populate existing embeddings cache: "機器學習" -> [1, 0]
    guardian._existing_embeddings = {"機器學習": [1.0, 0.0]}
    # New tag "ML" gets embedded as [0.95, 0.05] — very close to [1, 0]
    mock_embedding_svc.embed_query.return_value = [0.95, 0.05]

    result = await guardian.resolve_tags(["ML"], existing)

    assert result == ["機器學習"]


@pytest.mark.asyncio
async def test_new_tag_no_similar_creates_new(guardian, mock_embedding_svc):
    """新標籤與既有標籤都不相似 -> 建立新標籤"""
    existing = {"機器學習"}
    # Pre-populate: "機器學習" -> [1, 0]
    guardian._existing_embeddings = {"機器學習": [1.0, 0.0]}
    # New tag "料理" gets embedded as [0, 1] — orthogonal, distance = 1.0
    mock_embedding_svc.embed_query.return_value = [0.0, 1.0]

    result = await guardian.resolve_tags(["料理"], existing)

    assert result == ["料理"]


@pytest.mark.asyncio
async def test_empty_existing_tags_skips_guardian(guardian, mock_embedding_svc):
    """既有標籤為空 -> 全部當新標籤，不呼叫 embedding"""
    result = await guardian.resolve_tags(["Python", "AI"], set())

    assert result == ["Python", "AI"]
    mock_embedding_svc.embed_texts.assert_not_called()
    mock_embedding_svc.embed_query.assert_not_called()


@pytest.mark.asyncio
async def test_embedding_failure_keeps_original(guardian, mock_embedding_svc):
    """embedding 失敗 -> 直接採用 LLM 建議的標籤"""
    existing = {"Python"}
    mock_embedding_svc.embed_texts.side_effect = Exception("Ollama down")

    result = await guardian.resolve_tags(["NewTag"], existing)

    assert result == ["NewTag"]
