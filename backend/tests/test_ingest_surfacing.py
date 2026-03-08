"""Tests for the surfacing stage in the ingest pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import app.main  # noqa: F401

from app.routers.ingest import _run_surfacing_stage


@pytest.fixture
def mock_vs():
    vs = MagicMock()
    vs.search_by_embedding.return_value = [
        {
            "doc_id": "abc123",
            "title": "Related Article",
            "url": "https://example.com/related",
            "relation_group": "Python 開發",
            "tags": ["python", "fastapi"],
            "distance": 0.25,
        },
        {
            "doc_id": "def456",
            "title": "Another Article",
            "url": "https://example.com/another",
            "relation_group": "Web 框架",
            "tags": ["flask"],
            "distance": 0.40,
        },
    ]
    return vs


@pytest.mark.asyncio
async def test_surfacing_filters_by_threshold(mock_vs):
    """Only results with distance < 0.35 should be included."""
    results = await _run_surfacing_stage(
        vs=mock_vs,
        embeddings=[[0.1] * 10, [0.2] * 10],
        doc_id="self123",
    )

    mock_vs.search_by_embedding.assert_called_once_with(
        query_embedding=[0.1] * 10,
        exclude_doc_id="self123",
        top_k=5,
    )
    assert len(results) == 1
    assert results[0]["title"] == "Related Article"
    assert results[0]["distance"] == 0.25


@pytest.mark.asyncio
async def test_surfacing_max_3_results(mock_vs):
    """At most 3 results should be returned."""
    mock_vs.search_by_embedding.return_value = [
        {"doc_id": f"d{i}", "title": f"Doc {i}", "url": f"https://ex.com/{i}",
         "relation_group": "g", "tags": [], "distance": 0.1 + i * 0.05}
        for i in range(5)
    ]
    results = await _run_surfacing_stage(
        vs=mock_vs,
        embeddings=[[0.1] * 10],
        doc_id="self",
    )
    assert len(results) == 3


@pytest.mark.asyncio
async def test_surfacing_empty_embeddings(mock_vs):
    """Empty embeddings should return empty results without calling search."""
    results = await _run_surfacing_stage(
        vs=mock_vs,
        embeddings=[],
        doc_id="self123",
    )
    assert results == []
    mock_vs.search_by_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_surfacing_chromadb_error(mock_vs):
    """ChromaDB error should return empty results (graceful degradation)."""
    mock_vs.search_by_embedding.side_effect = Exception("ChromaDB down")
    results = await _run_surfacing_stage(
        vs=mock_vs,
        embeddings=[[0.1] * 10],
        doc_id="self123",
    )
    assert results == []
