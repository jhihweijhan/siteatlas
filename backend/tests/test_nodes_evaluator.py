import time
from unittest.mock import AsyncMock, patch

import pytest

from app.core.agentic_settings import AgenticConfig
from app.models.agentic_schemas import AgenticState, QueryPlan
from app.services.agentic.nodes_evaluator import execute


def make_chunk(distance: float, chunk_type: str = "text", doc: str = "doc1") -> dict:
    return {
        "id": f"{doc}_chunk_0",
        "distance": distance,
        "metadata": {"type": chunk_type, "source_url": f"https://{doc}.com"},
        "document": "some content",
    }


def make_state(chunks: list[dict], query: str = "test query", **overrides) -> AgenticState:
    defaults = dict(
        user_message=query,
        conversation_history=[],
        ui_hard_constraints={},
        existing_system_tags=[],
        config=AgenticConfig(),
        chat_model="qwen2.5:14b",
        embedding_model="bge-m3",
        start_time=time.time(),
        query_plan=QueryPlan(semantic_query=query),
        retrieved_chunks=chunks,
    )
    defaults.update(overrides)
    return AgenticState(**defaults)


@pytest.mark.asyncio
async def test_empty_chunks_insufficient():
    state = make_state([], query="anything")
    new_state, next_node = await execute(state)
    assert new_state.evidence_verdict == "insufficient"
    assert new_state.evidence_detail.reject_reason == "empty_result"
    assert next_node == "REFINE"


@pytest.mark.asyncio
async def test_red_zone_all_above_reject():
    chunks = [make_chunk(0.70), make_chunk(0.80)]
    state = make_state(chunks)
    new_state, next_node = await execute(state)
    assert new_state.evidence_verdict == "insufficient"
    assert new_state.evidence_detail.heuristic_zone == "red"
    assert next_node == "REFINE"


@pytest.mark.asyncio
async def test_green_zone_best_below_pass():
    chunks = [make_chunk(0.35), make_chunk(0.60)]
    state = make_state(chunks)
    new_state, next_node = await execute(state)
    assert new_state.evidence_verdict == "sufficient"
    assert new_state.evidence_detail.heuristic_zone == "green"
    assert next_node == "GENERATE"


@pytest.mark.asyncio
async def test_yellow_zone_in_between():
    chunks = [make_chunk(0.50)]
    state = make_state(chunks)
    new_state, next_node = await execute(state)
    assert new_state.evidence_verdict == "sufficient"
    assert new_state.evidence_detail.heuristic_zone == "yellow"
    assert next_node == "GENERATE"


@pytest.mark.asyncio
async def test_caption_dominance_with_code_query():
    chunks = [
        make_chunk(0.30, chunk_type="image_caption"),
        make_chunk(0.32, chunk_type="image_caption"),
        make_chunk(0.33, chunk_type="image_caption"),
        make_chunk(0.34, chunk_type="image_caption"),
        make_chunk(0.35, chunk_type="text"),
    ]
    state = make_state(chunks, query="這段程式碼怎麼用")
    new_state, next_node = await execute(state)
    assert new_state.evidence_verdict == "insufficient"
    assert new_state.evidence_detail.reject_reason == "caption_dominance"


@pytest.mark.asyncio
async def test_caption_dominance_not_triggered_for_image_query():
    chunks = [
        make_chunk(0.30, chunk_type="image_caption"),
        make_chunk(0.32, chunk_type="image_caption"),
        make_chunk(0.33, chunk_type="image_caption"),
        make_chunk(0.34, chunk_type="image_caption"),
        make_chunk(0.35, chunk_type="text"),
    ]
    state = make_state(chunks, query="這張圖片是什麼")
    new_state, next_node = await execute(state)
    # Should NOT trigger caption_dominance — query is about images
    assert new_state.evidence_detail.reject_reason != "caption_dominance"


@pytest.mark.asyncio
async def test_insufficient_refine_round_zero_returns_refine():
    state = make_state([], query="anything", refine_round=0)
    _, next_node = await execute(state)
    assert next_node == "REFINE"


@pytest.mark.asyncio
async def test_insufficient_refine_round_max_returns_no_answer():
    state = make_state([], query="anything", refine_round=1)
    _, next_node = await execute(state)
    assert next_node == "NO_ANSWER"


@pytest.mark.asyncio
async def test_yellow_zone_calls_llm_and_returns_verdict():
    chunks = [make_chunk(0.50)]
    state = make_state(chunks)
    with patch(
        "app.services.agentic.nodes_evaluator._call_evaluator_llm",
        new_callable=AsyncMock,
        return_value="insufficient",
    ):
        new_state, next_node = await execute(state)
    assert new_state.evidence_verdict == "insufficient"
    assert new_state.evidence_detail.heuristic_zone == "yellow"
    assert next_node == "REFINE"


@pytest.mark.asyncio
async def test_yellow_zone_llm_failure_defaults_sufficient():
    chunks = [make_chunk(0.50)]
    state = make_state(chunks)
    with patch(
        "app.services.agentic.nodes_evaluator._call_evaluator_llm",
        new_callable=AsyncMock,
        side_effect=RuntimeError("LLM unavailable"),
    ):
        new_state, next_node = await execute(state)
    assert new_state.evidence_verdict == "sufficient"
    assert new_state.evidence_detail.heuristic_zone == "yellow"
    assert next_node == "GENERATE"


@pytest.mark.asyncio
async def test_scope_pass_with_chunks():
    """有 scope constraints + 有 chunks → 自動通過（不走 heuristic）"""
    chunks = [make_chunk(0.55)]  # yellow zone distance
    state = make_state(
        chunks, query="這頁在說什麼",
        ui_hard_constraints={"relation_groups": ["example.com"], "source_urls": [], "tags": []},
    )
    new_state, next_node = await execute(state)
    assert new_state.evidence_verdict == "sufficient"
    assert new_state.evidence_detail.heuristic_zone == "scope_pass"
    assert next_node == "GENERATE"


@pytest.mark.asyncio
async def test_scope_pass_empty_chunks_still_insufficient():
    """有 scope 但無 chunks → 仍然 insufficient"""
    state = make_state(
        [], query="這頁在說什麼",
        ui_hard_constraints={"relation_groups": ["example.com"], "source_urls": [], "tags": []},
    )
    new_state, next_node = await execute(state)
    assert new_state.evidence_verdict == "insufficient"
    assert new_state.evidence_detail.reject_reason == "empty_result"


@pytest.mark.asyncio
async def test_no_scope_yellow_zone_still_evaluates():
    """無 scope → 正常走 heuristic yellow zone"""
    chunks = [make_chunk(0.55)]
    state = make_state(chunks, query="什麼是 RAG")
    new_state, _ = await execute(state)
    assert new_state.evidence_detail.heuristic_zone == "yellow"
