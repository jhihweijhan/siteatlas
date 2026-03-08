import time

import pytest

from app.core.agentic_settings import AgenticConfig
from app.models.agentic_schemas import AgenticState, QueryPlan
from app.services.agentic.nodes_refiner import execute


def make_state(query: str = "test query", **overrides) -> AgenticState:
    defaults = dict(
        user_message=query,
        conversation_history=[],
        ui_hard_constraints={},
        existing_system_tags=[],
        config=AgenticConfig(),
        chat_model="qwen2.5:14b",
        embedding_model="bge-m3",
        start_time=time.time(),
        query_plan=QueryPlan(
            semantic_query=query,
            tag_filters=["python", "docker"],
            relation_filters=["github.com"],
        ),
        previous_queries=[],
        refine_round=0,
    )
    defaults.update(overrides)
    return AgenticState(**defaults)


@pytest.mark.asyncio
async def test_soft_tags_removed_hard_tags_kept():
    state = make_state(
        ui_hard_constraints={"tags": ["python"]},
        query_plan=QueryPlan(
            semantic_query="test",
            tag_filters=["python", "docker"],
            relation_filters=[],
        ),
    )
    new_state, next_node = await execute(state)
    assert new_state.query_plan.tag_filters == ["python"]
    assert next_node == "RETRIEVE"


@pytest.mark.asyncio
async def test_soft_relations_removed_hard_relations_kept():
    state = make_state(
        ui_hard_constraints={"relation_groups": ["github.com"]},
        query_plan=QueryPlan(
            semantic_query="test",
            tag_filters=[],
            relation_filters=["github.com", "stackoverflow.com"],
        ),
    )
    new_state, next_node = await execute(state)
    assert new_state.query_plan.relation_filters == ["github.com"]


@pytest.mark.asyncio
async def test_refine_round_incremented():
    state = make_state(refine_round=0)
    new_state, _ = await execute(state)
    assert new_state.refine_round == 1


@pytest.mark.asyncio
async def test_duplicate_query_exhausted():
    state = make_state(
        query_plan=QueryPlan(semantic_query="same query"),
        previous_queries=["same query"],
    )
    new_state, next_node = await execute(state)
    assert new_state.evidence_verdict == "exhausted"
    assert next_node == "NO_ANSWER"


@pytest.mark.asyncio
async def test_no_soft_filters_still_proceeds():
    state = make_state(
        ui_hard_constraints={},
        query_plan=QueryPlan(
            semantic_query="test",
            tag_filters=[],
            relation_filters=[],
        ),
    )
    new_state, next_node = await execute(state)
    assert next_node == "RETRIEVE"
    assert new_state.refine_round == 1


@pytest.mark.asyncio
async def test_current_query_appended_to_previous():
    state = make_state(
        query_plan=QueryPlan(semantic_query="new question"),
        previous_queries=["old question"],
    )
    new_state, _ = await execute(state)
    assert "new question" in new_state.previous_queries
    assert "old question" in new_state.previous_queries
