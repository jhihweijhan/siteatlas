import time
from unittest.mock import AsyncMock, patch

import pytest

from app.core.agentic_settings import AgenticConfig
from app.models.agentic_schemas import AgenticState, QueryPlan, RouteAction


def make_state(query: str = "test query", **overrides) -> AgenticState:
    defaults = dict(
        user_message=query,
        conversation_history=[],
        ui_hard_constraints={},
        existing_system_tags=["Python", "Docker", "React"],
        config=AgenticConfig(),
        chat_model="qwen2.5:14b",
        embedding_model="bge-m3",
        start_time=time.time(),
        route_action=RouteAction.SEARCH_KNOWLEDGE,
        query_plan=QueryPlan(semantic_query=query),
        previous_queries=[],
    )
    defaults.update(overrides)
    return AgenticState(**defaults)


class TestValidateAndMergeFilters:
    def test_hallucinated_tags_removed(self):
        from app.services.agentic.nodes_retriever import validate_and_merge_filters

        plan = QueryPlan(
            semantic_query="test",
            tag_filters=["Python", "FakeTag", "Docker"],
        )
        result = validate_and_merge_filters(
            plan, existing_tags=["Python", "Docker", "React"], ui_hard={}
        )
        assert result["tags"] == ["Python", "Docker"]

    def test_ui_relation_groups_override_agent(self):
        from app.services.agentic.nodes_retriever import validate_and_merge_filters

        plan = QueryPlan(
            semantic_query="test",
            relation_filters=["agent-group"],
        )
        result = validate_and_merge_filters(
            plan,
            existing_tags=[],
            ui_hard={"relation_groups": ["ui-group"]},
        )
        assert result["relation_groups"] == ["ui-group"]

    def test_agent_groups_used_when_no_ui(self):
        from app.services.agentic.nodes_retriever import validate_and_merge_filters

        plan = QueryPlan(
            semantic_query="test",
            relation_filters=["agent-group"],
        )
        result = validate_and_merge_filters(plan, existing_tags=[], ui_hard={})
        assert result["relation_groups"] == ["agent-group"]

    def test_source_urls_only_from_ui(self):
        from app.services.agentic.nodes_retriever import validate_and_merge_filters

        plan = QueryPlan(semantic_query="test")
        result = validate_and_merge_filters(
            plan,
            existing_tags=[],
            ui_hard={"source_urls": ["https://example.com"]},
        )
        assert result["source_urls"] == ["https://example.com"]

    def test_source_urls_take_precedence_over_relation_groups(self):
        from app.services.agentic.nodes_retriever import validate_and_merge_filters

        plan = QueryPlan(
            semantic_query="test",
            relation_filters=["agent-group"],
        )
        result = validate_and_merge_filters(
            plan,
            existing_tags=[],
            ui_hard={
                "relation_groups": ["stale-group"],
                "source_urls": ["https://example.com/page"],
            },
        )
        assert result["source_urls"] == ["https://example.com/page"]
        assert result["relation_groups"] == []

    def test_empty_existing_tags(self):
        from app.services.agentic.nodes_retriever import validate_and_merge_filters

        plan = QueryPlan(
            semantic_query="test",
            tag_filters=["Python"],
        )
        result = validate_and_merge_filters(plan, existing_tags=[], ui_hard={})
        assert result["tags"] == []

    def test_ui_hard_tags_override_agent_tags(self):
        from app.services.agentic.nodes_retriever import validate_and_merge_filters

        plan = QueryPlan(
            semantic_query="test",
            tag_filters=["Python", "Docker"],
        )
        result = validate_and_merge_filters(
            plan,
            existing_tags=["Python", "Docker", "React"],
            ui_hard={"tags": ["React"]},
        )
        assert result["tags"] == ["React"]


class TestExecute:
    @pytest.mark.asyncio
    async def test_returns_evaluate_with_chunks(self):
        from app.services.agentic import nodes_retriever

        state = make_state("Docker 教學")
        mock_chunks = [
            {"id": "doc1_chunk_0", "document": "Docker 介紹", "metadata": {"source_url": "https://example.com"}, "distance": 0.3},
        ]

        with patch.object(
            nodes_retriever, "_embed_and_search", new_callable=AsyncMock, return_value=mock_chunks
        ):
            new_state, next_node = await nodes_retriever.execute(state)

        assert next_node == "EVALUATE"
        assert new_state.retrieved_chunks == mock_chunks
        assert new_state.final_query_params is not None

    @pytest.mark.asyncio
    async def test_does_not_track_previous_queries(self):
        """previous_queries 由 refiner 管理，retriever 不追蹤"""
        from app.services.agentic import nodes_retriever

        state = make_state(
            "new query",
            previous_queries=["old query"],
            query_plan=QueryPlan(semantic_query="new query"),
        )

        with patch.object(
            nodes_retriever, "_embed_and_search", new_callable=AsyncMock, return_value=[]
        ):
            new_state, _ = await nodes_retriever.execute(state)

        assert new_state.previous_queries == ["old query"]

    @pytest.mark.asyncio
    async def test_empty_chunks_still_returns_evaluate(self):
        from app.services.agentic import nodes_retriever

        state = make_state("obscure query")

        with patch.object(
            nodes_retriever, "_embed_and_search", new_callable=AsyncMock, return_value=[]
        ):
            new_state, next_node = await nodes_retriever.execute(state)

        assert next_node == "EVALUATE"
        assert new_state.retrieved_chunks == []
