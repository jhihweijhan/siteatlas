import time
from unittest.mock import AsyncMock, patch

import pytest

from app.core.agentic_settings import AgenticConfig
from app.models.agentic_schemas import AgenticState, RouteAction


def make_state(query: str = "test query", **overrides) -> AgenticState:
    defaults = dict(
        user_message=query,
        conversation_history=[],
        ui_hard_constraints={},
        existing_system_tags=["Python", "Docker"],
        config=AgenticConfig(),
        chat_model="qwen2.5:14b",
        embedding_model="bge-m3",
        start_time=time.time(),
    )
    defaults.update(overrides)
    return AgenticState(**defaults)


class TestParseRouterResponse:
    def test_valid_json(self):
        from app.services.agentic.nodes_router import parse_router_response

        raw = '{"action": "search_knowledge", "semantic_query": "Python 教學"}'
        action, query, tags = parse_router_response(raw, "fallback")
        assert action == "search_knowledge"
        assert query == "Python 教學"
        assert tags == []

    def test_malformed_json_fallback(self):
        from app.services.agentic.nodes_router import parse_router_response

        raw = "not valid json at all"
        action, query, tags = parse_router_response(raw, "fallback query")
        assert action == "search_knowledge"
        assert query == "fallback query"
        assert tags == []

    def test_invalid_action_fallback(self):
        from app.services.agentic.nodes_router import parse_router_response

        raw = '{"action": "fly_to_moon", "semantic_query": "test"}'
        action, query, _ = parse_router_response(raw, "fallback")
        assert action == "search_knowledge"
        assert query == "test"

    def test_empty_query_fallback(self):
        from app.services.agentic.nodes_router import parse_router_response

        raw = '{"action": "search_knowledge", "semantic_query": ""}'
        action, query, _ = parse_router_response(raw, "my fallback")
        assert action == "search_knowledge"
        assert query == "my fallback"

    def test_long_query_truncated(self):
        from app.services.agentic.nodes_router import parse_router_response

        long_q = "x" * 1000
        raw = f'{{"action": "search_knowledge", "semantic_query": "{long_q}"}}'
        action, query, _ = parse_router_response(raw, "fallback")
        assert len(query) <= 512

    def test_direct_chat_action(self):
        from app.services.agentic.nodes_router import parse_router_response

        raw = '{"action": "direct_chat", "semantic_query": "hello"}'
        action, query, _ = parse_router_response(raw, "fallback")
        assert action == "direct_chat"
        assert query == "hello"


class TestExecute:
    @pytest.mark.asyncio
    async def test_search_knowledge_route(self):
        from app.services.agentic import nodes_router

        state = make_state("什麼是 RAG")
        mock_response = '{"action": "search_knowledge", "semantic_query": "RAG 概念解釋"}'

        with patch.object(
            nodes_router, "_call_router_llm", new_callable=AsyncMock, return_value=mock_response
        ):
            new_state, next_node = await nodes_router.execute(state)

        assert new_state.route_action == RouteAction.SEARCH_KNOWLEDGE
        assert new_state.query_plan is not None
        assert new_state.query_plan.semantic_query == "RAG 概念解釋"
        assert next_node == "RETRIEVE"

    @pytest.mark.asyncio
    async def test_direct_chat_route(self):
        from app.services.agentic import nodes_router

        state = make_state("你好")
        mock_response = '{"action": "direct_chat", "semantic_query": "你好"}'

        with patch.object(
            nodes_router, "_call_router_llm", new_callable=AsyncMock, return_value=mock_response
        ):
            new_state, next_node = await nodes_router.execute(state)

        assert new_state.route_action == RouteAction.DIRECT_CHAT
        assert next_node == "GENERATE"

    @pytest.mark.asyncio
    async def test_need_clarification_route(self):
        from app.services.agentic import nodes_router

        state = make_state("???")
        mock_response = '{"action": "need_clarification", "semantic_query": "???"}'

        with patch.object(
            nodes_router, "_call_router_llm", new_callable=AsyncMock, return_value=mock_response
        ):
            new_state, next_node = await nodes_router.execute(state)

        assert new_state.route_action == RouteAction.NEED_CLARIFICATION
        assert next_node == "GENERATE"

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self):
        from app.services.agentic import nodes_router

        state = make_state("some question")

        with patch.object(
            nodes_router,
            "_call_router_llm",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM down"),
        ):
            new_state, next_node = await nodes_router.execute(state)

        assert new_state.route_action == RouteAction.SEARCH_KNOWLEDGE
        assert new_state.query_plan is not None
        assert new_state.query_plan.semantic_query == "some question"
        assert next_node == "RETRIEVE"

    @pytest.mark.asyncio
    async def test_router_extracts_tag_filters(self):
        from app.services.agentic import nodes_router

        state = make_state("Python 的 Docker 部署", existing_system_tags=["Python", "Docker", "React"])
        mock_response = '{"action": "search_knowledge", "semantic_query": "Python Docker 部署", "tag_filters": ["Python", "Docker"]}'

        with patch.object(
            nodes_router, "_call_router_llm", new_callable=AsyncMock, return_value=mock_response
        ):
            new_state, next_node = await nodes_router.execute(state)

        assert new_state.query_plan.tag_filters == ["Python", "Docker"]

    @pytest.mark.asyncio
    async def test_scope_constraints_override_direct_chat(self):
        from app.services.agentic import nodes_router

        state = make_state(
            "這頁在說什麼？",
            ui_hard_constraints={"relation_groups": ["antigravity.google"], "source_urls": [], "tags": []},
        )
        mock_response = '{"action": "direct_chat", "semantic_query": "這頁在說什麼", "tag_filters": []}'

        with patch.object(
            nodes_router, "_call_router_llm", new_callable=AsyncMock, return_value=mock_response
        ):
            new_state, next_node = await nodes_router.execute(state)

        assert new_state.route_action == RouteAction.SEARCH_KNOWLEDGE
        assert next_node == "RETRIEVE"

    @pytest.mark.asyncio
    async def test_scope_constraints_override_need_clarification(self):
        from app.services.agentic import nodes_router

        state = make_state(
            "內容有什麼？",
            ui_hard_constraints={"relation_groups": [], "source_urls": [], "tags": ["Python"]},
        )
        mock_response = '{"action": "need_clarification", "semantic_query": "內容有什麼", "tag_filters": []}'

        with patch.object(
            nodes_router, "_call_router_llm", new_callable=AsyncMock, return_value=mock_response
        ):
            new_state, next_node = await nodes_router.execute(state)

        assert new_state.route_action == RouteAction.SEARCH_KNOWLEDGE
        assert next_node == "RETRIEVE"

    @pytest.mark.asyncio
    async def test_no_scope_allows_direct_chat(self):
        from app.services.agentic import nodes_router

        state = make_state(
            "你好",
            ui_hard_constraints={"relation_groups": [], "source_urls": [], "tags": []},
        )
        mock_response = '{"action": "direct_chat", "semantic_query": "你好", "tag_filters": []}'

        with patch.object(
            nodes_router, "_call_router_llm", new_callable=AsyncMock, return_value=mock_response
        ):
            new_state, next_node = await nodes_router.execute(state)

        assert new_state.route_action == RouteAction.DIRECT_CHAT
        assert next_node == "GENERATE"
