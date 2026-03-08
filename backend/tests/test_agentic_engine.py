import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from app.core.agentic_settings import AgenticConfig
from app.models.agentic_schemas import AgenticState, QueryPlan, RouteAction


@pytest.fixture
def base_state():
    return AgenticState(
        user_message="Docker 怎麼用？",
        conversation_history=[],
        ui_hard_constraints={},
        existing_system_tags=["Docker"],
        config=AgenticConfig(),
        chat_model="qwen2.5:14b",
        embedding_model="bge-m3",
        start_time=time.monotonic(),
    )


class TestBuildSseEvent:
    def test_dict_data(self):
        from app.services.agentic.engine import build_sse_event

        result = build_sse_event("meta", {"key": "值"})
        assert result.startswith("event: meta\n")
        assert "data: " in result
        assert result.endswith("\n\n")
        payload = result.split("data: ", 1)[1].rstrip("\n")
        parsed = json.loads(payload)
        assert parsed == {"key": "值"}

    def test_list_data(self):
        from app.services.agentic.engine import build_sse_event

        result = build_sse_event("sources", [{"url": "https://a.com"}])
        payload = result.split("data: ", 1)[1].rstrip("\n")
        parsed = json.loads(payload)
        assert isinstance(parsed, list)

    def test_str_data(self):
        from app.services.agentic.engine import build_sse_event

        result = build_sse_event("token", "hello")
        assert result == "event: token\ndata: hello\n\n"

    def test_chinese_not_escaped(self):
        from app.services.agentic.engine import build_sse_event

        result = build_sse_event("status", {"msg": "路由中"})
        assert "路由中" in result
        assert "\\u" not in result


class TestRunAgenticLoop:
    @pytest.mark.asyncio
    async def test_happy_path_search_knowledge(self, base_state):
        from app.services.agentic.engine import run_agentic_loop

        chunks = [{"document": "Docker volume 說明", "distance": 0.3, "metadata": {"source_url": "https://a.com"}}]

        async def mock_router(state):
            state.route_action = RouteAction.SEARCH_KNOWLEDGE
            state.query_plan = QueryPlan(semantic_query="Docker 怎麼用")
            state.current_node = "ROUTER"
            return state, "RETRIEVE"

        async def mock_retriever(state):
            state.retrieved_chunks = chunks
            state.current_node = "RETRIEVER"
            return state, "EVALUATE"

        async def mock_evaluator(state):
            state.evidence_verdict = "sufficient"
            state.current_node = "EVALUATOR"
            return state, "GENERATE"

        async def mock_generator_stream(state):
            for token in ["Docker", " volume", " 用法"]:
                yield token

        with patch("app.services.agentic.engine.NODE_REGISTRY", {
            "ROUTE": mock_router,
            "RETRIEVE": mock_retriever,
            "EVALUATE": mock_evaluator,
            "REFINE": AsyncMock(),
            "NO_ANSWER": AsyncMock(),
        }), patch("app.services.agentic.engine.nodes_generator") as mock_gen_mod:
            mock_gen_mod.stream_execute = mock_generator_stream

            events = []
            async for event in run_agentic_loop(base_state):
                events.append(event)

        event_types = [e.split("\n")[0].replace("event: ", "") for e in events]

        assert "status" in event_types
        assert "token" in event_types
        assert "done" in event_types
        assert "transition" in event_types

    @pytest.mark.asyncio
    async def test_timeout_with_chunks_degrades_to_generate(self):
        from app.services.agentic.engine import run_agentic_loop

        state = AgenticState(
            user_message="test",
            conversation_history=[],
            ui_hard_constraints={},
            existing_system_tags=[],
            config=AgenticConfig(global_timeout_seconds=0.001),
            chat_model="qwen2.5:14b",
            embedding_model="bge-m3",
            start_time=time.monotonic() - 100,  # already expired
            retrieved_chunks=[{"document": "data", "distance": 0.3, "metadata": {}}],
        )

        async def mock_generator_stream(s):
            yield "answer"

        with patch("app.services.agentic.engine.nodes_generator") as mock_gen_mod:
            mock_gen_mod.stream_execute = mock_generator_stream

            events = []
            async for event in run_agentic_loop(state):
                events.append(event)

        all_text = "".join(events)
        assert "TIMEOUT_DEGRADE" in all_text

    @pytest.mark.asyncio
    async def test_timeout_no_chunks_goes_no_answer(self):
        from app.services.agentic.engine import run_agentic_loop

        state = AgenticState(
            user_message="test",
            conversation_history=[],
            ui_hard_constraints={},
            existing_system_tags=[],
            config=AgenticConfig(global_timeout_seconds=0.001),
            chat_model="qwen2.5:14b",
            embedding_model="bge-m3",
            start_time=time.monotonic() - 100,
        )

        async def mock_no_answer(s):
            s.final_answer = "抱歉"
            return s, "END"

        with patch("app.services.agentic.engine.NODE_REGISTRY", {
            "ROUTE": AsyncMock(),
            "RETRIEVE": AsyncMock(),
            "EVALUATE": AsyncMock(),
            "REFINE": AsyncMock(),
            "NO_ANSWER": mock_no_answer,
        }):
            events = []
            async for event in run_agentic_loop(state):
                events.append(event)

        all_text = "".join(events)
        assert "TIMEOUT_DEGRADE" in all_text


class TestBuildDonePayload:
    def test_payload_fields(self, base_state):
        from app.services.agentic.engine import _build_done_payload

        base_state.route_action = RouteAction.SEARCH_KNOWLEDGE
        base_state.evidence_verdict = "sufficient"
        base_state.refine_round = 1

        payload = _build_done_payload(base_state)
        assert payload["status"] == "done"
        assert "total_duration_ms" in payload
        assert payload["refine_rounds"] == 1
        assert payload["route_action"] == "search_knowledge"
        assert payload["evidence_verdict"] == "sufficient"
