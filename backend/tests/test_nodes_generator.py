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
        existing_system_tags=[],
        config=AgenticConfig(),
        chat_model="qwen2.5:14b",
        embedding_model="bge-m3",
        start_time=time.time(),
        route_action=RouteAction.SEARCH_KNOWLEDGE,
        query_plan=QueryPlan(semantic_query=query),
        retrieved_chunks=[],
    )
    defaults.update(overrides)
    return AgenticState(**defaults)


class TestExecuteNoAnswer:
    @pytest.mark.asyncio
    async def test_returns_template_and_end(self):
        from app.services.agentic.nodes_generator import execute_no_answer

        state = make_state("some question")
        new_state, next_node = await execute_no_answer(state)

        assert next_node == "END"
        assert "沒有找到" in new_state.final_answer
        assert len(new_state.final_answer) > 0


class TestBuildRagSystemPrompt:
    def test_contains_source_keyword(self):
        from app.services.agentic.nodes_generator import build_rag_system_prompt

        prompt = build_rag_system_prompt()
        assert "來源" in prompt

    def test_contains_no_pretrained_instruction(self):
        from app.services.agentic.nodes_generator import build_rag_system_prompt

        prompt = build_rag_system_prompt()
        assert "預訓練" in prompt or "知識庫" in prompt


class TestBuildRagMessages:
    def test_messages_include_chunks_and_history(self):
        from app.services.agentic.nodes_generator import build_rag_messages

        chunks = [
            {
                "document": "Docker 是容器技術",
                "metadata": {"source_url": "https://docker.com", "type": "text"},
                "distance": 0.2,
            },
            {
                "document": "Docker image 說明圖",
                "metadata": {"source_url": "https://docker.com", "type": "image_caption"},
                "distance": 0.3,
            },
        ]
        state = make_state(
            "Docker 是什麼",
            conversation_history=[
                {"role": "user", "content": "之前的問題"},
                {"role": "assistant", "content": "之前的回答"},
            ],
            retrieved_chunks=chunks,
        )

        messages = build_rag_messages(state)

        assert messages[0]["role"] == "system"
        # History should be present
        contents = [m["content"] for m in messages]
        joined = " ".join(contents)
        assert "之前的問題" in joined
        assert "Docker 是容器技術" in joined
        assert "Docker 是什麼" in joined

    def test_no_chunks_still_builds_messages(self):
        from app.services.agentic.nodes_generator import build_rag_messages

        state = make_state("hello", retrieved_chunks=[])
        messages = build_rag_messages(state)
        assert len(messages) >= 2  # system + user


class TestStreamExecute:
    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        from app.services.agentic import nodes_generator

        state = make_state(
            "Docker 是什麼",
            retrieved_chunks=[
                {"document": "Docker 介紹", "metadata": {"source_url": "https://docker.com", "type": "text"}, "distance": 0.2},
            ],
        )

        async def mock_stream(*args, **kwargs):
            for token in ["Docker ", "是一個 ", "容器工具。"]:
                yield {"message": {"content": token}}

        mock_client = AsyncMock()
        mock_client.stream_chat = mock_stream

        with patch.object(nodes_generator, "_ollama_client", mock_client):
            tokens = []
            async for token in nodes_generator.stream_execute(state):
                tokens.append(token)

        assert len(tokens) == 3
        assert "".join(tokens) == "Docker 是一個 容器工具。"
