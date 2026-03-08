from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.agentic_schemas import AgenticChatRequest
from app.services.agentic.engine import build_sse_event


class TestAgenticEndpointRegistration:
    def test_agentic_endpoint_exists(self):
        from app.main import app

        routes = [route.path for route in app.routes]
        assert "/api/agentic_chat" in routes

    def test_rag_endpoint_still_exists(self):
        from app.main import app

        routes = [route.path for route in app.routes]
        assert "/api/rag_chat" in routes


class TestAgenticModelResolution:
    @pytest.mark.asyncio
    async def test_agentic_chat_uses_available_chat_model_when_default_missing(self):
        import app.main as main_module
        from app.routers.agentic import agentic_chat

        fake_ollama = AsyncMock()
        fake_ollama.list_models.return_value = [{"name": "gemma3:12b"}]

        fake_vector_store = MagicMock()
        fake_vector_store.collect_existing_tags.return_value = []

        captured: dict[str, str] = {}

        async def fake_run_agentic_loop(state):
            captured["chat_model"] = state.chat_model
            yield build_sse_event("done", {"status": "done"})

        with (
            patch.object(main_module, "ollama_client", fake_ollama),
            patch.object(main_module, "vector_store", fake_vector_store),
            patch("app.routers.agentic.run_agentic_loop", fake_run_agentic_loop),
        ):
            response = await agentic_chat(AgenticChatRequest(query="這頁在說什麼？"))

            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))

        assert captured["chat_model"] == "gemma3:12b"
        assert any("event: done" in chunk for chunk in chunks)
