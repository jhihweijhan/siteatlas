import asyncio
import logging
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.config import settings
from app.core.agentic_settings import AgenticConfig
from app.models.agentic_schemas import AgenticChatRequest, AgenticState
from app.services.agentic import nodes_evaluator, nodes_generator, nodes_retriever, nodes_router
from app.services.agentic.engine import build_sse_event, run_agentic_loop
from app.services.model_resolver import resolve_chat_model_name

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/agentic_chat")
async def agentic_chat(request: AgenticChatRequest):
    from app.main import ollama_client, vector_store
    from app.services.embedding import EmbeddingService

    if ollama_client is None or vector_store is None:

        async def error_stream():
            yield build_sse_event("error", {"message": "Backend services not ready"})

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    # 初始化 node 依賴
    nodes_router.init_dependencies(ollama_client)
    embedding_model = request.embedding_model or settings.embedding_model
    embedding_service = EmbeddingService(ollama_client, model=embedding_model)
    nodes_retriever.init_dependencies(embedding_service, vector_store)
    nodes_generator.init_dependencies(ollama_client)
    nodes_evaluator.init_dependencies(ollama_client)

    # 收集 existing tags
    existing_tags = await asyncio.to_thread(vector_store.collect_existing_tags)

    # 模型
    available_models = await ollama_client.list_models()
    chat_model = resolve_chat_model_name(
        request.model,
        settings.llm_model,
        available_models,
    )

    # 初始 state
    state = AgenticState(
        user_message=request.query,
        conversation_history=request.conversation_history,
        ui_hard_constraints={
            "tags": request.tags or [],
            "relation_groups": request.relation_groups or [],
            "source_urls": request.source_urls or [],
        },
        existing_system_tags=list(existing_tags),
        config=AgenticConfig(),
        chat_model=chat_model,
        embedding_model=embedding_model,
        start_time=time.monotonic(),
    )

    return StreamingResponse(
        run_agentic_loop(state),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
