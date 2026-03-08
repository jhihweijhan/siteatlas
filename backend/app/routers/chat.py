from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.config import settings
from app.main import get_ollama
from app.models.schemas import ChatRequest
from app.services.model_resolver import resolve_chat_model_name
from app.services.ollama_client import OllamaClient

router = APIRouter()


@router.post("/api/chat")
async def chat(request: ChatRequest, ollama: OllamaClient = Depends(get_ollama)):
    available_models = await ollama.list_models()
    model = resolve_chat_model_name(request.model, settings.llm_model, available_models)
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            async for chunk in ollama.stream_chat(messages, model):
                if chunk.get("error"):
                    yield f"event: error\ndata: {chunk['error']}\n\n"
                    return
                content = chunk.get("message", {}).get("content")
                if content:
                    yield f"event: token\ndata: {content}\n\n"
                if chunk.get("done"):
                    yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
