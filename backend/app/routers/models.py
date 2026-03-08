import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.main import get_ollama
from app.services.model_resolver import classify_model_category
from app.services.ollama_client import OllamaClient

router = APIRouter()
logger = logging.getLogger(__name__)


class UnloadRequest(BaseModel):
    model: str = Field(min_length=1)


class PullRequest(BaseModel):
    model: str = Field(min_length=1)


@router.get("/api/models/local")
async def list_local_models(ollama: OllamaClient = Depends(get_ollama)):
    models = await ollama.list_models()
    categorized: dict[str, list[dict]] = {"chat": [], "vision": [], "embedding": []}

    for m in models:
        category = classify_model_category(m)
        entry = {
            "name": m.get("name", ""),
            "size": m.get("size", 0),
            "parameter_size": (m.get("details") or {}).get("parameter_size", ""),
            "family": (m.get("details") or {}).get("family", ""),
            "category": category,
        }
        categorized.setdefault(category, []).append(entry)

    return categorized


@router.post("/api/models/unload")
async def unload_model(
    request: UnloadRequest,
    ollama: OllamaClient = Depends(get_ollama),
):
    await ollama.unload_model(request.model)
    return {"status": "ok", "model": request.model}


@router.post("/api/models/pull")
async def pull_model(
    request: PullRequest,
    ollama: OllamaClient = Depends(get_ollama),
):
    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            async for event in ollama.pull_model(request.model):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
