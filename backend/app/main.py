import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.ollama_client import OllamaClient
from app.services.vector_store import VectorStoreService

ollama_client: OllamaClient | None = None
vector_store: VectorStoreService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ollama_client, vector_store
    ollama_client = OllamaClient(base_url=settings.ollama_base_url)
    vector_store = VectorStoreService(
        host=settings.chromadb_host,
        port=settings.chromadb_port,
    )
    app.state.ready = True
    yield
    app.state.ready = False
    if ollama_client is not None:
        await ollama_client.close()


app = FastAPI(title="SiteAtlas API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"(chrome-extension://.*|http://localhost(:\\d+)?|http://127\\.0\\.0\\.1(:\\d+)?)",
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_ollama() -> OllamaClient:
    assert ollama_client is not None
    return ollama_client


def get_vector_store() -> VectorStoreService:
    assert vector_store is not None
    return vector_store


@app.get("/health/readiness")
async def readiness(
    ollama: OllamaClient = Depends(get_ollama),
    vs: VectorStoreService = Depends(get_vector_store),
):
    health = await ollama.check_health()
    chroma_ok = await asyncio.to_thread(vs.check_health)
    all_ok = health.get("alive", False) and chroma_ok
    return {
        "status": "ok" if all_ok else "degraded",
        "ollama": health.get("alive", False),
        "chromadb": chroma_ok,
        "version": "1.0.0",
    }


from app.routers import agentic, chat, ingest, knowledge, models, rag  # noqa: E402

app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(rag.router)
app.include_router(knowledge.router)
app.include_router(models.router)
app.include_router(agentic.router)
