from app.services.ollama_client import OllamaClient


class EmbeddingService:
    def __init__(self, ollama_client: OllamaClient, model: str = "bge-m3"):
        self._ollama = ollama_client
        self._model = model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._ollama.embed(texts, model=self._model)

    async def embed_query(self, query: str) -> list[float]:
        embeddings = await self.embed_texts([query])
        return embeddings[0]
