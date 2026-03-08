from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = "http://127.0.0.1:11434"
    chromadb_host: str = "localhost"
    chromadb_port: int = 8100
    llm_model: str = "qwen2.5:14b"
    embedding_model: str = "bge-m3"
    chunk_size: int = 512
    chunk_overlap: int = 50
    log_level: str = "info"
    vision_model: str = "llama3.2-vision"
    max_images_per_page: int = 20
    min_image_dimension: int = 100
    max_image_size_bytes: int = 5_242_880  # 5MB

    model_config = {"env_prefix": ""}


settings = Settings()
