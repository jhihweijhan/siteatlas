from typing import Any

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: str = Field(pattern=r"^(system|user|assistant)$")
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None


class ImageData(BaseModel):
    base64: str = ""
    src: str = ""
    alt: str = ""
    width: int = 0
    height: int = 0
    index: int = 0


class IngestRequest(BaseModel):
    url: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str | None = None
    html: str | None = None
    language: str = "zh-TW"
    domain: str = "general"
    embedding_model: str | None = None
    relation_group: str | None = None
    vision_model: str | None = None
    images: list[ImageData] = Field(default_factory=list)
    source_type: str = "web"
    file_name: str | None = None
    file_ext: str | None = None
    content_hash: str | None = None

    @model_validator(mode="after")
    def require_content_or_html(self) -> "IngestRequest":
        if not self.content and not self.html:
            raise ValueError("至少需要提供 content 或 html 其中一個")
        return self


class IngestResponse(BaseModel):
    task_id: str
    chunks_count: int
    message: str


class IngestTaskStatus(BaseModel):
    task_id: str
    status: str
    stage: str
    message: str
    url: str
    title: str
    chunks_count: int = 0
    embedding_model: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str
    finished_at: str | None = None
    duration_ms: int | None = None
    image_frontend_total: int = 0
    image_filtered: int = 0
    image_total: int = 0
    image_caption_success: int = 0
    image_caption_failed: int = 0
    image_caption_failures: list[dict[str, Any]] = Field(default_factory=list)
    tagging_status: str | None = None
    tagging_tags: list[str] = Field(default_factory=list)
    tagging_error: str | None = None
    surfacing_results: list[dict[str, Any]] = Field(default_factory=list)


class IngestTaskStatusResponse(BaseModel):
    task: IngestTaskStatus


class FileIngestItem(BaseModel):
    task_id: str
    file_name: str
    status: str
    overwritten: bool = False
    message: str


class FileIngestBatchResponse(BaseModel):
    items: list[FileIngestItem]
    total: int


class RagChatRequest(BaseModel):
    query: str = Field(min_length=1)
    model: str | None = None
    embedding_model: str | None = None
    relation_groups: list[str] | None = None
    source_urls: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    tags: list[str] | None = None


class KnowledgeItem(BaseModel):
    id: str
    title: str
    url: str
    created_at: str
    chunks_count: int
    relation_group: str = "general"
    text_chunks_count: int = 0
    image_chunks_count: int = 0
    schema_keys: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class KnowledgeRelationUpdateRequest(BaseModel):
    relation_group: str = Field(min_length=1, max_length=120)


class KnowledgeTagsUpdateRequest(BaseModel):
    tags: list[str] = Field(default_factory=list, max_length=3)


class KnowledgeListResponse(BaseModel):
    items: list[KnowledgeItem]
    total: int


class KnowledgeDetailResponse(BaseModel):
    item: KnowledgeItem
    type_counts: dict[str, int] = Field(default_factory=dict)
    metadata_schema: dict[str, list[str]] = Field(default_factory=dict)
    text_samples: list[dict[str, Any]] = Field(default_factory=list)
    image_items: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    chromadb: bool
    version: str = "1.0.0"
