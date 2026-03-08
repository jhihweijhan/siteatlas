import asyncio
import hashlib
import logging
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pypdf import PdfReader

from app.config import settings
from app.main import get_ollama, get_vector_store
from app.models.schemas import (
    FileIngestBatchResponse,
    FileIngestItem,
    IngestRequest,
    IngestResponse,
    IngestTaskStatus,
    IngestTaskStatusResponse,
)
from app.services.chunker import ChineseTextChunker
from app.services.content_extractor import ContentExtractor
from app.services.embedding import EmbeddingService
from app.services.ollama_client import OllamaClient
from app.services.tag_guardian import TagGuardian
from app.services.vector_store import VectorStoreService

router = APIRouter()
logger = logging.getLogger(__name__)
content_extractor = ContentExtractor()
chunker = ChineseTextChunker(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)
ingest_tasks: dict[str, dict[str, Any]] = {}
ingest_tasks_lock = asyncio.Lock()
MAX_INGEST_TASKS = 300
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration_ms(started_at_iso: str, ended_at_iso: str) -> int | None:
    try:
        started = datetime.fromisoformat(started_at_iso)
        ended = datetime.fromisoformat(ended_at_iso)
        return max(0, int((ended - started).total_seconds() * 1000))
    except Exception:  # noqa: BLE001
        return None


def _short_error(exc: Exception, limit: int = 240) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text[:limit]


def _normalize_file_name(file_name: str) -> str:
    safe_name = Path(file_name).name.strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="檔名不可為空")
    return safe_name


def _build_file_source_url(file_name: str) -> str:
    return f"file://{_normalize_file_name(file_name).lower()}"


def _extract_text_from_upload(file_name: str, raw_bytes: bytes) -> tuple[str, str]:
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支援的檔案格式：{ext or 'unknown'}")

    if ext == ".pdf":
        reader = PdfReader(BytesIO(raw_bytes))
        content = "\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
    else:
        try:
            content = raw_bytes.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"無法解析檔案內容：{file_name}") from exc

    if not content:
        raise HTTPException(status_code=400, detail=f"檔案內容為空：{file_name}")

    return ext, content


async def _set_ingest_task(task_id: str, **fields: Any) -> dict[str, Any]:
    now = _now_iso()
    async with ingest_tasks_lock:
        task = ingest_tasks.get(task_id, {}).copy()
        task.update(fields)
        task["task_id"] = task_id
        task["updated_at"] = now
        ingest_tasks[task_id] = task
        return task.copy()


async def _create_ingest_task(task_id: str, request: IngestRequest, chunks_count: int) -> None:
    now = _now_iso()
    image_frontend_total = len([img for img in request.images if img.base64])
    task = {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "message": "任務已排入背景處理",
        "url": request.url,
        "title": request.title,
        "chunks_count": chunks_count,
        "embedding_model": request.embedding_model or settings.embedding_model,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "finished_at": None,
        "duration_ms": None,
        "image_frontend_total": image_frontend_total,
        "image_filtered": 0,
        "image_total": image_frontend_total,
        "image_caption_success": 0,
        "image_caption_failed": 0,
        "image_caption_failures": [],
        "tagging_status": None,
        "tagging_tags": [],
        "tagging_error": None,
        "surfacing_results": [],
    }

    async with ingest_tasks_lock:
        ingest_tasks[task_id] = task
        if len(ingest_tasks) > MAX_INGEST_TASKS:
            stale_ids = sorted(
                ingest_tasks.keys(),
                key=lambda tid: ingest_tasks[tid].get("updated_at", ""),
            )[: len(ingest_tasks) - MAX_INGEST_TASKS]
            for stale_id in stale_ids:
                ingest_tasks.pop(stale_id, None)


async def _get_ingest_task(task_id: str) -> dict[str, Any] | None:
    async with ingest_tasks_lock:
        task = ingest_tasks.get(task_id)
        return task.copy() if task else None


async def _run_tagging_stage(
    ollama: OllamaClient,
    embedding_svc: EmbeddingService,
    vs: VectorStoreService,
    title: str,
    content_preview: str,
    model: str,
) -> dict:
    """Execute tagging stage. Returns {"status": "success"|"degraded", "tags": [...], "error": str|None}"""
    try:
        existing_tags = await asyncio.to_thread(vs.collect_existing_tags)
        suggested = await ollama.generate_tags(
            title=title,
            content_preview=content_preview,
            existing_tags=sorted(existing_tags),
            model=model,
        )
        if suggested is None:
            return {"status": "degraded", "tags": [], "error": "LLM tagging failed"}

        guardian = TagGuardian(embedding_svc, threshold=0.3)
        resolved = await guardian.resolve_tags(suggested, existing_tags)
        return {"status": "success", "tags": resolved, "error": None}
    except Exception as e:
        logger.warning(f"Tagging stage failed: {e}")
        return {"status": "degraded", "tags": [], "error": str(e)}


SURFACING_DISTANCE_THRESHOLD = 0.35
SURFACING_MAX_RESULTS = 3


async def _run_surfacing_stage(
    vs: VectorStoreService,
    embeddings: list[list[float]],
    doc_id: str,
) -> list[dict]:
    """Search knowledge base for related documents using first chunk embedding."""
    if not embeddings:
        return []
    try:
        candidates = await asyncio.to_thread(
            vs.search_by_embedding,
            query_embedding=embeddings[0],
            exclude_doc_id=doc_id,
            top_k=5,
        )
        filtered = [c for c in candidates if c["distance"] < SURFACING_DISTANCE_THRESHOLD]
        return filtered[:SURFACING_MAX_RESULTS]
    except Exception:  # noqa: BLE001
        logger.warning("Surfacing stage failed, continuing pipeline", exc_info=True)
        return []


async def ingest_pipeline(task_id: str, request: IngestRequest, ollama: OllamaClient, vs: VectorStoreService):
    stage = "queued"
    try:
        image_captions: list[dict] = []
        image_caption_failed = 0
        image_caption_failures: list[dict[str, Any]] = []

        # Extract content from HTML if provided, otherwise use pre-extracted content
        extracted_text = request.content or ""
        extracted_image_urls: list[str] = []
        if request.html:
            stage = "extracting"
            await _set_ingest_task(
                task_id,
                status="running",
                stage=stage,
                message="正在用 trafilatura 提取頁面內容",
            )
            extracted = await asyncio.to_thread(
                content_extractor.extract, request.html, request.url
            )
            extracted_text = extracted.text
            extracted_image_urls = extracted.image_urls

            if not extracted_text:
                done_at = _now_iso()
                task = await _get_ingest_task(task_id)
                duration = _duration_ms((task or {}).get("created_at", done_at), done_at)
                await _set_ingest_task(
                    task_id,
                    status="error",
                    stage="extracting",
                    message="trafilatura 無法從 HTML 中提取有效內容",
                    error="無法從 HTML 提取內容，頁面可能缺乏文本",
                    chunks_count=0,
                    finished_at=done_at,
                    duration_ms=duration,
                )
                return

        stage = "chunking"
        await _set_ingest_task(
            task_id,
            status="running",
            stage=stage,
            message="正在切分文本",
        )
        chunks = chunker.split(extracted_text)
        if not chunks:
            done_at = _now_iso()
            task = await _get_ingest_task(task_id)
            duration = _duration_ms((task or {}).get("created_at", done_at), done_at)
            await _set_ingest_task(
                task_id,
                status="error",
                stage="chunking",
                message="文本切分結果為空，無可寫入內容",
                error="無法從內容切分出可寫入的文本區塊",
                chunks_count=0,
                finished_at=done_at,
                duration_ms=duration,
            )
            return

        # ── tagging stage ──
        stage = "tagging"
        await _set_ingest_task(task_id, stage="tagging", message="正在自動打標")
        embedding_model = request.embedding_model or settings.embedding_model
        embedder = EmbeddingService(ollama, model=embedding_model)
        tagging_result = await _run_tagging_stage(
            ollama=ollama,
            embedding_svc=embedder,
            vs=vs,
            title=request.title,
            content_preview=extracted_text[:500],
            model=settings.llm_model,
        )
        tag_values = (tagging_result["tags"] + ["", "", ""])[:3]
        await _set_ingest_task(
            task_id,
            tagging_status=tagging_result["status"],
            tagging_tags=tagging_result["tags"],
            tagging_error=tagging_result.get("error"),
        )

        all_images_with_data = [img for img in request.images if img.base64]
        # When using trafilatura, filter images to only those in main content
        if request.html and extracted_image_urls:
            content_url_set = {u.rstrip("/") for u in extracted_image_urls}
            valid_images = [
                img for img in all_images_with_data
                if img.src.rstrip("/") in content_url_set
            ]
            logger.info(
                "Image filter: %d frontend images -> %d in main content (trafilatura found %d)",
                len(all_images_with_data),
                len(valid_images),
                len(content_url_set),
            )
        else:
            valid_images = all_images_with_data
        image_filtered = len(all_images_with_data) - len(valid_images)
        await _set_ingest_task(
            task_id,
            image_frontend_total=len(all_images_with_data),
            image_filtered=image_filtered,
            image_total=len(valid_images),
        )
        if valid_images and request.vision_model:
            stage = "captioning"
            await _set_ingest_task(
                task_id,
                status="running",
                stage=stage,
                message=f"正在描述圖片（{len(valid_images)} 張）",
                image_total=len(valid_images),
                image_caption_success=0,
                image_caption_failed=0,
            )
            for idx, img in enumerate(valid_images):
                try:
                    await _set_ingest_task(
                        task_id,
                        message=f"正在描述圖片（{idx + 1}/{len(valid_images)}）",
                    )
                    context_parts = []
                    if request.title:
                        context_parts.append(f"此圖片來自文章：{request.title}")
                    if img.alt:
                        context_parts.append(f"圖片說明：{img.alt}")
                    contexts = [("\n".join(context_parts)).strip()]
                    if contexts[0]:
                        contexts.append("")

                    caption = ""
                    last_exc: Exception | None = None
                    for attempt_idx, caption_context in enumerate(contexts, 1):
                        try:
                            if attempt_idx > 1:
                                await _set_ingest_task(
                                    task_id,
                                    message=f"圖片 {idx + 1} 第 {attempt_idx} 次嘗試描述中",
                                )
                            caption = await ollama.describe_image(
                                img.base64,
                                model=request.vision_model,
                                context=caption_context,
                            )
                            break
                        except Exception as attempt_exc:  # noqa: BLE001
                            last_exc = attempt_exc

                    if not caption:
                        raise RuntimeError(_short_error(last_exc or RuntimeError("caption returned empty")))

                    image_captions.append({
                        "caption": caption,
                        "src": img.src,
                        "alt": img.alt,
                        "index": img.index,
                    })
                    await _set_ingest_task(
                        task_id,
                        image_caption_success=len(image_captions),
                        image_caption_failed=image_caption_failed,
                        image_caption_failures=image_caption_failures,
                    )
                except Exception as img_exc:  # noqa: BLE001
                    image_caption_failed += 1
                    failure_info = {
                        "index": img.index,
                        "src": img.src,
                        "error": _short_error(img_exc),
                    }
                    image_caption_failures.append(failure_info)
                    logger.warning(
                        "Failed to caption image (task_id=%s, index=%s, src=%s): %s",
                        task_id,
                        img.index,
                        img.src,
                        img_exc,
                    )
                    await _set_ingest_task(
                        task_id,
                        image_caption_success=len(image_captions),
                        image_caption_failed=image_caption_failed,
                        image_caption_failures=image_caption_failures,
                    )
            # Unload vision model to free VRAM for embedding
            try:
                await ollama.unload_model(request.vision_model)
            except Exception:  # noqa: BLE001
                pass

        caption_texts = [f"[圖片描述] {ic['caption']}" for ic in image_captions]
        all_texts = chunks + caption_texts

        relation_group = (request.relation_group or request.domain or "general").strip() or "general"
        stage = "embedding"
        await _set_ingest_task(
            task_id,
            status="running",
            stage=stage,
            chunks_count=len(chunks),
            embedding_model=embedding_model,
            message=f"正在產生向量（{len(chunks)} 文字 + {len(image_captions)} 圖片）",
            image_total=len(valid_images),
            image_caption_success=len(image_captions),
            image_caption_failed=image_caption_failed,
            image_caption_failures=image_caption_failures,
        )
        embeddings = await embedder.embed_texts(all_texts)

        doc_id = hashlib.md5(request.url.encode(), usedforsecurity=False).hexdigest()[:12]

        # ── surfacing stage ──
        stage = "surfacing"
        await _set_ingest_task(
            task_id,
            status="running",
            stage=stage,
            message="正在搜尋相關文件",
        )
        surfacing_results = await _run_surfacing_stage(vs, embeddings, doc_id)
        await _set_ingest_task(task_id, surfacing_results=surfacing_results)

        now = datetime.now(timezone.utc).isoformat()
        text_metadatas = [
            {
                "doc_id": doc_id,
                "source_url": request.url,
                "title": request.title,
                "language": request.language,
                "domain": request.domain,
                "relation_group": relation_group,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "created_at": now,
                "type": "text",
                "source_type": request.source_type,
                "file_name": request.file_name or "",
                "file_ext": request.file_ext or "",
                "content_hash": request.content_hash or "",
                "tag_1": tag_values[0],
                "tag_2": tag_values[1],
                "tag_3": tag_values[2],
            }
            for i in range(len(chunks))
        ]
        image_metadatas = [
            {
                "doc_id": doc_id,
                "source_url": request.url,
                "title": request.title,
                "language": request.language,
                "domain": request.domain,
                "relation_group": relation_group,
                "chunk_index": ic["index"],
                "total_chunks": len(chunks),
                "created_at": now,
                "type": "image_caption",
                "source_type": request.source_type,
                "file_name": request.file_name or "",
                "file_ext": request.file_ext or "",
                "content_hash": request.content_hash or "",
                "image_url": ic["src"],
                "image_alt": ic["alt"],
                "image_index": ic["index"],
                "vision_model": request.vision_model,
                "tag_1": tag_values[0],
                "tag_2": tag_values[1],
                "tag_3": tag_values[2],
            }
            for ic in image_captions
        ]
        all_metadatas = text_metadatas + image_metadatas

        text_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        image_ids = [f"{doc_id}_img_{ic['index']}" for ic in image_captions]
        all_ids = text_ids + image_ids

        stage = "writing"
        await _set_ingest_task(
            task_id,
            status="running",
            stage=stage,
            message="正在寫入向量資料庫",
        )
        await asyncio.to_thread(vs.upsert_with_ids, all_ids, all_texts, embeddings, all_metadatas)
        done_at = _now_iso()
        task = await _get_ingest_task(task_id)
        duration = _duration_ms((task or {}).get("created_at", done_at), done_at)
        await _set_ingest_task(
            task_id,
            status="done",
            stage="done",
            message=(
                f"寫入完成（{len(chunks)} 文字 + {len(image_captions)} 圖片）"
                if image_caption_failed == 0
                else f"寫入完成（{len(chunks)} 文字 + {len(image_captions)} 圖片，失敗 {image_caption_failed} 張）"
            ),
            error=None,
            finished_at=done_at,
            duration_ms=duration,
            image_total=len(valid_images),
            image_caption_success=len(image_captions),
            image_caption_failed=image_caption_failed,
            image_caption_failures=image_caption_failures,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "ingest pipeline failed (task_id=%s, url=%s, stage=%s, embedding_model=%s): %s",
            task_id,
            request.url,
            stage,
            request.embedding_model or settings.embedding_model,
            exc,
        )
        done_at = _now_iso()
        task = await _get_ingest_task(task_id)
        duration = _duration_ms((task or {}).get("created_at", done_at), done_at)
        await _set_ingest_task(
            task_id,
            status="error",
            stage=stage,
            message="背景任務執行失敗",
            error=str(exc),
            finished_at=done_at,
            duration_ms=duration,
            image_caption_success=len(image_captions),
            image_caption_failed=image_caption_failed,
            image_caption_failures=image_caption_failures,
        )
        # Background task failures should not crash request lifecycle.
        return


@router.post("/api/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    ollama: OllamaClient = Depends(get_ollama),
    vs: VectorStoreService = Depends(get_vector_store),
):
    doc_id = hashlib.md5(request.url.encode(), usedforsecurity=False).hexdigest()[:12]

    if request.html:
        estimated_chunks = max(1, len(request.html) // 1000)
    else:
        estimated_chunks = len(chunker.split(request.content or ""))

    await _create_ingest_task(doc_id, request, estimated_chunks)
    background_tasks.add_task(ingest_pipeline, doc_id, request, ollama, vs)

    return IngestResponse(
        task_id=doc_id,
        chunks_count=estimated_chunks,
        message=f"正在處理「{request.title}」",
    )


@router.post("/api/ingest/files", response_model=FileIngestBatchResponse, status_code=202)
async def ingest_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    overwrite_mode: str = Form("replace"),
    ollama: OllamaClient = Depends(get_ollama),
    vs: VectorStoreService = Depends(get_vector_store),
):
    if overwrite_mode != "replace":
        raise HTTPException(status_code=400, detail="目前僅支援 replace 覆蓋模式")
    if not files:
        raise HTTPException(status_code=400, detail="至少需要選擇一個檔案")

    items: list[FileIngestItem] = []
    for upload in files:
        file_name = _normalize_file_name(upload.filename or "")
        raw_bytes = await upload.read()
        file_ext, content = _extract_text_from_upload(file_name, raw_bytes)
        source_url = _build_file_source_url(file_name)
        content_hash = hashlib.md5(content.encode("utf-8"), usedforsecurity=False).hexdigest()
        doc_id = hashlib.md5(source_url.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]

        overwritten = False
        existing = await asyncio.to_thread(vs.find_document_by_source_url, source_url)
        if existing and existing.get("doc_id"):
            await asyncio.to_thread(vs.delete_document, existing["doc_id"])
            overwritten = True

        request = IngestRequest(
            url=source_url,
            title=file_name,
            content=content,
            language="zh-TW",
            domain="file_upload",
            images=[],
            source_type="file",
            file_name=file_name,
            file_ext=file_ext.lstrip("."),
            content_hash=content_hash,
        )
        estimated_chunks = len(chunker.split(request.content or ""))
        await _create_ingest_task(doc_id, request, estimated_chunks)
        background_tasks.add_task(ingest_pipeline, doc_id, request, ollama, vs)
        items.append(
            FileIngestItem(
                task_id=doc_id,
                file_name=file_name,
                status="queued",
                overwritten=overwritten,
                message=f"正在處理「{file_name}」",
            )
        )

    return FileIngestBatchResponse(items=items, total=len(items))


@router.get("/api/ingest/{task_id}", response_model=IngestTaskStatusResponse)
async def get_ingest_task_status(task_id: str):
    task = await _get_ingest_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return IngestTaskStatusResponse(task=IngestTaskStatus(**task))
