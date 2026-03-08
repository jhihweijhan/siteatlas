import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.config import settings
from app.main import get_ollama, get_vector_store
from app.models.schemas import RagChatRequest
from app.services.embedding import EmbeddingService
from app.services.filter_builder import (
    build_combined_where,
    build_image_caption_where,
    build_relation_where,
    build_source_where,
    build_tags_where,
)
from app.services.model_resolver import resolve_chat_model_name
from app.services.ollama_client import OllamaClient
from app.services.vector_store import VectorStoreService

router = APIRouter()

RAG_SYSTEM_PROMPT = """你是一個知識庫問答助手。請根據以下提供的參考資料回答使用者的問題。

規則：
1. 只根據提供的參考資料回答，不要使用你的預訓練知識。
2. 如果參考資料中找不到答案，請明確告知：「根據目前知識庫的資料，我無法回答這個問題。」
3. 在回答中引用來源，格式為 [1]、[2] 等。
4. 回答時使用繁體中文。
5. 適當使用條列式整理重點。
6. 若問題詢問「有哪些 relation group / 關聯群組 / 查詢範圍」，優先使用「知識庫關聯群組總覽」回答。
7. 若 context 中已有來源內容，請先嘗試摘要與回答；只有在 context 完全沒有相關資訊時，才可使用第2條的無法回答句式。
8. 如果參考資料中有圖片描述（type="image_caption"），在回答中標註「依據 [圖N]」並簡述圖片內容。"""


def _normalize_relation_group(value: str | None) -> str:
    return (value or "").strip()


def extract_result_relation_group(meta: dict) -> str:
    return _normalize_relation_group(meta.get("relation_group")) or _normalize_relation_group(meta.get("domain")) or "general"


def collect_relation_scope_info(
    docs: list[dict],
    selected_relation_groups: list[str],
) -> dict:
    known_groups = sorted(
        {
            _normalize_relation_group(d.get("relation_group")) or _normalize_relation_group(d.get("domain")) or "general"
            for d in docs
            if d.get("url")
        }
    )
    selected = sorted({_normalize_relation_group(g) for g in selected_relation_groups if _normalize_relation_group(g)})

    if selected:
        in_scope = [g for g in known_groups if g in set(selected)]
        mode = "filtered"
    else:
        in_scope = known_groups
        mode = "all"

    return {
        "mode": mode,
        "known_groups": known_groups,
        "in_scope_groups": in_scope,
        "selected_groups": selected,
    }


def build_relation_scope_context(scope_info: dict) -> str:
    known_groups = scope_info.get("known_groups", [])
    in_scope = scope_info.get("in_scope_groups", [])
    selected = scope_info.get("selected_groups", [])
    mode = scope_info.get("mode", "all")

    lines = [
        "知識庫關聯群組總覽：",
        f"- 模式: {mode}",
        f"- 全部群組: {', '.join(known_groups) if known_groups else 'none'}",
        f"- 本次範圍群組: {', '.join(in_scope) if in_scope else 'none'}",
    ]
    if selected:
        lines.append(f"- 使用者指定群組: {', '.join(selected)}")
    return "\n".join(lines)


def build_rag_context(results: list[dict]) -> str:
    if not results:
        return "<context>無相關資料</context>"

    parts = ["<context>"]
    for i, r in enumerate(results, 1):
        metadata = r.get("metadata", {})
        title = metadata.get("title", "")
        url = metadata.get("source_url", "")
        relation_group = metadata.get("relation_group", "")
        chunk_type = metadata.get("type", "text")
        image_url = metadata.get("image_url", "")
        parts.append(
            f'<source id="{i}" title="{title}" url="{url}" '
            f'relation_group="{relation_group}" type="{chunk_type}" '
            f'image_url="{image_url}">'
        )
        parts.append(r.get("document", ""))
        parts.append("</source>")
    parts.append("</context>")
    return "\n".join(parts)


def merge_results_with_image_captions(results: list[dict], image_caption_rows: list[dict]) -> list[dict]:
    if not image_caption_rows:
        return results

    merged = list(results)
    seen_ids = {r.get("id") for r in merged if r.get("id")}
    additions: list[dict] = []

    for row in image_caption_rows:
        meta = row.get("metadata", {})
        if meta.get("type") != "image_caption":
            continue
        row_id = row.get("id")
        if row_id and row_id in seen_ids:
            continue
        additions.append(row)
        if row_id:
            seen_ids.add(row_id)

    def _caption_sort_key(item: dict) -> tuple:
        meta = item.get("metadata", {})
        return (
            meta.get("source_url", ""),
            int(meta.get("image_index", meta.get("chunk_index", 0)) or 0),
        )

    additions.sort(key=_caption_sort_key)
    merged.extend(additions)
    return merged


def build_rag_meta(
    results: list[dict],
    top_k: int,
    model: str,
    embedding_model: str,
    scope_info: dict,
    selected_source_urls: list[str],
) -> dict:
    result_urls = {
        r.get("metadata", {}).get("source_url", "")
        for r in results
        if r.get("metadata", {}).get("source_url", "")
    }
    result_groups = {
        extract_result_relation_group(r.get("metadata", {}))
        for r in results
    }
    in_scope_groups = set(scope_info.get("in_scope_groups", []))
    hit_groups = sorted(in_scope_groups.intersection(result_groups))
    group_denominator = len(in_scope_groups)

    return {
        "mode": "rag",
        "model_used": model,
        "embedding_model_used": embedding_model,
        "top_k": top_k,
        "retrieved_chunks": len(results),
        "retrieval_hit_rate": round(min(len(results), top_k) / top_k, 4) if top_k > 0 else None,
        "unique_source_count": len(result_urls),
        "result_relation_groups": sorted(result_groups),
        "scope_mode": scope_info.get("mode", "all"),
        "scope_group_count": group_denominator,
        "scope_groups": scope_info.get("in_scope_groups", []),
        "selected_groups": scope_info.get("selected_groups", []),
        "selected_source_count": len(selected_source_urls),
        "hit_group_count": len(hit_groups),
        "hit_groups": hit_groups,
        "relation_group_hit_rate": (
            round(len(hit_groups) / group_denominator, 4)
            if group_denominator > 0
            else None
        ),
    }


@router.post("/api/rag_chat")
async def rag_chat(
    request: RagChatRequest,
    ollama: OllamaClient = Depends(get_ollama),
    vs: VectorStoreService = Depends(get_vector_store),
):
    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            available_models = await ollama.list_models()
            model = resolve_chat_model_name(
                request.model,
                settings.llm_model,
                available_models,
            )
            embedding_model = request.embedding_model or settings.embedding_model
            embedder = EmbeddingService(ollama, model=embedding_model)
            query_embedding = await embedder.embed_query(request.query)
            relation_groups = [g.strip() for g in (request.relation_groups or []) if g and g.strip()]
            source_urls = [u.strip() for u in (request.source_urls or []) if u and u.strip()]
            tags = [t.strip() for t in (request.tags or []) if t and t.strip()]
            where = build_combined_where(relation_groups, source_urls, tags)

            docs = await asyncio.to_thread(vs.list_documents)
            scope_info = collect_relation_scope_info(docs, relation_groups)
            relation_scope_context = build_relation_scope_context(scope_info)
            results = await asyncio.to_thread(vs.search, query_embedding, request.top_k, where)
            if source_urls:
                image_caption_where = build_image_caption_where(source_urls)
                if image_caption_where:
                    image_caption_rows = await asyncio.to_thread(vs.get_chunks, image_caption_where, True)
                    results = merge_results_with_image_captions(results, image_caption_rows)
            meta = build_rag_meta(
                results,
                request.top_k,
                model,
                embedding_model,
                scope_info,
                source_urls,
            )
            yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"

            context = build_rag_context(results)
            messages = [
                {"role": "system", "content": f"{RAG_SYSTEM_PROMPT}\n\n{relation_scope_context}\n\n{context}"},
                {"role": "user", "content": request.query},
            ]

            sources = [
                {
                    "id": i + 1,
                    "title": r.get("metadata", {}).get("title", ""),
                    "url": r.get("metadata", {}).get("source_url", ""),
                    "relation_group": extract_result_relation_group(r.get("metadata", {})),
                    "type": r.get("metadata", {}).get("type", "text"),
                    "image_url": r.get("metadata", {}).get("image_url", ""),
                    "image_alt": r.get("metadata", {}).get("image_alt", ""),
                }
                for i, r in enumerate(results)
            ]
            yield f"event: sources\ndata: {json.dumps(sources, ensure_ascii=False)}\n\n"

            async for chunk in ollama.stream_chat(messages, model):
                if chunk.get("error"):
                    yield f"event: error\ndata: {chunk['error']}\n\n"
                    return
                content = chunk.get("message", {}).get("content")
                if content:
                    yield f"event: token\ndata: {content}\n\n"
                if chunk.get("done"):
                    done_payload = {
                        "status": "done",
                        "total_duration": chunk.get("total_duration"),
                        "load_duration": chunk.get("load_duration"),
                        "prompt_eval_count": chunk.get("prompt_eval_count"),
                        "eval_count": chunk.get("eval_count"),
                    }
                    yield f"event: done\ndata: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
