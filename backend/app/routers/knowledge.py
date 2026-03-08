import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.main import get_vector_store
from app.models.schemas import (
    KnowledgeDetailResponse,
    KnowledgeItem,
    KnowledgeListResponse,
    KnowledgeRelationUpdateRequest,
    KnowledgeTagsUpdateRequest,
)
from app.services.vector_store import VectorStoreService

router = APIRouter()


@router.get("/api/knowledge", response_model=KnowledgeListResponse)
async def list_knowledge(vs: VectorStoreService = Depends(get_vector_store)):
    docs = await asyncio.to_thread(vs.list_documents)
    items = [
        KnowledgeItem(
            id=d.get("id", ""),
            title=d.get("title", ""),
            url=d.get("url", ""),
            created_at=d.get("created_at", ""),
            chunks_count=d.get("chunks_count", 0),
            relation_group=d.get("relation_group", "general"),
            text_chunks_count=d.get("text_chunks_count", 0),
            image_chunks_count=d.get("image_chunks_count", 0),
            schema_keys=d.get("schema_keys", []),
            tags=d.get("tags", []),
        )
        for d in docs
    ]
    return KnowledgeListResponse(items=items, total=len(items))


@router.get("/api/knowledge/{doc_id}", response_model=KnowledgeDetailResponse)
async def get_knowledge_detail(doc_id: str, vs: VectorStoreService = Depends(get_vector_store)):
    detail = await asyncio.to_thread(vs.get_document_detail, doc_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Document not found")

    item = detail.get("item", {})
    return KnowledgeDetailResponse(
        item=KnowledgeItem(
            id=item.get("id", ""),
            title=item.get("title", ""),
            url=item.get("url", ""),
            created_at=item.get("created_at", ""),
            chunks_count=item.get("chunks_count", 0),
            relation_group=item.get("relation_group", "general"),
            text_chunks_count=item.get("text_chunks_count", 0),
            image_chunks_count=item.get("image_chunks_count", 0),
            schema_keys=item.get("schema_keys", []),
            tags=item.get("tags", []),
        ),
        type_counts=detail.get("type_counts", {}),
        metadata_schema=detail.get("metadata_schema", {}),
        text_samples=detail.get("text_samples", []),
        image_items=detail.get("image_items", []),
    )


@router.put("/api/knowledge/{doc_id}/relation")
async def update_knowledge_relation(
    doc_id: str,
    payload: KnowledgeRelationUpdateRequest,
    vs: VectorStoreService = Depends(get_vector_store),
):
    updated = await asyncio.to_thread(vs.update_document_relation, doc_id, payload.relation_group)
    if not updated:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Relation group updated", "doc_id": doc_id, "relation_group": payload.relation_group}


@router.put("/api/knowledge/{doc_id}/tags")
async def update_knowledge_tags(
    doc_id: str,
    payload: KnowledgeTagsUpdateRequest,
    vs: VectorStoreService = Depends(get_vector_store),
):
    ok = await asyncio.to_thread(vs.update_document_tags, doc_id, payload.tags)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"message": "Tags updated", "doc_id": doc_id, "tags": payload.tags}


@router.delete("/api/knowledge/{doc_id}")
async def delete_knowledge(doc_id: str, vs: VectorStoreService = Depends(get_vector_store)):
    deleted = await asyncio.to_thread(vs.delete_document, doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted", "doc_id": doc_id}
