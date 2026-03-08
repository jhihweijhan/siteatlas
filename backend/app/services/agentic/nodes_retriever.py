import asyncio
import logging

from app.models.agentic_schemas import AgenticState, QueryPlan
from app.services.filter_builder import build_combined_where

logger = logging.getLogger(__name__)

_embedding_service = None
_vector_store = None


def init_dependencies(embedding_service, vector_store):
    global _embedding_service, _vector_store
    _embedding_service = embedding_service
    _vector_store = vector_store


def validate_and_merge_filters(
    query_plan: QueryPlan,
    existing_tags: list[str],
    ui_hard: dict,
) -> dict:
    existing_set = set(existing_tags)

    # Tags: UI hard overrides agent; agent tags intersected with existing
    ui_tags = ui_hard.get("tags", [])
    if ui_tags:
        tags = list(ui_tags)
    else:
        tags = [t for t in query_plan.tag_filters if t in existing_set]

    # Relation groups: UI hard overrides agent
    ui_relations = ui_hard.get("relation_groups", [])
    if ui_relations:
        relation_groups = list(ui_relations)
    else:
        relation_groups = list(query_plan.relation_filters)

    # Source URLs: only from UI
    source_urls = list(ui_hard.get("source_urls", []))

    # Page scope is already the narrowest constraint. If a stale relation_group
    # is kept alongside exact page URLs, it can produce an empty intersection
    # even though the target page exists in the knowledge base.
    if source_urls:
        relation_groups = []

    return {
        "tags": tags,
        "relation_groups": relation_groups,
        "source_urls": source_urls,
    }


async def _embed_and_search(query: str, top_k: int, where: dict | None) -> list[dict]:
    embedding = await _embedding_service.embed_query(query)
    results = await asyncio.to_thread(_vector_store.search, embedding, top_k, where)
    return results


async def execute(state: AgenticState) -> tuple[AgenticState, str]:
    query_plan = state.query_plan
    semantic_query = query_plan.semantic_query

    merged = validate_and_merge_filters(
        query_plan,
        existing_tags=state.existing_system_tags,
        ui_hard=state.ui_hard_constraints,
    )

    where = build_combined_where(
        relation_groups=merged["relation_groups"],
        source_urls=merged["source_urls"],
        tags=merged["tags"] or None,
    )

    top_k = state.ui_hard_constraints.get("top_k", 5)
    chunks = await _embed_and_search(semantic_query, top_k, where)

    state.retrieved_chunks = chunks
    state.final_query_params = {
        "semantic_query": semantic_query,
        "where": where,
        "top_k": top_k,
        **merged,
    }

    state.current_node = "RETRIEVER"
    return state, "EVALUATE"
