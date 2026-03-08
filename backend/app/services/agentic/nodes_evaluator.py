import logging
import re

from app.models.agentic_schemas import AgenticState, EvidenceDetail

logger = logging.getLogger(__name__)

_ollama_client = None


def init_dependencies(ollama_client):
    global _ollama_client
    _ollama_client = ollama_client


_CODE_KEYWORDS = re.compile(
    r"程式碼|code|config|設定檔|設定|配置|指令|command|script|腳本|snippet",
    re.IGNORECASE,
)


async def execute(state: AgenticState) -> tuple[AgenticState, str]:
    cfg = state.config
    chunks = state.retrieved_chunks
    query = state.query_plan.semantic_query if state.query_plan else state.user_message

    caption_count = sum(
        1 for c in chunks if c.get("metadata", {}).get("type") == "image_caption"
    )
    caption_ratio = caption_count / len(chunks) if chunks else 0.0
    best_distance = min((c["distance"] for c in chunks), default=1.0)

    verdict: str
    zone: str = ""
    reason: str | None = None

    # 0. scope-pass: 使用者已指定查詢範圍且有結果 → 信任 scope 過濾
    has_scope = bool(
        state.ui_hard_constraints.get("relation_groups")
        or state.ui_hard_constraints.get("source_urls")
        or state.ui_hard_constraints.get("tags")
    )

    # 1. empty
    if not chunks:
        verdict = "insufficient"
        reason = "empty_result"
    elif has_scope:
        verdict = "sufficient"
        zone = "scope_pass"
    # 2. caption dominance
    elif (
        caption_ratio >= cfg.caption_dominance_threshold
        and _CODE_KEYWORDS.search(query)
    ):
        verdict = "insufficient"
        reason = "caption_dominance"
    # 3. red zone
    elif best_distance > cfg.evidence_reject_threshold:
        verdict = "insufficient"
        zone = "red"
    # 4. green zone
    elif best_distance < cfg.evidence_auto_pass_threshold:
        verdict = "sufficient"
        zone = "green"
    # 5. yellow zone — LLM 判斷
    else:
        zone = "yellow"
        try:
            verdict = await _call_evaluator_llm(chunks, query, state)
        except Exception:
            logger.warning("yellow-zone LLM eval failed, fallback to sufficient")
            verdict = "sufficient"

    detail = EvidenceDetail(
        best_distance=best_distance,
        caption_ratio=caption_ratio,
        heuristic_zone=zone,
        reject_reason=reason,
    )

    state.evidence_verdict = verdict
    state.evidence_detail = detail
    state.current_node = "EVALUATOR"

    return state, _next_node(state)


async def _call_evaluator_llm(
    chunks: list[dict], query: str, state: AgenticState
) -> str:
    context_parts = []
    for c in chunks[:3]:
        text = c.get("document", "")[:200]
        context_parts.append(text)
    context = "\n---\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "判斷以下知識庫片段是否足以回答使用者的問題。"
                "只回覆 sufficient 或 insufficient"
            ),
        },
        {
            "role": "user",
            "content": f"問題：{query}\n\n知識庫片段：\n{context}",
        },
    ]

    parts: list[str] = []
    async for chunk in _ollama_client.stream_chat(
        messages, model=state.chat_model, num_ctx=2048,
        num_predict=state.config.evaluator_num_predict,
    ):
        content = chunk.get("message", {}).get("content", "")
        if content:
            parts.append(content)

    answer = "".join(parts).strip().lower()
    if "insufficient" in answer:
        return "insufficient"
    return "sufficient"


def _next_node(state: AgenticState) -> str:
    if state.evidence_verdict == "sufficient":
        return "GENERATE"
    if state.refine_round < state.config.max_refine_rounds:
        return "REFINE"
    return "NO_ANSWER"
