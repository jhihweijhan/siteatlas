import json
import logging
import time
from collections.abc import AsyncGenerator

from app.models.agentic_schemas import AgenticState
from app.services.agentic import (
    nodes_evaluator,
    nodes_generator,
    nodes_refiner,
    nodes_retriever,
    nodes_router,
)

logger = logging.getLogger(__name__)

NODE_REGISTRY = {
    "ROUTE": nodes_router.execute,
    "RETRIEVE": nodes_retriever.execute,
    "EVALUATE": nodes_evaluator.execute,
    "REFINE": nodes_refiner.execute,
    "NO_ANSWER": nodes_generator.execute_no_answer,
}

STREAMING_NODES = {"GENERATE"}
TERMINAL_NODES = {"END", "ERROR"}

STATUS_MESSAGES = {
    "ROUTE": "正在分析問題意圖...",
    "RETRIEVE": "正在搜尋知識庫...",
    "EVALUATE": "正在評估搜尋結果...",
    "REFINE": "正在調整搜尋策略...",
    "GENERATE": "正在生成回答...",
    "NO_ANSWER": "知識庫中未找到相關資訊...",
}


def build_sse_event(event_type: str, data) -> str:
    if isinstance(data, (dict, list)):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = str(data)
    return f"event: {event_type}\ndata: {payload}\n\n"


def _build_done_payload(state: AgenticState) -> dict:
    elapsed = time.monotonic() - state.start_time
    return {
        "status": "done",
        "total_duration_ms": round(elapsed * 1000),
        "refine_rounds": state.refine_round,
        "route_action": state.route_action.value if state.route_action else None,
        "evidence_verdict": state.evidence_verdict,
    }


def _build_sources(chunks: list[dict]) -> list[dict]:
    sources = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        sources.append(
            {
                "id": i,
                "title": meta.get("title", ""),
                "url": meta.get("source_url", ""),
                "relation_group": meta.get("relation_group", ""),
                "type": meta.get("type", "text"),
                "distance": c.get("distance"),
            }
        )
    return sources


async def run_agentic_loop(state: AgenticState) -> AsyncGenerator[str, None]:
    current_node = "ROUTE"
    timeout_degraded = False
    done_emitted = False

    while current_node not in TERMINAL_NODES:
        # --- streaming node (GENERATE) — checked before timeout so
        #     timeout-degraded GENERATE actually runs instead of looping ---
        if current_node in STREAMING_NODES:
            yield build_sse_event(
                "status",
                {
                    "node": current_node,
                    "message": STATUS_MESSAGES.get(current_node, ""),
                },
            )
            try:
                async for token in nodes_generator.stream_execute(state):
                    yield build_sse_event("token", token)
            except Exception:
                logger.exception("stream_execute error")
                yield build_sse_event("error", {"message": "生成回答時發生錯誤"})
                current_node = "ERROR"
                continue

            yield build_sse_event("done", _build_done_payload(state))
            done_emitted = True
            current_node = "END"
            continue

        # --- timeout check (after streaming so degraded GENERATE runs) ---
        elapsed = time.monotonic() - state.start_time
        if not timeout_degraded and elapsed > state.config.global_timeout_seconds:
            if state.retrieved_chunks:
                target = "GENERATE"
            else:
                target = "NO_ANSWER"
            yield build_sse_event(
                "transition",
                {
                    "from": current_node,
                    "to": target,
                    "reason": "TIMEOUT_DEGRADE",
                },
            )
            state.transition_log.append(
                {
                    "from": current_node,
                    "to": target,
                    "reason": "TIMEOUT_DEGRADE",
                }
            )
            current_node = target
            timeout_degraded = True
            continue

        # --- normal node ---
        node_func = NODE_REGISTRY.get(current_node)
        if node_func is None:
            yield build_sse_event("error", {"message": f"Unknown node: {current_node}"})
            current_node = "ERROR"
            continue

        yield build_sse_event(
            "status",
            {"node": current_node, "message": STATUS_MESSAGES.get(current_node, "")},
        )

        try:
            state, next_node = await node_func(state)
        except Exception:
            logger.exception("Node %s execution error", current_node)
            yield build_sse_event("error", {"message": f"節點 {current_node} 執行錯誤"})
            current_node = "ERROR"
            continue

        # record transition
        transition = {"from": current_node, "to": next_node}
        state.transition_log.append(transition)
        yield build_sse_event("transition", transition)

        # post-node extra events
        if current_node == "RETRIEVE":
            yield build_sse_event(
                "meta",
                {
                    "retrieved_chunks": len(state.retrieved_chunks),
                    "query_params": state.final_query_params,
                },
            )
            yield build_sse_event("sources", _build_sources(state.retrieved_chunks))

        if current_node == "EVALUATE":
            detail = state.evidence_detail
            yield build_sse_event(
                "evidence",
                {
                    "verdict": state.evidence_verdict,
                    "detail": detail.model_dump() if detail else None,
                },
            )

        # NO_ANSWER: emit token with final_answer + done, then END
        if current_node == "NO_ANSWER":
            yield build_sse_event("token", state.final_answer)
            yield build_sse_event("done", _build_done_payload(state))
            done_emitted = True
            current_node = "END"
            continue

        current_node = next_node

    # Safety: emit done if loop exited without GENERATE/NO_ANSWER doing it
    if not done_emitted:
        yield build_sse_event("done", _build_done_payload(state))
