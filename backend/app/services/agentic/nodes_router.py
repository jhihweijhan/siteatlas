import json
import logging

from app.models.agentic_schemas import AgenticState, QueryPlan, RouteAction

logger = logging.getLogger(__name__)

_ollama_client = None

_VALID_ACTIONS = {"search_knowledge", "direct_chat", "need_clarification"}

_ROUTER_SYSTEM_PROMPT = """\
意圖路由器。只輸出 JSON，不加說明。
action：search_knowledge（預設）| direct_chat（純閒聊）| need_clarification（太模糊）
格式：{"action":"...","semantic_query":"改寫後搜尋語句","tag_filters":["從現有標籤挑選"]}
規則：偏向 search_knowledge；tag_filters 只能從現有標籤選，不可捏造。
範例：
用戶：「Docker volume 怎麼設定？」 標籤：[Docker,Python] → {"action":"search_knowledge","semantic_query":"Docker volume 設定方式","tag_filters":["Docker"]}
用戶：「你好」 → {"action":"direct_chat","semantic_query":"你好","tag_filters":[]}
"""


def init_dependencies(ollama_client):
    global _ollama_client
    _ollama_client = ollama_client


def parse_router_response(raw: str, fallback_query: str) -> tuple[str, str, list[str]]:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Router JSON parse failed, fallback to search_knowledge")
        return "search_knowledge", fallback_query, []

    action = data.get("action", "")
    if action not in _VALID_ACTIONS:
        action = "search_knowledge"

    semantic_query = (data.get("semantic_query") or "").strip()
    if not semantic_query:
        semantic_query = fallback_query

    if len(semantic_query) > 512:
        semantic_query = semantic_query[:512]

    tag_filters = data.get("tag_filters", [])
    if not isinstance(tag_filters, list):
        tag_filters = []
    tag_filters = [t for t in tag_filters if isinstance(t, str) and t.strip()]

    return action, semantic_query, tag_filters


async def _call_router_llm(state: AgenticState) -> str:
    tags_str = ", ".join(state.existing_system_tags) if state.existing_system_tags else "(無)"
    system_content = _ROUTER_SYSTEM_PROMPT + f"\n\n現有標籤：[{tags_str}]"

    messages = [{"role": "system", "content": system_content}]

    history = state.conversation_history[-6:]
    for msg in history:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    messages.append({"role": "user", "content": state.user_message})

    parts = []
    async for chunk in _ollama_client.stream_chat(
        messages=messages,
        model=state.chat_model,
        num_ctx=4096,
        num_predict=state.config.router_num_predict,
    ):
        content = chunk.get("message", {}).get("content", "")
        if content:
            parts.append(content)

    return "".join(parts).strip()


_ACTION_TO_ROUTE = {
    "search_knowledge": RouteAction.SEARCH_KNOWLEDGE,
    "direct_chat": RouteAction.DIRECT_CHAT,
    "need_clarification": RouteAction.NEED_CLARIFICATION,
}

_ROUTE_TO_NODE = {
    RouteAction.SEARCH_KNOWLEDGE: "RETRIEVE",
    RouteAction.DIRECT_CHAT: "GENERATE",
    RouteAction.NEED_CLARIFICATION: "GENERATE",
}


def _has_scope_constraints(state: AgenticState) -> bool:
    ui = state.ui_hard_constraints
    return bool(
        ui.get("relation_groups")
        or ui.get("source_urls")
        or ui.get("tags")
    )


async def execute(state: AgenticState) -> tuple[AgenticState, str]:
    try:
        raw = await _call_router_llm(state)
        action, semantic_query, tag_filters = parse_router_response(raw, state.user_message)
    except Exception:
        logger.warning("Router LLM failed, fallback to search_knowledge", exc_info=True)
        action = "search_knowledge"
        semantic_query = state.user_message
        tag_filters = []

    # Guardrail: 使用者已指定查詢範圍 → 強制 search_knowledge
    if action != "search_knowledge" and _has_scope_constraints(state):
        logger.info("Scope constraints present, overriding %s → search_knowledge", action)
        action = "search_knowledge"

    route_action = _ACTION_TO_ROUTE[action]
    state.route_action = route_action
    state.query_plan = QueryPlan(
        semantic_query=semantic_query,
        tag_filters=tag_filters,
    )
    state.current_node = "ROUTER"

    next_node = _ROUTE_TO_NODE[route_action]
    return state, next_node
