import logging
from collections.abc import AsyncGenerator

from app.models.agentic_schemas import AgenticState

logger = logging.getLogger(__name__)

_ollama_client = None

_DIRECT_CHAT_SYSTEM_PROMPT = (
    "你是使用者的個人知識庫助手「摘頁 SiteGist」。"
    "使用者目前在閒聊，不涉及知識庫查詢。"
    "請友善地回應，使用繁體中文。"
)

_NO_ANSWER_TEMPLATE = (
    "很抱歉，根據目前知識庫的內容，沒有找到與您問題相關的資訊。\n\n"
    "建議您可以：\n"
    "1. 嘗試用不同的關鍵字重新提問\n"
    "2. 先將相關網頁加入知識庫\n"
    "3. 縮小搜尋範圍（指定特定標籤或來源）"
)


def init_dependencies(ollama_client):
    global _ollama_client
    _ollama_client = ollama_client


def build_rag_system_prompt() -> str:
    return (
        "你是一個知識庫問答助手。請嚴格根據以下提供的「知識庫內容」來回答使用者的問題。\n\n"
        "規則：\n"
        "1. 只能使用知識庫提供的資訊，不可使用預訓練知識\n"
        "2. 回答時必須標註 [來源: url]，標明資訊出處\n"
        "3. 如果知識庫中沒有相關資訊，請誠實告知\n"
        "4. 使用繁體中文回答\n"
        "5. 回答要簡潔、有條理"
    )


def _format_chunks_as_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(知識庫中無相關內容)"

    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source_url = meta.get("source_url", "未知來源")
        chunk_type = meta.get("type", "text")
        doc = chunk.get("document", "")

        prefix = f"[圖片描述] " if chunk_type == "image_caption" else ""
        parts.append(f"--- 片段 {i} [來源: {source_url}] ---\n{prefix}{doc}")

    return "\n\n".join(parts)


def build_rag_messages(state: AgenticState) -> list[dict]:
    messages = [{"role": "system", "content": build_rag_system_prompt()}]

    history = state.conversation_history[-6:]
    for msg in history:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    context = _format_chunks_as_context(state.retrieved_chunks)
    user_content = f"知識庫內容：\n{context}\n\n使用者問題：{state.user_message}"
    messages.append({"role": "user", "content": user_content})

    return messages


async def stream_execute(state: AgenticState) -> AsyncGenerator[str, None]:
    if state.route_action and state.route_action.value in ("direct_chat", "need_clarification"):
        messages = [
            {"role": "system", "content": _DIRECT_CHAT_SYSTEM_PROMPT},
            *[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in state.conversation_history[-6:]],
            {"role": "user", "content": state.user_message},
        ]
    else:
        messages = build_rag_messages(state)

    async for chunk in _ollama_client.stream_chat(
        messages=messages,
        model=state.chat_model,
    ):
        msg = chunk.get("message", {})
        content = msg.get("content", "")
        # 跳過 thinking model 的 thinking tokens（content 為空但 thinking 有值）
        if content:
            yield content


async def execute_no_answer(state: AgenticState) -> tuple[AgenticState, str]:
    state.final_answer = _NO_ANSWER_TEMPLATE
    state.current_node = "GENERATOR"
    return state, "END"
