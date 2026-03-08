def _normalize_model_name(name: str | None) -> str:
    return (name or "").strip()


def _is_cloud_model(meta: dict) -> bool:
    name = _normalize_model_name(meta.get("name"))
    return bool(meta.get("remote_host")) or name.endswith(":cloud")


def _is_embedding_model(meta: dict) -> bool:
    name = _normalize_model_name(meta.get("name")).lower()
    if "embed" in name:
        return True

    details = meta.get("details") or {}
    family = _normalize_model_name(details.get("family")).lower()
    families = [str(x).lower() for x in (details.get("families") or [])]
    return family == "bert" or "bert" in families


VISION_FAMILIES = {"mllama", "clip", "minicpm"}


def classify_model_category(meta: dict) -> str:
    """Classify a model as 'vision', 'embedding', or 'chat'."""
    if _is_embedding_model(meta):
        return "embedding"

    details = meta.get("details") or {}
    families = {str(x).lower() for x in (details.get("families") or [])}
    family = _normalize_model_name(details.get("family")).lower()

    if families & VISION_FAMILIES or family in VISION_FAMILIES:
        return "vision"

    name = _normalize_model_name(meta.get("name")).lower()
    vision_name_patterns = ("vision", "llava", "minicpm-v", "qwen2.5vl", "qwen3-vl")
    if any(p in name for p in vision_name_patterns):
        return "vision"

    return "chat"


def resolve_chat_model_name(
    requested_model: str | None,
    default_model: str | None,
    available_models: list[dict],
) -> str:
    requested = _normalize_model_name(requested_model)
    default = _normalize_model_name(default_model)
    by_name = {
        _normalize_model_name(m.get("name")): m
        for m in available_models
        if _normalize_model_name(m.get("name"))
    }
    names = set(by_name.keys())
    names.discard("")

    local_chat_names = [
        name
        for name, meta in by_name.items()
        if not _is_cloud_model(meta) and not _is_embedding_model(meta)
    ]

    if requested:
        if requested in names:
            if _is_embedding_model(by_name[requested]):
                raise ValueError(f"指定模型不支援聊天: {requested}")
            return requested
        raise ValueError(f"指定模型不存在: {requested}")

    if default and default in names and not _is_embedding_model(by_name[default]):
        return default

    if local_chat_names:
        return local_chat_names[0]

    any_chat_names = [name for name, meta in by_name.items() if not _is_embedding_model(meta)]
    if any_chat_names:
        return any_chat_names[0]

    if names:
        raise ValueError("目前只有 embedding 模型，請先 pull 一個聊天模型。")

    raise ValueError("Ollama 沒有可用模型，請先 pull 一個本機模型。")
