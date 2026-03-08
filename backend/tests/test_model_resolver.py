import pytest

from app.services.model_resolver import classify_model_category, resolve_chat_model_name


def test_resolve_uses_requested_model_when_available():
    models = [{"name": "qwen2.5-coder:3b"}, {"name": "phi4:latest"}]
    selected = resolve_chat_model_name("phi4:latest", "qwen2.5:14b", models)
    assert selected == "phi4:latest"


def test_resolve_raises_when_requested_model_not_found():
    models = [{"name": "qwen2.5-coder:3b"}]
    with pytest.raises(ValueError, match="指定模型不存在"):
        resolve_chat_model_name("qwen2.5:14b", "qwen2.5:14b", models)


def test_resolve_falls_back_to_first_local_when_default_missing():
    models = [
        {"name": "glm-5:cloud", "remote_host": "https://ollama.com:443"},
        {"name": "qwen2.5-coder:3b"},
    ]
    selected = resolve_chat_model_name(None, "qwen2.5:14b", models)
    assert selected == "qwen2.5-coder:3b"


def test_resolve_skips_embedding_models_for_chat_fallback():
    models = [
        {"name": "bge-m3:latest", "details": {"family": "bert"}},
        {"name": "qwen2.5-coder:3b", "details": {"family": "qwen2"}},
    ]
    selected = resolve_chat_model_name(None, "qwen2.5:14b", models)
    assert selected == "qwen2.5-coder:3b"


def test_classify_vision_model_mllama():
    meta = {"name": "llama3.2-vision", "details": {"families": ["mllama", "mllama"]}}
    assert classify_model_category(meta) == "vision"


def test_classify_vision_model_clip():
    meta = {"name": "minicpm-v", "details": {"families": ["minicpm", "clip"]}}
    assert classify_model_category(meta) == "vision"


def test_classify_vision_by_name():
    meta = {"name": "llava:13b", "details": {"families": ["llama"]}}
    assert classify_model_category(meta) == "vision"


def test_classify_embedding_model():
    meta = {"name": "bge-m3", "details": {"family": "bert", "families": ["bert"]}}
    assert classify_model_category(meta) == "embedding"


def test_classify_chat_model():
    meta = {"name": "qwen2.5:14b", "details": {"family": "qwen2", "families": ["qwen2"]}}
    assert classify_model_category(meta) == "chat"
