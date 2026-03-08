import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ollama_client import OllamaClient


@pytest.fixture
def client():
    return OllamaClient(base_url="http://localhost:11434")


@pytest.mark.asyncio
async def test_check_health_success(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"version": "0.5.0"}

    with patch.object(client._client, "get", return_value=mock_response):
        result = await client.check_health()
        assert result["alive"] is True
        assert result["version"] == "0.5.0"


@pytest.mark.asyncio
async def test_check_health_failure(client):
    with patch.object(client._client, "get", side_effect=Exception("Connection refused")):
        result = await client.check_health()
        assert result["alive"] is False


@pytest.mark.asyncio
async def test_list_models(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "qwen2.5:14b"}]}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "get", return_value=mock_response):
        models = await client.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "qwen2.5:14b"


# ---------- describe_image ----------

def _make_stream_mock(lines):
    """Create a mock streaming response for client.stream()."""
    async def _aiter_lines():
        for line in lines:
            yield line

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = _aiter_lines

    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
    stream_cm.__aexit__ = AsyncMock(return_value=False)
    return stream_cm


def _make_stream_error_mock(status_code=500, body=b'{"error":"internal server error"}'):
    async def _aread():
        return body

    async def _aiter_lines():
        if False:
            yield ""

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.aread = _aread
    mock_response.aiter_lines = _aiter_lines

    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
    stream_cm.__aexit__ = AsyncMock(return_value=False)
    return stream_cm


@pytest.mark.asyncio
async def test_describe_image_returns_text(client):
    lines = [
        '{"message":{"content":"這是"},"done":false}',
        '{"message":{"content":"一張貓的圖片"},"done":true}',
    ]
    client._client.stream = MagicMock(return_value=_make_stream_mock(lines))
    result = await client.describe_image("aGVsbG8=", model="llava")
    assert result == "這是一張貓的圖片"

    call_kwargs = client._client.stream.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert body["stream"] is True
    assert body["messages"][0]["images"] == ["aGVsbG8="]


@pytest.mark.asyncio
async def test_describe_image_includes_http_error_detail(client):
    client._client.stream = MagicMock(
        return_value=_make_stream_error_mock(
            status_code=500,
            body=b'{"error":"llava decoder crashed"}',
        )
    )
    with pytest.raises(RuntimeError, match=r"vision request failed \(500\): llava decoder crashed"):
        await client.describe_image("aGVsbG8=", model="llava")


@pytest.mark.asyncio
async def test_describe_image_retries_once_on_500_then_succeeds(client):
    lines = ['{"message":{"content":"成功描述"},"done":true}']
    client._client.stream = MagicMock(
        side_effect=[
            _make_stream_error_mock(status_code=500, body=b'{"error":"temporary overload"}'),
            _make_stream_mock(lines),
        ]
    )

    result = await client.describe_image("aGVsbG8=", model="llava")

    assert result == "成功描述"
    assert client._client.stream.call_count == 2


@pytest.mark.asyncio
async def test_describe_image_strips_data_uri_prefix(client):
    lines = ['{"message":{"content":"描述"},"done":true}']
    client._client.stream = MagicMock(return_value=_make_stream_mock(lines))
    await client.describe_image(
        "data:image/png;base64,aGVsbG8=", model="llava"
    )

    call_kwargs = client._client.stream.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert body["messages"][0]["images"] == ["aGVsbG8="]


# ---------- unload_model ----------

@pytest.mark.asyncio
async def test_unload_model_sends_keep_alive_zero(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    client._client.post = AsyncMock(return_value=mock_response)
    await client.unload_model("llava")

    call_kwargs = client._client.post.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert body["model"] == "llava"
    assert body["keep_alive"] == 0


@pytest.mark.asyncio
async def test_unload_model_ignores_404(client):
    mock_response = MagicMock()
    mock_response.status_code = 404

    client._client.post = AsyncMock(return_value=mock_response)
    # Should not raise
    await client.unload_model("nonexistent-model")


# ---------- pull_model ----------

@pytest.mark.asyncio
async def test_pull_model_yields_progress_dicts(client):
    lines = [
        '{"status":"pulling manifest"}',
        '{"status":"downloading","completed":50,"total":100}',
        '{"status":"success"}',
    ]

    # Build a fake async streaming response
    async def _aiter_lines():
        for line in lines:
            yield line

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.aiter_lines = _aiter_lines

    # Use an async context manager mock for client.stream()
    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
    stream_cm.__aexit__ = AsyncMock(return_value=False)

    client._client.stream = MagicMock(return_value=stream_cm)

    results = []
    async for chunk in client.pull_model("llava"):
        results.append(chunk)

    assert len(results) == 3
    assert results[0]["status"] == "pulling manifest"
    assert results[1]["completed"] == 50
    assert results[2]["status"] == "success"


@pytest.mark.asyncio
async def test_pull_model_raises_on_error_status(client):
    mock_response = MagicMock()
    mock_response.status_code = 500

    async def _aread():
        return b"internal server error"

    mock_response.aread = _aread

    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
    stream_cm.__aexit__ = AsyncMock(return_value=False)

    client._client.stream = MagicMock(return_value=stream_cm)

    with pytest.raises(RuntimeError, match="pull.*500"):
        async for _ in client.pull_model("bad-model"):
            pass


# ---------- generate_tags ----------


@pytest.mark.asyncio
async def test_generate_tags_success(client):
    """Valid JSON response returns tag list."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": '{"tags": ["Python", "Docker"]}'},
    }
    mock_response.raise_for_status = MagicMock()

    client._client.post = AsyncMock(return_value=mock_response)

    result = await client.generate_tags(
        title="Python Docker 教學",
        content_preview="本文介紹如何使用 Docker 部署 Python 應用",
        existing_tags=["Python", "Docker", "React"],
        model="qwen2.5:14b",
    )

    assert result == ["Python", "Docker"]

    # Verify request body
    call_kwargs = client._client.post.call_args
    body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert body["model"] == "qwen2.5:14b"
    assert body["stream"] is False
    assert body["format"] == "json"
    assert body["options"]["temperature"] == 0.3
    assert body["options"]["num_predict"] == 100


@pytest.mark.asyncio
async def test_generate_tags_invalid_json_retries_then_returns_none(client):
    """Invalid JSON twice returns None."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": "這不是JSON"},
    }
    mock_response.raise_for_status = MagicMock()

    client._client.post = AsyncMock(return_value=mock_response)

    result = await client.generate_tags(
        title="測試",
        content_preview="測試內容",
        existing_tags=["Python"],
        model="qwen2.5:14b",
    )

    assert result is None
    # Should have been called twice (original + 1 retry)
    assert client._client.post.call_count == 2


@pytest.mark.asyncio
async def test_generate_tags_timeout_returns_none(client):
    """Network timeout returns None immediately (no retry)."""
    import httpx

    client._client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

    result = await client.generate_tags(
        title="測試",
        content_preview="測試內容",
        existing_tags=[],
        model="qwen2.5:14b",
    )

    assert result is None
    # No retry on network/timeout errors
    assert client._client.post.call_count == 1
