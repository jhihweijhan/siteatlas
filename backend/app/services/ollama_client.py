import json
import logging
from collections.abc import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=120.0)

    async def check_health(self) -> dict:
        try:
            res = await self._client.get("/api/version", timeout=5.0)
            data = res.json()
            return {"alive": True, "version": data.get("version", "unknown")}
        except Exception as e:  # noqa: BLE001
            return {"alive": False, "error": str(e)}

    async def list_models(self) -> list[dict]:
        res = await self._client.get("/api/tags", timeout=10.0)
        res.raise_for_status()
        data = res.json()
        return data.get("models", [])

    async def embed(self, text: str | list[str], model: str = "bge-m3") -> list[list[float]]:
        texts = [text] if isinstance(text, str) else text

        # Newer Ollama API: /api/embed with batched input
        res = await self._client.post(
            "/api/embed",
            json={"model": model, "input": texts},
            timeout=60.0,
        )
        if res.status_code != 404:
            res.raise_for_status()
            data = res.json()
            return data.get("embeddings", [])

        # Fallback for older Ollama versions: /api/embeddings (single prompt)
        embeddings: list[list[float]] = []
        for item in texts:
            old_res = await self._client.post(
                "/api/embeddings",
                json={"model": model, "prompt": item},
                timeout=60.0,
            )
            old_res.raise_for_status()
            old_data = old_res.json()
            vector = old_data.get("embedding")
            if vector is not None:
                embeddings.append(vector)
        return embeddings

    async def stream_chat(
        self, messages: list[dict], model: str, num_ctx: int = 8192,
        num_predict: int | None = None,
    ) -> AsyncGenerator[dict, None]:
        options: dict = {"num_ctx": num_ctx}
        if num_predict is not None:
            options["num_predict"] = num_predict
        async with self._client.stream(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "options": options,
            },
            timeout=120.0,
        ) as response:
            if response.status_code >= 400:
                raw = await response.aread()
                detail = raw.decode("utf-8", errors="ignore").strip()
                short = detail[:280] if detail else "no response body"
                raise RuntimeError(f"Ollama error ({response.status_code}): {short}")
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    async def describe_image(
        self,
        base64_image: str,
        model: str,
        prompt: str = "Describe what you see in this image concisely.",
        context: str = "",
    ) -> str:
        # Strip data URI prefix if present
        if "," in base64_image and base64_image.startswith("data:"):
            base64_image = base64_image.split(",", 1)[1]

        logger.info(
            "describe_image: model=%s, base64_len=%d, starts_with=%s, context_len=%d",
            model,
            len(base64_image),
            base64_image[:40] if base64_image else "(empty)",
            len(context),
        )

        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\n{prompt}"

        def _extract_error_detail(raw: bytes) -> str:
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text:
                return "no response body"
            try:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    detail = payload.get("error") or payload.get("message")
                    if isinstance(detail, str) and detail.strip():
                        return detail.strip()
            except json.JSONDecodeError:
                pass
            return text

        def _is_retryable(exc: Exception) -> bool:
            if isinstance(exc, (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError)):
                return True
            if isinstance(exc, RuntimeError):
                msg = str(exc)
                return "vision request failed (500)" in msg or "response empty" in msg
            return False

        # Use streaming to avoid timeout during model loading
        import asyncio as _asyncio
        import json as _json

        max_retries = 3
        last_exc: Exception | None = None
        for _attempt in range(max_retries):
            try:
                content_parts: list[str] = []
                done_reason: str | None = None
                eval_count: int | None = None
                async with self._client.stream(
                    "POST",
                    "/api/chat",
                    json={
                        "model": model,
                        "stream": True,
                        "messages": [
                            {
                                "role": "user",
                                "content": full_prompt,
                                "images": [base64_image],
                            }
                        ],
                        # num_predict raised to 600 to accommodate thinking models
                        # (e.g. qwen3-vl) that consume ~200-400 invisible think tokens
                        # before producing visible output.
                        "options": {"num_predict": 600, "repeat_penalty": 1.3, "temperature": 0.3},
                    },
                    timeout=600.0,
                ) as response:
                    if response.status_code >= 400:
                        raw = await response.aread()
                        detail = _extract_error_detail(raw)
                        short = detail[:280]
                        raise RuntimeError(
                            f"Ollama vision request failed ({response.status_code}): {short}"
                        )
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = _json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("error"):
                            raise RuntimeError(str(chunk["error"]))
                        if chunk.get("message", {}).get("content"):
                            content_parts.append(chunk["message"]["content"])
                        if chunk.get("done"):
                            done_reason = chunk.get("done_reason")
                            eval_count = chunk.get("eval_count")
                            break
                caption = "".join(content_parts).strip()
                if caption:
                    return caption
                logger.warning(
                    "describe_image empty: model=%s, base64_len=%d, done_reason=%s, eval_count=%s",
                    model, len(base64_image), done_reason, eval_count,
                )
                raise RuntimeError("Ollama vision response empty")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _is_retryable(exc):
                    raise
                logger.warning(
                    "describe_image retry %d/%d: %s",
                    _attempt + 1, max_retries, exc,
                )
                await _asyncio.sleep(1)
                continue
        raise RuntimeError(str(last_exc) if last_exc else "Ollama vision request failed")

    async def unload_model(self, model: str) -> None:
        res = await self._client.post(
            "/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30.0,
        )
        if res.status_code == 404:
            return
        res.raise_for_status()

    async def pull_model(self, model: str) -> AsyncGenerator[dict, None]:
        async with self._client.stream(
            "POST",
            "/api/pull",
            json={"model": model, "stream": True},
            timeout=None,
        ) as response:
            if response.status_code >= 400:
                raw = await response.aread()
                detail = raw.decode("utf-8", errors="ignore").strip()
                raise RuntimeError(
                    f"pull model '{model}' failed ({response.status_code}): {detail}"
                )
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    async def generate_tags(
        self,
        title: str,
        content_preview: str,
        existing_tags: list[str],
        model: str,
        timeout: float = 30.0,
    ) -> list[str] | None:
        """Use LLM to generate 1-3 tags for a document.

        Returns a list of tag strings, or None on failure.
        - JSON parse failure: retries once
        - Network/timeout error: returns None immediately (no retry)
        """
        tags_str = ", ".join(existing_tags) if existing_tags else "(無)"
        system_prompt = (
            "你是一位文件分類專家。請根據以下文章內容，從「現有標籤」中挑選 1-3 個最符合的標籤。\n"
            "技術專有名詞請保留原文（如 Python、Docker、React），其餘用繁體中文。\n\n"
            "規則：\n"
            "1. 優先從「現有標籤」中選擇\n"
            "2. 只有當現有標籤都不適合時，才創建新標籤\n"
            "3. 新標籤應簡潔（2-5 個字）\n"
            "4. 強制輸出 JSON 格式\n\n"
            f"現有標籤：{tags_str}"
        )
        user_prompt = (
            f"標題：{title}\n"
            f"內容：{content_preview[:500]}\n\n"
            '請輸出：{"tags": ["標籤A", "標籤B"]}'
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3, "num_predict": 100},
        }

        max_attempts = 2  # 1 original + 1 retry on JSON parse failure
        for attempt in range(max_attempts):
            try:
                res = await self._client.post(
                    "/api/chat", json=payload, timeout=timeout
                )
                res.raise_for_status()
                data = res.json()
                raw_content = data.get("message", {}).get("content", "")
                parsed = json.loads(raw_content)
                tags = parsed.get("tags")
                if (
                    isinstance(tags, list)
                    and 1 <= len(tags) <= 3
                    and all(isinstance(t, str) and t.strip() for t in tags)
                ):
                    return [t.strip() for t in tags]
                # Invalid structure — treat as parse failure
                logger.warning("generate_tags: invalid structure: %s", parsed)
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                logger.warning("generate_tags: network/timeout error: %s", e)
                return None
            except json.JSONDecodeError:
                logger.warning(
                    "generate_tags: JSON parse failed (attempt %d/%d)",
                    attempt + 1,
                    max_attempts,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("generate_tags: unexpected error: %s", e)
                return None
        return None

    async def close(self):
        await self._client.aclose()
