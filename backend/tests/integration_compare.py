"""
擬真整合測試：比較 /api/rag_chat (傳統) vs /api/agentic_chat (Agentic FSM)
使用同一個 URL、同一個 query，比較 SSE 事件、回答品質、延遲。

用法：
    cd backend && uv run python tests/integration_compare.py

需要：
    - 舊版後端 on port 8000 (docker compose up)
    - 新版後端 on port 8001 (uv run uvicorn app.main:app --port 8001 --reload)
    - Ollama + ChromaDB 都在跑
    - hwmonitor-mqtt 已 ingest 進知識庫
"""

import asyncio
import json
import time

import httpx

OLD_BASE = "http://localhost:8000"
NEW_BASE = "http://localhost:8001"

# 明確指定模型（必須在 Ollama 中已存在）
CHAT_MODEL = "gemma3:12b"

TARGET_URL = "https://github.com/jhihweijhan/hwmonitor-mqtt"
TARGET_RELATION = "github.com"

# 測試情境：從簡單到邊界
TEST_CASES = [
    {
        "name": "基本語意查詢",
        "query": "hwmonitor-mqtt 這個專案是做什麼的？",
        "relation_groups": [TARGET_RELATION],
        "tags": [],
        "expect": "應該提到硬體監控和 MQTT",
    },
    {
        "name": "具體技術細節",
        "query": "hwmonitor-mqtt 要怎麼用 Docker 部署？",
        "relation_groups": [TARGET_RELATION],
        "tags": ["Docker"],
        "expect": "應該提到 docker-compose 或 Dockerfile",
    },
    {
        "name": "跨文件查詢（無 filter）",
        "query": "我有哪些 Python 相關的筆記？",
        "relation_groups": [],
        "tags": ["Python"],
        "expect": "應列出多個 Python 相關文件",
    },
    {
        "name": "知識庫不存在的主題",
        "query": "Kubernetes 的 pod 自動擴縮怎麼設定？",
        "relation_groups": [],
        "tags": [],
        "expect": "Agentic 應回 NO_ANSWER 或無充分證據",
    },
    {
        "name": "閒聊（非知識查詢）",
        "query": "你好，你是誰？",
        "relation_groups": [],
        "tags": [],
        "expect": "Agentic 應路由為 direct_chat",
    },
]


def parse_sse_events(raw: str) -> list[dict]:
    """解析 SSE 文本為事件列表"""
    events = []
    current_event = None
    current_data = []

    for line in raw.split("\n"):
        if line.startswith("event: "):
            if current_event and current_data:
                data_str = "\n".join(current_data)
                try:
                    data = json.loads(data_str)
                except (json.JSONDecodeError, ValueError):
                    data = data_str
                events.append({"event": current_event, "data": data})
            current_event = line[7:].strip()
            current_data = []
        elif line.startswith("data: "):
            current_data.append(line[6:])
        elif line == "" and current_event and current_data:
            data_str = "\n".join(current_data)
            try:
                data = json.loads(data_str)
            except (json.JSONDecodeError, ValueError):
                data = data_str
            events.append({"event": current_event, "data": data})
            current_event = None
            current_data = []

    if current_event and current_data:
        data_str = "\n".join(current_data)
        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, ValueError):
            data = data_str
        events.append({"event": current_event, "data": data})

    return events


async def call_rag_endpoint(base_url: str, query: str, relation_groups: list, tags: list) -> dict:
    """呼叫 /api/rag_chat 並收集 SSE 結果"""
    payload = {"query": query, "model": CHAT_MODEL, "top_k": 5}
    if relation_groups:
        payload["relation_groups"] = relation_groups
    if tags:
        payload["tags"] = tags

    start = time.monotonic()
    raw_text = ""
    first_token_time = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", f"{base_url}/api/rag_chat", json=payload) as resp:
            async for chunk in resp.aiter_text():
                raw_text += chunk
                if first_token_time is None and "event: token" in chunk:
                    first_token_time = time.monotonic() - start

    elapsed = time.monotonic() - start
    events = parse_sse_events(raw_text)

    tokens = [e["data"] for e in events if e["event"] == "token"]
    answer = "".join(str(t) for t in tokens)

    meta = next((e["data"] for e in events if e["event"] == "meta"), {})
    sources = next((e["data"] for e in events if e["event"] == "sources"), [])

    return {
        "answer": answer,
        "events": events,
        "event_types": [e["event"] for e in events],
        "elapsed_s": round(elapsed, 2),
        "first_token_s": round(first_token_time, 2) if first_token_time else None,
        "meta": meta,
        "sources": sources,
        "answer_len": len(answer),
    }


async def call_agentic_endpoint(base_url: str, query: str, relation_groups: list, tags: list) -> dict:
    """呼叫 /api/agentic_chat 並收集 SSE 結果"""
    payload = {"query": query, "model": CHAT_MODEL, "top_k": 5}
    if relation_groups:
        payload["relation_groups"] = relation_groups
    if tags:
        payload["tags"] = tags

    start = time.monotonic()
    raw_text = ""
    first_token_time = None

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", f"{base_url}/api/agentic_chat", json=payload) as resp:
            async for chunk in resp.aiter_text():
                raw_text += chunk
                if first_token_time is None and "event: token" in chunk:
                    first_token_time = time.monotonic() - start

    elapsed = time.monotonic() - start
    events = parse_sse_events(raw_text)

    tokens = [e["data"] for e in events if e["event"] == "token"]
    answer = "".join(str(t) for t in tokens)

    meta = next((e["data"] for e in events if e["event"] == "meta"), {})
    sources = next((e["data"] for e in events if e["event"] == "sources"), [])
    transitions = [e["data"] for e in events if e["event"] == "transition"]
    evidence = next((e["data"] for e in events if e["event"] == "evidence"), {})
    statuses = [e["data"] for e in events if e["event"] == "status"]
    done = next((e["data"] for e in events if e["event"] == "done"), {})

    return {
        "answer": answer,
        "events": events,
        "event_types": [e["event"] for e in events],
        "elapsed_s": round(elapsed, 2),
        "first_token_s": round(first_token_time, 2) if first_token_time else None,
        "meta": meta,
        "sources": sources,
        "transitions": transitions,
        "evidence": evidence,
        "statuses": statuses,
        "done": done,
        "answer_len": len(answer),
    }


def print_divider(char="=", width=80):
    print(char * width)


def print_comparison(case: dict, rag_result: dict, agentic_result: dict):
    print_divider()
    print(f"  測試情境: {case['name']}")
    print(f"  查詢: {case['query']}")
    print(f"  期望: {case['expect']}")
    print_divider("-")

    # --- 延遲比較 ---
    print(f"\n  [延遲比較]")
    print(f"  {'':20s} {'傳統 RAG':>12s}  {'Agentic RAG':>12s}")
    print(f"  {'總耗時':20s} {rag_result['elapsed_s']:>10.2f}s  {agentic_result['elapsed_s']:>10.2f}s")
    print(f"  {'首 Token':20s} {str(rag_result['first_token_s']):>11s}s  {str(agentic_result['first_token_s']):>11s}s")
    print(f"  {'回答長度':20s} {rag_result['answer_len']:>10d}字  {agentic_result['answer_len']:>10d}字")

    overhead = agentic_result["elapsed_s"] - rag_result["elapsed_s"]
    print(f"  Agentic 額外耗時: {overhead:+.2f}s")

    # --- SSE 事件比較 ---
    print(f"\n  [SSE 事件流]")
    print(f"  傳統: {' -> '.join(rag_result['event_types'][:10])}")
    agentic_types = agentic_result["event_types"]
    if len(agentic_types) > 15:
        print(f"  Agentic: {' -> '.join(agentic_types[:8])} ... {' -> '.join(agentic_types[-3:])}")
    else:
        print(f"  Agentic: {' -> '.join(agentic_types)}")

    # --- Agentic 獨有資訊 ---
    print(f"\n  [Agentic FSM 資訊]")
    for t in agentic_result.get("transitions", []):
        reason = f" ({t['reason']})" if "reason" in t else ""
        print(f"    {t.get('from', '?')} -> {t.get('to', '?')}{reason}")

    ev = agentic_result.get("evidence", {})
    if ev:
        detail = ev.get("detail", {}) or {}
        print(f"    證據判定: {ev.get('verdict', '?')}")
        print(f"    Heuristic Zone: {detail.get('heuristic_zone', '?')}")
        print(f"    Best Distance: {detail.get('best_distance', '?')}")
        print(f"    Caption Ratio: {detail.get('caption_ratio', '?')}")

    done = agentic_result.get("done", {})
    if done:
        print(f"    Route Action: {done.get('route_action', '?')}")
        print(f"    Refine Rounds: {done.get('refine_rounds', '?')}")

    # --- 回答預覽 ---
    print(f"\n  [傳統 RAG 回答 (前200字)]")
    print(f"  {rag_result['answer'][:200]}...")

    print(f"\n  [Agentic RAG 回答 (前200字)]")
    print(f"  {agentic_result['answer'][:200]}...")

    print()


async def run_single_case(case: dict):
    """執行單個測試情境，傳統和 Agentic 串行（避免 Ollama VRAM 競爭）"""
    print(f"\n  正在執行: {case['name']}...")

    rag_result = await call_rag_endpoint(
        OLD_BASE, case["query"], case["relation_groups"], case["tags"]
    )

    agentic_result = await call_agentic_endpoint(
        NEW_BASE, case["query"], case["relation_groups"], case["tags"]
    )

    print_comparison(case, rag_result, agentic_result)
    return rag_result, agentic_result


async def main():
    print_divider("=")
    print("  SiteGist Agentic RAG 擬真整合測試")
    print(f"  傳統 RAG: {OLD_BASE}/api/rag_chat")
    print(f"  Agentic:  {NEW_BASE}/api/agentic_chat")
    print(f"  測試情境: {len(TEST_CASES)} 個")
    print_divider("=")

    # 健康檢查
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r1 = await client.get(f"{OLD_BASE}/health/readiness")
            r2 = await client.get(f"{NEW_BASE}/health/readiness")
            assert r1.json()["status"] == "ok", f"舊版後端異常: {r1.text}"
            assert r2.json()["status"] == "ok", f"新版後端異常: {r2.text}"
            print("  兩個後端健康檢查通過\n")
        except Exception as e:
            print(f"  健康檢查失敗: {e}")
            print("  請確認兩個後端都在運行")
            return

    all_results = []
    for case in TEST_CASES:
        rag_r, agentic_r = await run_single_case(case)
        all_results.append((case, rag_r, agentic_r))

    # --- 總結 ---
    print_divider("=")
    print("  總結")
    print_divider("-")
    print(f"  {'情境':20s} {'傳統耗時':>10s} {'Agentic耗時':>12s} {'額外開銷':>10s} {'Agentic Route':>15s}")

    total_rag = 0
    total_agentic = 0
    for case, rag_r, agentic_r in all_results:
        overhead = agentic_r["elapsed_s"] - rag_r["elapsed_s"]
        route = agentic_r.get("done", {}).get("route_action", "?")
        print(f"  {case['name']:20s} {rag_r['elapsed_s']:>9.2f}s {agentic_r['elapsed_s']:>11.2f}s {overhead:>+9.2f}s {route:>15s}")
        total_rag += rag_r["elapsed_s"]
        total_agentic += agentic_r["elapsed_s"]

    print_divider("-")
    print(f"  {'合計':20s} {total_rag:>9.2f}s {total_agentic:>11.2f}s {total_agentic - total_rag:>+9.2f}s")
    print(f"  平均額外開銷: {(total_agentic - total_rag) / len(all_results):+.2f}s/query")
    print_divider("=")


if __name__ == "__main__":
    asyncio.run(main())
