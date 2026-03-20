<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Noto+Sans+TC&weight=900&size=40&pause=1000&color=0F766E&center=true&vCenter=true&width=920&lines=%E5%BE%9E%E7%B6%B2%E9%A0%81%E6%93%B7%E5%8F%96%E5%88%B0%E7%9F%A5%E8%AD%98%E6%AA%A2%E7%B4%A2%E7%9A%84%E5%AE%8C%E6%95%B4%E6%B5%81%E7%A8%8B;SiteAtlas+%E6%95%99%E5%AD%B8%E5%B0%88%E6%A1%88" alt="title" />
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/%E6%95%99%E5%AD%B8%E5%B0%88%E6%A1%88-Learning%20Project-FF6B6B?style=for-the-badge" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Python-FastAPI-3776AB?style=for-the-badge&logo=python&logoColor=white" /></a>
  <a href="#"><img src="https://img.shields.io/badge/JavaScript-Chrome%20Extension-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Vector%20Store-ChromaDB-10B981?style=for-the-badge" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Local%20AI-Ollama-0EA5E9?style=for-the-badge" /></a>
</p>

這不是單一 CRUD 範例，而是一個可以讓學生一路學到 `Chrome Extension`、`content extraction`、`chunking`、`vector search`、`SSE streaming` 與 `agentic state machine` 的完整本地 AI 系統。

```text
[Browser Page]
     │ 擷取 HTML / Readability
     ▼
[Clean Text + Images]
     │ split / tag / embed
     ▼
[Vector Store]
     │ retrieve / evaluate / refine
     ▼
[Agentic Answer]
```

<table>
  <tr>
    <td valign="top" width="33%">
      <b>Theme A<br>瀏覽器擴充與資料擷取</b>
      <br><br>
      - MV3 `service_worker` 訊息路由
      <br>- `chrome.scripting.executeScript`
      <br>- `Readability.js` 萃取主文
      <br>- 圖片抓取與 base64 正規化
      <br>- Side panel UI 狀態同步
    </td>
    <td valign="top" width="33%">
      <b>Theme B<br>RAG 前處理與檢索條件</b>
      <br><br>
      - 中文遞迴切塊 `ChineseTextChunker`
      <br>- chunk overlap 設計
      <br>- `where` filter 組裝
      <br>- tag / relation / source 三層約束
      <br>- 向量檢索結果整形
    </td>
    <td valign="top" width="33%">
      <b>Theme C<br>Agentic 決策與串流回應</b>
      <br><br>
      - `ROUTE → RETRIEVE → EVALUATE`
      <br>- yellow zone LLM evidence check
      <br>- timeout degrade 策略
      <br>- SSE token / status / transition event
      <br>- Pydantic state schema 建模
    </td>
  </tr>
</table>

| Typical Example | This Project |
|:---|:---|
| Todo List 主要在表單與 state | 同時涵蓋 browser、backend、vector DB、LLM pipeline |
| Chat demo 常只有 prompt → answer | 這裡有 ingest、retrieve、evaluate、refine 的完整鏈路 |
| 單頁面 toy project 很難做系統思考 | SiteAtlas 讓學生看到跨模組責任切分與資料流 |

## Quick Start

1. Clone 專案並安裝 backend 依賴。

```bash
git clone <your-repo-url>
cd siteatlas/backend
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

2. 啟動 ChromaDB 與 FastAPI。

```bash
docker compose up -d chromadb
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. 在 Chrome 載入 `extension/`，打開任意網頁後開啟 side panel 測試擷取與問答。

## Learning Roadmap

```text
Level 1 ─ Read (看懂結構)
│
├── 從 `backend/app/main.py` 看 FastAPI lifespan 與 dependency wiring
├── 從 `extension/manifest.json` 看 MV3 權限與 side panel 入口
└── 對照 `backend/app/routers/` 與 `extension/sidepanel/` 理解前後端分工

Level 2 ─ Understand (理解原理)
│
├── 讀 `backend/app/services/chunker.py` 的 `split()` / `_recursive_split()`
├── 讀 `backend/app/services/filter_builder.py` 的 `build_combined_where()`
└── 讀 `backend/app/services/content_extractor.py` 的 HTML → text/image_urls 轉換

Level 3 ─ Trace (追蹤流程)
│
├── 從 `extension/background.js` 的 `extractPage()` 追到內容擷取
├── 從 `backend/app/routers/ingest.py` 追到 chunk / embed / upsert
└── 從 `backend/app/services/agentic/engine.py` 追 `run_agentic_loop()` 的節點轉移

Level 4 ─ Modify (動手改)
│
├── 調整 `ChineseTextChunker` 的 `chunk_size` / `chunk_overlap`
├── 在 `filter_builder.py` 新增 metadata 篩選條件
└── 修改 `STATUS_MESSAGES` 或 evaluator 閾值，觀察 agentic 回答品質變化
```

<details>
<summary><b>演算法亮點</b>（點擊展開）</summary>

### 1. 中文遞迴切塊不是固定長度硬切

`backend/app/services/chunker.py` 先用段落、換行、中文標點逐層退化，盡量保住語意邊界。

```python
for sep in separators:
    if sep and sep in text:
        separator = sep
        break
```

學生會學到：`recursive split` 如何在「長度限制」與「語意完整」之間做工程折衷。

### 2. overlap 不是單純複製尾巴

它會先找標點，再決定從哪裡開始接回下一塊，避免 chunk 接縫太生硬。

```python
overlap_text = prev[-self.chunk_overlap :]
for sep in ["。", "！", "？", "；", "，", " "]:
    idx = overlap_text.find(sep)
    if idx >= 0:
        overlap_text = overlap_text[idx + 1 :]
```

學生會學到：為什麼資料前處理常需要 heuristic，而不是只靠理論上最簡單的切法。

</details>

<details>
<summary><b>系統設計亮點</b>（點擊展開）</summary>

### 1. 檢索條件組裝器很適合拿來教布林查詢

`backend/app/services/filter_builder.py` 把 `relation_groups`、`source_urls`、`tags` 動態組成 ChromaDB `where` 條件。

```python
if len(parts) == 1:
    return parts[0]
return {"$and": parts}
```

學生會學到：如何把 UI 選項轉成查詢語言，並維持函式可測試性。

### 2. 向量資料庫層有做結果整形

`backend/app/services/vector_store.py` 不只 query，還把 `ids / documents / metadatas / distances` 封裝成統一 item 結構。

```python
items.append({
    "id": ids[i],
    "document": docs[i] if i < len(docs) else "",
    "metadata": metadatas[i] if i < len(metadatas) else {},
})
```

學生會學到：資料層 abstraction 為什麼能降低 router 與 service 的耦合。

### 3. 前端不是單純 fetch，而是 browser capability orchestration

`extension/background.js` 要處理 active tab、腳本注入、跨網域圖片抓取與快取。

```javascript
const results = await chrome.scripting.executeScript({
  target: { tabId },
  files: ['content/Readability.js', 'content/content.js']
});
```

學生會學到：Extension 開發中的 capability boundary 與 message routing。

</details>

<details>
<summary><b>Agentic RAG 亮點</b>（點擊展開）</summary>

### 1. 這裡有一個很清楚的小型 state machine

`backend/app/services/agentic/engine.py` 用 `NODE_REGISTRY` 和 `current_node` 驅動流程，而不是把所有判斷塞進一個大函式。

```python
NODE_REGISTRY = {
    "ROUTE": nodes_router.execute,
    "RETRIEVE": nodes_retriever.execute,
    "EVALUATE": nodes_evaluator.execute,
}
```

學生會學到：何時該把 LLM pipeline 寫成節點式流程，而不是線性 script。

### 2. evaluator 有 heuristic zone + LLM fallback

`backend/app/services/agentic/nodes_evaluator.py` 先看 `best_distance` 與 `caption_ratio`，只有黃區才呼叫模型判斷。

```python
elif best_distance < cfg.evidence_auto_pass_threshold:
    verdict = "sufficient"
else:
    verdict = await _call_evaluator_llm(chunks, query, state)
```

學生會學到：如何把 deterministic rule 與 LLM judgment 混合，控制成本與穩定性。

### 3. SSE 事件流適合教可觀測性

這個專案不是只回最後答案，而是把 `status`、`transition`、`token`、`done` 都串流給前端。

```python
yield build_sse_event("status", {...})
yield build_sse_event("transition", transition)
yield build_sse_event("done", _build_done_payload(state))
```

學生會學到：如何把長任務拆成可觀察、可中斷、可除錯的互動流程。

</details>

<details>
<summary><b>建議教學方式</b>（點擊展開）</summary>

### 課堂切法

1. 先跑起來，讓學生看到「頁面擷取 → 知識庫 → 問答」的完整效果。
2. 再拆成三段教：`extension`、`ingest + retrieval`、`agentic loop`。
3. 最後讓學生各自改一個點，例如 chunking 規則、filter 條件、UI 狀態提示。

### 適合的作業題

- 把 `SUPPORTED_FILE_EXTENSIONS` 擴充到更多格式。
- 增加新的 `metadata` filter，例如時間或作者。
- 替 `run_agentic_loop()` 新增一個自訂節點並設計 transition 條件。

</details>

---

支援本地 `Ollama + ChromaDB + Chrome Extension` 的完整學習路徑。授權限制與非商用條款請見 [LICENSE.md](LICENSE.md)。
