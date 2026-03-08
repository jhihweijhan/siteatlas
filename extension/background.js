// Service Worker: Message hub, CORS proxy, streaming relay
// Ref: @chrome-extension-mv3 skill, @ollama-integration skill

// --- Initialization ---
chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });

const DEFAULT_SETTINGS = {
  ollamaUrl: 'http://127.0.0.1:11434',
  model: '',
  embeddingModel: '',
  queryRelationGroups: [],
  querySourceUrls: [],
  queryRelationGroup: '',
  backendUrl: 'http://localhost:8000',
  ragMode: false,
  visionModel: ''
};
const EXTRACT_CACHE_KEY = 'extractedPageCache';
const EXTRACT_CACHE_LIMIT = 15;

// --- Message Router ---
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message, sender).then(sendResponse).catch(err => {
    sendResponse({ error: err.message });
  });
  return true;
});

async function handleMessage(message, sender) {
  switch (message.type) {
    case 'EXTRACT_PAGE':
      return extractPage(message.tabId);
    case 'GET_CACHED_EXTRACT_FOR_ACTIVE_TAB':
      return getCachedExtractForActiveTab();
    case 'EXTRACT_HTML':
      return extractHtml(message.tabId);
    case 'INGEST_PAGE':
      return ingestPage(message.data);
    case 'GET_MODELS':
      return getModels();
    case 'CHECK_SERVER':
      return checkServer();
    case 'GET_SETTINGS':
      return getSettings();
    case 'SAVE_SETTINGS':
      return saveSettings(message.settings);
    case 'PULL_MODEL':
      return pullModel(message.model);
    default:
      return { error: `Unknown message type: ${message.type}` };
  }
}

// --- Page Extraction ---
async function extractPage(tabId) {
  if (!tabId) {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) return { error: 'No active tab found' };
    tabId = tab.id;
  }

  try {
    const tab = await chrome.tabs.get(tabId);
    const tabUrl = tab?.url || '';
    if (
      !tabUrl ||
      tabUrl.startsWith('chrome://') ||
      tabUrl.startsWith('chrome-extension://') ||
      tabUrl.startsWith('edge://') ||
      tabUrl.startsWith('about:') ||
      tabUrl.startsWith('devtools://')
    ) {
      return {
        error: '此頁面類型不支援擷取，請切換到一般 http/https 網頁。'
      };
    }

    const results = await chrome.scripting.executeScript({
      target: { tabId },
      files: ['content/Readability.js', 'content/content.js']
    });

    if (results && results.length > 0) {
      const lastResult = results[results.length - 1];
      if (lastResult.result) {
        // Fetch cross-origin images that canvas couldn't convert
        const imgCount = Array.isArray(lastResult.result.images) ? lastResult.result.images.length : 0;
        console.log('[SiteGist] extractPage: content script returned', imgCount, 'images');
        if (imgCount > 0) {
          await fetchMissingImageData(lastResult.result.images);
          const fetched = lastResult.result.images.filter(i => i.base64).length;
          console.log('[SiteGist] extractPage: fetched base64 for', fetched, '/', imgCount, 'images');
        }
        await cacheExtractedPage(lastResult.result);
        return lastResult.result;
      }
    }
    return { error: 'No extraction result returned' };
  } catch (e) {
    if (
      typeof e?.message === 'string' &&
      e.message.includes('Extension manifest must request permission')
    ) {
      return {
        error: '沒有此網站的存取權限，請重新載入擴充功能後再試。'
      };
    }
    return { error: `Injection failed: ${e.message}` };
  }
}

async function extractHtml(tabId) {
  if (!tabId) {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) return { error: 'No active tab found' };
    tabId = tab.id;
  }

  try {
    const tab = await chrome.tabs.get(tabId);
    const tabUrl = tab?.url || '';
    if (
      !tabUrl ||
      tabUrl.startsWith('chrome://') ||
      tabUrl.startsWith('chrome-extension://') ||
      tabUrl.startsWith('edge://') ||
      tabUrl.startsWith('about:') ||
      tabUrl.startsWith('devtools://')
    ) {
      return {
        error: '此頁面類型不支援擷取，請切換到一般 http/https 網頁。'
      };
    }

    const results = await chrome.scripting.executeScript({
      target: { tabId },
      files: ['content/extract_html.js']
    });

    if (results && results.length > 0) {
      const lastResult = results[results.length - 1];
      if (lastResult.result) {
        return lastResult.result;
      }
    }
    return { error: 'No HTML extraction result returned' };
  } catch (e) {
    if (
      typeof e?.message === 'string' &&
      e.message.includes('Extension manifest must request permission')
    ) {
      return {
        error: '沒有此網站的存取權限，請重新載入擴充功能後再試。'
      };
    }
    return { error: `HTML extraction failed: ${e.message}` };
  }
}

async function fetchMissingImageData(images) {
  const MAX_SIZE_BYTES = 5 * 1024 * 1024;
  const TRANSCODE_MAX_DIMENSION = 1600;
  const TRANSCODE_QUALITY = 0.9;

  async function blobToDataUri(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = () => reject(new Error('FileReader failed'));
      reader.readAsDataURL(blob);
    });
  }

  async function transcodeBlobForVision(blob) {
    if (typeof createImageBitmap !== 'function' || typeof OffscreenCanvas === 'undefined') {
      return blob;
    }

    const bitmap = await createImageBitmap(blob);
    try {
      const maxSide = Math.max(bitmap.width, bitmap.height);
      // Skip transcoding for images already within size/dimension limits —
      // OffscreenCanvas re-encoding can produce data that confuses some
      // vision models (empty responses). Ollama accepts JPEG/PNG/WebP natively.
      if (maxSide <= TRANSCODE_MAX_DIMENSION && blob.size <= MAX_SIZE_BYTES) {
        return blob;
      }
      const scale = maxSide > TRANSCODE_MAX_DIMENSION ? (TRANSCODE_MAX_DIMENSION / maxSide) : 1;
      const width = Math.max(1, Math.round(bitmap.width * scale));
      const height = Math.max(1, Math.round(bitmap.height * scale));
      const canvas = new OffscreenCanvas(width, height);
      const ctx = canvas.getContext('2d', { alpha: false });
      if (!ctx) {
        return blob;
      }
      ctx.drawImage(bitmap, 0, 0, width, height);
      return await canvas.convertToBlob({
        type: 'image/jpeg',
        quality: TRANSCODE_QUALITY
      });
    } finally {
      if (typeof bitmap.close === 'function') {
        bitmap.close();
      }
    }
  }

  const promises = images.map(async (img) => {
    if (img.base64 || !img.needsFetch || !img.src) return;
    try {
      console.log('[SiteGist] fetching image:', img.src.substring(0, 80));
      const res = await fetch(img.src, { signal: AbortSignal.timeout(15000) });
      if (!res.ok) { console.log('[SiteGist] fetch failed:', res.status); return; }
      const contentLength = res.headers.get('content-length');
      if (contentLength && Number(contentLength) > MAX_SIZE_BYTES) return;
      const blob = await res.blob();
      if (!blob.type.startsWith('image/') || blob.size > MAX_SIZE_BYTES) return;
      // Re-encode into normalized JPEG to reduce model-side decode/runtime failures.
      const visionBlob = await transcodeBlobForVision(blob);
      img.base64 = await blobToDataUri(visionBlob);
    } catch (_e) {
      // Network error or timeout — leave without base64
    }
  });
  await Promise.all(promises);
}

async function getCachedExtractForActiveTab() {
  const tab = await getActiveTab();
  const tabUrl = normalizeUrl(tab?.url || '');
  if (!tabUrl) {
    return { page: null, activeUrl: '' };
  }

  const stored = await chrome.storage.local.get(EXTRACT_CACHE_KEY);
  const list = Array.isArray(stored[EXTRACT_CACHE_KEY]) ? stored[EXTRACT_CACHE_KEY] : [];
  const match = list.find((item) => normalizeUrl(item?.url || '') === tabUrl);
  if (!match) {
    return { page: null, activeUrl: tabUrl };
  }

  return {
    activeUrl: tabUrl,
    page: {
      url: match.url || tab.url,
      title: match.title || tab.title || '',
      textContent: String(match.textContent || ''),
      images: Array.isArray(match.images) ? match.images : []
    },
    cachedAt: match.cachedAt || null
  };
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab || null;
}

function normalizeUrl(url) {
  return String(url || '').replace(/\/+$/, '');
}

async function cacheExtractedPage(page) {
  const url = normalizeUrl(page?.url || '');
  if (!url) {
    return;
  }

  // Strip base64 image data before caching to avoid exceeding
  // chrome.storage.local quota (10MB). Only metadata is kept.
  const lightImages = Array.isArray(page.images)
    ? page.images.map(({ base64: _b64, ...meta }) => meta)
    : [];
  const entry = {
    url,
    title: String(page.title || ''),
    textContent: String(page.textContent || ''),
    images: lightImages,
    cachedAt: new Date().toISOString()
  };

  const stored = await chrome.storage.local.get(EXTRACT_CACHE_KEY);
  const list = Array.isArray(stored[EXTRACT_CACHE_KEY]) ? stored[EXTRACT_CACHE_KEY] : [];
  const next = [entry, ...list.filter((item) => normalizeUrl(item?.url || '') !== url)]
    .slice(0, EXTRACT_CACHE_LIMIT);

  await chrome.storage.local.set({ [EXTRACT_CACHE_KEY]: next });
}

// --- Ollama API ---
async function getSettings() {
  const stored = await chrome.storage.local.get('settings');
  const raw = stored.settings || {};
  const merged = { ...DEFAULT_SETTINGS, ...raw };
  let groups = [];

  if (Array.isArray(raw.queryRelationGroups)) {
    groups = normalizeRelationGroupList(raw.queryRelationGroups);
  } else if (typeof raw.queryRelationGroup === 'string') {
    const legacy = normalizeRelationGroup(raw.queryRelationGroup);
    groups = legacy ? [legacy] : [];
  }

  return {
    ...merged,
    queryRelationGroups: groups,
    querySourceUrls: normalizeSourceUrlList(raw.querySourceUrls),
    queryRelationGroup: groups[0] || ''
  };
}

async function saveSettings(settings) {
  const previous = await getSettings();
  const incoming = settings || {};
  const next = { ...previous, ...incoming };
  let groups = [];
  let sourceUrls = [];

  const hasArray = Object.prototype.hasOwnProperty.call(incoming, 'queryRelationGroups');
  const hasLegacy = Object.prototype.hasOwnProperty.call(incoming, 'queryRelationGroup');

  if (hasArray) {
    groups = normalizeRelationGroupList(incoming.queryRelationGroups);
  } else if (hasLegacy) {
    const legacy = normalizeRelationGroup(incoming.queryRelationGroup);
    groups = legacy ? [legacy] : [];
  } else if (Array.isArray(next.queryRelationGroups)) {
    groups = normalizeRelationGroupList(next.queryRelationGroups);
  } else if (typeof next.queryRelationGroup === 'string') {
    const legacy = normalizeRelationGroup(next.queryRelationGroup);
    groups = legacy ? [legacy] : [];
  }

  const hasSourceUrls = Object.prototype.hasOwnProperty.call(incoming, 'querySourceUrls');
  if (hasSourceUrls) {
    sourceUrls = normalizeSourceUrlList(incoming.querySourceUrls);
  } else if (Array.isArray(next.querySourceUrls)) {
    sourceUrls = normalizeSourceUrlList(next.querySourceUrls);
  }

  next.queryRelationGroups = groups;
  next.querySourceUrls = sourceUrls;
  next.queryRelationGroup = groups[0] || '';

  await chrome.storage.local.set({
    settings: next
  });
  return { success: true };
}

function normalizeRelationGroup(value) {
  return String(value || '').trim();
}

function normalizeRelationGroupList(values) {
  if (!Array.isArray(values)) {
    return [];
  }

  const dedup = new Set();
  for (const value of values) {
    const normalized = normalizeRelationGroup(value);
    if (normalized) {
      dedup.add(normalized);
    }
  }
  return [...dedup];
}

function normalizeSourceUrlList(values) {
  if (!Array.isArray(values)) {
    return [];
  }
  const dedup = new Set();
  const result = [];
  for (const value of values) {
    const normalized = String(value || '').trim();
    if (!normalized) continue;
    const key = normalizeUrl(normalized);
    if (!key || dedup.has(key)) continue;
    dedup.add(key);
    result.push(normalized);
  }
  return result;
}

async function getOllamaUrl() {
  const settings = await getSettings();
  return settings.ollamaUrl;
}

async function getModels() {
  const baseUrl = await getOllamaUrl();
  try {
    const res = await fetch(`${baseUrl}/api/tags`, {
      signal: AbortSignal.timeout(5000)
    });
    if (!res.ok) return { error: `Ollama returned ${res.status}` };
    const data = await res.json();
    return { models: data.models || [] };
  } catch (e) {
    return { error: `Cannot connect to Ollama: ${e.message}` };
  }
}

async function checkServer() {
  const baseUrl = await getOllamaUrl();
  try {
    const res = await fetch(`${baseUrl}/api/version`, {
      signal: AbortSignal.timeout(3000)
    });
    if (!res.ok) return { alive: false, error: `HTTP ${res.status}` };
    const data = await res.json();
    return { alive: true, version: data.version };
  } catch (e) {
    return { alive: false, error: e.message };
  }
}

async function ingestPage(data) {
  const settings = await getSettings();
  if (!settings.backendUrl) {
    return { error: 'Backend URL is empty' };
  }

  const payload = { ...(data || {}) };
  if (!payload.embedding_model && settings.embeddingModel) {
    payload.embedding_model = settings.embeddingModel;
  }
  if (!payload.vision_model && settings.visionModel) {
    payload.vision_model = settings.visionModel;
  }

  try {
    const res = await fetch(`${settings.backendUrl}/api/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(30000)
    });

    if (!res.ok) {
      const body = await res.text();
      return { error: `Backend error (${res.status})${body ? `: ${body.slice(0, 200)}` : ''}` };
    }

    return await res.json();
  } catch (e) {
    return { error: `Cannot connect to backend: ${e.message}` };
  }
}

async function pullModel(model) {
  const settings = await getSettings();
  if (!settings.backendUrl) {
    return { error: 'Backend URL is empty' };
  }

  try {
    const res = await fetch(`${settings.backendUrl}/api/models/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model })
    });

    if (!res.ok) {
      const text = await res.text();
      return { error: `Pull failed (${res.status}): ${text.slice(0, 200)}` };
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let lastEvent = {};

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            lastEvent = JSON.parse(line.slice(6));
          } catch (_e) { /* ignore */ }
        }
      }
    }

    return lastEvent;
  } catch (e) {
    return { error: `Pull error: ${e.message}` };
  }
}

// --- Streaming Chat via Port ---
chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== 'ollama-stream') return;

  let abortController = null;

  port.onMessage.addListener(async (msg) => {
    if (msg.type === 'CHAT_REQUEST') {
      abortController = new AbortController();
      await streamChat(port, msg.messages, msg.model, abortController.signal);
    } else if (msg.type === 'ABORT') {
      if (abortController) abortController.abort();
    }
  });

  port.onDisconnect.addListener(() => {
    if (abortController) abortController.abort();
  });
});

async function streamChat(port, messages, model, signal) {
  const settings = await getSettings();
  const baseUrl = settings.ollamaUrl;

  if (!model) model = settings.model;
  if (!model) {
    port.postMessage({ type: 'ERROR', error: '請先在設定中選擇模型' });
    return;
  }
  if (typeof model === 'string' && model.endsWith(':cloud')) {
    port.postMessage({
      type: 'ERROR',
      error: '目前選到雲端模型，需要 Ollama Connect 登入。請改選本機模型再試。'
    });
    return;
  }

  try {
    const res = await fetch(`${baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        messages,
        stream: true,
        options: { num_ctx: 8192 }
      }),
      signal
    });

    if (!res.ok) {
      const text = await res.text();
      if (res.status === 401 && text.includes('unauthorized')) {
        port.postMessage({
          type: 'ERROR',
          error: 'Ollama 回傳 401（雲端模型未登入）。請改選本機模型（如 qwen2.5-coder:3b、phi4:latest）。'
        });
        return;
      }
      port.postMessage({ type: 'ERROR', error: `Ollama error (${res.status}): ${text}` });
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const chunk = JSON.parse(line);
          if (chunk.error) {
            port.postMessage({ type: 'ERROR', error: chunk.error });
            return;
          }
          if (chunk.message?.content) {
            port.postMessage({ type: 'TOKEN', content: chunk.message.content });
          }
          if (chunk.done) {
            port.postMessage({
              type: 'DONE',
              totalDuration: chunk.total_duration,
              evalCount: chunk.eval_count
            });
          }
        } catch (e) {
          // Skip malformed JSON lines
        }
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      port.postMessage({ type: 'ABORTED' });
    } else {
      port.postMessage({ type: 'ERROR', error: e.message });
    }
  }
}
