// Side Panel: Chat UI, settings management, streaming render
// Supports both direct Ollama mode and backend RAG mode.

const $settingsPanel = document.getElementById('settings-panel');
const $btnSettings = document.getElementById('btn-settings');
const $ollamaUrl = document.getElementById('ollama-url');
const $modelSelect = document.getElementById('model-select');
const $btnRefreshModels = document.getElementById('btn-refresh-models');
const $serverStatus = document.getElementById('server-status');
const $btnSaveSettings = document.getElementById('btn-save-settings');

const $pageTitle = document.getElementById('page-title');
const $currentPageRelations = document.getElementById('current-page-relations');
const $btnExtract = document.getElementById('btn-extract');
const $messages = document.getElementById('messages');
const $chatContainer = document.getElementById('chat-container');
const $userInput = document.getElementById('user-input');
const $btnSend = document.getElementById('btn-send');
const $btnStop = document.getElementById('btn-stop');

const $btnCloseSettings = document.getElementById('btn-close-settings');
const $contextBanner = document.getElementById('context-surfacing-banner');
const $contextBannerText = document.getElementById('context-banner-text');
const $btnContextBannerView = document.getElementById('btn-context-banner-view');
const $btnContextBannerClose = document.getElementById('btn-context-banner-close');

const $taskStatusContainer = document.getElementById('task-status');
const $taskState = document.getElementById('task-state');
const $taskText = document.getElementById('task-text');
const $taskProgress = document.getElementById('task-progress');
const $taskError = document.getElementById('task-error');

const $ragToggle = document.getElementById('rag-toggle');
const $agenticToggle = document.getElementById('agentic-toggle');
const $backendUrl = document.getElementById('backend-url');

const $workspaceTabs = [...document.querySelectorAll('.workspace-tab')];
const $panelQa = document.getElementById('panel-qa');
const $panelModel = document.getElementById('panel-model');
const $panelDb = document.getElementById('panel-db');

const $runtimeMode = document.getElementById('runtime-mode');
const $runtimeActiveModel = document.getElementById('runtime-active-model');
const $runtimeActiveEmbeddingModel = document.getElementById('runtime-active-embedding-model');
const $runtimeDefaultModel = document.getElementById('runtime-default-model');
const $runtimeDefaultEmbeddingModel = document.getElementById('runtime-default-embedding-model');
const $runtimeModelSelect = document.getElementById('runtime-model-select');
const $runtimeEmbeddingModelSelect = document.getElementById('runtime-embedding-model-select');
const $btnApplyModel = document.getElementById('btn-apply-model');
const $btnApplyEmbeddingModel = document.getElementById('btn-apply-embedding-model');
const $runtimeVisionModelSelect = document.getElementById('runtime-vision-model-select');
const $btnApplyVisionModel = document.getElementById('btn-apply-vision-model');
const $runtimeActiveVisionModel = document.getElementById('runtime-active-vision-model');
const $modelPullInput = document.getElementById('model-pull-input');
const $btnPullModel = document.getElementById('btn-pull-model');
const $pullProgress = document.getElementById('pull-progress');
const $pullStatusText = document.getElementById('pull-status-text');
const $pullProgressBar = document.getElementById('pull-progress-bar');

const $dbHealth = document.getElementById('db-health');
const $dbDocCount = document.getElementById('db-doc-count');
const $dbChunkCount = document.getElementById('db-chunk-count');
const $knowledgeSection = document.getElementById('knowledge-section');
const $knowledgeList = document.getElementById('knowledge-list');
const $knowledgeSearchInput = document.getElementById('knowledge-search-input');
const $btnKnowledgeExpand = document.getElementById('btn-knowledge-expand');
const $btnKnowledgeCollapse = document.getElementById('btn-knowledge-collapse');
const $knowledgeListContainer = document.getElementById('knowledge-list-container');
const $knowledgeListStats = document.getElementById('knowledge-list-stats');
const $btnDbRefresh = document.getElementById('btn-db-refresh');
const $btnDbDiagnose = document.getElementById('btn-db-diagnose');
const $btnExtensionReload = document.getElementById('btn-extension-reload');
const $dbBackendUrl = document.getElementById('db-backend-url');
const $dbRagMode = document.getElementById('db-rag-mode');
const $dbLastCheck = document.getElementById('db-last-check');
const $dbLastError = document.getElementById('db-last-error');
const $dbErrorDetail = document.getElementById('db-error-detail');
const $fileUploadInput = document.getElementById('file-upload-input');
const $btnFileChoose = document.getElementById('btn-file-choose');
const $btnFileUpload = document.getElementById('btn-file-upload');
const $fileUploadHint = document.getElementById('file-upload-hint');
const $fileUploadSummary = document.getElementById('file-upload-summary');
const $fileUploadList = document.getElementById('file-upload-list');
const $dropzoneContent = document.getElementById('dropzone-content');
const $fileUploadActionsPanel = document.getElementById('file-upload-actions-panel');
const $queryRelationCheckboxes = document.getElementById('query-relation-checkboxes');
const $queryScopeStatus = document.getElementById('query-scope-status');

// Knowledge Detail Modal elements
const $knowledgeDetailModal = document.getElementById('knowledge-detail-modal');
const $btnKdClose = document.getElementById('btn-kd-close');
const $kdTitle = document.getElementById('kd-title');
const $kdUrlLink = document.getElementById('kd-url-link');
const $kdStats = document.getElementById('kd-stats');
const $kdSchemaPreview = document.getElementById('kd-schema-preview');
const $kdRelationInput = document.getElementById('kd-relation-input');
const $btnKdSaveRelation = document.getElementById('btn-kd-save-relation');
const $kdTagsEditor = document.getElementById('kd-tags-editor');
const $kdContentPreview = document.getElementById('kd-content-preview');
const $btnKdDelete = document.getElementById('btn-kd-delete');

let currentKnowledgeItem = null;

let pageContent = null;
let chatHistory = [];
let currentPort = null;
let isStreaming = false;
let currentAbortController = null;
let currentTaskProgress = 0;
let availableModels = [];
let currentWorkspace = 'qa';
let currentActiveModel = '(待命)';
let currentActiveEmbeddingModel = '(待命)';
let ragTimeoutHandle = null;
let manualStopRequested = false;
let ragTimedOut = false;
let selectedRelationGroups = [];
let selectedSourceUrls = [];
let cachedKnowledgeItems = [];
let selectedTags = [];
let pendingUploadFiles = [];

const SYSTEM_PROMPT = `你是一個網頁閱讀助手。使用者會提供一篇網頁的內容，請根據該內容回答使用者的問題。

規則：
1. 只根據提供的網頁內容回答，不要使用你的預訓練知識。
2. 如果網頁內容中找不到答案，請明確告知使用者。
3. 回答時使用繁體中文。
4. 適當使用條列式整理重點。`;

const MAX_CONTENT_CHARS = 4000;
const RAG_TIMEOUT_MS = 120000;
const AUTO_INGEST_ON_EXTRACT = true;
const SUPPORTED_FILE_EXTENSIONS = ['pdf', 'txt', 'md', 'csv'];

function setTaskStatus(state, text, options = {}) {
  const { progress, error } = options;
  const labelMap = {
    idle: '待命',
    working: '進行中',
    success: '完成',
    error: '錯誤'
  };

  if ($taskState) {
    $taskState.textContent = labelMap[state] || '待命';
    $taskState.className = `task-pill ${state}`;
  }
  if ($taskText) {
    $taskText.textContent = text;
  }
  if ($taskProgress && typeof progress === 'number') {
    currentTaskProgress = Math.max(0, Math.min(100, progress));
    $taskProgress.style.width = `${currentTaskProgress}%`;
  }
  if ($taskError) {
    if (error) {
      $taskError.textContent = `⚠ ${error}`;
      $taskError.classList.remove('hidden');
    } else {
      $taskError.textContent = '';
      $taskError.classList.add('hidden');
    }
  }

  if ($taskStatusContainer) {
    if (state === 'idle') {
      setTimeout(() => {
        if ($taskState.textContent === labelMap.idle) {
          $taskStatusContainer.classList.add('hidden');
        }
      }, 3000);
    } else {
      $taskStatusContainer.classList.remove('hidden');
    }
  }
}

function updateTaskProgress(progress, text) {
  const next = Math.max(currentTaskProgress, progress);
  setTaskStatus('working', text, { progress: next });
}

function isCloudModelObject(model) {
  if (!model) return false;
  const name = String(model.name || '');
  return Boolean(model.remote_host) || name.endsWith(':cloud');
}

function isCloudModelName(name) {
  const hit = availableModels.find((m) => m.name === name);
  if (hit) return isCloudModelObject(hit);
  return String(name || '').endsWith(':cloud');
}

function isEmbeddingModelObject(model) {
  if (!model) return false;
  const name = String(model.name || '').toLowerCase();
  if (name.includes('embed') || name.includes('bge')) return true;

  const family = String(model.details?.family || '').toLowerCase();
  const families = Array.isArray(model.details?.families)
    ? model.details.families.map((x) => String(x).toLowerCase())
    : [];
  return family === 'bert' || families.includes('bert') || family === 'nomic-bert';
}

function isVisionModelObject(model) {
  if (!model) return false;
  const name = String(model.name || '').toLowerCase();
  if (name.includes('vision') || name.includes('llava') || name.includes('minicpm-v')
    || name.includes('qwen2.5vl') || name.includes('qwen3-vl')) return true;

  const families = Array.isArray(model.details?.families)
    ? model.details.families.map((x) => String(x).toLowerCase())
    : [];
  const VISION_FAMILIES = new Set(['mllama', 'clip', 'minicpm']);
  return families.some((f) => VISION_FAMILIES.has(f));
}

function activateWorkspaceTab(tabName) {
  const next = ['qa', 'model', 'db'].includes(tabName) ? tabName : 'qa';
  currentWorkspace = next;

  for (const tabBtn of $workspaceTabs) {
    tabBtn.classList.toggle('active', tabBtn.dataset.workspace === next);
  }

  $panelQa?.classList.toggle('hidden', next !== 'qa');
  $panelModel?.classList.toggle('hidden', next !== 'model');
  $panelDb?.classList.toggle('hidden', next !== 'db');
}

function syncRuntimeModelSelect() {
  if (!$runtimeModelSelect || !$modelSelect) return;
  $runtimeModelSelect.innerHTML = $modelSelect.innerHTML;
  $runtimeModelSelect.value = $modelSelect.value || '';
}

function syncRuntimeEmbeddingModelSelect(renderModels) {
  if (!$runtimeEmbeddingModelSelect) return;

  const options = [
    '<option value="">(後端預設嵌入模型)</option>',
    ...renderModels.map((m) => {
      const size = m.details?.parameter_size || '';
      const quant = m.details?.quantization_level || '';
      const suffix = size && quant ? ` (${size} ${quant})` : size ? ` (${size})` : '';
      return `<option value="${escapeHtml(m.name)}">${escapeHtml(m.name + suffix)}</option>`;
    })
  ];

  $runtimeEmbeddingModelSelect.innerHTML = options.join('');

  const savedEmbedding = $runtimeEmbeddingModelSelect.dataset.savedModel;
  if (savedEmbedding) {
    const hit = [...$runtimeEmbeddingModelSelect.options].find((o) => o.value === savedEmbedding);
    if (hit) {
      $runtimeEmbeddingModelSelect.value = savedEmbedding;
      return;
    }
  }

  if ($runtimeEmbeddingModelSelect.value &&
    [...$runtimeEmbeddingModelSelect.options].some((o) => o.value === $runtimeEmbeddingModelSelect.value)) {
    return;
  }

  $runtimeEmbeddingModelSelect.value = '';
}

function getSelectedEmbeddingModel() {
  return $runtimeEmbeddingModelSelect?.value || '';
}

function getSelectedVisionModel() {
  return $runtimeVisionModelSelect?.value || '';
}

function syncRuntimeVisionModelSelect(visionModels = []) {
  if (!$runtimeVisionModelSelect) return;
  const saved = $runtimeVisionModelSelect.dataset.savedModel || '';
  let html = '<option value="">（不使用視覺模型）</option>';
  for (const m of visionModels) {
    const name = m.name || '';
    const params = m.details?.parameter_size || '';
    const sel = name === saved ? ' selected' : '';
    html += `<option value="${name}"${sel}>${name} ${params ? '(' + params + ')' : ''}</option>`;
  }
  $runtimeVisionModelSelect.innerHTML = html;
}

function renderRuntimeInfo(activeModel = null, activeEmbeddingModel = null) {
  const ragMode = Boolean($ragToggle?.checked);
  const defaultChatModel = $modelSelect?.value || '(未選擇)';
  const defaultEmbeddingModel = getSelectedEmbeddingModel() || '(後端預設嵌入模型)';

  if (activeModel !== null) {
    currentActiveModel = activeModel;
  }
  if (activeEmbeddingModel !== null) {
    currentActiveEmbeddingModel = activeEmbeddingModel;
  }

  if ($runtimeMode) {
    $runtimeMode.textContent = ragMode ? 'RAG' : 'Direct';
  }
  if ($runtimeDefaultModel) {
    $runtimeDefaultModel.textContent = defaultChatModel;
  }
  if ($runtimeDefaultEmbeddingModel) {
    $runtimeDefaultEmbeddingModel.textContent = defaultEmbeddingModel;
  }
  if ($runtimeActiveModel) {
    $runtimeActiveModel.textContent = currentActiveModel;
  }
  if ($runtimeActiveEmbeddingModel) {
    $runtimeActiveEmbeddingModel.textContent = currentActiveEmbeddingModel;
  }
}

async function saveCurrentSettings() {
  const settings = {
    ollamaUrl: $ollamaUrl.value.replace(/\/+$/, ''),
    model: $modelSelect.value,
    embeddingModel: getSelectedEmbeddingModel(),
    queryRelationGroups: getSelectedRelationGroups(),
    querySourceUrls: getSelectedSourceUrls(),
    queryTags: selectedTags,
    ragMode: Boolean($ragToggle?.checked),
    agenticMode: Boolean($agenticToggle?.checked),
    backendUrl: ($backendUrl?.value || 'http://localhost:8000').replace(/\/+$/, '')
  };
  return await sendRuntimeMessage({ type: 'SAVE_SETTINGS', settings }, 10000);
}

function getBackendBaseUrl() {
  return ($backendUrl?.value || 'http://localhost:8000').replace(/\/+$/, '');
}

function syncDbDiagnosticsContext() {
  if ($dbBackendUrl) {
    $dbBackendUrl.textContent = getBackendBaseUrl();
  }
  if ($dbRagMode) {
    const ragOn = $ragToggle?.checked;
    const agenticOn = $agenticToggle?.checked;
    $dbRagMode.textContent = ragOn ? (agenticOn ? 'Agentic RAG' : 'RAG') : '停用';
  }
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

function getSelectedRelationGroups() {
  return normalizeRelationGroupList(selectedRelationGroups);
}

function normalizeSourceUrl(value) {
  return String(value || '').trim();
}

function normalizeSourceUrlList(values) {
  if (!Array.isArray(values)) {
    return [];
  }
  const dedup = new Set();
  const result = [];
  for (const value of values) {
    const normalized = normalizeSourceUrl(value);
    if (!normalized) {
      continue;
    }
    const key = normalizeUrl(normalized);
    if (!key || dedup.has(key)) {
      continue;
    }
    dedup.add(key);
    result.push(normalized);
  }
  return result;
}

function getSelectedSourceUrls() {
  return normalizeSourceUrlList(selectedSourceUrls);
}

function normalizeHierarchicalScopeSelection(groups, sourceUrls, items = []) {
  const normalizedGroups = normalizeRelationGroupList(groups);
  const normalizedPages = normalizeSourceUrlList(sourceUrls);

  if (normalizedPages.length === 0) {
    return {
      relationGroups: normalizedGroups,
      sourceUrls: normalizedPages
    };
  }

  const pageGroupByKey = new Map();
  for (const item of items) {
    const url = normalizeSourceUrl(item?.url);
    if (!url) continue;
    pageGroupByKey.set(normalizeUrl(url), normalizeRelationGroup(item?.relation_group) || getDomainFromUrl(url));
  }

  const inferredGroups = [];
  normalizedPages.forEach((url) => {
    const key = normalizeUrl(url);
    const inferredGroup = pageGroupByKey.get(key) || getDomainFromUrl(url);
    if (inferredGroup && !inferredGroups.includes(inferredGroup)) {
      inferredGroups.push(inferredGroup);
    }
  });

  return {
    relationGroups: inferredGroups.length > 0 ? inferredGroups : normalizedGroups,
    sourceUrls: normalizedPages
  };
}

function normalizeSelectedScope(items = cachedKnowledgeItems) {
  const normalized = normalizeHierarchicalScopeSelection(
    selectedRelationGroups,
    selectedSourceUrls,
    items
  );
  selectedRelationGroups = normalized.relationGroups;
  selectedSourceUrls = normalized.sourceUrls;
  return normalized;
}

async function addSourceUrlToQueryScope(url, options = {}) {
  const { persist = true } = options;
  const normalized = normalizeSourceUrl(url);
  if (!normalized) {
    return false;
  }

  const current = getSelectedSourceUrls();
  const key = normalizeUrl(normalized);
  const exists = current.some((item) => normalizeUrl(item) === key);
  if (!exists) {
    selectedSourceUrls = [...current, normalized];
  }
  normalizeSelectedScope(cachedKnowledgeItems);

  syncRelationScopeOptions(cachedKnowledgeItems);
  renderRelationScopeStatus();

  if (persist && !exists) {
    const saved = await saveCurrentSettings();
    if (saved?.error) {
      addErrorMessage(`查詢範圍儲存失敗: ${saved.error}`);
    }
  }

  return !exists;
}

function getSourceLabel(url) {
  const key = normalizeUrl(url);
  const hit = cachedKnowledgeItems.find((item) => normalizeUrl(item.url) === key);
  return hit?.title || url;
}

function formatRelationScope(groups) {
  const list = normalizeRelationGroupList(groups);
  return list.length > 0 ? list.join('、') : '全部資料';
}

function formatHierarchicalScope(groups, sourceUrls) {
  const pages = normalizeSourceUrlList(sourceUrls);
  const inferredGroups = pages.length > 0
    ? [...new Set(pages.map((url) => getDomainFromUrl(url)).filter(Boolean))]
    : [];
  const effectiveGroups = normalizeRelationGroupList(groups).length > 0
    ? normalizeRelationGroupList(groups)
    : inferredGroups;
  const relationText = effectiveGroups.length > 0 ? effectiveGroups.join('、') : '全部資料';
  if (relationText === '全部資料') {
    return '全部資料';
  }
  if (pages.length === 0) {
    return `${relationText}（全部頁面）`;
  }
  return `${relationText}（指定 ${pages.length} 頁）`;
}

function formatSelectedPagePreview(sourceUrls, max = 2) {
  const pages = normalizeSourceUrlList(sourceUrls);
  if (!pages.length) {
    return '';
  }
  const labels = pages.slice(0, max).map((url) => getSourceLabel(url));
  const suffix = pages.length > max ? '…' : '';
  return `${labels.join('、')}${suffix}`;
}

function parseJsonSafe(raw, fallback = null) {
  try {
    return JSON.parse(raw);
  } catch (_e) {
    return fallback;
  }
}

function formatPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '-';
  }
  return `${(value * 100).toFixed(1)}%`;
}

function formatDurationMs(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '-';
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(2)} s`;
  }
  return `${Math.round(value)} ms`;
}

function nsToMs(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return null;
  }
  return value / 1_000_000;
}

function renderRelationScopeStatus() {
  if (!$queryScopeStatus) {
    return;
  }
  const groups = getSelectedRelationGroups();
  const pages = getSelectedSourceUrls();
  const tags = selectedTags || [];

  if (groups.length === 0 && pages.length === 0 && tags.length === 0) {
    $queryScopeStatus.textContent = '目前：全部資料';
    return;
  }

  const parts = [];

  if (groups.length > 0) {
    const preview = groups.slice(0, 3).join('、');
    const suffix = groups.length > 3 ? '…' : '';
    parts.push(`${groups.length} 個群組（${preview}${suffix}）`);
  }

  if (pages.length > 0) {
    const pagePreview = pages.slice(0, 2).map((url) => getSourceLabel(url)).join('、');
    const pageSuffix = pages.length > 2 ? '…' : '';
    parts.push(`${pages.length} 個頁面（${pagePreview}${pageSuffix}）`);
  }

  if (tags.length > 0) {
    const tagPreview = tags.slice(0, 3).join('、');
    const tagSuffix = tags.length > 3 ? '…' : '';
    parts.push(`標籤：${tagPreview}${tagSuffix}`);
  }

  $queryScopeStatus.textContent = `目前：${parts.join(' / ')}`;
}

function updateCurrentPageRelationInfo() {
  if (!$currentPageRelations) {
    return;
  }

  if (!pageContent?.url) {
    $currentPageRelations.textContent = '目前頁面關聯群組：-';
    return;
  }

  const targetUrl = normalizeUrl(pageContent.url);
  const groups = new Set();
  for (const item of cachedKnowledgeItems) {
    if (normalizeUrl(item.url) !== targetUrl) {
      continue;
    }
    const group = normalizeRelationGroup(item.relation_group);
    if (group) {
      groups.add(group);
    }
  }

  const list = [...groups];
  if (!list.length) {
    $currentPageRelations.textContent = '目前頁面關聯群組：尚未建立';
    return;
  }

  const chips = list
    .map((group) => `<span class="relation-chip">${escapeHtml(group)}</span>`)
    .join('');
  $currentPageRelations.innerHTML = `目前頁面關聯群組：${chips}`;
}

let currentRelationSearch = '';
const $queryRelationSearch = document.getElementById('query-relation-search');
const $queryRelationList = document.getElementById('query-relation-list');

function renderRelationList(groupMap, sortedGroups, pageGroupByKey) {
  if (!$queryRelationList) return;

  const searchLow = currentRelationSearch.toLowerCase();
  let html = '';

  const isAllSelected = selectedRelationGroups.length === 0 && selectedSourceUrls.length === 0;

  // "All Data" row
  if (!searchLow) {
    html += `
        <div class="relation-option relation-all-row ${isAllSelected ? 'selected' : ''}" data-action="select-all">
          <span class="relation-group-label" style="text-align: center; width: 100%;">全部資料</span>
        </div>
      `;
  }

  let hasVisibleItems = false;
  const selectedPageKeySet = new Set(selectedSourceUrls.map(url => normalizeUrl(url)));

  for (const group of sortedGroups) {
    const pages = groupMap.get(group) || [];

    // Filter pages by search
    const visiblePages = pages.filter(p => !searchLow || p.title.toLowerCase().includes(searchLow) || p.url.toLowerCase().includes(searchLow));
    const isGroupVisible = !searchLow || group.toLowerCase().includes(searchLow) || visiblePages.length > 0;

    if (!isGroupVisible) continue;
    hasVisibleItems = true;

    const isGroupSelected = selectedRelationGroups.includes(group);

    html += `
      <div class="relation-option relation-group-row ${isGroupSelected ? 'selected' : ''}" data-group="${escapeHtml(group)}">
        <span class="relation-group-label">${escapeHtml(group)}</span>
        <span class="relation-group-count">(${pages.length})</span>
      </div>
    `;

    if (visiblePages.length > 0) {
      html += `<div class="relation-children">`;
      for (const page of visiblePages) {
        const pageKey = normalizeUrl(page.url);
        // Page is visually selected if the group is selected AND (we aren't using explicit pages OR we explicitly selected this page)
        const selectedWithinGroup = [...selectedPageKeySet].filter(key => {
          const matchGroup = pages.find(p => normalizeUrl(p.url) === key);
          return !!matchGroup;
        });
        const useExplicitPages = selectedWithinGroup.length > 0;

        let isPageSelected = false;
        if (isGroupSelected) {
          isPageSelected = useExplicitPages ? selectedPageKeySet.has(pageKey) : true;
        } else if (selectedPageKeySet.has(pageKey)) {
          isPageSelected = true;
        }

        html += `
          <div class="relation-option relation-page-row ${isPageSelected ? 'selected' : ''}" 
               data-group="${escapeHtml(group)}" data-page-url="${escapeHtml(page.url)}">
            <span class="relation-page-label" title="${escapeHtml(page.url)}">${escapeHtml(page.title || page.url)}</span>
          </div>
        `;
      }
      html += `</div>`;
    } else if (!searchLow) {
      html += `<div class="relation-children"><div class="relation-option-empty relation-page-empty">此群組尚無頁面</div></div>`;
    }
  }

  if (!hasVisibleItems) {
    html += `<div class="relation-option-empty">${searchLow ? '找不到符合的群組或網頁' : '尚無可用群組'}</div>`;
  }

  $queryRelationList.innerHTML = html;

  // Event Listeners for Rows
  const bindEvents = () => {
    // Select All
    const allRow = $queryRelationList.querySelector('[data-action="select-all"]');
    if (allRow) {
      allRow.addEventListener('click', () => {
        selectedRelationGroups = [];
        selectedSourceUrls = [];
        renderRelationScopeStatus();
        renderRelationList(groupMap, sortedGroups, pageGroupByKey);
        saveCurrentSettings();
      });
    }

    // Toggle Group
    $queryRelationList.querySelectorAll('.relation-group-row').forEach(row => {
      row.addEventListener('click', (e) => {
        const group = row.getAttribute('data-group');
        const currentlySelected = selectedRelationGroups.includes(group);

        if (currentlySelected) {
          // Deselect group and all its explicit pages
          selectedRelationGroups = selectedRelationGroups.filter(g => g !== group);
          const pagesInGroupKeys = (groupMap.get(group) || []).map(p => normalizeUrl(p.url));
          selectedSourceUrls = selectedSourceUrls.filter(u => !pagesInGroupKeys.includes(normalizeUrl(u)));
        } else {
          // Select group, clear any explicit pages for this group, let it inherit
          selectedRelationGroups.push(group);
          const pagesInGroupKeys = (groupMap.get(group) || []).map(p => normalizeUrl(p.url));
          selectedSourceUrls = selectedSourceUrls.filter(u => !pagesInGroupKeys.includes(normalizeUrl(u)));
        }

        renderRelationScopeStatus();
        renderRelationList(groupMap, sortedGroups, pageGroupByKey);
        saveCurrentSettings();
      });
    });

    // Toggle Page
    $queryRelationList.querySelectorAll('.relation-page-row').forEach(row => {
      row.addEventListener('click', (e) => {
        const group = row.getAttribute('data-group');
        const pageUrl = row.getAttribute('data-page-url');
        const pageKey = normalizeUrl(pageUrl);

        const isGroupSelected = selectedRelationGroups.includes(group);
        const pagesInGroup = groupMap.get(group) || [];
        const pagesInGroupKeys = pagesInGroup.map(p => normalizeUrl(p.url));

        const selectedPageSet = new Set(selectedSourceUrls.map(u => normalizeUrl(u)));
        const explicitPagesInGroup = pagesInGroupKeys.filter(k => selectedPageSet.has(k));

        const isCurrentlySelected = isGroupSelected && (explicitPagesInGroup.length === 0 || selectedPageSet.has(pageKey));

        if (!isGroupSelected) {
          selectedRelationGroups.push(group);
        }

        if (isCurrentlySelected) {
          // Deselect this page
          if (explicitPagesInGroup.length === 0 && isGroupSelected) {
            // We were inheriting all, now we want all EXCEPT this one.
            // Which means we must explicitly select all others.
            pagesInGroupKeys.forEach(k => {
              if (k !== pageKey) {
                const origUrl = pagesInGroup.find(p => normalizeUrl(p.url) === k)?.url;
                if (origUrl) selectedSourceUrls.push(origUrl);
              }
            });
          } else {
            // Simply remove it from explicit list
            selectedSourceUrls = selectedSourceUrls.filter(u => normalizeUrl(u) !== pageKey);
          }

          // If we deselected the ONLY explicit page, the group would revert to "inherit all"
          // But the user intent was to deselect the group ultimately.
          const newExplicitCount = explicitPagesInGroup.filter(k => k !== pageKey).length;
          const willRevertToAll = newExplicitCount === 0;
          if (willRevertToAll && explicitPagesInGroup.length > 0) {
            selectedRelationGroups = selectedRelationGroups.filter(g => g !== group);
          }

        } else {
          // Select this page
          if (explicitPagesInGroup.length === 0 && !isGroupSelected) {
            // First page in group being selected
            selectedSourceUrls.push(pageUrl);
          } else {
            selectedSourceUrls.push(pageUrl);
          }
        }

        // Clean up: if all pages in a group are explicitly selected, we can clear the explicit list and just rely on group selection
        const newSelectedPageSet = new Set(selectedSourceUrls.map(u => normalizeUrl(u)));
        const newExplicitCount = pagesInGroupKeys.filter(k => newSelectedPageSet.has(k)).length;
        if (newExplicitCount === pagesInGroupKeys.length && pagesInGroupKeys.length > 0) {
          selectedSourceUrls = selectedSourceUrls.filter(u => !pagesInGroupKeys.includes(normalizeUrl(u)));
          if (!selectedRelationGroups.includes(group)) {
            selectedRelationGroups.push(group);
          }
        }

        renderRelationScopeStatus();
        renderRelationList(groupMap, sortedGroups, pageGroupByKey);
        saveCurrentSettings();
      });
    });
  };

  bindEvents();
}

function syncRelationScopeOptions(items = []) {
  if (!$queryRelationList) return;

  normalizeSelectedScope(items);

  const groupMap = new Map();
  const pageGroupByKey = new Map();

  for (const item of items) {
    const group = normalizeRelationGroup(item.relation_group) || 'general';
    const url = normalizeSourceUrl(item.url);
    if (!groupMap.has(group)) {
      groupMap.set(group, []);
    }
    if (url) {
      pageGroupByKey.set(normalizeUrl(url), group);
      groupMap.get(group).push({
        url,
        title: String(item.title || '').trim(),
      });
    }
  }

  const selectedGroups = getSelectedRelationGroups();
  const selectedPages = getSelectedSourceUrls();
  const selectedPageKeySet = new Set(selectedPages.map((url) => normalizeUrl(url)));

  for (const key of selectedPageKeySet) {
    const inferredGroup = pageGroupByKey.get(key);
    if (inferredGroup && !selectedGroups.includes(inferredGroup)) {
      selectedGroups.push(inferredGroup);
    }
  }

  const sortedGroups = [...groupMap.keys()].sort((a, b) => a.localeCompare(b, 'zh-Hant'));
  for (const group of selectedGroups) {
    if (!sortedGroups.includes(group)) {
      sortedGroups.push(group);
      groupMap.set(group, []);
    }
  }
  sortedGroups.sort((a, b) => a.localeCompare(b, 'zh-Hant'));

  // Bind Search Event once
  if ($queryRelationSearch && !$queryRelationSearch.dataset.bound) {
    $queryRelationSearch.dataset.bound = 'true';
    $queryRelationSearch.addEventListener('input', (e) => {
      currentRelationSearch = e.target.value;
      renderRelationList(groupMap, sortedGroups, pageGroupByKey);
    });
  }

  renderRelationList(groupMap, sortedGroups, pageGroupByKey);
  renderRelationScopeStatus();
  updateCurrentPageRelationInfo();
}

let currentTagSearch = '';
const $queryTagSearch = document.getElementById('query-tag-search');
const $queryTagsSelected = document.getElementById('query-tags-selected');
const $queryTagSuggestions = document.getElementById('query-tag-suggestions');

function renderTagChips(allTags) {
  if (!$queryTagSearch || !$queryTagsSelected || !$queryTagSuggestions) return;

  const searchLow = currentTagSearch.toLowerCase();

  // Render Selected
  if (selectedTags.length === 0) {
    $queryTagsSelected.innerHTML = '';
  } else {
    $queryTagsSelected.innerHTML = selectedTags.map(tag => `
      <span class="tag-chip active" data-tag-remove="${escapeHtml(tag)}">
        ${escapeHtml(tag)}
        <span class="remove-icon">✕</span>
      </span>
    `).join('');
  }

  // Render Available (Filtered by search and excluding selected)
  const availableTags = [...allTags]
    .filter(tag => !selectedTags.includes(tag))
    .filter(tag => !searchLow || tag.toLowerCase().includes(searchLow))
    .sort();

  if (allTags.size === 0) {
    $queryTagSuggestions.innerHTML = '<span class="scope-status">尚無標籤，系統會在擷取頁面後自動產生</span>';
  } else if (availableTags.length === 0) {
    if (searchLow) {
      $queryTagSuggestions.innerHTML = '<span class="scope-status">找不到符合的標籤</span>';
    } else {
      $queryTagSuggestions.innerHTML = '<span class="scope-status">沒有更多可用標籤</span>';
    }
  } else {
    $queryTagSuggestions.innerHTML = availableTags.map(tag => `
      <span class="tag-chip" data-tag-add="${escapeHtml(tag)}">
        ${escapeHtml(tag)}
      </span>
    `).join('');
  }

  // Event Listeners for Chips
  $queryTagsSelected.querySelectorAll('[data-tag-remove]').forEach(el => {
    el.addEventListener('click', () => {
      const tag = el.getAttribute('data-tag-remove');
      selectedTags = selectedTags.filter(t => t !== tag);
      syncRelationGroupsFromTags();
      renderTagChips(allTags);
      renderRelationScopeStatus();
      syncRelationScopeOptions(cachedKnowledgeItems);
      saveCurrentSettings();
    });
  });

  $queryTagSuggestions.querySelectorAll('[data-tag-add]').forEach(el => {
    el.addEventListener('click', () => {
      const tag = el.getAttribute('data-tag-add');
      if (!selectedTags.includes(tag)) {
        selectedTags.push(tag);
      }
      currentTagSearch = '';
      $queryTagSearch.value = '';
      syncRelationGroupsFromTags();
      renderTagChips(allTags);
      renderRelationScopeStatus();
      syncRelationScopeOptions(cachedKnowledgeItems);
      saveCurrentSettings();
    });
  });
}

function syncRelationGroupsFromTags() {
  if (selectedTags.length === 0) {
    // 沒有標籤篩選時，清除自動勾選的群組
    selectedRelationGroups = [];
    selectedSourceUrls = [];
    return;
  }
  // 找出包含被選標籤的文件所屬 relation groups
  const matchingGroups = new Set();
  for (const item of cachedKnowledgeItems) {
    const itemTags = item.tags || [];
    if (selectedTags.some(t => itemTags.includes(t))) {
      const group = normalizeRelationGroup(item.relation_group) || 'general';
      matchingGroups.add(group);
    }
  }
  selectedRelationGroups = [...matchingGroups];
  selectedSourceUrls = [];
}

function syncTagScopeOptions() {
  const allTags = new Set();
  cachedKnowledgeItems.forEach(item => {
    (item.tags || []).forEach(t => allTags.add(t));
  });

  renderTagChips(allTags);

  if ($queryTagSearch && !$queryTagSearch.dataset.bound) {
    $queryTagSearch.dataset.bound = 'true';
    $queryTagSearch.addEventListener('input', (e) => {
      currentTagSearch = e.target.value;
      renderTagChips(allTags);
    });
  }
}

function setDbActionsDisabled(disabled) {
  if ($btnDbRefresh) $btnDbRefresh.disabled = disabled;
  if ($btnDbDiagnose) $btnDbDiagnose.disabled = disabled;
  if ($btnExtensionReload) $btnExtensionReload.disabled = disabled;
  if ($btnFileChoose) $btnFileChoose.disabled = disabled;
  if ($btnFileUpload) $btnFileUpload.disabled = disabled || pendingUploadFiles.length === 0;
  if ($fileUploadInput) $fileUploadInput.disabled = disabled;
}

function getFileExtension(name) {
  const raw = String(name || '');
  const idx = raw.lastIndexOf('.');
  return idx >= 0 ? raw.slice(idx + 1).toLowerCase() : '';
}

function renderPendingUploadFiles() {
  if (!$fileUploadSummary || !$fileUploadList) return;

  if (!pendingUploadFiles.length) {
    $fileUploadSummary.textContent = '尚未選擇文件';
    $fileUploadList.innerHTML = '<div class="knowledge-empty">尚未選擇文件</div>';
    $fileUploadList.classList.add('hidden');
    if ($fileUploadActionsPanel) $fileUploadActionsPanel.classList.add('hidden');
    if ($btnFileUpload) $btnFileUpload.disabled = true;
    return;
  }

  if ($fileUploadActionsPanel) $fileUploadActionsPanel.classList.remove('hidden');
  $fileUploadList.classList.remove('hidden');

  $fileUploadSummary.textContent = `已選擇 ${pendingUploadFiles.length} 個文件`;
  $fileUploadList.innerHTML = pendingUploadFiles.map((file) => `
    <div class="file-upload-item">
      <div class="file-upload-main">
        <div class="file-upload-name">${escapeHtml(file.name)}</div>
        <div class="file-upload-meta">${escapeHtml(getFileExtension(file.name).toUpperCase() || 'UNKNOWN')} · ${Math.max(1, Math.round(file.size / 1024))} KB</div>
      </div>
      <span class="file-upload-badge queued">待上傳</span>
    </div>
  `).join('');
  if ($btnFileUpload) $btnFileUpload.disabled = false;
}

function renderUploadResults(items = []) {
  if (!$fileUploadList || !$fileUploadSummary) return;
  if (!items.length) {
    renderPendingUploadFiles();
    return;
  }

  if ($fileUploadActionsPanel) $fileUploadActionsPanel.classList.remove('hidden');
  $fileUploadList.classList.remove('hidden');

  const successCount = items.filter((item) => item.status === 'success').length;
  const errorCount = items.filter((item) => item.status === 'error').length;
  const overwrittenCount = items.filter((item) => item.overwritten).length;
  $fileUploadSummary.textContent = `完成 ${successCount} / ${items.length}${errorCount ? `，失敗 ${errorCount}` : ''}${overwrittenCount ? `，覆蓋 ${overwrittenCount}` : ''}`;
  $fileUploadList.innerHTML = items.map((item) => {
    const badgeClass = item.status === 'error'
      ? 'error'
      : item.overwritten
        ? 'overwritten'
        : item.status === 'success'
          ? 'success'
          : 'queued';
    const badgeText = item.status === 'error'
      ? '失敗'
      : item.overwritten
        ? '已覆蓋'
        : item.status === 'success'
          ? '完成'
          : '排隊中';
    return `
      <div class="file-upload-item">
        <div class="file-upload-main">
          <div class="file-upload-name">${escapeHtml(item.file_name || '(未知檔名)')}</div>
          <div class="file-upload-meta">${escapeHtml(item.message || '')}</div>
        </div>
        <span class="file-upload-badge ${badgeClass}">${badgeText}</span>
      </div>
    `;
  }).join('');
}

function markDbCheckTime() {
  if ($dbLastCheck) {
    $dbLastCheck.textContent = new Date().toLocaleString('zh-TW', { hour12: false });
  }
}

function clearDbErrorState() {
  if ($dbLastError) {
    $dbLastError.textContent = '無';
  }
  if ($dbErrorDetail) {
    $dbErrorDetail.textContent = '';
    $dbErrorDetail.classList.add('hidden');
  }
}

function buildDbErrorInfo(error, endpointPath) {
  const baseUrl = getBackendBaseUrl();
  const raw = String(error?.message || error || '未知錯誤');
  const endpoint = `${baseUrl}${endpointPath}`;

  let summary = raw;
  if (raw.includes('Failed to fetch')) {
    summary = `無法連線到後端 (${baseUrl})`;
  } else if (raw.includes('AbortError')) {
    summary = `連線逾時 (${endpointPath})`;
  } else if (raw.startsWith('HTTP ')) {
    summary = `後端回傳 ${raw}`;
  }

  const detail = [
    `時間: ${new Date().toLocaleString('zh-TW', { hour12: false })}`,
    `Endpoint: ${endpoint}`,
    `Backend URL: ${baseUrl}`,
    `RAG 模式: ${$ragToggle?.checked ? '啟用' : '停用'}`,
    `原始錯誤: ${raw}`,
    '',
    '排障建議：',
    `1) 先開啟 ${baseUrl}/health/readiness，確認是否可連線`,
    '2) 檢查側欄設定中的 Backend URL 與實際埠號是否一致',
    '3) 若是 Docker 啟動，請在專案根目錄執行：',
    '   cd backend && docker compose ps',
    '   cd backend && docker compose restart api chromadb'
  ].join('\n');

  return { summary, detail, raw };
}

function showDbError(errorInfo) {
  if ($dbLastError) {
    $dbLastError.textContent = errorInfo.summary;
  }
  if ($dbErrorDetail) {
    $dbErrorDetail.textContent = errorInfo.detail;
    $dbErrorDetail.classList.remove('hidden');
  }
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 8000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function refreshDbStatus(options = {}) {
  const { silent = false } = options;
  const baseUrl = getBackendBaseUrl();
  syncDbDiagnosticsContext();

  try {
    const res = await fetchWithTimeout(`${baseUrl}/health/readiness`, { method: 'GET' }, 7000);
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`HTTP ${res.status}${body ? `: ${body.slice(0, 180)}` : ''}`);
    }

    const data = await res.json();
    if ($dbHealth) {
      $dbHealth.textContent = data.chromadb ? '正常' : '異常';
    }
    if ($fileUploadHint) {
      $fileUploadHint.textContent = '支援 PDF / TXT / MD / CSV，可多選';
    }
    if ($btnFileChoose) $btnFileChoose.disabled = false;
    if ($btnFileUpload) $btnFileUpload.disabled = pendingUploadFiles.length === 0;
    if ($fileUploadInput) $fileUploadInput.disabled = false;
    markDbCheckTime();
    clearDbErrorState();
    return { ok: true, data };
  } catch (error) {
    if ($dbHealth) {
      $dbHealth.textContent = '離線';
    }
    markDbCheckTime();
    const errorInfo = buildDbErrorInfo(error, '/health/readiness');
    showDbError(errorInfo);
    if ($fileUploadHint) {
      $fileUploadHint.textContent = '後端離線，暫時無法上傳文件';
    }
    if ($btnFileChoose) $btnFileChoose.disabled = true;
    if ($btnFileUpload) $btnFileUpload.disabled = true;
    if ($fileUploadInput) $fileUploadInput.disabled = true;
    if (!silent) {
      setTaskStatus('error', '資料庫狀態檢查失敗', { progress: 0, error: errorInfo.summary });
    }
    return { ok: false, error: errorInfo };
  }
}

async function uploadSelectedFiles() {
  if (!pendingUploadFiles.length) {
    addErrorMessage('請先選擇文件');
    return;
  }

  const health = await refreshDbStatus({ silent: true });
  if (!health.ok) {
    addErrorMessage(`無法上傳文件：${health.error.summary}`);
    return;
  }

  setDbActionsDisabled(true);
  setTaskStatus('working', `準備上傳 ${pendingUploadFiles.length} 個文件...`, { progress: 10 });

  try {
    const baseUrl = getBackendBaseUrl();
    const formData = new FormData();
    formData.append('overwrite_mode', 'replace');
    pendingUploadFiles.forEach((file) => {
      formData.append('files', file, file.name);
    });

    const res = await fetchWithTimeout(`${baseUrl}/api/ingest/files`, {
      method: 'POST',
      body: formData
    }, 60000);

    if (!res.ok) {
      const body = await res.text();
      throw new Error(`HTTP ${res.status}${body ? `: ${body.slice(0, 180)}` : ''}`);
    }

    const data = await res.json();
    const items = Array.isArray(data.items) ? data.items.map((item) => ({ ...item })) : [];
    renderUploadResults(items);

    for (let index = 0; index < items.length; index += 1) {
      const item = items[index];
      setTaskStatus('working', `等待文件寫入知識庫 (${index + 1}/${items.length})`, {
        progress: Math.min(90, 20 + Math.round((index / Math.max(items.length, 1)) * 60))
      });
      const waitResult = await waitForKnowledgeIngest({
        taskId: item.task_id,
        targetUrl: `file://${String(item.file_name || '').toLowerCase()}`,
        timeoutMs: 45000,
        intervalMs: 1200
      });
      if (waitResult.ok) {
        item.status = 'success';
        item.message = item.overwritten
          ? '覆蓋舊文件並完成寫入'
          : '文件已完成寫入';
      } else {
        item.status = 'error';
        const failure = buildIngestFailureInfo(waitResult, {
          taskId: item.task_id,
          targetUrl: `file://${String(item.file_name || '').toLowerCase()}`
        });
        item.message = failure.summary;
      }
      renderUploadResults(items);
    }

    pendingUploadFiles = [];
    if ($fileUploadInput) {
      $fileUploadInput.value = '';
    }
    await refreshDbStatus({ silent: true });
    await refreshKnowledgeList({ silent: true });

    const failed = items.filter((item) => item.status === 'error').length;
    if (failed > 0) {
      setTaskStatus('error', '部分文件上傳失敗', {
        progress: 100,
        error: `${failed} 個文件未成功寫入知識庫`
      });
    } else {
      setTaskStatus('success', `文件上傳完成（${items.length} 個）`, { progress: 100 });
    }
  } catch (error) {
    const errorInfo = buildDbErrorInfo(error, '/api/ingest/files');
    showDbError(errorInfo);
    setTaskStatus('error', '文件上傳失敗', { progress: 0, error: errorInfo.summary });
    addErrorMessage(`文件上傳失敗: ${errorInfo.summary}`);
  } finally {
    setDbActionsDisabled(false);
    if (pendingUploadFiles.length > 0) {
      renderPendingUploadFiles();
    }
  }
}

async function init() {
  setTaskStatus('working', '初始化中...', { progress: 10 });
  activateWorkspaceTab('qa');
  renderPendingUploadFiles();
  if ($userInput) {
    $userInput.disabled = false;
  }
  setupEventListeners();
  syncDbDiagnosticsContext();

  await loadSettings();
  await restoreCachedPageForActiveTab({ silent: true });
  updateTaskProgress(25, '載入設定...');

  await checkServerStatus();
  updateTaskProgress(45, '檢查服務狀態...');

  await loadModels();
  updateTaskProgress(100, '初始化完成');

  setTaskStatus('idle', '準備就緒', { progress: 0 });
  renderRuntimeInfo();
  updateSendState();

  await refreshDbStatus();
  await refreshKnowledgeList();
}

function sendRuntimeMessage(message, timeoutMs = 15000) {
  return new Promise((resolve) => {
    let done = false;
    const timer = setTimeout(() => {
      if (done) return;
      done = true;
      resolve({ error: `請求逾時（>${Math.round(timeoutMs / 1000)} 秒）` });
    }, timeoutMs);

    chrome.runtime.sendMessage(message).then((resp) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve(resp || {});
    }).catch((e) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve({ error: e?.message || '訊息傳送失敗' });
    });
  });
}

async function loadSettings() {
  const response = await sendRuntimeMessage({ type: 'GET_SETTINGS' }, 8000);

  if (response.ollamaUrl) {
    $ollamaUrl.value = response.ollamaUrl;
  }
  if (response.model) {
    $modelSelect.dataset.savedModel = response.model;
  }
  if (typeof response.embeddingModel === 'string' && $runtimeEmbeddingModelSelect) {
    $runtimeEmbeddingModelSelect.dataset.savedModel = response.embeddingModel;
  }
  if (Array.isArray(response.queryRelationGroups)) {
    selectedRelationGroups = normalizeRelationGroupList(response.queryRelationGroups);
  } else if (typeof response.queryRelationGroup === 'string') {
    const legacy = normalizeRelationGroup(response.queryRelationGroup);
    selectedRelationGroups = legacy ? [legacy] : [];
  } else {
    selectedRelationGroups = [];
  }
  if (Array.isArray(response.querySourceUrls)) {
    selectedSourceUrls = normalizeSourceUrlList(response.querySourceUrls);
  } else {
    selectedSourceUrls = [];
  }
  selectedTags = Array.isArray(response.queryTags) ? response.queryTags : [];
  if ($ragToggle && typeof response.ragMode === 'boolean') {
    $ragToggle.checked = response.ragMode;
  }
  if ($agenticToggle && typeof response.agenticMode === 'boolean') {
    $agenticToggle.checked = response.agenticMode;
  }
  if ($backendUrl && response.backendUrl) {
    $backendUrl.value = response.backendUrl;
  }

  syncDbDiagnosticsContext();
  syncRelationScopeOptions(cachedKnowledgeItems);
  syncTagScopeOptions();
  renderRuntimeInfo();
}

async function restoreCachedPageForActiveTab(options = {}) {
  const { silent = true } = options;
  const response = await sendRuntimeMessage({ type: 'GET_CACHED_EXTRACT_FOR_ACTIVE_TAB' }, 7000);
  if (response?.error || !response?.page) {
    return false;
  }

  const restored = response.page;
  if (!restored?.textContent) {
    return false;
  }

  pageContent = {
    url: restored.url || '',
    title: restored.title || '已擷取頁面',
    textContent: String(restored.textContent || '')
  };

  if ($pageTitle) {
    $pageTitle.textContent = `${pageContent.title || '已擷取頁面'}（快取）`;
  }
  updateCurrentPageRelationInfo();

  if (!silent) {
    addSystemMessage('已載入此頁先前擷取內容，可直接在 Direct 模式問答。');
  }
  return true;
}

async function getActiveTabUrlFromBackground() {
  const response = await sendRuntimeMessage({ type: 'GET_CACHED_EXTRACT_FOR_ACTIVE_TAB' }, 5000);
  return normalizeUrl(response?.activeUrl || '');
}

async function checkServerStatus() {
  if ($ragToggle?.checked) {
    $serverStatus.textContent = 'RAG 模式';
    $serverStatus.className = 'status-badge online';
    setTaskStatus('idle', 'RAG 模式已啟用', { progress: 0 });
    return;
  }

  const response = await sendRuntimeMessage({ type: 'CHECK_SERVER' }, 8000);
  if (response.alive) {
    $serverStatus.textContent = `已連線 (v${response.version})`;
    $serverStatus.className = 'status-badge online';
    setTaskStatus('idle', `Ollama 已連線 (v${response.version})`, { progress: 0 });
  } else {
    $serverStatus.textContent = `離線: ${response.error || '未知錯誤'}`;
    $serverStatus.className = 'status-badge offline';
    setTaskStatus('error', '服務離線', {
      progress: 0,
      error: response.error || '未知錯誤'
    });
  }
}

function renderChatModelOptions(renderModels) {
  return renderModels
    .map((m) => {
      const size = m.details?.parameter_size || '';
      const quant = m.details?.quantization_level || '';
      const suffix = size && quant ? ` (${size} ${quant})` : size ? ` (${size})` : '';
      return `<option value="${escapeHtml(m.name)}">${escapeHtml(m.name + suffix)}</option>`;
    })
    .join('');
}

async function loadModels() {
  setTaskStatus('working', '載入模型列表...', { progress: 30 });
  $modelSelect.innerHTML = '<option value="">載入中...</option>';

  const response = await sendRuntimeMessage({ type: 'GET_MODELS' }, 12000);
  if (response.error) {
    $modelSelect.innerHTML = '<option value="">無法取得模型列表</option>';
    if ($runtimeModelSelect) {
      $runtimeModelSelect.innerHTML = '<option value="">無法取得模型列表</option>';
    }
    if ($runtimeEmbeddingModelSelect) {
      $runtimeEmbeddingModelSelect.innerHTML = '<option value="">(後端預設嵌入模型)</option>';
    }
    setTaskStatus('error', '模型列表載入失敗', {
      progress: 0,
      error: response.error
    });
    renderRuntimeInfo();
    updateSendState();
    return;
  }

  const models = response.models || [];
  availableModels = models;

  if (models.length === 0) {
    $modelSelect.innerHTML = '<option value="">尚無已安裝的模型</option>';
    syncRuntimeModelSelect();
    syncRuntimeEmbeddingModelSelect([]);
    syncRuntimeVisionModelSelect([]);
    setTaskStatus('idle', '尚未找到可用模型', { progress: 0 });
    renderRuntimeInfo();
    updateSendState();
    return;
  }

  const localModels = models.filter((m) => !isCloudModelObject(m));
  let renderModels = [];
  let renderEmbeddingModels = [];

  if (localModels.length > 0) {
    const localChatModels = localModels.filter((m) => !isEmbeddingModelObject(m));
    const embeddingCandidates = localModels.filter((m) => isEmbeddingModelObject(m));
    renderModels = localChatModels.length > 0 ? localChatModels : localModels;
    renderEmbeddingModels = embeddingCandidates.length > 0 ? embeddingCandidates : localModels;
  } else {
    const cloudChatModels = models.filter((m) => !isEmbeddingModelObject(m));
    const cloudEmbeddingModels = models.filter((m) => isEmbeddingModelObject(m));
    renderModels = cloudChatModels.length > 0 ? cloudChatModels : models;
    renderEmbeddingModels = cloudEmbeddingModels.length > 0 ? cloudEmbeddingModels : models;
  }

  $modelSelect.innerHTML = renderChatModelOptions(renderModels);

  const savedModel = $modelSelect.dataset.savedModel;
  if (savedModel) {
    const option = [...$modelSelect.options].find((o) => o.value === savedModel);
    if (option) {
      option.selected = true;
    }
  } else if ($modelSelect.options.length > 0) {
    $modelSelect.selectedIndex = 0;
  }

  syncRuntimeModelSelect();
  syncRuntimeEmbeddingModelSelect(renderEmbeddingModels);

  const visionCandidates = localModels.filter((m) => isVisionModelObject(m));
  syncRuntimeVisionModelSelect(visionCandidates);

  const cloudCount = models.length - localModels.length;
  if (localModels.length > 0 && cloudCount > 0) {
    addSystemMessage(`已隱藏 ${cloudCount} 個雲端模型，預設使用本機模型。`);
  }

  setTaskStatus('success', `模型載入完成（本機 ${renderModels.length} 個）`, { progress: 100 });
  renderRuntimeInfo();
  updateSendState();
}

function setupEventListeners() {
  $btnSettings.addEventListener('click', () => {
    $settingsPanel.classList.remove('hidden');
  });

  $btnCloseSettings?.addEventListener('click', () => {
    $settingsPanel.classList.add('hidden');
  });

  $btnContextBannerClose?.addEventListener('click', () => {
    $contextBanner.classList.add('hidden');
  });

  $btnContextBannerView?.addEventListener('click', () => {
    activateWorkspaceTab('db');
    $contextBanner.classList.add('hidden');
  });

  $btnSaveSettings.addEventListener('click', async () => {
    setTaskStatus('working', '儲存設定...', { progress: 20 });

    const saved = await saveCurrentSettings();
    if (saved?.error) {
      setTaskStatus('error', '設定儲存失敗', { progress: 0, error: saved.error });
      addErrorMessage(saved.error);
      return;
    }

    await checkServerStatus();
    await refreshDbStatus();
    renderRuntimeInfo();
    updateSendState();
    await refreshKnowledgeList();

    $settingsPanel.classList.add('hidden');
    addSystemMessage('設定已儲存');
    setTaskStatus('success', '設定已儲存', { progress: 100 });
  });

  $btnRefreshModels.addEventListener('click', async () => {
    await loadModels();
  });

  $backendUrl?.addEventListener('input', () => {
    syncDbDiagnosticsContext();
  });

  $modelSelect.addEventListener('change', () => {
    syncRuntimeModelSelect();
    renderRuntimeInfo($modelSelect.value || '(未選擇)');
    updateSendState();
  });

  $runtimeModelSelect?.addEventListener('change', () => {
    $modelSelect.value = $runtimeModelSelect.value;
    renderRuntimeInfo($modelSelect.value || '(未選擇)');
    updateSendState();
  });

  $runtimeEmbeddingModelSelect?.addEventListener('change', () => {
    renderRuntimeInfo(null, getSelectedEmbeddingModel() || '(後端預設嵌入模型)');
  });



  $btnApplyModel?.addEventListener('click', async () => {
    if (!$runtimeModelSelect?.value) {
      addErrorMessage('請先選擇聊天模型');
      return;
    }

    setTaskStatus('working', '切換聊天模型中...', { progress: 35 });
    $modelSelect.value = $runtimeModelSelect.value;

    const saved = await saveCurrentSettings();
    if (saved?.error) {
      setTaskStatus('error', '聊天模型切換失敗', { progress: 0, error: saved.error });
      addErrorMessage(saved.error);
      return;
    }

    renderRuntimeInfo($modelSelect.value || '(未選擇)');
    updateSendState();
    setTaskStatus('success', `已切換聊天模型：${$modelSelect.value}`, { progress: 100 });
  });

  $btnApplyEmbeddingModel?.addEventListener('click', async () => {
    const embeddingModel = getSelectedEmbeddingModel();
    if (embeddingModel && isCloudModelName(embeddingModel)) {
      addErrorMessage('嵌入模型請使用本機模型，避免 Ollama Connect 401 錯誤。');
      return;
    }

    setTaskStatus('working', '切換嵌入模型中...', { progress: 35 });
    const saved = await saveCurrentSettings();
    if (saved?.error) {
      setTaskStatus('error', '嵌入模型切換失敗', { progress: 0, error: saved.error });
      addErrorMessage(saved.error);
      return;
    }

    renderRuntimeInfo(null, embeddingModel || '(後端預設嵌入模型)');
    setTaskStatus('success', `已切換嵌入模型：${embeddingModel || '後端預設'}`, { progress: 100 });
  });

  $btnApplyVisionModel?.addEventListener('click', async () => {
    const selected = $runtimeVisionModelSelect?.value || '';
    await sendRuntimeMessage({ type: 'SAVE_SETTINGS', settings: { visionModel: selected } });
    if ($runtimeActiveVisionModel) {
      $runtimeActiveVisionModel.textContent = selected || '(未設定)';
    }
    addSystemMessage(`視覺模型已更新為：${selected || '(未設定)'}`);
  });

  $btnPullModel?.addEventListener('click', async () => {
    const modelName = $modelPullInput?.value?.trim();
    if (!modelName) return;

    $btnPullModel.disabled = true;
    $pullProgress?.classList.remove('hidden');
    if ($pullStatusText) $pullStatusText.textContent = `正在下載 ${modelName}...`;
    if ($pullProgressBar) $pullProgressBar.style.width = '0%';

    const result = await sendRuntimeMessage({ type: 'PULL_MODEL', model: modelName }, 600000);

    if (result.error) {
      if ($pullStatusText) $pullStatusText.textContent = `下載失敗：${result.error}`;
      addErrorMessage(`模型下載失敗：${result.error}`);
    } else {
      if ($pullStatusText) $pullStatusText.textContent = `${modelName} 下載完成`;
      if ($pullProgressBar) $pullProgressBar.style.width = '100%';
      addSystemMessage(`模型 ${modelName} 下載完成，重新載入模型列表...`);
      await loadModels();
    }

    $btnPullModel.disabled = false;
    setTimeout(() => $pullProgress?.classList.add('hidden'), 3000);
  });

  if ($ragToggle) {
    $ragToggle.addEventListener('change', async () => {
      syncDbDiagnosticsContext();
      await checkServerStatus();
      await refreshDbStatus();
      renderRuntimeInfo();
      updateSendState();
      await refreshKnowledgeList();
    });
  }

  for (const tabBtn of $workspaceTabs) {
    tabBtn.addEventListener('click', async () => {
      const tabName = tabBtn.dataset.workspace || 'qa';
      activateWorkspaceTab(tabName);
      if (tabName === 'model') {
        renderRuntimeInfo();
      }
      if (tabName === 'db') {
        await refreshDbStatus();
        await refreshKnowledgeList();
      }
    });
  }

  $btnDbRefresh?.addEventListener('click', async () => {
    await runDbRecoveryFlow({ includeServerCheck: false, showSuccessMessage: true });
  });

  $btnDbDiagnose?.addEventListener('click', async () => {
    await runDbRecoveryFlow({ includeServerCheck: true, showSuccessMessage: true });
  });

  $btnExtensionReload?.addEventListener('click', () => {
    chrome.runtime.reload();
  });

  if ($dropzoneContent) {
    $dropzoneContent.addEventListener('click', (e) => {
      if (e.target !== $btnFileChoose && !$btnFileChoose?.contains(e.target)) {
        toggleKnowledgeFocus(false);
        $fileUploadInput?.click();
      }
    });
    
    $dropzoneContent.addEventListener('dragover', (e) => {
      e.preventDefault();
      $dropzoneContent.classList.add('dragover');
    });

    $dropzoneContent.addEventListener('dragleave', (e) => {
      e.preventDefault();
      $dropzoneContent.classList.remove('dragover');
    });

    $dropzoneContent.addEventListener('drop', (e) => {
      e.preventDefault();
      $dropzoneContent.classList.remove('dragover');
      toggleKnowledgeFocus(false);
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        if ($fileUploadInput) {
          $fileUploadInput.files = e.dataTransfer.files;
          $fileUploadInput.dispatchEvent(new Event('change'));
        }
      }
    });
  }

  $btnFileChoose?.addEventListener('click', () => {
    toggleKnowledgeFocus(false);
    $fileUploadInput?.click();
  });

  $fileUploadInput?.addEventListener('change', (event) => {
    toggleKnowledgeFocus(false);
    const files = [...(event.target?.files || [])];
    const invalid = files.find((file) => !SUPPORTED_FILE_EXTENSIONS.includes(getFileExtension(file.name)));
    if (invalid) {
      addErrorMessage(`不支援的檔案格式：${invalid.name}`);
      pendingUploadFiles = [];
      renderPendingUploadFiles();
      return;
    }
    pendingUploadFiles = files;
    renderPendingUploadFiles();
  });

  $btnFileUpload?.addEventListener('click', async () => {
    toggleKnowledgeFocus(false);
    await uploadSelectedFiles();
  });

  $btnExtract.addEventListener('click', extractCurrentPage);
  $btnSend.addEventListener('click', sendMessage);

  $userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  $userInput.addEventListener('input', () => {
    $userInput.style.height = 'auto';
    $userInput.style.height = `${Math.min($userInput.scrollHeight, 120)}px`;
    updateSendState();
  });

  $btnStop.addEventListener('click', stopStreaming);
}

function updateSendState() {
  const hasInput = Boolean($userInput.value.trim());
  const hasPage = Boolean(pageContent);
  const ragMode = Boolean($ragToggle?.checked);
  const hasModel = Boolean($modelSelect.value);

  const canSend = ragMode
    ? (!isStreaming && hasInput)
    : (!isStreaming && hasInput && hasPage && hasModel);

  $btnSend.disabled = !canSend;
  $userInput.disabled = isStreaming;

  if (hasPage && !ragMode && !hasModel) {
    addTransientHint('請先在設定中選擇模型');
  }
}

async function extractCurrentPage() {
  setTaskStatus('working', '正在擷取頁面內容...', { progress: 10 });
  $btnExtract.disabled = true;
  $btnExtract.textContent = '擷取中...';
  $pageTitle.textContent = '擷取中...';

  const response = await sendRuntimeMessage({ type: 'EXTRACT_PAGE' }, 30000);

  if (response.error) {
    pageContent = null;
    $pageTitle.textContent = `擷取失敗: ${response.error}`;
    addErrorMessage(response.error);
    setTaskStatus('error', '頁面擷取失敗', { progress: 0, error: response.error });
    updateCurrentPageRelationInfo();
  } else {
    pageContent = response;
    $pageTitle.textContent = response.title || '已擷取';
    updateCurrentPageRelationInfo();
    await addSourceUrlToQueryScope(response.url, { persist: true });

    chatHistory = [];
    $messages.innerHTML = '';

    const textContent = String(response.textContent || '');
    const charCount = textContent.length;
    const imgCount = Array.isArray(response.images) ? response.images.filter(i => i.base64).length : 0;
    const imgTotal = Array.isArray(response.images) ? response.images.length : 0;
    const imgInfo = imgTotal > 0 ? `，${imgCount}/${imgTotal} 張圖片` : '';
    addSystemMessage(`已擷取「${response.title || '未命名頁面'}」（${charCount} 字${imgInfo}）`);
    setTaskStatus('success', `擷取完成（${charCount} 字${imgInfo}）`, { progress: 100 });

    if (AUTO_INGEST_ON_EXTRACT) {
      setTaskStatus('working', '擷取完成，正在自動存入知識庫...', { progress: 30 });
      await saveToKnowledgeBase({ auto: true });
    }
  }

  updateSendState();
  $btnExtract.disabled = false;
  $btnExtract.textContent = '擷取此頁';
}

async function saveToKnowledgeBase(options = {}) {
  const { auto = false } = options;
  if (!pageContent) {
    addErrorMessage('請先擷取頁面內容');
    return;
  }

  const embeddingModel = getSelectedEmbeddingModel();
  if (embeddingModel && isCloudModelName(embeddingModel)) {
    addErrorMessage('嵌入模型目前選到雲端模型，請改用本機模型。');
    return;
  }

  setTaskStatus('working', (auto ? '正在自動擷取頁面' : '正在擷取頁面') + '...', { progress: 10 });

  const pageUrl = pageContent.url || '';
  const domain = getDomainFromUrl(pageUrl);
  const baseline = await refreshKnowledgeList({ silent: true });
  const baselineItem = (baseline.items || []).find((item) => normalizeUrl(item.url) === normalizeUrl(pageUrl));

  // Extract raw HTML for backend trafilatura processing
  const htmlResult = await sendRuntimeMessage({ type: 'EXTRACT_HTML' }, 15000);

  // Build payload: HTML for text extraction (trafilatura), images from frontend (Readability)
  const payload = {
    url: htmlResult?.url || pageUrl,
    title: htmlResult?.title || pageContent.title || document.title,
    language: 'zh-TW',
    domain: domain || 'general',
    relation_group: domain || 'general',
    images: pageContent.images || []
  };

  if (htmlResult && htmlResult.html) {
    payload.html = htmlResult.html;
  } else {
    // Fallback: use Readability-extracted content if HTML extraction fails
    console.warn('[SiteGist] HTML extraction failed, falling back to Readability content');
    payload.content = pageContent.textContent || '';
  }

  if (embeddingModel) {
    payload.embedding_model = embeddingModel;
  }

  let visionModel = getSelectedVisionModel();
  // Auto-select vision model if images are present but no model chosen
  const imgCount = Array.isArray(payload.images) ? payload.images.filter(i => i.base64).length : 0;
  if (!visionModel && imgCount > 0 && $runtimeVisionModelSelect) {
    const visionOpts = $runtimeVisionModelSelect.querySelectorAll('option[value]');
    for (const opt of visionOpts) {
      if (opt.value) { visionModel = opt.value; break; }
    }
  }
  if (visionModel) {
    payload.vision_model = visionModel;
  }

  const imgNote = imgCount > 0 ? `（含 ${imgCount} 張圖片）` : '';
  setTaskStatus('working', (auto ? '正在自動存入知識庫' : '正在存入知識庫') + imgNote + '...', { progress: 20 });

  const baselineChunks = baselineItem ? Number(baselineItem.chunks_count || 0) : null;

  const response = await sendRuntimeMessage({
    type: 'INGEST_PAGE',
    data: payload
  }, 30000);

  if (response.error) {
    addErrorMessage(`知識庫存入失敗: ${response.error}`);
    setTaskStatus('error', '知識庫存入失敗', { progress: 0, error: response.error });
    return;
  }

  renderRuntimeInfo(null, embeddingModel || '(後端預設嵌入模型)');
  addSystemMessage(`已送出知識庫存入：${response.task_id || 'processing'}`);
  setTaskStatus('working', '已送出存入請求，等待寫入完成...', { progress: 40 });

  const imageCount = payload.vision_model ? imgCount : 0;
  // HTML mode adds extra time for trafilatura extraction
  const baseTimeout = payload.html ? 60000 : 45000;
  const ingestTimeout = baseTimeout + imageCount * 30000;

  const waitResult = await waitForKnowledgeIngest({
    taskId: response.task_id || '',
    targetUrl: payload.url,
    previousChunks: baselineChunks,
    timeoutMs: ingestTimeout,
    intervalMs: 1200
  });

  if (!waitResult.ok) {
    const failure = buildIngestFailureInfo(waitResult, {
      taskId: response.task_id || '',
      targetUrl: payload.url
    });
    addErrorMessage(failure.summary);
    showDbError(failure);
    setTaskStatus('error', failure.summary, {
      progress: 0,
      error: failure.summary
    });
    return;
  }

  const seconds = (waitResult.elapsedMs / 1000).toFixed(1);
  const chunkCount = waitResult.item?.chunks_count ?? response.chunks_count ?? '?';
  const imageFrontendTotal = Number(waitResult.task?.image_frontend_total || 0);
  const imageFiltered = Number(waitResult.task?.image_filtered || 0);
  const imageTotal = Number(waitResult.task?.image_total || 0);
  const imageSuccess = Number(waitResult.task?.image_caption_success || 0);
  const imageFailed = Number(waitResult.task?.image_caption_failed || 0);
  const imageFailures = Array.isArray(waitResult.task?.image_caption_failures)
    ? waitResult.task.image_caption_failures
    : [];
  const doneLabel = auto ? '自動存入完成' : '知識庫寫入完成';
  const imageParts = [];
  if (imageFrontendTotal > 0) {
    imageParts.push(`擷取 ${imageFrontendTotal} 張圖片`);
    if (imageFiltered > 0) {
      imageParts.push(`過濾 ${imageFiltered} 張無關圖片`);
    }
    if (imageSuccess > 0 || imageFailed > 0) {
      imageParts.push(`描述 ${imageSuccess}/${imageTotal}`);
    }
    if (imageFailed > 0) {
      imageParts.push(`失敗 ${imageFailed}`);
    }
  }
  const imageNote = imageParts.length > 0 ? `；${imageParts.join('，')}` : '';
  setTaskStatus('success', `${doneLabel}（${chunkCount} chunks, ${seconds}s${imageNote}）`, { progress: 100 });
  if (imageFailed > 0) {
    const details = imageFailures.slice(0, 2).map((f) => {
      const idx = Number.isFinite(Number(f?.index)) ? Number(f.index) : '?';
      const src = String(f?.src || '');
      const shortSrc = src ? src.slice(0, 72) + (src.length > 72 ? '...' : '') : '';
      const err = String(f?.error || 'unknown error');
      return `圖#${idx} ${err}${shortSrc ? ` (${shortSrc})` : ''}`;
    });
    const overflow = imageFailures.length > 2 ? `\n其餘 ${imageFailures.length - 2} 張可在任務詳情查看。` : '';
    addSystemMessage(`圖片描述有 ${imageFailed} 張失敗，已保留可用內容。\n${details.join('\n')}${overflow}`);
  }

  // ── Surfacing: show related docs system message ──
  if (waitResult.task) {
    renderSurfacingMessage(waitResult.task);
  }
}

// ── Knowledge Detail Modal Logic ──
function openKnowledgeModal(item) {
  if (!item || !$knowledgeDetailModal) return;
  currentKnowledgeItem = item;

  $kdTitle.textContent = item.title || '(無標題)';
  $kdUrlLink.href = item.url || '#';
  $kdUrlLink.textContent = item.url || '';

  $kdStats.innerHTML = `
    <span>總計 ${item.chunks_count || 0} chunks</span>
    <span>文字 ${item.text_chunks_count || 0}</span>
    <span>圖片 ${item.image_chunks_count || 0}</span>
  `;

  const schemaKeys = Array.isArray(item.schema_keys) ? item.schema_keys : [];
  $kdSchemaPreview.textContent = `Schema: ${schemaKeys.length ? schemaKeys.join(', ') : '-'}`;

  $kdRelationInput.value = normalizeRelationGroup(item.relation_group) || 'general';

  // Tag Editor Binding (using the existing bindTagEditor but passing the modal container)
  $kdTagsEditor.innerHTML = `
    ${(item.tags || []).map(t =>
    `<span class="tag-badge editable" data-tag-val="${escapeHtml(t)}">${escapeHtml(t)}<span class="tag-remove" title="移除標籤">✕</span></span>`
  ).join('')}
    ${(item.tags || []).length < 3 ? `<button class="tag-add-btn" title="新增標籤">+</button>` : ''}
  `;
  $kdTagsEditor.dataset.tagsEditor = item.id || '';
  $kdTagsEditor.dataset.tags = JSON.stringify(item.tags || []);

  // Re-bind tag actions inside the modal
  bindTagEditor($knowledgeDetailModal, item);

  // Show Loading state for preview
  $kdContentPreview.innerHTML = '載入中...';
  $knowledgeDetailModal.classList.remove('hidden');

  // Fetch full details
  fetchKnowledgeDetail(item.id).then(detail => {
    $kdContentPreview.innerHTML = renderKnowledgeDetailBody(detail);
  }).catch(e => {
    const errorInfo = buildDbErrorInfo(e, `/api/knowledge/${encodeURIComponent(item.id)}`);
    $kdContentPreview.innerHTML = `<div class="knowledge-detail-error">載入失敗: ${escapeHtml(errorInfo.summary)}</div>`;
  });
}

function closeKnowledgeModal() {
  if ($knowledgeDetailModal) {
    $knowledgeDetailModal.classList.add('hidden');
    currentKnowledgeItem = null;
  }
}

if ($btnKdClose) {
  $btnKdClose.addEventListener('click', closeKnowledgeModal);
}

if ($knowledgeDetailModal) {
  $knowledgeDetailModal.addEventListener('click', (e) => {
    // Click on overlay background to close
    if (e.target === $knowledgeDetailModal) {
      closeKnowledgeModal();
    }
  });

  // Handle ESC key to close
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !$knowledgeDetailModal.classList.contains('hidden')) {
      closeKnowledgeModal();
    }
  });
}

if ($btnKdSaveRelation) {
  $btnKdSaveRelation.addEventListener('click', async () => {
    if (!currentKnowledgeItem) return;
    const nextGroup = normalizeRelationGroup($kdRelationInput.value);
    if (!nextGroup) {
      addErrorMessage('relation group 不可為空');
      return;
    }
    const originalText = $btnKdSaveRelation.textContent;
    $btnKdSaveRelation.textContent = '儲存中...';
    $btnKdSaveRelation.disabled = true;
    try {
      await updateKnowledgeRelation(currentKnowledgeItem.id, nextGroup);
      $btnKdSaveRelation.textContent = '已儲存';
      setTimeout(() => {
        $btnKdSaveRelation.textContent = originalText;
        $btnKdSaveRelation.disabled = false;
        closeKnowledgeModal();
        refreshKnowledgeList();
      }, 1000);
    } catch (e) {
      $btnKdSaveRelation.textContent = originalText;
      $btnKdSaveRelation.disabled = false;
      addErrorMessage(`儲存失敗: ${e.message}`);
    }
  });
}

if ($btnKdDelete) {
  $btnKdDelete.addEventListener('click', async () => {
    if (!currentKnowledgeItem) return;
    if (confirm('確定要刪除這筆文件嗎？此操作無法復原。')) {
      await deleteKnowledge(currentKnowledgeItem.id);
      closeKnowledgeModal();
      refreshKnowledgeList();
    }
  });
}

// ── Focus Mode Handlers ──
function toggleKnowledgeFocus(forceExpand) {
  if (!$knowledgeSection) return;
  const isCurrentlyFocused = $knowledgeSection.classList.contains('focused');
  const shouldExpand = typeof forceExpand === 'boolean' ? forceExpand : !isCurrentlyFocused;

  if (shouldExpand) {
    if (!$knowledgeSection.classList.contains('focused')) {
      $knowledgeSection.classList.add('focused');
      $btnKnowledgeExpand?.classList.add('hidden');
      $btnKnowledgeCollapse?.classList.remove('hidden');
      renderKnowledgeItems(cachedKnowledgeItems, $knowledgeSearchInput?.value || '');
    }
  } else {
    if ($knowledgeSection.classList.contains('focused')) {
      $knowledgeSection.classList.remove('focused');
      $btnKnowledgeExpand?.classList.remove('hidden');
      $btnKnowledgeCollapse?.classList.add('hidden');
    }
  }
}

if ($btnKnowledgeExpand) {
  $btnKnowledgeExpand.addEventListener('click', () => toggleKnowledgeFocus(true));
}
if ($btnKnowledgeCollapse) {
  $btnKnowledgeCollapse.addEventListener('click', () => toggleKnowledgeFocus(false));
}
if ($knowledgeSearchInput) {
  $knowledgeSearchInput.addEventListener('focus', () => toggleKnowledgeFocus(true));
  $knowledgeSearchInput.addEventListener('input', (e) => {
    renderKnowledgeItems(cachedKnowledgeItems, e.target.value);
  });
}

function renderKnowledgeItems(items, searchTerm = '') {
  if (!$knowledgeList) return;
  $knowledgeList.innerHTML = '';

  const lowerSearch = (searchTerm || '').toLowerCase();
  const filtered = items.filter(item => {
    if (!lowerSearch) return true;
    return (item.title || '').toLowerCase().includes(lowerSearch) ||
      (item.url || '').toLowerCase().includes(lowerSearch) ||
      (item.relation_group || '').toLowerCase().includes(lowerSearch);
  });

  if ($knowledgeListStats) {
    if (lowerSearch) {
      $knowledgeListStats.textContent = `搜尋結果: 找到 ${filtered.length} 篇 (共 ${items.length} 篇)`;
    } else {
      $knowledgeListStats.textContent = `庫內共有 ${items.length} 篇文件`;
    }
  }

  if (!filtered.length) {
    $knowledgeList.innerHTML = '<div class="knowledge-empty">找不到符合的資料</div>';
    return;
  }

  for (const item of filtered) {
    const row = document.createElement('div');
    row.className = 'knowledge-item';
    const relationGroup = normalizeRelationGroup(item.relation_group) || 'general';

    const MAX_VISIBLE_TAGS = 3;
    const tags = item.tags || [];
    const visibleTags = tags.slice(0, MAX_VISIBLE_TAGS);
    const hiddenTagsCount = tags.length - MAX_VISIBLE_TAGS;

    row.innerHTML = `
      <div class="knowledge-main">
        <div class="knowledge-title" title="${escapeHtml(item.title || '')}">${escapeHtml(item.title || '(無標題)')}</div>
        <div class="knowledge-sub" title="${escapeHtml(item.url || '')}">${escapeHtml(item.url || '')}</div>
      </div>
      <div class="knowledge-card-footer">
        <div class="knowledge-stats">
          <span class="knowledge-schema-preview" title="${escapeHtml(relationGroup)}">${escapeHtml(relationGroup)}</span>
          <span>${item.chunks_count || 0} chunks</span>
        </div>
        <div class="knowledge-badges">
          ${visibleTags.map(t => `<span class="knowledge-badge">${escapeHtml(t)}</span>`).join('')}
          ${hiddenTagsCount > 0 ? `<span class="knowledge-badge">+${hiddenTagsCount}</span>` : ''}
        </div>
      </div>
    `;

    // Click card to open modal
    row.addEventListener('click', () => {
      openKnowledgeModal(item);
    });

    $knowledgeList.appendChild(row);
  }
}

async function refreshKnowledgeList(options = {}) {
  const { silent = false } = options;
  if (!$knowledgeList) {
    return { ok: false, items: [] };
  }

  // If not focused, we don't need to show loading in the main list natively to avoid jumping UX
  if ($knowledgeSection?.classList.contains('focused')) {
    $knowledgeList.innerHTML = '<div class="knowledge-empty">載入中...</div>';
  }

  try {
    const baseUrl = getBackendBaseUrl();
    const res = await fetchWithTimeout(`${baseUrl}/api/knowledge`, { method: 'GET' }, 9000);
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`HTTP ${res.status}${body ? `: ${body.slice(0, 180)}` : ''}`);
    }

    const data = await res.json();
    const items = data.items || [];
    cachedKnowledgeItems = items;
    syncRelationScopeOptions(items);
    syncTagScopeOptions();
    const chunksTotal = items.reduce((sum, item) => sum + (item.chunks_count || 0), 0);

    if ($dbDocCount) $dbDocCount.textContent = String(items.length);
    if ($dbChunkCount) $dbChunkCount.textContent = String(chunksTotal);
    markDbCheckTime();
    clearDbErrorState();

    if (!items.length) {
      cachedKnowledgeItems = [];
      syncRelationScopeOptions([]);
      syncTagScopeOptions();
      if ($knowledgeListStats) $knowledgeListStats.textContent = '共 0 篇文件';
      if ($knowledgeSection?.classList.contains('focused')) {
        $knowledgeList.innerHTML = '<div class="knowledge-empty">目前沒有資料</div>';
      }
      return { ok: true, count: 0, chunks: 0, items };
    }

    // Dynamic render to support Search-Centric approach
    if ($knowledgeSection?.classList.contains('focused')) {
      renderKnowledgeItems(items, $knowledgeSearchInput?.value || '');
    } else if ($knowledgeListStats) {
      $knowledgeListStats.textContent = `庫內共有 ${items.length} 篇文件`;
    }

    return { ok: true, count: items.length, chunks: chunksTotal, items };
  } catch (e) {
    cachedKnowledgeItems = [];
    syncRelationScopeOptions([]);
    syncTagScopeOptions();
    if ($dbDocCount) $dbDocCount.textContent = '0';
    if ($dbChunkCount) $dbChunkCount.textContent = '0';
    markDbCheckTime();
    const errorInfo = buildDbErrorInfo(e, '/api/knowledge');
    showDbError(errorInfo);
    $knowledgeList.innerHTML = `<div class="knowledge-empty">載入失敗: ${escapeHtml(errorInfo.summary)}</div>`;
    if (!silent) {
      setTaskStatus('error', '知識庫列表載入失敗', {
        progress: 0,
        error: errorInfo.summary
      });
    }
    return { ok: false, error: errorInfo, items: [] };
  }
}

async function fetchKnowledgeDetail(docId) {
  const baseUrl = getBackendBaseUrl();
  const res = await fetchWithTimeout(`${baseUrl}/api/knowledge/${encodeURIComponent(docId)}`, {
    method: 'GET'
  }, 12000);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`HTTP ${res.status}${err ? `: ${err.slice(0, 220)}` : ''}`);
  }
  return await res.json();
}

function bindTagEditor(row, item) {
  const editor = row.querySelector('[data-tags-editor]');
  if (!editor) return;
  const docId = editor.dataset.tagsEditor;

  function getCurrentTags() {
    return JSON.parse(editor.dataset.tags || '[]');
  }

  function rerenderEditor(tags) {
    editor.dataset.tags = JSON.stringify(tags);
    editor.innerHTML = tags.map(t =>
      `<span class="tag-badge editable" data-tag-val="${escapeHtml(t)}">` +
      `${escapeHtml(t)}<span class="tag-remove" title="移除標籤">✕</span></span>`
    ).join('') + (tags.length < 3
      ? '<button class="tag-add-btn" title="新增標籤">+</button>'
      : '');
    attachListeners();
  }

  async function saveTags(tags) {
    try {
      const baseUrl = getBackendBaseUrl();
      const res = await fetchWithTimeout(`${baseUrl}/api/knowledge/${encodeURIComponent(docId)}/tags`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tags }),
      }, 8000);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // Update cached item
      const cached = cachedKnowledgeItems.find(c => c.id === docId);
      if (cached) cached.tags = [...tags];
      syncTagScopeOptions();
    } catch (e) {
      addErrorMessage(`標籤儲存失敗: ${e.message}`);
    }
  }

  function attachListeners() {
    // Remove tag
    editor.querySelectorAll('.tag-remove').forEach(el => {
      el.addEventListener('click', (ev) => {
        ev.stopPropagation();
        const badge = el.closest('[data-tag-val]');
        if (!badge) return;
        const tag = badge.dataset.tagVal;
        const tags = getCurrentTags().filter(t => t !== tag);
        rerenderEditor(tags);
        saveTags(tags);
      });
    });

    // Add tag button
    const addBtn = editor.querySelector('.tag-add-btn');
    if (addBtn) {
      addBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        showTagPicker(editor, docId);
      });
    }
  }

  attachListeners();
}

function showTagPicker(editor, docId) {
  // Remove any existing picker
  const existing = editor.querySelector('.tag-picker-dropdown');
  if (existing) { existing.remove(); return; }

  const currentTags = JSON.parse(editor.dataset.tags || '[]');
  const allTags = new Set();
  cachedKnowledgeItems.forEach(item => (item.tags || []).forEach(t => allTags.add(t)));
  const available = [...allTags].filter(t => !currentTags.includes(t)).sort();

  const dropdown = document.createElement('div');
  dropdown.className = 'tag-picker-dropdown';

  const searchInput = document.createElement('input');
  searchInput.type = 'text';
  searchInput.placeholder = '搜尋或輸入新標籤...';
  searchInput.className = 'tag-picker-search';
  dropdown.appendChild(searchInput);

  const listEl = document.createElement('div');
  listEl.className = 'tag-picker-list';
  dropdown.appendChild(listEl);

  function renderOptions(filter) {
    const filterLow = (filter || '').toLowerCase();
    const filtered = available.filter(t => !filterLow || t.toLowerCase().includes(filterLow));
    let html = filtered.map(t =>
      `<div class="tag-picker-option" data-pick="${escapeHtml(t)}">${escapeHtml(t)}</div>`
    ).join('');
    if (filterLow && !allTags.has(filter) && !currentTags.includes(filter)) {
      html += `<div class="tag-picker-option tag-picker-new" data-pick="${escapeHtml(filter)}">新增「${escapeHtml(filter)}」</div>`;
    }
    if (!html) {
      html = '<div class="tag-picker-empty">無可用標籤</div>';
    }
    listEl.innerHTML = html;
    listEl.querySelectorAll('[data-pick]').forEach(el => {
      el.addEventListener('click', () => {
        const tag = el.dataset.pick;
        const tags = JSON.parse(editor.dataset.tags || '[]');
        if (tags.length >= 3 || tags.includes(tag)) return;
        tags.push(tag);
        dropdown.remove();
        // Re-render & save
        editor.dataset.tags = JSON.stringify(tags);
        editor.innerHTML = tags.map(t =>
          `<span class="tag-badge editable" data-tag-val="${escapeHtml(t)}">` +
          `${escapeHtml(t)}<span class="tag-remove" title="移除標籤">✕</span></span>`
        ).join('') + (tags.length < 3
          ? '<button class="tag-add-btn" title="新增標籤">+</button>'
          : '');
        bindTagEditor(editor.closest('.knowledge-item'), { id: docId, tags });
        // Save to backend
        const baseUrl = getBackendBaseUrl();
        fetchWithTimeout(`${baseUrl}/api/knowledge/${encodeURIComponent(docId)}/tags`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tags }),
        }, 8000).then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const cached = cachedKnowledgeItems.find(c => c.id === docId);
          if (cached) cached.tags = [...tags];
          syncTagScopeOptions();
        }).catch(e => addErrorMessage(`標籤儲存失敗: ${e.message}`));
      });
    });
  }

  renderOptions('');
  searchInput.addEventListener('input', () => renderOptions(searchInput.value.trim()));
  searchInput.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') dropdown.remove();
    if (ev.key === 'Enter') {
      const val = searchInput.value.trim();
      if (val) {
        const firstOption = listEl.querySelector('[data-pick]');
        if (firstOption) firstOption.click();
      }
    }
  });

  editor.appendChild(dropdown);
  searchInput.focus();

  // Close on outside click
  function onOutsideClick(ev) {
    if (!dropdown.contains(ev.target) && ev.target !== dropdown) {
      dropdown.remove();
      document.removeEventListener('click', onOutsideClick, true);
    }
  }
  setTimeout(() => document.addEventListener('click', onOutsideClick, true), 0);
}

function renderKnowledgeDetailBody(detail) {
  const typeCounts = detail?.type_counts || {};
  const metadataSchema = detail?.metadata_schema || {};
  const textSamples = Array.isArray(detail?.text_samples) ? detail.text_samples : [];
  const imageItems = Array.isArray(detail?.image_items) ? detail.image_items : [];
  const schemaEntries = Object.entries(metadataSchema);

  const typeCountHtml = Object.entries(typeCounts)
    .map(([k, v]) => `<span class="knowledge-badge">${escapeHtml(k)}: ${escapeHtml(String(v))}</span>`)
    .join('');

  const schemaHtml = schemaEntries.length
    ? schemaEntries
      .map(([k, types]) => {
        const joined = Array.isArray(types) ? types.join(' | ') : String(types || '');
        return `<div class="knowledge-schema-row"><code>${escapeHtml(k)}</code><span>${escapeHtml(joined)}</span></div>`;
      })
      .join('')
    : '<div class="knowledge-detail-empty">無 schema 資訊</div>';

  const textHtml = textSamples.length
    ? textSamples.map((sample) => {
      const preview = String(sample?.preview || '').trim();
      const idx = Number.isFinite(Number(sample?.chunk_index)) ? Number(sample.chunk_index) : '?';
      return `<div class="knowledge-text-row"><span class="knowledge-text-index">#${idx}</span><span>${escapeHtml(preview || '(空白)')}</span></div>`;
    }).join('')
    : '<div class="knowledge-detail-empty">無文字樣本</div>';

  const imageHtml = imageItems.length
    ? imageItems.map((img) => {
      const url = String(img?.image_url || '');
      const alt = String(img?.image_alt || '');
      const caption = String(img?.caption || '');
      const idx = Number.isFinite(Number(img?.image_index)) ? Number(img.image_index) : '?';
      const thumb = url
        ? `<img src="${escapeHtml(url)}" alt="${escapeHtml(alt || '圖片')}" class="knowledge-image-thumb" loading="lazy">`
        : '<div class="knowledge-image-thumb placeholder">無圖片</div>';
      return `
        <div class="knowledge-image-row">
          ${thumb}
          <div class="knowledge-image-meta">
            <div class="knowledge-image-head">圖 ${escapeHtml(String(idx))} ${alt ? `· ${escapeHtml(alt)}` : ''}</div>
            <div class="knowledge-image-caption">${escapeHtml(caption || '(無圖片描述)')}</div>
            ${url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noreferrer">開啟原圖</a>` : ''}
          </div>
        </div>
      `;
    }).join('')
    : '<div class="knowledge-detail-empty">無圖片描述</div>';

  return `
    <div class="knowledge-detail-section">
      <div class="knowledge-detail-title">類型統計</div>
      <div class="knowledge-badges">${typeCountHtml || '<span class="knowledge-detail-empty">無資料</span>'}</div>
    </div>
    <div class="knowledge-detail-section">
      <div class="knowledge-detail-title">Metadata Schema（動態）</div>
      <div class="knowledge-schema-grid">${schemaHtml}</div>
    </div>
    <div class="knowledge-detail-section">
      <div class="knowledge-detail-title">文字樣本</div>
      <div class="knowledge-text-list">${textHtml}</div>
    </div>
    <div class="knowledge-detail-section">
      <div class="knowledge-detail-title">圖片描述</div>
      <div class="knowledge-image-list">${imageHtml}</div>
    </div>
  `;
}

function normalizeUrl(url) {
  return String(url || '').replace(/\/+$/, '');
}

function getIngestStageLabel(stage) {
  const stageMap = {
    queued: '排隊中',
    chunking: '切分文本',
    captioning: '描述圖片',
    embedding: '產生向量',
    writing: '寫入資料庫',
    done: '完成',
    error: '失敗'
  };
  return stageMap[String(stage || '')] || (stage || '未知階段');
}

async function fetchIngestTaskStatus(taskId) {
  const id = String(taskId || '').trim();
  if (!id) {
    return { ok: false, notFound: false, error: 'missing task_id' };
  }

  const baseUrl = getBackendBaseUrl();
  const endpoint = `${baseUrl}/api/ingest/${encodeURIComponent(id)}`;
  try {
    const res = await fetchWithTimeout(endpoint, { method: 'GET' }, 7000);
    if (res.status === 404) {
      return { ok: false, notFound: true, error: 'Task not found' };
    }
    if (!res.ok) {
      const body = await res.text();
      return {
        ok: false,
        notFound: false,
        error: `HTTP ${res.status}${body ? `: ${body.slice(0, 180)}` : ''}`
      };
    }
    const data = await res.json();
    return { ok: true, task: data.task || null };
  } catch (e) {
    return { ok: false, notFound: false, error: e?.message || String(e) };
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForKnowledgeIngest(options) {
  const {
    taskId = '',
    targetUrl,
    previousChunks = null,
    timeoutMs = 45000,
    intervalMs = 1200
  } = options;

  const startedAt = Date.now();
  let latestTask = null;
  let latestStatusError = '';
  while (Date.now() - startedAt < timeoutMs) {
    const elapsed = Date.now() - startedAt;
    const progress = Math.min(94, 40 + Math.floor((elapsed / timeoutMs) * 54));
    let progressText = '等待背景任務寫入 DB...';

    if (taskId) {
      const status = await fetchIngestTaskStatus(taskId);
      if (status.ok && status.task) {
        latestTask = status.task;
        const stageLabel = getIngestStageLabel(status.task.stage);
        const msg = status.task.message ? `：${status.task.message}` : '';
        const filtered = Number(status.task.image_filtered || 0);
        const filterNote = filtered > 0 ? `｜已過濾 ${filtered} 張無關圖片` : '';
        progressText = `等待背景任務寫入 DB...（${stageLabel}${msg}${filterNote}）`;
        if (status.task.status === 'error') {
          return {
            ok: false,
            reason: 'task_error',
            task: latestTask,
            elapsedMs: elapsed
          };
        }
      } else if (!status.notFound) {
        latestStatusError = status.error || '';
      }
    }

    updateTaskProgress(progress, progressText);

    const list = await refreshKnowledgeList({ silent: true });
    if (list.ok) {
      const found = (list.items || []).find((item) => normalizeUrl(item.url) === normalizeUrl(targetUrl));
      if (found) {
        const chunks = Number(found.chunks_count || 0);
        if (previousChunks === null || chunks !== previousChunks || elapsed > 1500) {
          return { ok: true, item: found, elapsedMs: elapsed, task: latestTask };
        }
      }
    }

    await sleep(intervalMs);
  }

  return {
    ok: false,
    reason: 'timeout',
    task: latestTask,
    statusError: latestStatusError,
    elapsedMs: Date.now() - startedAt
  };
}

function buildIngestFailureInfo(waitResult, context = {}) {
  const task = waitResult?.task || null;
  const elapsedMs = Number(waitResult?.elapsedMs || 0);
  const elapsedText = `${(elapsedMs / 1000).toFixed(1)}s`;
  const reason = waitResult?.reason || 'unknown';
  const taskId = context.taskId || task?.task_id || '-';
  const targetUrl = context.targetUrl || task?.url || '-';
  const stageLabel = getIngestStageLabel(task?.stage);
  const taskMsg = task?.message ? String(task.message) : '';
  const taskErr = task?.error ? String(task.error) : '';
  const imageFailures = Array.isArray(task?.image_caption_failures) ? task.image_caption_failures : [];
  const statusError = waitResult?.statusError ? String(waitResult.statusError) : '';

  let summary = '背景任務尚未完成或失敗';
  if (reason === 'task_error') {
    summary = `背景任務失敗（${stageLabel}）${taskErr ? `: ${taskErr}` : ''}`;
  } else if (reason === 'timeout') {
    summary = `背景任務逾時（${elapsedText}）${task ? `，停在「${stageLabel}」` : ''}`;
  } else if (statusError) {
    summary = `背景任務狀態查詢失敗: ${statusError}`;
  }

  const detail = [
    `時間: ${new Date().toLocaleString('zh-TW', { hour12: false })}`,
    `task_id: ${taskId}`,
    `目標 URL: ${targetUrl}`,
    `失敗類型: ${reason}`,
    `逾時/耗時: ${elapsedText}`,
    `目前階段: ${stageLabel}`,
    `任務狀態: ${task?.status || '-'}`,
    `階段訊息: ${taskMsg || '-'}`,
    `錯誤訊息: ${taskErr || '-'}`,
    `圖片描述失敗: ${Number(task?.image_caption_failed || 0)}`,
    `圖片描述失敗明細: ${imageFailures.length ? '' : '-'}`,
    ...imageFailures.slice(0, 5).map((f) => `  - #${f?.index ?? '?'} ${String(f?.error || '')} ${String(f?.src || '')}`),
    `狀態查詢錯誤: ${statusError || '-'}`,
    '',
    '排障建議：',
    '1) 在資料庫頁按「連線診斷」，確認 backend/ollama/chromadb 都是正常',
    '2) 檢查嵌入模型是否存在於本機 Ollama（例如 bge-m3/nomic-embed-text）',
    '3) 若錯誤含 unauthorized，代表命中雲端模型，請改成本機模型',
    '4) 若持續卡在「產生向量」，請檢查 backend api 容器日誌'
  ].join('\n');

  return {
    summary,
    detail,
    raw: taskErr || statusError || reason
  };
}

async function runDbRecoveryFlow(options = {}) {
  const {
    includeServerCheck = false,
    showSuccessMessage = false
  } = options;

  setDbActionsDisabled(true);
  try {
    setTaskStatus('working', '執行資料庫排障流程...', { progress: 10 });
    syncDbDiagnosticsContext();

    if (includeServerCheck) {
      await checkServerStatus();
      updateTaskProgress(28, '檢查服務狀態...');
    }

    const health = await refreshDbStatus({ silent: true });
    if (!health.ok) {
      setTaskStatus('error', '資料庫狀態檢查失敗', {
        progress: 0,
        error: health.error.summary
      });
      return false;
    }

    updateTaskProgress(62, '重新載入知識庫列表...');
    const list = await refreshKnowledgeList({ silent: true });
    if (!list.ok) {
      setTaskStatus('error', '知識庫列表載入失敗', {
        progress: 0,
        error: list.error.summary
      });
      return false;
    }

    const summary = `資料庫連線正常（${list.count} 筆文件 / ${list.chunks} chunks）`;
    setTaskStatus('success', summary, { progress: 100 });
    if (showSuccessMessage) {
      addSystemMessage(summary);
    }
    return true;
  } finally {
    setDbActionsDisabled(false);
  }
}

async function deleteKnowledge(docId) {
  try {
    const baseUrl = getBackendBaseUrl();
    const res = await fetchWithTimeout(`${baseUrl}/api/knowledge/${encodeURIComponent(docId)}`, {
      method: 'DELETE'
    }, 9000);

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status}${err ? `: ${err.slice(0, 180)}` : ''}`);
    }

    addSystemMessage(`已刪除知識項目 ${docId}`);
    await refreshKnowledgeList();
  } catch (e) {
    const errorInfo = buildDbErrorInfo(e, `/api/knowledge/${encodeURIComponent(docId)}`);
    showDbError(errorInfo);
    addErrorMessage(`刪除失敗: ${errorInfo.summary}`);
  }
}

async function updateKnowledgeRelation(docId, relationGroup) {
  try {
    const baseUrl = getBackendBaseUrl();
    const res = await fetchWithTimeout(`${baseUrl}/api/knowledge/${encodeURIComponent(docId)}/relation`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ relation_group: relationGroup })
    }, 10000);

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`HTTP ${res.status}${err ? `: ${err.slice(0, 180)}` : ''}`);
    }

    addSystemMessage(`已更新關聯群組：${docId} -> ${relationGroup}`);
    await refreshKnowledgeList();
  } catch (e) {
    const errorInfo = buildDbErrorInfo(e, `/api/knowledge/${encodeURIComponent(docId)}/relation`);
    showDbError(errorInfo);
    addErrorMessage(`更新關聯群組失敗: ${errorInfo.summary}`);
  }
}

async function sendMessage() {
  const text = $userInput.value.trim();
  if (!text || isStreaming) {
    return;
  }

  const ragMode = Boolean($ragToggle?.checked);
  const model = $modelSelect.value;
  const embeddingModel = getSelectedEmbeddingModel();
  const relationGroups = getSelectedRelationGroups();
  const sourceUrls = getSelectedSourceUrls();
  const relationScopeText = formatHierarchicalScope(relationGroups, sourceUrls);
  const sourceScopePreview = formatSelectedPagePreview(sourceUrls);
  const forceRagByScope = !ragMode && (relationGroups.length > 0 || sourceUrls.length > 0 || selectedTags.length > 0);
  const effectiveRagMode = ragMode || forceRagByScope;

  if (!effectiveRagMode && !model) {
    addErrorMessage('請先在設定中選擇模型');
    return;
  }
  if (!effectiveRagMode && isCloudModelName(model)) {
    addErrorMessage('目前選到雲端模型，請改選本機模型（例如 qwen2.5-coder:3b）。');
    return;
  }
  if (effectiveRagMode && model && isCloudModelName(model)) {
    addErrorMessage('RAG 聊天模型目前選到雲端模型，請改選本機模型。');
    return;
  }
  if (effectiveRagMode && embeddingModel && isCloudModelName(embeddingModel)) {
    addErrorMessage('RAG 嵌入模型目前選到雲端模型，請改選本機模型。');
    return;
  }

  addUserMessage(text);
  $userInput.value = '';
  $userInput.style.height = 'auto';

  chatHistory.push({ role: 'user', content: text });
  setTaskStatus('working', '正在送出問題...', { progress: 8 });

  const activeChatModel = model || (effectiveRagMode ? '(後端預設模型)' : '(未選擇)');
  const activeEmbeddingModel = effectiveRagMode
    ? (embeddingModel || '(後端預設嵌入模型)')
    : '(未使用)';
  renderRuntimeInfo(activeChatModel, activeEmbeddingModel);

  if (effectiveRagMode) {
    if (forceRagByScope) {
      if ($ragToggle) {
        $ragToggle.checked = true;
      }
      syncDbDiagnosticsContext();
      addSystemMessage('偵測到已指定查詢範圍，已自動切換為 RAG（Direct 模式不使用知識庫）。');
      const saved = await saveCurrentSettings();
      if (saved?.error) {
        addErrorMessage(`RAG 模式設定儲存失敗: ${saved.error}`);
      }
    }
    const tagsScopeText = selectedTags.length > 0 ? `；標籤：${selectedTags.slice(0, 3).join('、')}${selectedTags.length > 3 ? '…' : ''}` : '';
    addSystemMessage(
      `本次查詢範圍：${relationScopeText}${sourceScopePreview ? `；指定頁面：${sourceScopePreview}` : ''}${tagsScopeText}`
    );
    if ($agenticToggle?.checked) {
      sendAgenticMessage(text, model, embeddingModel, relationGroups, sourceUrls);
    } else {
      sendRagMessage(text, model, embeddingModel, relationGroups, sourceUrls);
    }
    updateSendState();
    return;
  }

  if (!pageContent) {
    const restored = await restoreCachedPageForActiveTab({ silent: false });
    if (restored) {
      updateSendState();
    }
  }

  if (!pageContent) {
    if (!cachedKnowledgeItems.length) {
      await refreshKnowledgeList({ silent: true });
    }
    const activeUrl = await getActiveTabUrlFromBackground();
    const hasDocInKb = Boolean(
      activeUrl &&
      cachedKnowledgeItems.some((item) => normalizeUrl(item.url) === activeUrl)
    );

    if (hasDocInKb) {
      if (model && isCloudModelName(model)) {
        addErrorMessage('切換到 RAG 時檢測到雲端聊天模型，請改選本機模型。');
        return;
      }
      if (embeddingModel && isCloudModelName(embeddingModel)) {
        addErrorMessage('切換到 RAG 時檢測到雲端嵌入模型，請改選本機模型。');
        return;
      }

      if ($ragToggle) {
        $ragToggle.checked = true;
      }
      syncDbDiagnosticsContext();
      addSystemMessage('偵測到此頁已在知識庫，已自動切換 RAG 模式回答。');
      addSystemMessage(
        `本次查詢範圍：${relationScopeText}${sourceScopePreview ? `；指定頁面：${sourceScopePreview}` : ''}`
      );
      sendRagMessage(text, model, embeddingModel, relationGroups, sourceUrls);
      updateSendState();
      return;
    }

    addErrorMessage('Direct 模式需要先擷取目前頁面');
    return;
  }

  let contentText = pageContent.textContent || '';
  if (contentText.length > MAX_CONTENT_CHARS) {
    contentText = `${contentText.slice(0, MAX_CONTENT_CHARS)}\n\n[...內容已截斷]`;
  }

  const systemMessage = {
    role: 'system',
    content:
      `${SYSTEM_PROMPT}\n\n<webpage>\n<title>${pageContent.title || ''}</title>\n` +
      `<content>\n${contentText}\n</content>\n</webpage>`
  };

  const messages = [systemMessage, ...chatHistory];
  startDirectStreaming(messages, model, {
    queryLength: text.length,
    contextChars: contentText.length
  });
  updateSendState();
}

function startDirectStreaming(messages, model, tipContext = {}) {
  isStreaming = true;
  manualStopRequested = false;
  ragTimedOut = false;
  const startedAt = Date.now();
  let firstTokenAt = null;

  $btnStop.classList.remove('hidden');
  $userInput.disabled = true;
  setTaskStatus('working', '模型回覆中...', { progress: 15 });

  const tipCard = createAssistantTipCard({
    mode: 'Direct',
    query_length: tipContext.queryLength,
    context_chars_sent: tipContext.contextChars,
    model_used: model || '(設定預設)',
    status: 'streaming'
  });
  const msgEl = tipCard.messageEl;
  const stream = initStreamParser(msgEl);
  $messages.appendChild(tipCard.container);
  scrollToBottom();

  let fullResponse = '';
  currentPort = chrome.runtime.connect({ name: 'ollama-stream' });

  currentPort.onMessage.addListener((msg) => {
    if (msg.type === 'TOKEN') {
      if (!firstTokenAt) {
        firstTokenAt = Date.now();
        tipCard.setMetrics({
          first_token_ms: firstTokenAt - startedAt
        });
      }
      fullResponse += msg.content;
      feedStreamChunk(stream, msg.content);
      scrollToBottom();
      const p = Math.min(92, 15 + Math.floor(fullResponse.length / 80) * 2);
      updateTaskProgress(p, '模型回覆中...');
      return;
    }

    if (msg.type === 'DONE') {
      tipCard.setMetrics({
        response_chars: fullResponse.length,
        total_latency_ms: Date.now() - startedAt,
        ollama_total_duration_ms: nsToMs(msg.totalDuration),
        eval_count: msg.evalCount || null,
        status: 'done'
      });
      finalizeStream(stream);
      chatHistory.push({ role: 'assistant', content: fullResponse });
      setTaskStatus('success', '回覆完成', { progress: 100 });
      finishStreaming();
      return;
    }

    if (msg.type === 'ERROR') {
      tipCard.setMetrics({
        response_chars: fullResponse.length,
        total_latency_ms: Date.now() - startedAt,
        status: 'error',
        error: msg.error
      });
      finalizeStream(stream);
      if (fullResponse) {
        chatHistory.push({ role: 'assistant', content: fullResponse });
      }
      addErrorMessage(msg.error);
      setTaskStatus('error', '模型回覆失敗', { progress: 0, error: msg.error });
      finishStreaming();
      return;
    }

    if (msg.type === 'ABORTED') {
      tipCard.setMetrics({
        response_chars: fullResponse.length,
        total_latency_ms: Date.now() - startedAt,
        status: 'aborted'
      });
      finalizeStream(stream);
      if (fullResponse) {
        chatHistory.push({ role: 'assistant', content: fullResponse });
      }
      setTaskStatus('idle', '已停止生成', { progress: 0 });
      finishStreaming();
    }
  });

  currentPort.postMessage({
    type: 'CHAT_REQUEST',
    messages,
    model
  });
}

function clearRagTimeout() {
  if (ragTimeoutHandle) {
    clearTimeout(ragTimeoutHandle);
    ragTimeoutHandle = null;
  }
}

async function sendRagMessage(query, model, embeddingModel, relationGroups = [], sourceUrls = []) {
  const scopeText = formatHierarchicalScope(relationGroups, sourceUrls);
  const requestedTopK = 5;
  isStreaming = true;
  manualStopRequested = false;
  ragTimedOut = false;
  const startedAt = Date.now();
  let firstTokenAt = null;

  $btnStop.classList.remove('hidden');
  $userInput.disabled = true;
  setTaskStatus('working', `RAG 檢索中（範圍：${scopeText}）...`, { progress: 15 });

  const tipCard = createAssistantTipCard({
    mode: 'RAG',
    status: 'retrieving',
    scope: scopeText,
    selected_groups: relationGroups.join(', ') || '(全部)',
    selected_source_count: sourceUrls.length,
    model_used: model || '(後端自動選擇)',
    embedding_model_used: embeddingModel || '(後端預設)',
    top_k: requestedTopK,
    query_length: query.length
  });
  const msgEl = tipCard.messageEl;
  const stream = initStreamParser(msgEl);
  $messages.appendChild(tipCard.container);
  scrollToBottom();

  const baseUrl = ($backendUrl?.value || 'http://localhost:8000').replace(/\/+$/, '');
  currentAbortController = new AbortController();
  ragTimeoutHandle = setTimeout(() => {
    ragTimedOut = true;
    currentAbortController?.abort();
  }, RAG_TIMEOUT_MS);

  try {
    const payload = { query };
    payload.top_k = requestedTopK;
    if (model) {
      payload.model = model;
    }
    if (embeddingModel) {
      payload.embedding_model = embeddingModel;
    }
    if (relationGroups.length > 0) {
      payload.relation_groups = relationGroups;
    }
    if (sourceUrls.length > 0) {
      payload.source_urls = sourceUrls;
    }
    if (selectedTags.length > 0) {
      payload.tags = selectedTags;
    }
    updateTaskProgress(18, `RAG 檢索中（範圍：${scopeText}）...`);

    const res = await fetch(`${baseUrl}/api/rag_chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: currentAbortController.signal
    });

    if (!res.ok || !res.body) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let currentEvent = '';
    let fullResponse = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        if (!line) {
          continue;
        }
        if (line.startsWith('event:')) {
          currentEvent = line.slice(6).trim();
          continue;
        }
        if (!line.startsWith('data:')) {
          continue;
        }

        const data = line.slice(5).trim();
        if (currentEvent === 'meta') {
          const meta = parseJsonSafe(data, {});
          if (meta && typeof meta === 'object') {
            tipCard.setMetrics({
              mode: 'RAG',
              model_used: meta.model_used || model || '(後端自動選擇)',
              embedding_model_used: meta.embedding_model_used || embeddingModel || '(後端預設)',
              top_k: meta.top_k ?? requestedTopK,
              selected_source_count: meta.selected_source_count ?? sourceUrls.length,
              retrieved_chunks: meta.retrieved_chunks,
              retrieval_hit_rate: meta.retrieval_hit_rate,
              unique_source_count: meta.unique_source_count,
              scope: (meta.scope_groups || []).join(', ') || scopeText,
              scope_group_count: meta.scope_group_count,
              hit_group_count: meta.hit_group_count,
              relation_group_hit_rate: meta.relation_group_hit_rate,
              status: 'generating'
            });
          }
          continue;
        }
        if (currentEvent === 'sources') {
          const sources = renderSources(data);
          const groups = new Set((sources || []).map((s) => s.relation_group).filter(Boolean));
          const urls = new Set((sources || []).map((s) => s.url).filter(Boolean));
          tipCard.setMetrics({
            retrieved_chunks: Array.isArray(sources) ? sources.length : null,
            unique_source_count: urls.size,
            result_relation_groups: [...groups].join(', ')
          });
          updateTaskProgress(40, '已取得來源，生成中...');
          continue;
        }
        if (currentEvent === 'token') {
          if (!firstTokenAt) {
            firstTokenAt = Date.now();
            tipCard.setMetrics({
              first_token_ms: firstTokenAt - startedAt,
              status: 'generating'
            });
          }
          fullResponse += data;
          feedStreamChunk(stream, data);
          scrollToBottom();
          const p = Math.min(92, 40 + Math.floor(fullResponse.length / 80) * 2);
          updateTaskProgress(p, 'RAG 生成中...');
          continue;
        }
        if (currentEvent === 'done') {
          const doneData = parseJsonSafe(data, {});
          tipCard.setMetrics({
            response_chars: fullResponse.length,
            total_latency_ms: Date.now() - startedAt,
            ollama_total_duration_ms: nsToMs(doneData?.total_duration),
            ollama_load_duration_ms: nsToMs(doneData?.load_duration),
            prompt_eval_count: doneData?.prompt_eval_count ?? null,
            eval_count: doneData?.eval_count ?? null,
            status: 'done'
          });
          finalizeStream(stream);
          chatHistory.push({ role: 'assistant', content: fullResponse });
          setTaskStatus('success', 'RAG 回覆完成', { progress: 100 });
          finishStreaming();
          return;
        }
        if (currentEvent === 'error') {
          throw new Error(data || 'RAG 回應失敗');
        }
      }
    }

    chatHistory.push({ role: 'assistant', content: fullResponse || '' });
    tipCard.setMetrics({
      response_chars: fullResponse.length,
      total_latency_ms: Date.now() - startedAt,
      status: 'done'
    });
    setTaskStatus('success', 'RAG 回覆完成', { progress: 100 });
    finishStreaming();
  } catch (e) {
    finalizeStream(stream);
    tipCard.setMetrics({
      response_chars: fullResponse?.length || 0,
      total_latency_ms: Date.now() - startedAt,
      status: 'error',
      error: e?.message || String(e)
    });
    if (e.name === 'AbortError') {
      if (ragTimedOut) {
        addErrorMessage(`RAG 回覆逾時（${Math.round(RAG_TIMEOUT_MS / 1000)} 秒）`);
        setTaskStatus('error', 'RAG 回覆逾時', {
          progress: 0,
          error: `請稍後重試，或先縮小問題範圍。`
        });
      } else if (manualStopRequested) {
        addSystemMessage('已停止生成');
        setTaskStatus('idle', '已停止生成', { progress: 0 });
      } else {
        addErrorMessage('RAG 連線已中斷');
      }
    } else {
      addErrorMessage(`RAG 模式錯誤: ${e.message || e}`);
      setTaskStatus('error', 'RAG 回覆失敗', {
        progress: 0,
        error: e.message || String(e)
      });
    }
    finishStreaming();
  } finally {
    clearRagTimeout();
  }
}

// ─── Agentic RAG ───────────────────────────────────────────────

async function sendAgenticMessage(query, model, embeddingModel, relationGroups = [], sourceUrls = []) {
  const scopeText = formatHierarchicalScope(relationGroups, sourceUrls);
  isStreaming = true;
  manualStopRequested = false;
  ragTimedOut = false;
  const startedAt = Date.now();
  let firstTokenAt = null;

  $btnStop.classList.remove('hidden');
  $userInput.disabled = true;
  setTaskStatus('working', `Agentic RAG 分析中（範圍：${scopeText}）...`, { progress: 10 });

  const agenticCard = createAgenticTipCard({
    mode: 'Agentic RAG',
    status: 'routing',
    scope: scopeText,
    model_used: model || '(後端自動選擇)',
    query_length: query.length
  });
  const msgEl = agenticCard.messageEl;
  const stream = initStreamParser(msgEl);
  $messages.appendChild(agenticCard.container);
  scrollToBottom();

  const baseUrl = ($backendUrl?.value || 'http://localhost:8000').replace(/\/+$/, '');
  currentAbortController = new AbortController();
  ragTimeoutHandle = setTimeout(() => {
    ragTimedOut = true;
    currentAbortController?.abort();
  }, RAG_TIMEOUT_MS);

  let fullResponse = '';

  try {
    const payload = { query, top_k: 5 };
    if (model) payload.model = model;
    if (embeddingModel) payload.embedding_model = embeddingModel;
    if (relationGroups.length > 0) payload.relation_groups = relationGroups;
    if (sourceUrls.length > 0) payload.source_urls = sourceUrls;
    if (selectedTags.length > 0) payload.tags = selectedTags;
    if (chatHistory.length > 0) payload.conversation_history = chatHistory.slice(-6);

    const res = await fetch(`${baseUrl}/api/agentic_chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: currentAbortController.signal
    });

    if (!res.ok || !res.body) {
      const text = await res.text();
      throw new Error(text || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let currentEvent = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        if (!line) continue;
        if (line.startsWith('event:')) {
          currentEvent = line.slice(6).trim();
          continue;
        }
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();

        if (currentEvent === 'status') {
          const status = parseJsonSafe(data, {});
          agenticCard.addPathNode(status.node, status.message);
          const statusMsg = status.message || status.node || '';
          updateTaskProgress(15, statusMsg);
          continue;
        }
        if (currentEvent === 'transition') {
          const t = parseJsonSafe(data, {});
          agenticCard.addTransition(t);
          continue;
        }
        if (currentEvent === 'meta') {
          const meta = parseJsonSafe(data, {});
          agenticCard.setDebugInfo('meta', meta);
          updateTaskProgress(30, '已取得檢索結果...');
          continue;
        }
        if (currentEvent === 'sources') {
          renderSources(data);
          const sources = parseJsonSafe(data, []);
          agenticCard.setDebugInfo('sources', sources);
          updateTaskProgress(40, '已取得來源，評估中...');
          continue;
        }
        if (currentEvent === 'evidence') {
          const ev = parseJsonSafe(data, {});
          agenticCard.setDebugInfo('evidence', ev);
          agenticCard.setEvidenceVerdict(ev.verdict, ev.detail);
          continue;
        }
        if (currentEvent === 'token') {
          if (!firstTokenAt) {
            firstTokenAt = Date.now();
            agenticCard.setDebugInfo('first_token_ms', firstTokenAt - startedAt);
          }
          fullResponse += data;
          feedStreamChunk(stream, data);
          scrollToBottom();
          const p = Math.min(92, 40 + Math.floor(fullResponse.length / 80) * 2);
          updateTaskProgress(p, 'Agentic RAG 生成中...');
          continue;
        }
        if (currentEvent === 'done') {
          const doneData = parseJsonSafe(data, {});
          agenticCard.finalize({
            total_latency_ms: Date.now() - startedAt,
            response_chars: fullResponse.length,
            route_action: doneData.route_action,
            refine_rounds: doneData.refine_rounds,
            evidence_verdict: doneData.evidence_verdict,
            total_duration_ms: doneData.total_duration_ms
          });
          finalizeStream(stream);
          chatHistory.push({ role: 'assistant', content: fullResponse });
          setTaskStatus('success', 'Agentic RAG 回覆完成', { progress: 100 });
          finishStreaming();
          return;
        }
        if (currentEvent === 'error') {
          const errData = parseJsonSafe(data, {});
          throw new Error(errData.message || data || 'Agentic RAG 回應失敗');
        }
      }
    }

    chatHistory.push({ role: 'assistant', content: fullResponse || '' });
    agenticCard.finalize({
      total_latency_ms: Date.now() - startedAt,
      response_chars: fullResponse.length
    });
    setTaskStatus('success', 'Agentic RAG 回覆完成', { progress: 100 });
    finishStreaming();
  } catch (e) {
    finalizeStream(stream);
    agenticCard.finalize({
      total_latency_ms: Date.now() - startedAt,
      response_chars: fullResponse?.length || 0,
      status: 'error',
      error: e?.message || String(e)
    });
    if (e.name === 'AbortError') {
      if (ragTimedOut) {
        addErrorMessage(`Agentic RAG 回覆逾時（${Math.round(RAG_TIMEOUT_MS / 1000)} 秒）`);
        setTaskStatus('error', 'Agentic RAG 回覆逾時', { progress: 0 });
      } else if (manualStopRequested) {
        addSystemMessage('已停止生成');
        setTaskStatus('idle', '已停止生成', { progress: 0 });
      } else {
        addErrorMessage('Agentic RAG 連線已中斷');
      }
    } else {
      addErrorMessage(`Agentic RAG 錯誤: ${e.message || e}`);
      setTaskStatus('error', 'Agentic RAG 回覆失敗', { progress: 0, error: e.message || String(e) });
    }
    finishStreaming();
  } finally {
    clearRagTimeout();
  }
}

function createAgenticTipCard(initialMetrics = {}) {
  const container = document.createElement('div');
  container.className = 'assistant-card';

  const header = document.createElement('div');
  header.className = 'assistant-card-header';

  const messageEl = createMessageElement('assistant');

  // --- Path summary bar ---
  const pathBar = document.createElement('div');
  pathBar.className = 'agentic-path-bar';

  const modeBadge = document.createElement('span');
  modeBadge.className = 'agentic-mode-badge';
  modeBadge.textContent = 'Agentic RAG';
  pathBar.appendChild(modeBadge);

  const pathNodes = document.createElement('span');
  pathNodes.className = 'agentic-path-nodes';
  pathBar.appendChild(pathNodes);

  const verdictBadge = document.createElement('span');
  verdictBadge.className = 'agentic-verdict-badge hidden';
  pathBar.appendChild(verdictBadge);

  const durationBadge = document.createElement('span');
  durationBadge.className = 'agentic-duration-badge hidden';
  pathBar.appendChild(durationBadge);

  // --- Debug panel (expandable) ---
  const debugToggle = document.createElement('button');
  debugToggle.className = 'tip-toggle';
  debugToggle.innerHTML = `
    <span style="display:flex; align-items:center; gap:4px;">
      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/></svg>
      FSM 詳情
    </span>
  `;
  debugToggle.title = '查看 Agentic RAG 處理詳情';

  const debugPanel = document.createElement('div');
  debugPanel.className = 'tip-panel agentic-debug-panel hidden';

  debugToggle.addEventListener('click', () => {
    debugPanel.classList.toggle('hidden');
  });

  // Internal state
  const visitedNodes = [];
  const transitions = [];
  const debugInfo = {};

  function renderPathNodes() {
    const nodeLabels = {
      ROUTE: '路由',
      RETRIEVE: '檢索',
      EVALUATE: '評估',
      REFINE: '調整',
      GENERATE: '生成',
      NO_ANSWER: '無結果'
    };
    pathNodes.innerHTML = visitedNodes
      .map((n, i) => {
        const label = nodeLabels[n] || n;
        const isLast = i === visitedNodes.length - 1;
        const nodeHtml = `<span class="agentic-path-node ${isLast ? 'active' : ''}">${label}</span>`;
        return i > 0 ? `<span class="agentic-path-arrow">→</span>${nodeHtml}` : nodeHtml;
      })
      .join('');
  }

  function renderDebugPanel() {
    const sections = [];

    // Transitions
    if (transitions.length > 0) {
      const tLines = transitions.map(t => {
        const reason = t.reason ? ` (${t.reason})` : '';
        return `  ${t.from || '?'} → ${t.to || '?'}${reason}`;
      });
      sections.push(`狀態轉移:\n${tLines.join('\n')}`);
    }

    // Evidence
    if (debugInfo.evidence) {
      const ev = debugInfo.evidence;
      const detail = ev.detail || {};
      const evLines = [
        `  判定: ${ev.verdict || '?'}`,
        detail.heuristic_zone ? `  Heuristic Zone: ${detail.heuristic_zone}` : null,
        detail.best_distance != null ? `  Best Distance: ${detail.best_distance}` : null,
        detail.caption_ratio != null ? `  Caption Ratio: ${detail.caption_ratio}` : null
      ].filter(Boolean);
      sections.push(`證據評估:\n${evLines.join('\n')}`);
    }

    // Meta
    if (debugInfo.meta) {
      const m = debugInfo.meta;
      const mLines = [
        m.retrieved_chunks != null ? `  命中 Chunks: ${m.retrieved_chunks}` : null,
        m.query_params ? `  查詢參數: ${JSON.stringify(m.query_params)}` : null
      ].filter(Boolean);
      if (mLines.length) sections.push(`檢索資訊:\n${mLines.join('\n')}`);
    }

    // Timing
    const timingLines = [];
    if (debugInfo.first_token_ms != null) timingLines.push(`  首 Token: ${debugInfo.first_token_ms}ms`);
    if (debugInfo.total_latency_ms != null) timingLines.push(`  總延遲: ${debugInfo.total_latency_ms}ms`);
    if (debugInfo.total_duration_ms != null) timingLines.push(`  後端耗時: ${debugInfo.total_duration_ms}ms`);
    if (debugInfo.refine_rounds != null) timingLines.push(`  Refine 輪次: ${debugInfo.refine_rounds}`);
    if (timingLines.length) sections.push(`效能:\n${timingLines.join('\n')}`);

    if (debugInfo.error) sections.push(`錯誤: ${debugInfo.error}`);

    debugPanel.textContent = sections.join('\n\n') || '(處理中...)';
  }

  header.appendChild(messageEl);
  header.appendChild(debugToggle);
  container.appendChild(pathBar);
  container.appendChild(header);
  container.appendChild(debugPanel);
  renderPathNodes();
  renderDebugPanel();

  return {
    container,
    messageEl,
    addPathNode(node, message) {
      if (node && !visitedNodes.includes(node)) {
        visitedNodes.push(node);
        renderPathNodes();
      }
    },
    addTransition(t) {
      transitions.push(t);
      renderDebugPanel();
    },
    setDebugInfo(key, value) {
      debugInfo[key] = value;
      renderDebugPanel();
    },
    setEvidenceVerdict(verdict, detail) {
      if (!verdict) return;
      verdictBadge.classList.remove('hidden');
      const verdictLabels = { sufficient: '充分', insufficient: '不足', uncertain: '不確定' };
      const verdictClasses = { sufficient: 'verdict-pass', insufficient: 'verdict-fail', uncertain: 'verdict-warn' };
      verdictBadge.textContent = verdictLabels[verdict] || verdict;
      verdictBadge.className = `agentic-verdict-badge ${verdictClasses[verdict] || ''}`;
    },
    finalize(info = {}) {
      Object.assign(debugInfo, info);
      if (info.route_action) {
        const actionLabels = {
          search_knowledge: '知識查詢',
          direct_chat: '閒聊',
          no_answer: '無結果'
        };
        modeBadge.textContent = `Agentic: ${actionLabels[info.route_action] || info.route_action}`;
      }
      if (info.total_latency_ms != null) {
        durationBadge.classList.remove('hidden');
        durationBadge.textContent = `${(info.total_latency_ms / 1000).toFixed(1)}s`;
      }
      if (info.status === 'error') {
        modeBadge.classList.add('error');
      }
      renderDebugPanel();
    }
  };
}

function renderSources(raw) {
  try {
    const sources = JSON.parse(raw);
    if (!Array.isArray(sources) || sources.length === 0) {
      return [];
    }

    const sourceMsg = createMessageElement('system');
    sourceMsg.textContent = '';

    const header = document.createElement('div');
    header.className = 'sources-header';
    header.textContent = '引用來源';
    sourceMsg.appendChild(header);

    const container = document.createElement('div');
    container.className = 'sources-container';

    for (const s of sources) {
      const groupText = s.relation_group ? ` [${s.relation_group}]` : '';

      if (s.type === 'image_caption' && s.image_url) {
        const row = document.createElement('div');
        row.className = 'source-row source-image';

        const thumb = document.createElement('img');
        thumb.className = 'source-thumbnail';
        thumb.src = s.image_url;
        thumb.alt = s.image_alt || '圖片';
        thumb.title = s.image_alt || s.title || '';
        thumb.loading = 'lazy';
        thumb.addEventListener('click', () => window.open(s.image_url, '_blank'));

        const label = document.createElement('span');
        label.className = 'source-label';
        label.textContent = `[圖${s.id}]${groupText} ${s.image_alt || s.title || '圖片'}`;

        thumb.addEventListener('error', () => {
          thumb.style.display = 'none';
          label.textContent = `[圖${s.id}]${groupText} ${s.title || '圖片'} (圖片已失效)`;
        });

        row.appendChild(thumb);
        row.appendChild(label);
        container.appendChild(row);
      } else {
        const row = document.createElement('div');
        row.className = 'source-row';

        const label = document.createElement('span');
        label.className = 'source-label';
        label.textContent = `[${s.id}]${groupText} ${s.title || '無標題'} ${s.url || ''}`;

        row.appendChild(label);
        container.appendChild(row);
      }
    }

    sourceMsg.appendChild(container);
    $messages.appendChild(sourceMsg);
    scrollToBottom();
    return sources;
  } catch (_e) {
    return [];
  }
}

function stopStreaming() {
  manualStopRequested = true;

  if (currentPort) {
    currentPort.postMessage({ type: 'ABORT' });
  }
  if (currentAbortController) {
    currentAbortController.abort();
  }
}

function finishStreaming() {
  isStreaming = false;
  currentPort = null;
  currentAbortController = null;
  clearRagTimeout();

  $btnStop.classList.add('hidden');
  $userInput.disabled = false;

  renderRuntimeInfo();
  updateSendState();
  $userInput.focus();

  manualStopRequested = false;
  ragTimedOut = false;
}

function normalizeTipValue(value) {
  if (value === null || value === undefined || value === '') {
    return '-';
  }
  return String(value);
}

function formatTipMetrics(metrics) {
  const statusMap = {
    retrieving: '檢索中',
    generating: '生成中',
    streaming: '生成中',
    done: '完成',
    error: '錯誤',
    aborted: '已停止'
  };

  const rows = [
    ['狀態', statusMap[metrics.status] || normalizeTipValue(metrics.status)],
    ['模式', metrics.mode],
    ['聊天模型', metrics.model_used],
    ['嵌入模型', metrics.embedding_model_used],
    ['查詢範圍', metrics.scope],
    ['Top-K', metrics.top_k],
    ['指定頁面數', metrics.selected_source_count],
    ['命中 chunks', metrics.retrieved_chunks],
    ['檢索命中率', metrics.retrieval_hit_rate !== undefined ? formatPercent(metrics.retrieval_hit_rate) : '-'],
    ['命中來源數', metrics.unique_source_count],
    ['範圍群組數', metrics.scope_group_count],
    ['命中群組數', metrics.hit_group_count],
    ['群組命中率', metrics.relation_group_hit_rate !== undefined ? formatPercent(metrics.relation_group_hit_rate) : '-'],
    ['結果群組', metrics.result_relation_groups],
    ['首 token 延遲', metrics.first_token_ms !== undefined ? formatDurationMs(metrics.first_token_ms) : '-'],
    ['總延遲', metrics.total_latency_ms !== undefined ? formatDurationMs(metrics.total_latency_ms) : '-'],
    ['Ollama 總耗時', metrics.ollama_total_duration_ms !== undefined ? formatDurationMs(metrics.ollama_total_duration_ms) : '-'],
    ['Ollama 載入耗時', metrics.ollama_load_duration_ms !== undefined ? formatDurationMs(metrics.ollama_load_duration_ms) : '-'],
    ['Prompt tokens', metrics.prompt_eval_count],
    ['Eval tokens', metrics.eval_count],
    ['回覆字數', metrics.response_chars],
    ['問題字數', metrics.query_length],
    ['上下文字數', metrics.context_chars_sent]
  ];

  if (metrics.error) {
    rows.push(['錯誤', metrics.error]);
  }

  return rows
    .filter(([, value]) => value !== undefined)
    .map(([label, value]) => `${label}: ${normalizeTipValue(value)}`)
    .join('\n');
}

function createAssistantTipCard(initialMetrics = {}) {
  const container = document.createElement('div');
  container.className = 'assistant-card';

  const header = document.createElement('div');
  header.className = 'assistant-card-header';

  const messageEl = createMessageElement('assistant');
  const tipToggle = document.createElement('button');
  tipToggle.className = 'tip-toggle';
  tipToggle.innerHTML = `
    <span style="display:flex; align-items:center; gap:4px;">
      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
      詳細指標
    </span>
  `;
  tipToggle.title = '查看問答指標';

  const tipPanel = document.createElement('div');
  tipPanel.className = 'tip-panel hidden';

  const metrics = { ...initialMetrics };

  const render = () => {
    tipPanel.textContent = formatTipMetrics(metrics);
  };

  tipToggle.addEventListener('click', () => {
    tipPanel.classList.toggle('hidden');
  });

  header.appendChild(messageEl);
  header.appendChild(tipToggle);
  container.appendChild(header);
  container.appendChild(tipPanel);
  render();

  return {
    container,
    messageEl,
    setMetrics(patch = {}) {
      Object.assign(metrics, patch);
      render();
    }
  };
}

function createMessageElement(role) {
  const el = document.createElement('div');
  el.className = `message ${role}`;
  return el;
}

// ─── Streaming Markdown Renderer ────────────────────────────────

function createThinkFilter() {
  let inThinkBlock = false;
  let buffer = '';

  return function filterChunk(chunk) {
    let result = '';
    buffer += chunk;

    while (buffer.length > 0) {
      if (inThinkBlock) {
        const closeIdx = buffer.indexOf('</think>');
        if (closeIdx >= 0) {
          buffer = buffer.slice(closeIdx + 8);
          inThinkBlock = false;
        } else {
          if (buffer.length > 7) {
            buffer = buffer.slice(-7);
          }
          break;
        }
      } else {
        const openIdx = buffer.indexOf('<think>');
        if (openIdx >= 0) {
          result += buffer.slice(0, openIdx);
          buffer = buffer.slice(openIdx + 7);
          inThinkBlock = true;
        } else {
          if (buffer.length > 6) {
            result += buffer.slice(0, -6);
            buffer = buffer.slice(-6);
          }
          break;
        }
      }
    }

    return result;
  };
}

function initStreamParser(el) {
  el.classList.add('streaming');
  el.innerHTML = '';

  const renderer = smd.default_renderer(el);
  const p = smd.parser(renderer);
  const thinkFilter = createThinkFilter();

  return {
    parser: p,
    el,
    thinkFilter,
    totalLength: 0,
    lastSanitizeAt: 0,
    fullResponse: '',
    aborted: false
  };
}

function feedStreamChunk(stream, chunk) {
  if (stream.aborted) return;

  stream.fullResponse += chunk;
  const filtered = stream.thinkFilter(chunk);
  if (!filtered) return;

  stream.totalLength += filtered.length;
  if (stream.totalLength - stream.lastSanitizeAt > 500) {
    stream.lastSanitizeAt = stream.totalLength;
    if (typeof DOMPurify !== 'undefined') {
      DOMPurify.sanitize(stream.fullResponse);
      if (DOMPurify.removed.length > 0) {
        console.warn('[Security] Potentially malicious content detected in LLM output');
        stream.aborted = true;
        smd.parser_end(stream.parser);
        stream.el.classList.remove('streaming');
        const warning = document.createElement('div');
        warning.className = 'message error';
        warning.textContent = '⚠ 偵測到不安全內容，已中斷渲染';
        stream.el.parentElement?.appendChild(warning);
        return;
      }
    }
  }

  smd.parser_write(stream.parser, filtered);
}

function finalizeStream(stream) {
  if (!stream) return;

  if (!stream.aborted) {
    smd.parser_end(stream.parser);
  }

  stream.el.classList.remove('streaming');

  if (typeof hljs !== 'undefined') {
    const codeBlocks = stream.el.querySelectorAll('pre code');
    codeBlocks.forEach((block) => {
      hljs.highlightElement(block);
    });
  }
}

function addUserMessage(text) {
  const el = createMessageElement('user');
  el.textContent = text;
  $messages.appendChild(el);
  scrollToBottom();
}

function addSystemMessage(text) {
  const el = createMessageElement('system');
  el.textContent = text;
  $messages.appendChild(el);
  scrollToBottom();
}

function addErrorMessage(text) {
  const el = createMessageElement('error');
  el.textContent = `⚠ ${text}`;
  $messages.appendChild(el);
  scrollToBottom();
  setTaskStatus('error', '發生錯誤', { progress: 0, error: text });
}

let lastHint = '';
function addTransientHint(text) {
  if (lastHint === text) {
    return;
  }
  lastHint = text;
  setTimeout(() => {
    lastHint = '';
  }, 2000);
}

function scrollToBottom() {
  $chatContainer.scrollTop = $chatContainer.scrollHeight;
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function getDomainFromUrl(url) {
  try {
    return new URL(url).hostname;
  } catch (_e) {
    return 'general';
  }
}

function renderSurfacingMessage(taskData) {
  const results = taskData.surfacing_results || [];
  const tags = taskData.tagging_tags || [];
  const title = taskData.title || '';

  const el = createMessageElement('system');
  el.classList.add('surfacing-msg');
  el.style.textAlign = 'left';

  let html = '';
  html += '<div class="surfacing-title">' + escapeHtml('擷取完成：「' + title + '」') + '</div>';

  if (tags.length > 0) {
    html += '<div class="surfacing-tags">自動歸類：' + tags.map(t => escapeHtml(t)).join('、') + '</div>';
  } else {
    html += '<div class="surfacing-tags">未能自動歸類</div>';
  }

  if (results.length > 0) {
    html += '<div class="surfacing-related">知識庫中相關文件：</div>';
    html += '<ol class="surfacing-list">';
    results.forEach(r => {
      const similarity = Math.round((1 - r.distance) * 100) + '%';
      const rTags = (r.tags || []).length > 0 ? '（' + r.tags.join('、') + '）' : '';
      html += '<li>「' + escapeHtml(r.title) + '」';
      html += '<span class="surfacing-group">' + escapeHtml(r.relation_group) + '</span>';
      html += escapeHtml(rTags);
      html += ' — 相似度 ' + similarity + '</li>';
    });
    html += '</ol>';
    html += '<button class="btn-primary btn-small surfacing-apply">套用為查詢範圍</button>';
  }

  el.innerHTML = html;
  $messages.appendChild(el);
  scrollToBottom();

  const applyBtn = el.querySelector('.surfacing-apply');
  if (applyBtn) {
    applyBtn.addEventListener('click', () => {
      if (applyBtn.disabled) return;

      const allTags = new Set(tags);
      results.forEach(r => (r.tags || []).forEach(t => allTags.add(t)));

      const allGroups = new Set();
      results.forEach(r => {
        if (r.relation_group) allGroups.add(r.relation_group);
      });

      allTags.forEach(t => {
        if (!selectedTags.includes(t)) selectedTags.push(t);
      });

      allGroups.forEach(g => {
        if (!selectedRelationGroups.includes(g)) selectedRelationGroups.push(g);
      });

      syncTagScopeOptions();
      renderRelationScopeStatus();
      syncRelationScopeOptions();
      saveCurrentSettings();

      applyBtn.textContent = '已套用';
      applyBtn.disabled = true;
    });
  }
}

init();
