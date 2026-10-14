(() => {
  "use strict";

  if (window.__mindflowFinalLoaded) return;
  window.__mindflowFinalLoaded = true;

  const TARGET_FPS = 30;
  const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;
  const FRAME_WIDTH = 640;
  const FRAME_HEIGHT = 480;
  const MAX_ANALYZED_SOURCES = 5;
  const SOURCE_GRACE_MS = 12_000;
  const HISTORY_WINDOW_MS = 5 * 60 * 1000;
  const HISTORY_SAMPLE_MS = 2_000;
  const PANEL_POSITION_KEY = "mindflow-final-panel-position-v1";
  const PREFERENCES_KEY = "mindflowExtensionPreferences";
  const SESSION_STARTED_AT = performance.now();

  const DECISIONS = Object.freeze({
    ANALYZE: "ANALYZE",
    IGNORE: "IGNORE"
  });

  const LABELS = Object.freeze({
    engagement: "Engajamento",
    boredom: "Tédio",
    confusion: "Confusão",
    frustration: "Frustração"
  });
  const DEFAULT_PREFERENCES = Object.freeze({
    theme: "light",
    density: "compact",
    scale: "normal",
    showTrend: true,
    showConfidence: true,
    showSecondary: true,
    showChart: true
  });

  const sources = new Map();
  const aliases = new Map();
  const videoBindings = new WeakMap();
  const overlays = new Map();
  const predictionHistory = new Map();
  const groupHistory = [];
  const localCameras = { deviceIds: new Set(), labels: new Set() };

  let sourceCounter = 1;
  let panel = null;
  let pythonHealth = null;
  let pythonError = "Aguardando o servidor local";
  let scanScheduled = false;
  let lastGroupSampleAt = 0;
  let preferences = { ...DEFAULT_PREFERENCES };

  function normalizeText(value) {
    return String(value || "")
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .toLowerCase()
      .replace(/\s+/g, " ")
      .trim();
  }

  function escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function getVideoTrack(video) {
    try {
      return video.srcObject?.getVideoTracks?.()[0] || null;
    } catch {
      return null;
    }
  }

  function getTrackSettings(track) {
    try {
      return track?.getSettings?.() || {};
    } catch {
      return {};
    }
  }

  function isVisible(video) {
    if (!video?.isConnected) return false;
    const style = getComputedStyle(video);
    const rect = video.getBoundingClientRect();
    return style.display !== "none" &&
      style.visibility !== "hidden" &&
      Number(style.opacity || 1) > 0 &&
      rect.width >= 48 && rect.height >= 36 &&
      rect.bottom > 0 && rect.right > 0 &&
      rect.top < innerHeight && rect.left < innerWidth;
  }

  function ancestorChain(video, limit = 7) {
    const chain = [];
    let node = video;
    for (let index = 0; node && index < limit; index += 1, node = node.parentElement) {
      chain.push(node);
    }
    return chain;
  }

  function identityKeys(video) {
    const keys = [];
    const track = getVideoTrack(video);
    const stream = video.srcObject;
    const settings = getTrackSettings(track);

    if (track?.id) keys.push(`track:${track.id}`);
    if (stream?.id) keys.push(`stream:${stream.id}`);

    const usefulAttributes = [
      "data-participant-id",
      "data-requested-participant-id",
      "data-ssrc",
      "data-stream-id"
    ];
    for (const node of ancestorChain(video)) {
      for (const attribute of usefulAttributes) {
        const value = node.getAttribute?.(attribute);
        if (value) keys.push(`dom:${attribute}:${value}`);
      }
    }
    if (settings.deviceId) keys.push(`device:${settings.deviceId}`);
    return [...new Set(keys)];
  }

  function createSourceId() {
    return `participant_${String(sourceCounter++).padStart(2, "0")}`;
  }

  function sourceForVideo(video) {
    const keys = identityKeys(video);
    const signature = keys.filter(key => key.startsWith("track:") || key.startsWith("stream:"))[0] || keys[0] || "";
    const bound = videoBindings.get(video);

    if (bound && (!signature || bound.signature === signature) && sources.has(bound.id)) {
      return sources.get(bound.id);
    }

    let id = keys.map(key => aliases.get(key)).find(candidate => sources.has(candidate));
    if (!id) id = createSourceId();

    let source = sources.get(id);
    if (!source) {
      const canvas = document.createElement("canvas");
      canvas.width = FRAME_WIDTH;
      canvas.height = FRAME_HEIGHT;
      source = {
        id,
        video,
        canvas,
        decision: null,
        automaticRole: "PARTICIPANT",
        automaticReason: "",
        identityKeys: new Set(),
        lastSeen: Date.now(),
        lastCaptureAt: 0,
        inFlight: false,
        lastResponse: null,
        lastError: "",
        lastPredictionFrame: null,
        measuredFps: 0
      };
      sources.set(id, source);
    }

    source.video = video;
    source.lastSeen = Date.now();
    for (const key of keys) {
      aliases.set(key, id);
      source.identityKeys.add(key);
    }
    videoBindings.set(video, { id, signature });
    return source;
  }

  function tileRoot(video) {
    const videoRect = video.getBoundingClientRect();
    let best = video.parentElement || video;
    for (const node of ancestorChain(video, 8).slice(1)) {
      const rect = node.getBoundingClientRect();
      if (rect.width < videoRect.width * 0.85 || rect.height < videoRect.height * 0.85) continue;
      if (rect.width > Math.max(videoRect.width * 2.2, 900)) break;
      if (rect.height > Math.max(videoRect.height * 2.2, 700)) break;
      best = node;
    }
    return best;
  }

  function nearbyText(video) {
    const root = tileRoot(video);
    return normalizeText([
      root?.innerText || "",
      root?.getAttribute?.("aria-label") || "",
      getVideoTrack(video)?.label || ""
    ].join(" ")).slice(0, 1500);
  }

  function anonymousSourceLabel(sourceId) {
    const number = String(sourceId || "").match(/(\d+)$/)?.[1] || "--";
    return `Fonte ${number}`;
  }

  function detectAutomaticRole(video) {
    const track = getVideoTrack(video);
    const settings = getTrackSettings(track);
    const label = normalizeText(track?.label);
    const text = nearbyText(video);
    const displaySurface = normalizeText(settings.displaySurface);
    const screenWords = [
      "tela apresentada", "apresentando", "apresentacao de", "presentation from",
      "presenting", "screen share", "shared screen", "compartilhamento de tela"
    ];

    if (["monitor", "window", "browser"].includes(displaySurface)) {
      return { role: "SCREEN", reason: `displaySurface=${displaySurface}` };
    }
    if (/screen|display|window|monitor|tela/i.test(label) && !/camera|webcam/i.test(label)) {
      return { role: "SCREEN", reason: "rótulo da faixa indica compartilhamento" };
    }
    if (screenWords.some(word => text.includes(normalizeText(word)))) {
      return { role: "SCREEN", reason: "texto do Meet indica apresentação" };
    }

    const deviceId = String(settings.deviceId || "");
    const normalizedLabel = normalizeText(track?.label).replace(/\([^)]*\)/g, "").trim();
    if (deviceId && localCameras.deviceIds.has(deviceId)) {
      return { role: "INTERVIEWER", reason: "mesmo dispositivo da câmera local" };
    }
    if (normalizedLabel && localCameras.labels.has(normalizedLabel)) {
      return { role: "INTERVIEWER", reason: "mesmo nome da câmera local" };
    }
    if (/\b(você|you)\b/.test(text) || text.includes("sua apresentacao") || text.includes("your presentation")) {
      return { role: "INTERVIEWER", reason: "tile marcado como você pelo Meet" };
    }
    return { role: "PARTICIPANT", reason: "sem classificação automática confiável" };
  }

  function effectiveState(source) {
    if (source.automaticRole === "SCREEN") return "AUTO_SCREEN";
    if (source.automaticRole === "INTERVIEWER") return "AUTO_INTERVIEWER";
    if (source.decision === DECISIONS.ANALYZE) return "ANALYZING";
    if (source.decision === DECISIONS.IGNORE) return "IGNORED";
    return "PENDING";
  }

  function stateCopy(state) {
    const copy = {
      ANALYZING: ["IA ATIVA", "Frames enviados somente ao Python local"],
      IGNORED: ["IGNORADO", "Não enviado à IA"],
      AUTO_SCREEN: ["TELA DETECTADA", "Bloqueada automaticamente"],
      AUTO_INTERVIEWER: ["APRESENTADOR DETECTADO", "Bloqueado automaticamente"],
      PENDING: ["REVISAR", "Não enviado até você decidir"]
    };
    return copy[state] || copy.PENDING;
  }

  function analyzedCount(exceptId = "") {
    return [...sources.values()].filter(source => source.id !== exceptId && source.decision === DECISIONS.ANALYZE).length;
  }

  function sendMessage(message) {
    return new Promise(resolve => {
      try {
        chrome.runtime.sendMessage(message, response => {
          if (chrome.runtime.lastError) {
            resolve({ success: false, error: chrome.runtime.lastError.message });
            return;
          }
          resolve(response || { success: false, error: "Sem resposta" });
        });
      } catch (error) {
        resolve({ success: false, error: String(error) });
      }
    });
  }

  function resetRemote(sourceId) {
    void sendMessage({ type: "MINDFLOW_RESET_PARTICIPANT", participantId: sourceId });
  }

  function clearSourceResults(source) {
    source.lastResponse = null;
    source.lastError = "";
    source.lastPredictionFrame = null;
    source.measuredFps = 0;
    predictionHistory.delete(source.id);
  }

  function setDecision(sourceId, requestedDecision) {
    const source = sources.get(sourceId);
    if (!source) return;
    if (source.automaticRole !== "PARTICIPANT") return;
    const nextDecision = source.decision === requestedDecision ? null : requestedDecision;
    if (nextDecision === DECISIONS.ANALYZE && analyzedCount(sourceId) >= MAX_ANALYZED_SOURCES) {
      window.alert("O protótipo permite analisar no máximo 5 fontes ao mesmo tempo.");
      return;
    }
    source.decision = nextDecision;
    source.inFlight = false;
    clearSourceResults(source);
    resetRemote(source.id);
    renderOverlay(source);
    renderPanel();
  }

  function ensureOverlay(source) {
    let overlay = overlays.get(source.id);
    if (overlay?.isConnected) return overlay;
    overlay = document.createElement("section");
    overlay.className = "mindflow-source-overlay";
    overlay.dataset.sourceId = source.id;
    overlay.addEventListener("pointerdown", event => event.stopPropagation());
    overlay.addEventListener("click", event => {
      event.stopPropagation();
      const button = event.target.closest("button[data-decision]");
      if (!button) return;
      setDecision(source.id, button.dataset.decision);
    });
    document.body.appendChild(overlay);
    overlays.set(source.id, overlay);
    return overlay;
  }

  function positionOverlay(source, overlay) {
    const rect = source.video.getBoundingClientRect();
    overlay.style.left = `${Math.max(4, rect.left + 6)}px`;
    overlay.style.top = `${Math.max(4, rect.top + 6)}px`;
    overlay.style.maxWidth = `${Math.max(160, rect.width - 12)}px`;
    overlay.classList.toggle("mindflow-source-overlay--small", rect.width < 250);
  }

  function renderOverlay(source) {
    if (!source.video?.isConnected || !isVisible(source.video)) return;
    const overlay = ensureOverlay(source);
    const state = effectiveState(source);
    const [title, detail] = stateCopy(state);
    const quality = source.lastResponse
      ? `${source.lastResponse.faceDetected ? "rosto" : "sem rosto"} · ${source.lastResponse.poseDetected ? "pose" : "sem pose"}`
      : source.lastError ? "Python local indisponível" : detail;
    const buttons = source.automaticRole === "PARTICIPANT"
      ? [
          [DECISIONS.ANALYZE, "Analisar"],
          [DECISIONS.IGNORE, "Ignorar"]
        ].map(([decision, label]) =>
          `<button type="button" data-decision="${decision}" class="${source.decision === decision ? "is-active" : ""}">${label}</button>`
        ).join("")
      : `<span class="mf-auto-block">Bloqueio automático</span>`;

    overlay.dataset.state = state;
    overlay.innerHTML = `
      <div class="mf-source-head">
        <span>ANÔNIMA</span>
        <strong>${escapeHtml(anonymousSourceLabel(source.id))}</strong>
      </div>
      <div class="mf-source-status"><b>${escapeHtml(title)}</b><small>${escapeHtml(quality)}</small></div>
      <div class="mf-source-actions">${buttons}</div>`;
    positionOverlay(source, overlay);
  }

  function removeSource(source) {
    resetRemote(source.id);
    overlays.get(source.id)?.remove();
    overlays.delete(source.id);
    sources.delete(source.id);
    predictionHistory.delete(source.id);
    for (const key of source.identityKeys) {
      if (aliases.get(key) === source.id) aliases.delete(key);
    }
  }

  function scanVideos() {
    const now = Date.now();
    const activeIds = new Set();
    for (const video of document.querySelectorAll("video")) {
      if (!isVisible(video)) continue;
      const source = sourceForVideo(video);
      const automatic = detectAutomaticRole(video);
      const wasAuthorized = source.decision === DECISIONS.ANALYZE;
      source.automaticRole = automatic.role;
      source.automaticReason = automatic.reason;
      if (automatic.role !== "PARTICIPANT" && source.decision !== null) {
        source.decision = null;
        source.inFlight = false;
        clearSourceResults(source);
        if (wasAuthorized) resetRemote(source.id);
      }
      source.lastSeen = now;
      activeIds.add(source.id);
      renderOverlay(source);
    }

    for (const source of [...sources.values()]) {
      if (!activeIds.has(source.id)) {
        overlays.get(source.id)?.remove();
        overlays.delete(source.id);
        if (now - source.lastSeen > SOURCE_GRACE_MS) removeSource(source);
      }
    }
    renderPanel();
  }

  function scheduleScan() {
    if (scanScheduled) return;
    scanScheduled = true;
    setTimeout(() => {
      scanScheduled = false;
      scanVideos();
    }, 120);
  }

  function extractPrediction(inference) {
    if (!inference?.ready || !inference.states) return null;
    const scores = {};
    const confidences = {};
    for (const key of Object.keys(LABELS)) {
      const value = Number(inference.states[key]?.scorePercent);
      if (!Number.isFinite(value)) return null;
      scores[key] = Math.max(0, Math.min(100, value));
      const confidence = Number(inference.states[key]?.confidencePercent);
      confidences[key] = Number.isFinite(confidence) ? Math.max(0, Math.min(100, confidence)) : 0;
    }
    return { scores, confidences };
  }

  function recordPrediction(source, response) {
    const prediction = extractPrediction(response.inference);
    if (!prediction) return;
    const inferenceFrame = Number(response.inference.updatedAtFrame ?? response.frameIdx);
    if (source.lastPredictionFrame === inferenceFrame) return;
    source.lastPredictionFrame = inferenceFrame;
    const history = predictionHistory.get(source.id) || [];
    history.push({ ...prediction, at: Date.now() });
    while (history.length > 5) history.shift();
    predictionHistory.set(source.id, history);
  }

  function smoothedScores(sourceId) {
    const history = predictionHistory.get(sourceId);
    if (!history?.length || Date.now() - history.at(-1).at > 6_000) return null;
    const result = { scores: {}, confidences: {} };
    for (const key of Object.keys(LABELS)) {
      result.scores[key] = history.reduce((sum, item) => sum + item.scores[key], 0) / history.length;
      result.confidences[key] = history.reduce((sum, item) => sum + item.confidences[key], 0) / history.length;
    }
    return result;
  }

  function groupScores() {
    const values = [...sources.values()]
      .filter(source => effectiveState(source) === "ANALYZING")
      .map(source => smoothedScores(source.id))
      .filter(Boolean);
    if (!values.length) return null;
    const result = { participants: values.length, confidences: {} };
    for (const key of Object.keys(LABELS)) {
      result[key] = values.reduce((sum, item) => sum + item.scores[key], 0) / values.length;
      result.confidences[key] = values.reduce((sum, item) => sum + item.confidences[key], 0) / values.length;
    }
    return result;
  }

  function intensityLabel(score) {
    if (score >= 83) return "muito alto";
    if (score >= 50) return "alto";
    if (score >= 17) return "baixo";
    return "muito baixo";
  }

  function farol(scores) {
    if (!scores) return { color: "gray", title: "Coletando dados", confidence: 0, key: null };
    const key = Object.keys(LABELS).reduce((best, candidate) => scores[candidate] > scores[best] ? candidate : best, "engagement");
    const colors = { engagement: "green", confusion: "yellow", boredom: "red", frustration: "red" };
    return {
      color: scores[key] >= 50 ? colors[key] : "gray",
      title: `${LABELS[key]} ${intensityLabel(scores[key])}`,
      confidence: scores.confidences?.[key] || 0,
      key
    };
  }

  function trendArrow() {
    if (groupHistory.length < 2) return "→";
    const recent = groupHistory.at(-1).value;
    const previous = groupHistory[Math.max(0, groupHistory.length - 4)].value;
    const delta = recent - previous;
    if (delta > 3) return "↗";
    if (delta < -3) return "↘";
    return "→";
  }

  function recordGroupPoint(scores) {
    if (!scores || Date.now() - lastGroupSampleAt < HISTORY_SAMPLE_MS) return;
    lastGroupSampleAt = Date.now();
    groupHistory.push({ at: lastGroupSampleAt, value: scores.engagement });
    const cutoff = lastGroupSampleAt - HISTORY_WINDOW_MS;
    while (groupHistory.length && groupHistory[0].at < cutoff) groupHistory.shift();
  }

  function chartPoints() {
    if (!groupHistory.length) return "";
    const now = Date.now();
    return groupHistory.map(point => {
      const x = Math.max(0, Math.min(100, ((point.at - (now - HISTORY_WINDOW_MS)) / HISTORY_WINDOW_MS) * 100));
      const y = 34 - Math.max(0, Math.min(100, point.value)) * 0.34;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
  }

  function historyStatus() {
    const history = pythonHealth?.history;
    if (!pythonHealth) return "Histórico: aguardando Python";
    if (history?.enabled) return "Histórico Supabase: ativo";
    return "Histórico Supabase: desativado";
  }

  function ensurePanel() {
    if (panel?.isConnected) return panel;
    panel = document.createElement("aside");
    panel.id = "mindflow-panel";
    panel.tabIndex = 0;
    panel.innerHTML = `<header data-drag-handle><strong>MindFlow AI</strong><span>GRUPO</span></header><div class="mf-body"></div>`;
    document.body.appendChild(panel);
    makePanelDraggable(panel);
    return panel;
  }

  function makePanelDraggable(element) {
    const handle = element.querySelector("[data-drag-handle]");
    let dragging = false;
    let offsetX = 0;
    let offsetY = 0;
    handle.addEventListener("pointerdown", event => {
      if (event.button !== 0) return;
      const rect = element.getBoundingClientRect();
      dragging = true;
      offsetX = event.clientX - rect.left;
      offsetY = event.clientY - rect.top;
      handle.setPointerCapture?.(event.pointerId);
      event.preventDefault();
    });
    handle.addEventListener("pointermove", event => {
      if (!dragging) return;
      const left = Math.max(8, Math.min(innerWidth - element.offsetWidth - 8, event.clientX - offsetX));
      const top = Math.max(8, Math.min(innerHeight - 60, event.clientY - offsetY));
      element.style.left = `${left}px`;
      element.style.top = `${top}px`;
      element.style.right = "auto";
    });
    const finish = () => {
      if (!dragging) return;
      dragging = false;
      const rect = element.getBoundingClientRect();
      try { localStorage.setItem(PANEL_POSITION_KEY, JSON.stringify({ left: rect.left, top: rect.top })); } catch { /* nada */ }
    };
    handle.addEventListener("pointerup", finish);
    handle.addEventListener("pointercancel", finish);
    try {
      const saved = JSON.parse(localStorage.getItem(PANEL_POSITION_KEY) || "null");
      if (Number.isFinite(saved?.left) && Number.isFinite(saved?.top)) {
        element.style.left = `${Math.max(8, Math.min(innerWidth - 308, saved.left))}px`;
        element.style.top = `${Math.max(8, saved.top)}px`;
        element.style.right = "auto";
      }
    } catch { /* nada */ }
  }

  function renderPanel() {
    const element = ensurePanel();
    const body = element.querySelector(".mf-body");
    const scores = groupScores();
    const light = farol(scores);
    recordGroupPoint(scores);
    element.dataset.light = light.color;
    element.dataset.theme = preferences.theme;
    element.dataset.density = preferences.density;
    element.dataset.scale = preferences.scale;

    const scoreHtml = scores
      ? Object.entries(LABELS).map(([key, label]) => `<span><b>${Math.round(scores[key])}%</b><small>${label}</small></span>`).join("")
      : `<em>Aguardando 30 frames de alguma fonte autorizada.</em>`;
    const sourceHtml = [...sources.values()].map(source => {
      const state = effectiveState(source);
      const title = stateCopy(state)[0];
      const fps = source.measuredFps ? ` · ${source.measuredFps.toFixed(1)} fps reais` : "";
      return `<li data-state="${state}"><b>${escapeHtml(anonymousSourceLabel(source.id))}</b><small>${escapeHtml(title + fps)}</small></li>`;
    }).join("") || `<li class="mf-empty">Nenhum vídeo visível no Meet.</li>`;

    const pythonText = pythonHealth
      ? `Python conectado · modelo ${pythonHealth.modelLoaded ? "carregado" : "indisponível"} · alvo ${pythonHealth.targetFps || TARGET_FPS} fps`
      : `Python desconectado · ${pythonError}`;

    body.innerHTML = `
      <div class="mf-status"><i id="mindflow-light"></i><div><b>${escapeHtml(light.title)} ${preferences.showTrend && scores ? `<span class="mf-trend">${trendArrow()}</span>` : ""}</b><small>${preferences.showConfidence && scores ? `Confiança: ${Math.round(light.confidence)}% · ` : ""}${scores ? `${scores.participants} fonte(s)` : "Nenhuma previsão ativa"}</small></div></div>
      <div class="mf-details">
        ${preferences.showSecondary ? `<div class="mf-scores">${scoreHtml}</div>` : ""}
        ${preferences.showChart ? `<div class="mf-chart-title">Engajamento do grupo · últimos 5 minutos</div><svg viewBox="0 0 100 34" preserveAspectRatio="none" aria-label="Histórico de engajamento"><polyline points="${chartPoints()}"></polyline></svg>` : ""}
        <div class="mf-source-list-title">Fontes anônimas detectadas (${sources.size})</div>
        <ul class="mf-source-list">${sourceHtml}</ul>
        <small id="mindflow-python">${escapeHtml(pythonText)}<br>${escapeHtml(historyStatus())}<br>Nenhum nome de participante é coletado.</small>
      </div>`;
  }

  async function captureSource(source, now) {
    if (source.inFlight || now - source.lastCaptureAt < FRAME_INTERVAL_MS) return;
    if (effectiveState(source) !== "ANALYZING" || !isVisible(source.video)) return;
    if (source.video.readyState < 2 || !source.video.videoWidth || !source.video.videoHeight) return;
    source.inFlight = true;
    source.lastCaptureAt = now;
    const authorizedVideo = source.video;
    const ctx = source.canvas.getContext("2d", { alpha: false });

    try {
      ctx.drawImage(authorizedVideo, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
      const payload = {
        participantId: source.id,
        capturedAtUnixMs: Date.now(),
        sessionTimestampMs: performance.now() - SESSION_STARTED_AT,
        imageDataUrl: source.canvas.toDataURL("image/jpeg", 0.72)
      };
      const response = await sendMessage({ type: "MINDFLOW_PROCESS_FRAME", payload });

      const stillAuthorized = source.decision === DECISIONS.ANALYZE && source.video === authorizedVideo && sources.has(source.id);
      if (!stillAuthorized) return;
      if (!response?.success) throw new Error(response?.error || "Falha ao chamar o Python");
      source.lastResponse = response.python;
      source.lastError = "";
      source.measuredFps = Number(response.python.processingFps || 0);
      recordPrediction(source, response.python);
      renderOverlay(source);
      renderPanel();
    } catch (error) {
      if (source.decision === DECISIONS.ANALYZE) {
        source.lastError = String(error?.message || error);
        pythonError = source.lastError;
      }
    } finally {
      source.inFlight = false;
    }
  }

  function captureLoop(now) {
    for (const source of sources.values()) void captureSource(source, now);
    requestAnimationFrame(captureLoop);
  }

  async function checkPython() {
    const response = await sendMessage({ type: "MINDFLOW_HEALTH" });
    if (response?.success) {
      pythonHealth = response.python;
      pythonError = "";
    } else {
      pythonHealth = null;
      pythonError = response?.error || "servidor local indisponível";
    }
    renderPanel();
  }

  async function loadLocalCameraInfo() {
    localCameras.deviceIds.clear();
    localCameras.labels.clear();
    try {
      const devices = await navigator.mediaDevices?.enumerateDevices?.();
      for (const device of devices || []) {
        if (device.kind !== "videoinput") continue;
        if (device.deviceId) localCameras.deviceIds.add(device.deviceId);
        const label = normalizeText(device.label).replace(/\([^)]*\)/g, "").trim();
        if (label) localCameras.labels.add(label);
      }
    } catch { /* sem permissão para enumerar; revisão manual continua segura */ }
  }

  function normalizePreferences(value = {}) {
    return {
      theme: ["light", "dark"].includes(value.theme) ? value.theme : DEFAULT_PREFERENCES.theme,
      density: ["compact", "comfortable"].includes(value.density) ? value.density : DEFAULT_PREFERENCES.density,
      scale: ["small", "normal", "large"].includes(value.scale) ? value.scale : DEFAULT_PREFERENCES.scale,
      showTrend: value.showTrend !== false,
      showConfidence: value.showConfidence !== false,
      showSecondary: value.showSecondary !== false,
      showChart: value.showChart !== false
    };
  }

  async function loadPreferences() {
    try {
      const stored = await chrome.storage.local.get(PREFERENCES_KEY);
      preferences = normalizePreferences(stored[PREFERENCES_KEY]);
    } catch {
      preferences = { ...DEFAULT_PREFERENCES };
    }
    renderPanel();
  }

  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName !== "local" || !changes[PREFERENCES_KEY]) return;
    preferences = normalizePreferences(changes[PREFERENCES_KEY].newValue);
    renderPanel();
  });

  const observer = new MutationObserver(records => {
    const meetChanged = records.some(record => {
      const target = record.target?.nodeType === Node.ELEMENT_NODE
        ? record.target
        : record.target?.parentElement;
      return !target?.closest?.("#mindflow-panel, .mindflow-source-overlay");
    });
    if (meetChanged) scheduleScan();
  });
  observer.observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ["src", "style", "class"] });
  addEventListener("resize", scheduleScan);
  addEventListener("scroll", scheduleScan, true);
  navigator.mediaDevices?.addEventListener?.("devicechange", () => void loadLocalCameraInfo().then(scheduleScan));
  addEventListener("beforeunload", () => {
    for (const source of sources.values()) resetRemote(source.id);
  });

  void loadLocalCameraInfo().finally(scanVideos);
  void loadPreferences();
  void sendMessage({ type: "MINDFLOW_SYNC_PREFERENCES" });
  void checkPython();
  setInterval(scanVideos, 900);
  setInterval(checkPython, 5_000);
  setInterval(renderPanel, 1_000);
  requestAnimationFrame(captureLoop);
})();
