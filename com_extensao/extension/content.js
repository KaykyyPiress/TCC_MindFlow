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
  const PANEL_SIZE_KEY = "mindflow-final-panel-size-v1";
  const PREFERENCES_KEY = "mindflowExtensionPreferences";
  const AUTH_KEY = "mindflowAuthSession";
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
  const REACTION_HELP = Object.freeze({
    engagement: "Intensidade estimada de sinais de atenção e participação.",
    boredom: "Intensidade estimada de sinais associados a desinteresse ou baixa estimulação.",
    confusion: "Intensidade estimada de sinais associados a dúvida ou dificuldade de acompanhar.",
    frustration: "Intensidade estimada de sinais associados a tensão ou desconforto."
  });
  const DEFAULT_PREFERENCES = Object.freeze({
    preferencesVersion: 2,
    theme: "light",
    scale: "normal",
    fontSize: "normal",
    trendDisplay: "hover",
    confidenceDisplay: "hover",
    scoresDisplay: "always",
    chartDisplay: "always",
    sourcesDisplay: "hover",
    technicalDisplay: "hover"
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
  let accountState = { checked: false, signedIn: false, email: "" };
  let accountLoginOpen = false;
  let accountSyncState = "idle";
  let databaseMeeting = { state: "idle", meetKey: "", title: "", code: "", error: "" };
  let meetingWasActive = false;
  let meetingLifecycleBusy = false;

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
    if (!accountState.signedIn) return "LOGIN_REQUIRED";
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
      LOGIN_REQUIRED: ["LOGIN NECESSÁRIO", "Entre no MindFlow para liberar a análise"],
      PENDING: ["REVISAR", "Não enviado até você decidir"]
    };
    return copy[state] || copy.PENDING;
  }

  function analyzedCount(exceptId = "") {
    return [...sources.values()].filter(source => source.id !== exceptId && source.decision === DECISIONS.ANALYZE).length;
  }

  function sendMessage(message) {
    return new Promise(resolve => {
      const timeout = setTimeout(() => resolve({ success: false, error: "A extensão não respondeu em 12 segundos." }), 12_000);
      try {
        chrome.runtime.sendMessage(message, response => {
          clearTimeout(timeout);
          if (chrome.runtime.lastError) {
            resolve({ success: false, error: chrome.runtime.lastError.message });
            return;
          }
          resolve(response || { success: false, error: "Sem resposta" });
        });
      } catch (error) {
        clearTimeout(timeout);
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
    if (requestedDecision === DECISIONS.ANALYZE && !accountState.signedIn) {
      accountLoginOpen = true;
      renderPanel();
      window.alert("Entre na sua conta do MindFlow antes de iniciar a análise. Assim, a reunião e o histórico ficam vinculados ao usuário correto.");
      return;
    }
    if (requestedDecision === DECISIONS.ANALYZE && databaseMeeting.state !== "ready") {
      if (databaseMeeting.state !== "saving") void ensureDatabaseMeeting();
      window.alert(databaseMeeting.state === "error"
        ? `A reunião ainda não pôde ser salva: ${databaseMeeting.error}`
        : "Aguarde alguns segundos enquanto a reunião é registrada no Supabase.");
      return;
    }
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
          [DECISIONS.ANALYZE, accountState.signedIn ? "Analisar" : "Entrar para analisar"],
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
    const recent = groupHistory.at(-1).engagement;
    const previous = groupHistory[Math.max(0, groupHistory.length - 4)].engagement;
    const delta = recent - previous;
    if (delta > 3) return "↗";
    if (delta < -3) return "↘";
    return "→";
  }

  function recordGroupPoint(scores) {
    if (!scores || Date.now() - lastGroupSampleAt < HISTORY_SAMPLE_MS) return;
    lastGroupSampleAt = Date.now();
    groupHistory.push({
      at: lastGroupSampleAt,
      engagement: scores.engagement,
      boredom: scores.boredom,
      confusion: scores.confusion,
      frustration: scores.frustration
    });
    const cutoff = lastGroupSampleAt - HISTORY_WINDOW_MS;
    while (groupHistory.length && groupHistory[0].at < cutoff) groupHistory.shift();
  }

  function chartPoints(metric) {
    if (!groupHistory.length) return "";
    const now = Date.now();
    return groupHistory.map(point => {
      const x = Math.max(0, Math.min(100, ((point.at - (now - HISTORY_WINDOW_MS)) / HISTORY_WINDOW_MS) * 100));
      const y = 36 - Math.max(0, Math.min(100, point[metric])) * 0.36;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
  }

  function historyStatus() {
    const history = pythonHealth?.history;
    if (databaseMeeting.state === "saving") return "Histórico: preparando reunião no Supabase";
    if (databaseMeeting.state === "ending") return "Histórico: concluindo reunião";
    if (databaseMeeting.state === "ended") return "Histórico: reunião concluída";
    if (databaseMeeting.state === "error") return `Histórico: ${databaseMeeting.error}`;
    if (!pythonHealth) return "Histórico: aguardando Python";
    if (history?.enabled) {
      const code = history.sessionCode || databaseMeeting.code;
      const saved = Number(history.saved || 0);
      if (history.lastError) return "Histórico Supabase: ativo, mas o último envio falhou";
      return `Histórico Supabase: ativo${code ? ` · ${code}` : ""} · ${saved} registros`;
    }
    if (history?.available && databaseMeeting.state === "ready") {
      return "Histórico Supabase: conectando à reunião";
    }
    if (history?.available) return "Histórico Supabase: pronto · aguardando reunião";
    return "Histórico Supabase: desativado no Python";
  }

  function cleanTitleCandidate(value, meetCode) {
    const candidate = String(value || "")
      .replace(/\s*[|–—]\s*Google Meet\s*$/i, "")
      .replace(/^Google Meet\s*[-–—|:]?\s*/i, "")
      .replace(/^Meet\s*[-–—|:]?\s*/i, "")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 140);
    const normalized = normalizeText(candidate);
    const code = normalizeText(meetCode);
    const rejectedExact = ["meet", "google meet", "reuniao", "meeting"];
    const rejectedFragments = [
      "voce saiu da reuniao", "you left the meeting", "participar agora", "join now",
      "pronto para participar", "ready to join", "configurar dispositivos", "getting ready"
    ];
    if (candidate.length < 3 || rejectedExact.includes(normalized) || rejectedFragments.some(text => normalized.includes(text))) return "";
    if (code && (normalized === code || normalized.replace(/\s/g, "-") === code)) return "";
    if (/^[a-z]{3}-[a-z]{4}-[a-z]{3}$/i.test(candidate)) return "";
    return candidate;
  }

  function detectedMeetingTitle(meetCode) {
    const selectors = [
      "[data-meeting-title]",
      "[data-call-title]",
      "[aria-label*='título da reunião' i]",
      "[aria-label*='titulo da reuniao' i]",
      "[aria-label*='meeting title' i]"
    ];
    for (const selector of selectors) {
      for (const element of document.querySelectorAll(selector)) {
        const candidate = cleanTitleCandidate(
          element.getAttribute("data-meeting-title") ||
          element.getAttribute("data-call-title") ||
          element.getAttribute("aria-label") ||
          element.textContent,
          meetCode
        );
        if (candidate) return candidate;
      }
    }

    const fromDocument = cleanTitleCandidate(document.title, meetCode);
    if (fromDocument) return fromDocument;

    for (const heading of document.querySelectorAll("[role='heading'], h1, h2")) {
      const rect = heading.getBoundingClientRect();
      if (rect.width < 1 || rect.height < 1) continue;
      const candidate = cleanTitleCandidate(heading.textContent, meetCode);
      if (candidate && !/^(detalhes|participantes|atividades|chat|controles|reacoes)$/i.test(normalizeText(candidate))) {
        return candidate;
      }
    }
    return "";
  }

  function meetingIdentity() {
    const meetCode = location.pathname.split("/").filter(Boolean)[0] || "sala-sem-codigo";
    const detectedTitle = detectedMeetingTitle(meetCode);
    return {
      meetKey: `meet:${meetCode}`,
      title: detectedTitle || `Google Meet · ${meetCode}`
    };
  }

  function isFallbackMeetingTitle(value) {
    return /^Google Meet\s*·/i.test(String(value || "").trim());
  }

  async function ensureDatabaseMeeting() {
    if (!accountState.signedIn || databaseMeeting.state === "saving" || databaseMeeting.state === "ready") return;
    const identity = meetingIdentity();
    databaseMeeting = { state: "saving", ...identity, code: "", error: "" };
    renderPanel();
    const response = await sendMessage({ type: "MINDFLOW_ENSURE_MEETING", meeting: identity });
    if (!response?.success) {
      databaseMeeting = { state: "error", ...identity, code: "", error: response?.error || "não foi possível salvar a reunião" };
      renderPanel();
      return;
    }
    databaseMeeting = {
      state: "ready",
      ...identity,
      title: response.meeting?.title || identity.title,
      code: response.meeting?.code || "",
      error: response.history?.ok === false ? response.history.message || "Python sem histórico" : ""
    };
    void checkPython();
    renderPanel();
  }

  async function sendMeetingHeartbeat() {
    if (!accountState.signedIn || databaseMeeting.state !== "ready" || !databaseMeeting.meetKey) return;
    const response = await sendMessage({
      type: "MINDFLOW_MEETING_HEARTBEAT",
      meetKey: databaseMeeting.meetKey
    });
    const refreshedHistory = response?.history?.history;
    if (response?.success && refreshedHistory && pythonHealth) {
      pythonHealth = { ...pythonHealth, history: refreshedHistory };
      renderPanel();
    }
  }

  function hasActiveCallControl() {
    const labels = [...document.querySelectorAll("button[aria-label], [role='button'][aria-label]")]
      .map(element => normalizeText(element.getAttribute("aria-label")));
    return labels.some(label =>
      label.includes("sair da chamada") ||
      label.includes("encerrar chamada") ||
      label.includes("leave call") ||
      label.includes("end call")
    );
  }

  function hasEndedMeetingView() {
    const text = normalizeText(document.body?.innerText || "");
    return [
      "voce saiu da reuniao",
      "a reuniao terminou",
      "esta reuniao terminou",
      "a chamada terminou",
      "voce foi removido da reuniao",
      "participar novamente",
      "voltar para a tela inicial",
      "you left the meeting",
      "the meeting has ended",
      "this meeting has ended",
      "the call has ended",
      "you were removed from the meeting",
      "rejoin",
      "return to home screen"
    ].some(phrase => text.includes(phrase));
  }

  async function finishDatabaseMeeting() {
    if (meetingLifecycleBusy || databaseMeeting.state !== "ready" || !databaseMeeting.meetKey) return;
    meetingLifecycleBusy = true;
    const current = { ...databaseMeeting };
    databaseMeeting = { ...current, state: "ending" };
    renderPanel();
    const response = await sendMessage({ type: "MINDFLOW_END_MEETING", meetKey: current.meetKey });
    databaseMeeting = response?.success
      ? { ...current, state: "ended", error: "" }
      : { ...current, state: "error", error: response?.error || "não foi possível concluir a reunião" };
    for (const source of sources.values()) {
      if (source.decision === DECISIONS.ANALYZE) resetRemote(source.id);
      source.decision = null;
      source.inFlight = false;
      clearSourceResults(source);
      renderOverlay(source);
    }
    meetingLifecycleBusy = false;
    renderPanel();
  }

  async function updateMeetingTitle(title) {
    const meetCode = location.pathname.split("/").filter(Boolean)[0] || "";
    const cleaned = cleanTitleCandidate(title, meetCode);
    if (!cleaned || databaseMeeting.state !== "ready") return false;
    const response = await sendMessage({
      type: "MINDFLOW_UPDATE_MEETING_TITLE",
      meetKey: databaseMeeting.meetKey,
      title: cleaned
    });
    if (!response?.success) return false;
    databaseMeeting.title = response.meeting?.title || cleaned;
    renderPanel();
    return true;
  }

  async function monitorMeetingLifecycle() {
    if (!accountState.signedIn || meetingLifecycleBusy) return;
    const active = hasActiveCallControl();
    if (active) {
      meetingWasActive = true;
      if (["idle", "ended", "error"].includes(databaseMeeting.state)) {
        if (databaseMeeting.state === "ended") {
          databaseMeeting = { state: "idle", meetKey: "", title: "", code: "", error: "" };
        }
        await ensureDatabaseMeeting();
      }
      if (databaseMeeting.state === "ready" && isFallbackMeetingTitle(databaseMeeting.title)) {
        const detected = meetingIdentity().title;
        if (!isFallbackMeetingTitle(detected)) await updateMeetingTitle(detected);
      }
      return;
    }
    if (meetingWasActive && hasEndedMeetingView()) await finishDatabaseMeeting();
  }

  function ensurePanel() {
    if (panel?.isConnected) return panel;
    panel = document.createElement("aside");
    panel.id = "mindflow-panel";
    panel.tabIndex = 0;
    panel.innerHTML = `<header data-drag-handle><div class="mf-brand"><strong>MindFlow AI</strong><button id="mindflow-meeting-title" type="button" title="Clique para corrigir o título da reunião">Google Meet</button></div><span>GRUPO</span></header><div class="mf-body"></div><button class="mf-resize-handle" type="button" aria-label="Redimensionar painel" title="Arraste para aumentar ou diminuir">↘</button>`;
    panel.addEventListener("click", event => {
      const action = event.target.closest("[data-account-action]")?.dataset.accountAction;
      if (action === "login") {
        accountLoginOpen = !accountLoginOpen;
        renderPanel();
      }
      if (action === "sync") {
        accountSyncState = "syncing";
        renderPanel();
        void sendMessage({ type: "MINDFLOW_SYNC_PREFERENCES" }).then(response => {
          accountSyncState = response?.success ? "success" : "error";
          if (response?.success) void loadPreferences();
          renderPanel();
          setTimeout(() => {
            accountSyncState = "idle";
            renderPanel();
          }, 2500);
        });
      }
      if (event.target.closest("#mindflow-meeting-title")) {
        event.stopPropagation();
        const title = window.prompt("Título desta reunião no histórico:", databaseMeeting.title || meetingIdentity().title);
        if (title?.trim()) void updateMeetingTitle(title).then(success => {
          if (!success) window.alert("Não foi possível atualizar o título. Verifique se a reunião já foi registrada e tente novamente.");
        });
      }
    });
    document.body.appendChild(panel);
    makePanelDraggable(panel);
    makePanelResizable(panel);
    return panel;
  }

  function makePanelDraggable(element) {
    const handle = element.querySelector("[data-drag-handle]");
    let dragging = false;
    let offsetX = 0;
    let offsetY = 0;
    handle.addEventListener("pointerdown", event => {
      if (event.button !== 0) return;
      if (event.target.closest("button, input, select, textarea, a")) return;
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

  function makePanelResizable(element) {
    const handle = element.querySelector(".mf-resize-handle");
    let resizing = false;
    let startX = 0;
    let startY = 0;
    let startWidth = 0;
    let startHeight = 0;

    handle.addEventListener("pointerdown", event => {
      if (event.button !== 0) return;
      resizing = true;
      startX = event.clientX;
      startY = event.clientY;
      startWidth = element.offsetWidth;
      startHeight = element.offsetHeight;
      handle.setPointerCapture?.(event.pointerId);
      event.stopPropagation();
      event.preventDefault();
    });
    handle.addEventListener("pointermove", event => {
      if (!resizing) return;
      const maxWidth = Math.max(250, Math.min(520, innerWidth - element.getBoundingClientRect().left - 8));
      const maxHeight = Math.max(160, innerHeight - element.getBoundingClientRect().top - 8);
      element.style.width = `${Math.max(250, Math.min(maxWidth, startWidth + event.clientX - startX))}px`;
      element.style.height = `${Math.max(160, Math.min(maxHeight, startHeight + event.clientY - startY))}px`;
    });
    const finish = () => {
      if (!resizing) return;
      resizing = false;
      try { localStorage.setItem(PANEL_SIZE_KEY, JSON.stringify({ width: element.offsetWidth, height: element.offsetHeight })); } catch { /* nada */ }
    };
    handle.addEventListener("pointerup", finish);
    handle.addEventListener("pointercancel", finish);

    try {
      const saved = JSON.parse(localStorage.getItem(PANEL_SIZE_KEY) || "null");
      if (Number.isFinite(saved?.width)) element.style.width = `${Math.max(250, Math.min(520, saved.width))}px`;
      if (Number.isFinite(saved?.height)) element.style.height = `${Math.max(160, Math.min(innerHeight - 16, saved.height))}px`;
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
    element.dataset.scale = preferences.scale;
    element.dataset.fontSize = preferences.fontSize;
    const titleElement = element.querySelector("#mindflow-meeting-title");
    if (titleElement) {
      titleElement.textContent = databaseMeeting.title || meetingIdentity().title;
      titleElement.title = databaseMeeting.title || meetingIdentity().title;
    }

    // O iframe precisa permanecer conectado enquanto o usuário digita.
    // As leituras continuam sendo processadas, mas a parte visual aguarda o login terminar.
    if (accountLoginOpen && !accountState.signedIn && body.querySelector(".mf-login-frame")) return;

    const scoreHtml = scores
      ? Object.entries(LABELS).map(([key, label]) => {
        const value = Math.round(scores[key]);
        const help = `${REACTION_HELP[key]} ${value}% representa a intensidade desta leitura; não é porcentagem de pessoas nem chance de acerto.`;
        return `<span class="mf-score-card" tabindex="0"><b>${value}%</b><small>${label}</small><em class="mf-score-help">${escapeHtml(help)}</em></span>`;
      }).join("")
      : `<em>Aguardando 30 frames de alguma fonte autorizada.</em>`;
    const sourceHtml = [...sources.values()].map(source => {
      const state = effectiveState(source);
      const title = stateCopy(state)[0];
      const fps = source.measuredFps ? ` · ${source.measuredFps.toFixed(1)} fps reais` : "";
      return `<li data-state="${state}"><b>${escapeHtml(anonymousSourceLabel(source.id))}</b><small>${escapeHtml(title + fps)}</small></li>`;
    }).join("") || `<li class="mf-empty">Nenhum vídeo visível no Meet.</li>`;
    const sourceStates = [...sources.values()].map(source => effectiveState(source));
    const analyzingSources = sourceStates.filter(state => state === "ANALYZING").length;
    const ignoredSources = Math.max(0, sourceStates.length - analyzingSources);

    const chartHtml = `
      <div class="mf-section mf-chart" data-display="${escapeHtml(preferences.chartDisplay)}">
        <div class="mf-chart-head"><b>Reações · últimos 5 minutos</b><span>AO VIVO</span></div>
        <div class="mf-chart-legend">
          <span class="engagement"><i></i>Eng. ${scores ? Math.round(scores.engagement) : 0}%</span>
          <span class="boredom"><i></i>Tédio ${scores ? Math.round(scores.boredom) : 0}%</span>
          <span class="confusion"><i></i>Conf. ${scores ? Math.round(scores.confusion) : 0}%</span>
          <span class="frustration"><i></i>Frus. ${scores ? Math.round(scores.frustration) : 0}%</span>
        </div>
        <svg viewBox="0 0 100 36" preserveAspectRatio="none" aria-label="Histórico das quatro reações nos últimos cinco minutos">
          <g class="mf-chart-grid"><line x1="0" y1="9" x2="100" y2="9"></line><line x1="0" y1="18" x2="100" y2="18"></line><line x1="0" y1="27" x2="100" y2="27"></line></g>
          <polyline class="engagement" points="${chartPoints("engagement")}"></polyline>
          <polyline class="boredom" points="${chartPoints("boredom")}"></polyline>
          <polyline class="confusion" points="${chartPoints("confusion")}"></polyline>
          <polyline class="frustration" points="${chartPoints("frustration")}"></polyline>
        </svg>
        <div class="mf-chart-axis"><span>5 min atrás</span><span>agora</span></div>
      </div>`;

    const pythonText = pythonHealth
      ? `Python conectado · modelo ${pythonHealth.modelLoaded ? "carregado" : "indisponível"} · alvo ${pythonHealth.targetFps || TARGET_FPS} fps`
      : `Python desconectado · ${pythonError}`;

    const syncLabel = accountSyncState === "syncing" ? "Sincronizando..." : accountSyncState === "success" ? "Sincronizado" : accountSyncState === "error" ? "Tentar novamente" : "Sincronizar";
    const accountHtml = !accountState.checked
      ? `<div class="mf-account-note"><div><b>Verificando sua conta...</b><small>A análise será liberada após a autenticação.</small></div></div>`
      : accountState.signedIn
        ? `<div class="mf-account-note is-connected"><div><b>Conta conectada</b><small>${escapeHtml(accountState.email || "Reunião vinculada à sua conta")}</small></div><button type="button" data-account-action="sync" ${accountSyncState === "syncing" ? "disabled" : ""}>${syncLabel}</button></div>`
        : `<div class="mf-account-note is-warning"><div><b>Login obrigatório</b><small>Entre para analisar e salvar esta reunião.</small></div><button type="button" data-account-action="login">${accountLoginOpen ? "Fechar" : "Entrar"}</button></div>`;
    const loginHtml = accountLoginOpen && !accountState.signedIn
      ? `<iframe class="mf-login-frame" src="${escapeHtml(chrome.runtime.getURL("login-panel.html"))}" title="Entrar no MindFlow AI"></iframe>`
      : "";

    body.innerHTML = `
      ${accountHtml}
      ${loginHtml}
      <div class="mf-status"><i id="mindflow-light"></i><div><b>${escapeHtml(light.title)} ${scores ? `<span class="mf-trend mf-conditional" data-display="${escapeHtml(preferences.trendDisplay)}">${trendArrow()}</span>` : ""}</b><small>${scores ? `<span class="mf-confidence mf-conditional" data-display="${escapeHtml(preferences.confidenceDisplay)}">Confiança: ${Math.round(light.confidence)}% · </span>${scores.participants} fonte(s)` : "Nenhuma previsão ativa"}</small></div></div>
      <div class="mf-details">
        <div class="mf-section mf-scores" data-display="${escapeHtml(preferences.scoresDisplay)}">${scoreHtml}</div>
        ${chartHtml}
        <div class="mf-section mf-sources" data-display="${escapeHtml(preferences.sourcesDisplay)}">
          <div class="mf-source-list-title">Fontes anônimas detectadas (${sources.size})</div>
          <small class="mf-source-summary">${analyzingSources} analisada(s) · ${ignoredSources} ignorada(s)</small>
          <ul class="mf-source-list">${sourceHtml}</ul>
        </div>
        <small class="mf-section" data-display="${escapeHtml(preferences.technicalDisplay)}" id="mindflow-python">${escapeHtml(pythonText)}<br>${escapeHtml(historyStatus())}<br>Nenhum nome de participante é coletado.</small>
      </div>`;
  }

  async function captureSource(source, now) {
    if (source.inFlight || now - source.lastCaptureAt < FRAME_INTERVAL_MS) return;
    if (!accountState.signedIn || databaseMeeting.state !== "ready") return;
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
    const displayMode = (mode, legacy, fallback) => {
      if (["hidden", "hover", "always"].includes(mode)) return mode;
      if (legacy === false) return "hidden";
      if (legacy === true) return "always";
      return fallback;
    };
    const legacyFontSizes = { normal: "small", large: "normal", extra: "large" };
    const fontSize = value.preferencesVersion === 2
      ? value.fontSize
      : legacyFontSizes[value.fontSize] || DEFAULT_PREFERENCES.fontSize;
    return {
      preferencesVersion: 2,
      theme: ["light", "dark"].includes(value.theme) ? value.theme : DEFAULT_PREFERENCES.theme,
      scale: ["small", "normal", "large"].includes(value.scale) ? value.scale : DEFAULT_PREFERENCES.scale,
      fontSize: ["small", "normal", "large"].includes(fontSize) ? fontSize : DEFAULT_PREFERENCES.fontSize,
      trendDisplay: displayMode(value.trendDisplay, value.showTrend, DEFAULT_PREFERENCES.trendDisplay),
      confidenceDisplay: displayMode(value.confidenceDisplay, value.showConfidence, DEFAULT_PREFERENCES.confidenceDisplay),
      scoresDisplay: displayMode(value.scoresDisplay, value.showSecondary, DEFAULT_PREFERENCES.scoresDisplay),
      chartDisplay: displayMode(value.chartDisplay, value.showChart, DEFAULT_PREFERENCES.chartDisplay),
      sourcesDisplay: displayMode(value.sourcesDisplay, value.showSources, DEFAULT_PREFERENCES.sourcesDisplay),
      technicalDisplay: displayMode(value.technicalDisplay, value.showTechnical, DEFAULT_PREFERENCES.technicalDisplay)
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

  async function checkAccount() {
    const response = await sendMessage({ type: "MINDFLOW_ACCOUNT_STATE" });
    const wasSignedIn = accountState.signedIn;
    accountState = response?.success
      ? { checked: true, signedIn: Boolean(response.signedIn), email: response.user?.email || "" }
      : { checked: true, signedIn: false, email: "" };
    accountLoginOpen = !accountState.signedIn;
    if (accountState.signedIn) {
      void monitorMeetingLifecycle();
    } else {
      databaseMeeting = { state: "idle", meetKey: "", title: "", code: "", error: "" };
      if (wasSignedIn) {
        for (const source of sources.values()) {
          if (source.decision === DECISIONS.ANALYZE) resetRemote(source.id);
          source.decision = null;
          source.inFlight = false;
          clearSourceResults(source);
          renderOverlay(source);
        }
      }
    }
    renderPanel();
  }

  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName !== "local") return;
    if (changes[PREFERENCES_KEY]) {
      preferences = normalizePreferences(changes[PREFERENCES_KEY].newValue);
      renderPanel();
    }
    if (changes[AUTH_KEY]) void checkAccount();
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
  void checkAccount();
  void checkPython();
  setInterval(scanVideos, 900);
  setInterval(checkPython, 5_000);
  setInterval(() => void monitorMeetingLifecycle(), 2_000);
  setInterval(() => void sendMeetingHeartbeat(), 15_000);
  setInterval(renderPanel, 1_000);
  requestAnimationFrame(captureLoop);
})();
