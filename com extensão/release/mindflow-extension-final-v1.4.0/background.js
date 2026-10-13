const PYTHON_BASE_URL = "http://127.0.0.1:8765";
const SUPABASE_URL = "https://xeykjfytdvqtgcewvoii.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhleWtqZnl0ZHZxdGdjZXd2b2lpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODg4MTA2MzMsImV4cCI6MjEwNDM4NjYzM30.DHrm3C3Zl7ZnckpEv1FsngbkkUWC1a94q8577l3aLc0";
const AUTH_KEY = "mindflowAuthSession";
const PREFERENCES_KEY = "mindflowExtensionPreferences";
const ACTIVE_MEETINGS_KEY = "mindflowActiveMeetings";
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

async function fetchJson(path, options = {}, timeoutMs = 8000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${PYTHON_BASE_URL}${path}`, {
      ...options,
      signal: controller.signal
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.message || payload.error || `HTTP ${response.status}`);
    }
    return payload;
  } finally {
    clearTimeout(timeout);
  }
}

async function supabaseJson(path, options = {}, accessToken = "", timeoutMs = 10_000) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${SUPABASE_URL}${path}`, {
      ...options,
      signal: controller.signal,
      headers: {
        apikey: SUPABASE_ANON_KEY,
        Authorization: `Bearer ${accessToken || SUPABASE_ANON_KEY}`,
        "Content-Type": "application/json",
        ...(options.headers || {})
      }
    });
    const payload = response.status === 204 ? null : await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload?.msg || payload?.message || payload?.error_description || payload?.error || `HTTP ${response.status}`);
    return payload;
  } catch (error) {
    if (error?.name === "AbortError") throw new Error("O Supabase demorou mais de 10 segundos para responder.");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function normalizedPreferences(value = {}) {
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

async function saveSession(payload) {
  const session = {
    accessToken: payload.access_token,
    refreshToken: payload.refresh_token,
    expiresAt: Date.now() + Number(payload.expires_in || 3600) * 1000,
    user: { id: payload.user?.id, email: payload.user?.email || "" }
  };
  await chrome.storage.local.set({ [AUTH_KEY]: session });
  return session;
}

async function activeSession() {
  const stored = await chrome.storage.local.get(AUTH_KEY);
  let session = stored[AUTH_KEY];
  if (!session?.refreshToken) return null;
  if (Number(session.expiresAt || 0) > Date.now() + 60_000) return session;
  const refreshed = await supabaseJson("/auth/v1/token?grant_type=refresh_token", {
    method: "POST",
    body: JSON.stringify({ refresh_token: session.refreshToken })
  });
  session = await saveSession(refreshed);
  return session;
}

async function login(email, password) {
  const payload = await supabaseJson("/auth/v1/token?grant_type=password", {
    method: "POST",
    body: JSON.stringify({ email, password })
  });
  const session = await saveSession(payload);
  let sync = null;
  let syncError = "";
  try {
    sync = await syncPreferences(session);
  } catch (error) {
    syncError = String(error?.message || error);
  }
  return { session, sync, syncError };
}

async function syncPreferences(providedSession = null) {
  const session = providedSession || await activeSession();
  if (!session?.user?.id) throw new Error("Entre na sua conta para sincronizar.");
  const rows = await supabaseJson(`/rest/v1/extensao_preferencia?select=configuracao,atualizado_em&comunicador_id=eq.${encodeURIComponent(session.user.id)}&limit=1`, {}, session.accessToken);
  const preferences = normalizedPreferences(rows?.[0]?.configuracao || DEFAULT_PREFERENCES);
  await chrome.storage.local.set({ [PREFERENCES_KEY]: preferences });
  return { preferences, updatedAt: rows?.[0]?.atualizado_em || null };
}

function cleanMeetingTitle(value, meetKey) {
  const normalized = String(value || "")
    .replace(/[\u0000-\u001f\u007f]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 140);
  if (normalized) return normalized;
  const when = new Intl.DateTimeFormat("pt-BR", {
    dateStyle: "short",
    timeStyle: "short"
  }).format(new Date());
  return `Google Meet · ${meetKey || when}`;
}

async function configurePythonHistory(meeting) {
  return fetchJson("/history/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sessionCode: meeting.code,
      title: meeting.title
    })
  }, 5000);
}

async function ensureMeeting(payload = {}) {
  const session = await activeSession();
  if (!session?.user?.id) throw new Error("Entre na sua conta para salvar a reunião.");

  const meetKey = String(payload.meetKey || "").trim().slice(0, 120);
  if (!meetKey) throw new Error("Não foi possível identificar esta sala do Google Meet.");

  const stored = await chrome.storage.local.get(ACTIVE_MEETINGS_KEY);
  const meetings = stored[ACTIVE_MEETINGS_KEY] || {};
  const existing = meetings[meetKey];
  const stillRecent = existing && Date.now() - Number(existing.lastSeenAt || 0) < 8 * 60 * 60 * 1000;

  if (stillRecent && existing.userId === session.user.id && existing.code) {
    existing.lastSeenAt = Date.now();
    meetings[meetKey] = existing;
    await chrome.storage.local.set({ [ACTIVE_MEETINGS_KEY]: meetings });
    const history = await configurePythonHistory(existing).catch(error => ({
      ok: false,
      message: String(error?.message || error)
    }));
    return { meeting: existing, history };
  }

  if (existing?.id && existing.userId === session.user.id) {
    await supabaseJson(
      `/rest/v1/sessao?id=eq.${encodeURIComponent(existing.id)}`,
      {
        method: "PATCH",
        headers: { Prefer: "return=minimal" },
        body: JSON.stringify({
          status: "encerrada",
          encerrada_em: new Date().toISOString()
        })
      },
      session.accessToken
    ).catch(() => null);
  }

  const title = cleanMeetingTitle(payload.title, meetKey);
  const startedAt = new Date().toISOString();
  const rows = await supabaseJson(
    "/rest/v1/sessao?select=id,codigo,titulo,status,iniciada_em",
    {
      method: "POST",
      headers: { Prefer: "return=representation" },
      body: JSON.stringify({
        comunicador_id: session.user.id,
        titulo: title,
        plataforma: "Google Meet"
      })
    },
    session.accessToken
  );
  const row = rows?.[0];
  if (!row?.id || !row?.codigo) throw new Error("O Supabase não devolveu o identificador da reunião.");

  await supabaseJson(
    `/rest/v1/sessao?id=eq.${encodeURIComponent(row.id)}`,
    {
      method: "PATCH",
      headers: { Prefer: "return=minimal" },
      body: JSON.stringify({
        status: "em_andamento",
        iniciada_em: startedAt,
        titulo: title,
        plataforma: "Google Meet"
      })
    },
    session.accessToken
  );

  const meeting = {
    id: row.id,
    code: row.codigo,
    title,
    meetKey,
    userId: session.user.id,
    startedAt,
    lastSeenAt: Date.now()
  };
  meetings[meetKey] = meeting;
  await chrome.storage.local.set({ [ACTIVE_MEETINGS_KEY]: meetings });
  const history = await configurePythonHistory(meeting).catch(error => ({
    ok: false,
    message: String(error?.message || error)
  }));
  return { meeting, history };
}

async function endMeeting(meetKey) {
  const key = String(meetKey || "").trim();
  if (!key) return { ended: false };
  const [session, stored] = await Promise.all([
    activeSession(),
    chrome.storage.local.get(ACTIVE_MEETINGS_KEY)
  ]);
  const meetings = stored[ACTIVE_MEETINGS_KEY] || {};
  const meeting = meetings[key];
  if (!session?.accessToken || !meeting?.id || meeting.userId !== session.user?.id) {
    return { ended: false };
  }
  await supabaseJson(
    `/rest/v1/sessao?id=eq.${encodeURIComponent(meeting.id)}`,
    {
      method: "PATCH",
      headers: { Prefer: "return=minimal" },
      body: JSON.stringify({
        status: "encerrada",
        encerrada_em: new Date().toISOString()
      })
    },
    session.accessToken
  );
  delete meetings[key];
  await chrome.storage.local.set({ [ACTIVE_MEETINGS_KEY]: meetings });
  return { ended: true, meetingId: meeting.id };
}

async function logout() {
  const session = await activeSession().catch(() => null);
  if (session?.accessToken) {
    await supabaseJson("/auth/v1/logout", { method: "POST" }, session.accessToken).catch(() => null);
  }
  await chrome.storage.local.remove(AUTH_KEY);
  await chrome.storage.local.set({ [PREFERENCES_KEY]: { ...DEFAULT_PREFERENCES } });
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "MINDFLOW_ACCOUNT_STATE") {
    Promise.all([activeSession(), chrome.storage.local.get(PREFERENCES_KEY)])
      .then(([session, stored]) => sendResponse({ success: true, signedIn: Boolean(session), user: session?.user || null, preferences: normalizedPreferences(stored[PREFERENCES_KEY]) }))
      .catch(error => sendResponse({ success: false, error: String(error) }));
    return true;
  }

  if (message?.type === "MINDFLOW_LOGIN") {
    login(String(message.email || "").trim(), String(message.password || ""))
      .then(result => sendResponse({ success: true, user: result.session.user, sync: result.sync, syncError: result.syncError }))
      .catch(error => sendResponse({ success: false, error: String(error) }));
    return true;
  }

  if (message?.type === "MINDFLOW_OPEN_LOGIN") {
    chrome.tabs.create({ url: chrome.runtime.getURL("popup.html") })
      .then(() => sendResponse({ success: true }))
      .catch(error => sendResponse({ success: false, error: String(error) }));
    return true;
  }

  if (message?.type === "MINDFLOW_SYNC_PREFERENCES") {
    syncPreferences()
      .then(data => sendResponse({ success: true, ...data }))
      .catch(error => sendResponse({ success: false, error: String(error) }));
    return true;
  }

  if (message?.type === "MINDFLOW_ENSURE_MEETING") {
    ensureMeeting(message.meeting)
      .then(data => sendResponse({ success: true, ...data }))
      .catch(error => sendResponse({ success: false, error: String(error?.message || error) }));
    return true;
  }

  if (message?.type === "MINDFLOW_END_MEETING") {
    endMeeting(message.meetKey)
      .then(data => sendResponse({ success: true, ...data }))
      .catch(error => sendResponse({ success: false, error: String(error?.message || error) }));
    return true;
  }

  if (message?.type === "MINDFLOW_LOGOUT") {
    logout()
      .then(() => sendResponse({ success: true }))
      .catch(error => sendResponse({ success: false, error: String(error) }));
    return true;
  }

  if (message?.type === "MINDFLOW_HEALTH") {
    fetchJson("/health", {}, 2500)
      .then(data => sendResponse({ success: true, python: data }))
      .catch(error => sendResponse({ success: false, error: String(error) }));
    return true;
  }

  if (message?.type === "MINDFLOW_PROCESS_FRAME") {
    fetchJson("/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message.payload)
    })
      .then(data => sendResponse({ success: true, python: data }))
      .catch(error => sendResponse({ success: false, error: String(error) }));
    return true;
  }

  if (message?.type === "MINDFLOW_RESET_PARTICIPANT") {
    const id = encodeURIComponent(message.participantId || "");
    fetchJson(`/reset/${id}`, { method: "POST" })
      .then(data => sendResponse({ success: true, python: data }))
      .catch(error => sendResponse({ success: false, error: String(error) }));
    return true;
  }

  return false;
});

