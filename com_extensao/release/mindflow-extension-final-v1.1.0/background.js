const PYTHON_BASE_URL = "http://127.0.0.1:8765";
const SUPABASE_URL = "https://xeykjfytdvqtgcewvoii.supabase.co";
const SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhleWtqZnl0ZHZxdGdjZXd2b2lpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODg4MTA2MzMsImV4cCI6MjEwNDM4NjYzM30.DHrm3C3Zl7ZnckpEv1FsngbkkUWC1a94q8577l3aLc0";
const AUTH_KEY = "mindflowAuthSession";
const PREFERENCES_KEY = "mindflowExtensionPreferences";
const DEFAULT_PREFERENCES = Object.freeze({
  theme: "light",
  density: "compact",
  scale: "normal",
  showTrend: true,
  showConfidence: true,
  showSecondary: true,
  showChart: true
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

async function supabaseJson(path, options = {}, accessToken = "") {
  const response = await fetch(`${SUPABASE_URL}${path}`, {
    ...options,
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
}

function normalizedPreferences(value = {}) {
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
  await syncPreferences(session).catch(() => null);
  return session;
}

async function syncPreferences(providedSession = null) {
  const session = providedSession || await activeSession();
  if (!session?.user?.id) throw new Error("Entre na sua conta para sincronizar.");
  const rows = await supabaseJson(`/rest/v1/extensao_preferencia?select=configuracao,atualizado_em&comunicador_id=eq.${encodeURIComponent(session.user.id)}&limit=1`, {}, session.accessToken);
  const preferences = normalizedPreferences(rows?.[0]?.configuracao || DEFAULT_PREFERENCES);
  await chrome.storage.local.set({ [PREFERENCES_KEY]: preferences });
  return { preferences, updatedAt: rows?.[0]?.atualizado_em || null };
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
      .then(session => sendResponse({ success: true, user: session.user }))
      .catch(error => sendResponse({ success: false, error: String(error) }));
    return true;
  }

  if (message?.type === "MINDFLOW_SYNC_PREFERENCES") {
    syncPreferences()
      .then(data => sendResponse({ success: true, ...data }))
      .catch(error => sendResponse({ success: false, error: String(error) }));
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

