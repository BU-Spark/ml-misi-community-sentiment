const runtimeBaseUrl =
  window.APP_CONFIG?.apiBaseUrl ||
  window.APP_CONFIG?.API_BASE_URL ||
  (window.location.port === '8000' ? 'http://127.0.0.1:8888' : '');

function parseTimeoutMs(value, fallback) {
  if (value === null || value === 'null' || value === 'none' || value === 'off') {
    return null;
  }
  if (value === undefined || value === '') {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback;
  }
  return parsed;
}

const ApiConfig = {
  baseUrl: runtimeBaseUrl,
  timeoutMs: parseTimeoutMs(
    window.APP_CONFIG?.requestTimeoutMs ?? window.APP_CONFIG?.REQUEST_TIMEOUT_MS,
    30000
  ),
  chatTimeoutMs: parseTimeoutMs(
    window.APP_CONFIG?.chatTimeoutMs ?? window.APP_CONFIG?.CHAT_TIMEOUT_MS,
    null
  ),
  // Opt-in: stream assistant replies token-by-token via Server-Sent Events.
  streamingEnabled: Boolean(window.APP_CONFIG?.streaming ?? window.APP_CONFIG?.STREAMING ?? false),
};

function getCookie(name) {
  const encoded = `${name}=`;
  const parts = document.cookie.split(';');
  for (const part of parts) {
    const trimmed = part.trim();
    if (trimmed.startsWith(encoded)) {
      return decodeURIComponent(trimmed.slice(encoded.length));
    }
  }
  return '';
}

const GUEST_SESSION_KEY = 'otp_guest_session_token';
const GUEST_CSRF_KEY = 'otp_guest_csrf_token';

function getGuestSessionToken() {
  try {
    return sessionStorage.getItem(GUEST_SESSION_KEY) || '';
  } catch (error) {
    return '';
  }
}

function getGuestCsrfToken() {
  try {
    return sessionStorage.getItem(GUEST_CSRF_KEY) || '';
  } catch (error) {
    return '';
  }
}

function storeGuestSession({ session_token: sessionToken, csrf_token: csrfToken }) {
  try {
    if (sessionToken) {
      sessionStorage.setItem(GUEST_SESSION_KEY, sessionToken);
    }
    if (csrfToken) {
      sessionStorage.setItem(GUEST_CSRF_KEY, csrfToken);
    }
  } catch (error) {
    // sessionStorage may be unavailable in restrictive contexts.
  }
}

function clearGuestSession() {
  try {
    sessionStorage.removeItem(GUEST_SESSION_KEY);
    sessionStorage.removeItem(GUEST_CSRF_KEY);
  } catch (error) {
    // ignore
  }
}

function hasGuestSession() {
  return Boolean(getGuestSessionToken());
}

function applyAuthHeaders(headers, method) {
  const guestToken = getGuestSessionToken();
  const isMutating = !['GET', 'HEAD', 'OPTIONS'].includes(method);

  if (guestToken) {
    headers.Authorization = `Bearer ${guestToken}`;
    if (isMutating) {
      const guestCsrf = getGuestCsrfToken();
      if (guestCsrf) {
        headers['X-CSRF-Token'] = guestCsrf;
      }
    }
    return;
  }

  if (isMutating) {
    const csrfToken = getCookie('otp_csrf');
    if (csrfToken) {
      headers['X-CSRF-Token'] = csrfToken;
    }
  }
}

async function apiRequest(path, options = {}) {
  const { timeoutMs: requestTimeoutMs, ...fetchOptions } = options;
  const timeoutMs = requestTimeoutMs === undefined ? ApiConfig.timeoutMs : requestTimeoutMs;
  const controller = new AbortController();
  const shouldAbort = Number.isFinite(timeoutMs) && timeoutMs > 0;
  const timeoutId = shouldAbort ? setTimeout(() => controller.abort(), timeoutMs) : null;
  const method = (fetchOptions.method || 'GET').toUpperCase();
  const headers = {
    ...(fetchOptions.headers || {}),
  };

  if (!headers['Content-Type'] && !['GET', 'HEAD'].includes(method)) {
    headers['Content-Type'] = 'application/json';
  }

  applyAuthHeaders(headers, method);

  try {
    const response = await fetch(`${ApiConfig.baseUrl}${path}`, {
      credentials: 'include',
      signal: shouldAbort ? controller.signal : undefined,
      ...fetchOptions,
      method,
      headers,
    });

    let payload = null;
    const text = await response.text();
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (error) {
        payload = { raw: text };
      }
    }

    if (!response.ok) {
      let errorMessage = (payload && (payload.error || payload.message)) || `Request failed with status ${response.status}`;
      if (response.status === 429) {
        const retryAfter = payload?.retry_after;
        errorMessage = retryAfter
          ? `Too many requests. Please wait ${retryAfter} seconds and try again.`
          : 'Too many requests. Please try again later.';
      }
      return {
        success: false,
        status: response.status,
        data: payload,
        error: errorMessage,
        retryAfter: payload?.retry_after ?? null,
      };
    }

    return {
      success: true,
      status: response.status,
      data: payload,
      error: null,
    };
  } catch (error) {
    const message = error.name === 'AbortError'
      ? 'The request took too long. Please try again.'
      : (error.message || 'Unable to reach the server.');
    return {
      success: false,
      status: null,
      data: null,
      error: message,
    };
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
}

async function sendMessageStream(threadId, payload, { onDelta, onCorrection, timeoutMs } = {}) {
  const streamTimeout = timeoutMs === undefined
    ? (ApiConfig.chatTimeoutMs ?? 120000)
    : timeoutMs;
  const headers = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  };
  applyAuthHeaders(headers, 'POST');

  const controller = new AbortController();
  const shouldAbort = Number.isFinite(streamTimeout) && streamTimeout > 0;
  const timeoutId = shouldAbort ? setTimeout(() => controller.abort(), streamTimeout) : null;

  let response;
  try {
    response = await fetch(`${ApiConfig.baseUrl}/conversations/${threadId}/messages/stream`, {
      method: 'POST',
      credentials: 'include',
      headers,
      body: JSON.stringify(payload),
      signal: shouldAbort ? controller.signal : undefined,
    });
  } catch (error) {
    if (timeoutId) clearTimeout(timeoutId);
    const timedOut = error?.name === 'AbortError';
    return {
      success: false,
      status: null,
      data: null,
      error: timedOut
        ? 'The response took too long. Please try again.'
        : (error.message || 'Unable to reach the server.'),
    };
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }

  if (!response.ok || !response.body) {
    let data = null;
    try {
      data = await response.json();
    } catch (error) {
      data = null;
    }
    let errorMessage = (data && (data.error || data.message)) || `Request failed with status ${response.status}`;
    if (response.status === 429) {
      const retryAfter = data?.retry_after;
      errorMessage = retryAfter
        ? `Too many requests. Please wait ${retryAfter} seconds and try again.`
        : 'Too many requests. Please try again later.';
    }
    return {
      success: false,
      status: response.status,
      data,
      error: errorMessage,
      retryAfter: data?.retry_after ?? null,
    };
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalEvent = null;
  let errorEvent = null;

  const handleEvent = (rawEvent) => {
    const dataLine = rawEvent.split('\n').find((line) => line.startsWith('data:'));
    if (!dataLine) return;
    const jsonStr = dataLine.slice(5).trim();
    if (!jsonStr) return;
    let evt;
    try {
      evt = JSON.parse(jsonStr);
    } catch (error) {
      return;
    }
    if (evt.type === 'delta') {
      if (onDelta) onDelta(evt.text || '');
    } else if (evt.type === 'correction') {
      if (onCorrection) onCorrection(evt.text || '');
    } else if (evt.type === 'final') {
      finalEvent = evt;
    } else if (evt.type === 'error') {
      errorEvent = evt;
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf('\n\n')) >= 0) {
        const rawEvent = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        handleEvent(rawEvent);
      }
    }
    if (buffer.trim()) {
      handleEvent(buffer);
    }
  } catch (error) {
    return { success: false, status: null, data: null, error: error.message || 'Stream interrupted.' };
  }

  if (errorEvent) {
    return { success: false, status: 200, data: errorEvent, error: errorEvent.error || 'Generation failed.' };
  }
  if (finalEvent) {
    return { success: true, status: 200, data: finalEvent, error: null };
  }
  return { success: false, status: 200, data: null, error: 'The response stream ended unexpectedly.' };
}

window.ApiClient = {
  config: ApiConfig,
  request: apiRequest,
  hasGuestSession,
  clearGuestSession,
  storeGuestSession,
  getGuestSessionToken,
  health: () => apiRequest('/health', { method: 'GET' }),
  getSession: () => apiRequest('/auth/me', { method: 'GET' }),
  createGuestSession: async () => {
    const result = await apiRequest('/auth/guest', { method: 'POST' });
    if (result.success && result.data) {
      storeGuestSession(result.data);
    }
    return result;
  },
  signup: (payload) => apiRequest('/auth/signup', { method: 'POST', body: JSON.stringify(payload) }),
  login: (payload) => apiRequest('/auth/login', { method: 'POST', body: JSON.stringify(payload) }),
  logout: () => apiRequest('/auth/logout', { method: 'POST', body: JSON.stringify({}) }),
  completeProfile: (payload) => apiRequest('/auth/complete-profile', { method: 'POST', body: JSON.stringify(payload) }),
  fetchThreads: () => apiRequest('/conversations', { method: 'GET' }),
  createThread: (payload = {}) => apiRequest('/conversations', { method: 'POST', body: JSON.stringify(payload) }),
  updateThread: (threadId, payload) => apiRequest(`/conversations/${threadId}`, { method: 'PATCH', body: JSON.stringify(payload) }),
  deleteThread: (threadId) => apiRequest(`/conversations/${threadId}`, { method: 'DELETE', body: JSON.stringify({}) }),
  fetchMessages: (threadId, params = {}) => {
    const qs = new URLSearchParams();
    if (params.limit) qs.set('limit', String(params.limit));
    if (params.before) qs.set('before', params.before);
    const suffix = qs.toString() ? `?${qs.toString()}` : '';
    return apiRequest(`/conversations/${threadId}/messages${suffix}`, { method: 'GET' });
  },
  sendMessage: (threadId, payload) => apiRequest(`/conversations/${threadId}/messages`, {
    method: 'POST',
    body: JSON.stringify(payload),
    timeoutMs: ApiConfig.chatTimeoutMs,
  }),
  sendMessageStream,
  fetchEvents: (daysAhead = 14, limit = 10) => {
    const qs = new URLSearchParams({ days_ahead: String(daysAhead), limit: String(limit) });
    return apiRequest(`/events?${qs.toString()}`, { method: 'GET' });
  },
  flagInteraction: (logId, flagReason, flagDetails) => apiRequest('/log', {
    method: 'PUT',
    body: JSON.stringify({ log_id: logId, flag_reason: flagReason, flag_details: flagDetails }),
  }),
  adminStats: () => apiRequest('/admin/stats', { method: 'GET' }),
  adminFlags: () => apiRequest('/admin/flags', { method: 'GET' }),
  adminInteractions: () => apiRequest('/admin/interactions', { method: 'GET' }),
  adminNoResults: () => apiRequest('/admin/no-results', { method: 'GET' }),
  adminCommentFlag: (flagId, comment, resolved) => apiRequest(`/admin/flags/${flagId}/comment`, {
    method: 'PUT',
    body: JSON.stringify({ moderator_comment: comment, resolved }),
  }),
  adminAddKnowledge: (payload) => apiRequest('/admin/knowledge', {
    method: 'POST',
    body: JSON.stringify(payload),
  }),
  adminGetKnowledge: () => apiRequest('/admin/knowledge', { method: 'GET' }),
  adminDeleteKnowledge: (id) => apiRequest(`/admin/knowledge/${id}`, { method: 'DELETE' }),
  adminEditKnowledge: (id, payload) => apiRequest(`/admin/knowledge/${id}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  }),
  submitCommunityNote: (content, category) => apiRequest('/community/notes', {
    method: 'POST',
    body: JSON.stringify({ content, category }),
  }),
  communityNotesChat: (messages) => apiRequest('/community/notes/chat', {
    method: 'POST',
    body: JSON.stringify({ messages }),
    timeoutMs: null,
  }),
adminGetPending: () => apiRequest('/admin/knowledge/pending', { method: 'GET' }),
adminApproveNote: (id) => apiRequest(`/admin/knowledge/${id}/approve`, {
  method: 'PUT',
  body: JSON.stringify({}),
}),
};
