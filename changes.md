# On The Porch — Complete Change Log

This document records **every major change** made during project onboarding: reliability hardening, performance work, guest authentication, production security, and UX improvements.

**Status:** All changes are **uncommitted** as of this writing.

**Scope:** 23 modified files (+2,855 / −767 lines), 8 new files.

---

## Architecture: Before vs After

### Before
```
Browser → must sign in (password or Google OAuth)
       → Flask dev server (debug)
       → unified_chatbot (sequential RAG, 3–6 Gemini calls/question)
       → in-memory session dict (per worker)
       → no streaming, no guest access
```

### After
```
Browser → guest-by-default (tab-scoped Bearer) OR password login (optional)
       → Flask dev OR Gunicorn (gthread, multi-worker)
       → api_v2
            ├─ rate limits (Redis or memory)
            ├─ semantic cache (optional)
            ├─ SSE streaming + Boston.gov post-stream fallback
            ├─ guest sessions + admin gating
            └─ _execute_agent_response / stream_agent_response
                  ├─ instant chitchat (no LLM)
                  ├─ parallel RAG / hybrid
                  ├─ events fast SQL path
                  └─ Chroma WAL + lock retry
       → Redis session cache (optional, recommended for multi-worker)
       → cron: flock + guest cleanup
```

---

## Major Change 1 — Stop 500 Crashes (Reliability Phase 1)

**Goal:** Fix confirmed bugs that caused intermittent HTTP 500 responses.

### `on_the_porch/rag stuff/boston_gov.py`

- **Added `_empty_boston_gov_result(final_query, search_url)`** — returns a canonical dict (`query`, `search_url`, `paragraphs`, `bullet_points`, `links`, `text`) instead of inconsistent shapes.
- **Error paths** (network failure, empty page, parse failure) now return `_empty_boston_gov_result(...)` instead of a bare `[]` list.
- **Why:** Downstream code called `.get("text")` on the return value. A list caused `AttributeError` → 500. `unified_chatbot._build_boston_gov_fallback_answer()` already checks `isinstance(ai_result, dict)`; this makes the producer always safe.

### `on_the_porch/sql_chat/app4.py`

- **Added env tunables:** `GEMINI_MAX_ATTEMPTS`, `DB_MAX_ATTEMPTS`, `GEMINI_REQUEST_TIMEOUT`, `GEMINI_FAST_MODEL`.
- **Added `_is_transient_error()`** — detects 429, 5xx, timeout strings for retry decisions.
- **Added `_safe_extract_response_text()`** — extracts Gemini text without crashing on blocked or multi-part responses (thinking models).
- **Added `_generate_content()`** — wraps Gemini calls with per-request timeout via `request_options`.
- **Added `_chat_with_model()`** — retry loop with exponential backoff.
- **Changed `_connect_mysql()`** — retries with backoff; raises a normal `Exception` instead of `sys.exit(1)`.
- **Why:** `sys.exit(1)` is a `BaseException` that kills Gunicorn/Flask workers on transient DB blips. Gemini multi-part responses raised `ValueError` when accessing `.text`.

### `on_the_porch/unified_chatbot.py`

- **Mirrored Gemini timeout wrapper** — `_generate_content()` with `GEMINI_REQUEST_TIMEOUT` and `_GEMINI_SUPPORTS_REQUEST_OPTIONS`.
- **Boston.gov classifier/regenerator** — routed through `_generate_content()` for consistent timeout/retry behavior.

---

## Major Change 2 — Database Connection Resilience (Reliability Phase 2)

**Goal:** Eliminate `MySQL server has gone away` on idle pooled connections.

### `api/api_v2.py`

- **Config:** Added `MYSQL_POOL_SIZE` (capped 1–32) and `MYSQL_CONNECT_TIMEOUT`.
- **`get_db_connection()`** — calls `connection.ping(reconnect=True)` before returning a pooled connection so stale sockets are transparently replaced.
- **Pool creation** — uses env-sized pool with `pool_reset_session=True` and bounded connect timeout.

### `on_the_porch/sql_chat/app4.py`

- **Added `_db_thread_local`** — one MySQL connection per worker thread.
- **Added `_get_db_connection()`** — reuses thread-local connection; `ping(reconnect=True)` on reuse.
- **`_execute_sql()`** — no longer closes the thread-local connection after every query.
- **Why:** Opening a new connection per SQL query exhausted the pool and triggered gone-away errors under hybrid load.

---

## Major Change 3 — Vector Store & Cron Safety (Reliability Phase 3)

**Goal:** Stop Chroma/SQLite lock errors during cron ingestion; fix wrong vectordb path.

**Scope note:** MySQL cron writes are incremental (`INSERT … ON DUPLICATE KEY UPDATE`) — no table rebuild window. Changes focus on Chroma only.

### `on_the_porch/rag stuff/retrieval.py`

- **`VECTORDB_DIR`** — resolved to an absolute path relative to the file (not cwd-dependent `../vectordb_new`).
- **Added `_CHROMA_MAX_ATTEMPTS`, `_WAL_ENABLED_PATHS`** — retry and WAL configuration.
- **Added `_VECTORDB_SINGLETON_ENABLED`** — reuse Chroma client across requests (env `VECTORDB_SINGLETON`).
- **Added `_is_locked_error()`, `_ensure_sqlite_wal()`** — detect SQLite lock errors; enable WAL mode on the Chroma SQLite file.
- **`load_vectordb()`** — singleton client + exponential backoff retry on lock errors.
- **`_keyword_retrieve()`** — falls back to keyword scan when Chroma is locked (logged; production risk at scale).

### `on_the_porch/rag stuff/build_vectordb.py`

- **Path alignment** — uses the same resolved `VECTORDB_DIR` as `retrieval.py` so builds land in `on_the_porch/vectordb_new`.

### `on_the_porch/rag stuff/ingest_rss.py`

- **Path alignment** — imports/uses resolved vectordb path so RSS ingestion writes to the live store.

### `on_the_porch/unified_chatbot.py`

- **Added `_fix_retrieval_vectordb_path()`** — called at startup; overrides `retrieval.VECTORDB_DIR` to `on_the_porch/vectordb_new` regardless of cwd.

### `cron_ingest.sh`

- **Added `flock` overlap guard** — second scheduled run exits immediately if another ingest is in progress.
- **Structured logging** — `run_step` helper appends to `logs/cron_ingest.log`.

---

## Major Change 4 — Production Server (Reliability Phase 4)

**Goal:** Replace Flask debug dev server with a production-grade WSGI stack.

### `api/gunicorn_conf.py` *(new)*

- **Worker model:** `gthread` (3 workers × 8 threads by default) for I/O-bound Gemini/MySQL/Chroma work.
- **Timeouts:** 120s worker timeout, 30s graceful shutdown, worker recycling after 1000 requests.
- **`preload_app = False`** — each worker builds its own MySQL pool and Chroma client after fork (required for correctness).
- **All settings env-overridable:** `GUNICORN_BIND`, `GUNICORN_WORKERS`, `GUNICORN_THREADS`, etc.

### `start_api.sh` *(new)*

- Finds project venv `gunicorn` and runs: `gunicorn -c api/gunicorn_conf.py api_v2:app`.

### `api/api_v2.py`

- **`Config.DEBUG`** — driven by `FLASK_DEBUG` env (default `false`); removed hardcoded debug mode.
- **`app.run(..., threaded=True)`** — dev server only; startup banner points to `./start_api.sh`.
- **Background threads** — `_trigger_ingest()` and `_remove_from_chroma()` run community-note ingest/Chroma deletes in daemon threads so admin mutations return immediately.

### `api/requirements.txt`

- **Added `gunicorn`** and later **`flask-cors`** (required import that was missing from API-only installs).

### `on_the_porch/unified_chatbot.py`

- **Boston.gov vectordb write** — moved to a background daemon thread so the user response is not blocked while saving scraped content.

---

## Major Change 5 — Redis Session Cache (Reliability Phase 5)

**Goal:** Share legacy `/chat` retrieval cache and rate-limit counters across Gunicorn workers.

### `start_redis.sh` *(new)*

- Starts project-local Redis on `127.0.0.1:6379`, data in `.redis/`, no persistence (cache-only).

### `api/api_v2.py`

- **Config:** `CACHE_BACKEND` (`memory` | `redis`), `REDIS_URL`, `CACHE_TTL_MINUTES`, `CACHE_MAX_SESSIONS`, `CACHE_KEY_PREFIX`.
- **Added `_BaseSessionCache`, `_MemorySessionCache`, `_RedisSessionCache`** — pluggable backends with thread-safe memory fallback.
- **`_build_session_cache()`** — constructs backend at startup; logs which backend is active.
- **Redis outage** — cache get/set failures degrade to empty cache (never 500).

### `.gitignore`

- **Ignores `.redis/`** — local Redis data directory.

---

## Major Change 6 — Performance, Streaming & Semantic Cache (Reliability Phase 6)

**Goal:** Faster answers, token streaming, optional near-duplicate caching.

### `on_the_porch/unified_chatbot.py`

- **`_PARALLEL_RETRIEVAL` env flag** (default true).
- **`_retrieve_rag()`** — `ThreadPoolExecutor` runs RSS, Boston.gov, transcript, and policy retrievals concurrently.
- **`_run_hybrid_parts()`** — SQL and RAG halves run in parallel before merge.
- **Model split:** `GEMINI_FAST_MODEL` for routing/classifiers; `GEMINI_MODEL` for SQL/RAG compose; `GEMINI_SUMMARY_MODEL` for SQL summarization.
- **`stream_agent_response()`** *(new)* — yields `('delta', text)` tokens and one `('final', payload)`; true Gemini streaming for RAG/hybrid/history; pseudo-stream (word chunks) for SQL after full compute.
- **`apply_post_stream_fallback()`** *(added in production hardening)* — applies Boston.gov fallback after streaming completes, matching non-streaming quality.

### `on_the_porch/sql_chat/app4.py`

- **Schema cache** — `_fetch_schema_snapshot()` with TTL (`SCHEMA_CACHE_TTL_SECONDS`) avoids repeated `information_schema` queries per SQL question.

### `api/semantic_cache.py` *(new)*

- **`SemanticResponseCache`** — embeds questions, cosine-similarity match, TTL eviction, LRU cap.
- **Off by default** (`SEMANTIC_CACHE_ENABLED=false`); embedding failure degrades to no cache.

### `api/api_v2.py`

- **`_semantic_cache_eligible()`** — first-turn only; excludes time-sensitive and calendar questions.
- **`_execute_agent_response()`** — semantic cache lookup before routing; stores fresh answers after generation.
- **`POST /conversations/<id>/messages/stream`** — SSE endpoint emitting `delta`, `correction`, `final`, and `error` events.
- **Streaming generator** — releases DB before LLM work; persists messages after stream; applies semantic cache + `apply_post_stream_fallback()` before save.

### `public/api.js`

- **`sendMessageStream()`** — parses SSE; `onDelta` / `onCorrection` callbacks; 120s abort timeout; 429 retry-after messaging.
- **Guest auth headers** — `Authorization: Bearer` + `X-CSRF-Token` from `sessionStorage` for mutating requests.

### `public/config.js`

- **`streaming: true`** — enables SSE path in the client.
- **`chatTimeoutMs: 120000`** — timeout for non-streaming chat and streaming abort.

### `public/app.js`

- **Streaming UX** — rAF-batched in-place DOM updates on `.message-text`; typing indicator until first token; stale-request guard via `messageVersion`.
- **`onCorrection` handler** — replaces streamed text when Boston.gov fallback enriches the answer post-stream.

---

## Major Change 7 — Chat UI Layout Fixes

**Goal:** Stop message overflow and input bar overlapping message footers (Sources, Flag).

### `public/styles.css`

- **Flex constraint chain:** `.chat-container` → `.chat-stage` → `.chat-messages` → `.message` → `.message-content` all use `min-width: 0`, `overflow` control, and word-break rules.
- **`.chat-input-container`** — `flex: 0 0 auto` inside `.chat-stage` (not a sibling below the scroll region).
- **`.meta-pill`** — `max-width: 100%` with wrap for long source labels.
- **Mobile (`@media max-width 720px`)** — fixed double horizontal padding on input.

### `public/index.html`

- **DOM restructure:** moved `.chat-input-container` inside `.chat-stage` so messages scroll above a fixed input footer within the column flex layout.
- **Script load order:** `instantReplies.js` before `app.js`.

---

## Major Change 8 — Response Speed Configuration

**Goal:** Reduce perceived latency via faster models and tighter timeouts.

### `example_env.txt` / local `.env`

- Documented and recommended: `GEMINI_MODEL=gemini-2.5-flash-lite`, `GEMINI_FAST_MODEL`, `GEMINI_REQUEST_TIMEOUT=20`, `GEMINI_MAX_ATTEMPTS=2`.
- `SEMANTIC_CACHE_ENABLED=false` by default to avoid embedding latency on cache miss.

### `on_the_porch/sql_chat/app4.py` & `on_the_porch/unified_chatbot.py`

- **Defaults aligned** with env tunables so code behavior matches configured models without hardcoding preview-model names.

---

## Major Change 9 — Events Fast Path (No LLM SQL)

**Goal:** Answer “events this week” without 3+ Gemini calls and without streaming timeouts.

### `on_the_porch/sql_chat/app4.py`

- **`is_generic_events_list_question()`** — detects broad calendar list queries.
- **`_events_list_days_ahead()`, `_fast_weekly_events_sql()`** — builds direct SQL on `weekly_events` for the next N days.
- **`_format_events_answer_fallback()`** — template answer if Gemini summarization fails.
- **`run_generic_events_query()`** — executes fast SQL + one summarize call (or template fallback).

### `on_the_porch/unified_chatbot.py`

- **`_run_sql()`** — tries `app4.run_generic_events_query()` first for generic event questions.
- **SQL generation failure** — retries fast path or returns a friendly error instead of raising.
- **Streaming SQL path** — wrapped in try/except so a SQL pipeline failure yields a user-visible message, not a worker crash.

---

## Major Change 10 — Instant Chitchat (No LLM, No DB)

**Goal:** Greetings like “hi” should not trigger routing, retrieval, or multi-second latency.

### `on_the_porch/unified_chatbot.py`

- **`_SMALL_TALK_*` constants, `is_small_talk()`, `small_talk_response()`, `build_small_talk_result()`** — server-side detection and canned replies; domain-hint blocklist prevents “hi, what events…” from short-circuiting.
- **`_execute_agent_response()`** — checks `is_small_talk()` before semantic cache and routing; returns `mode: "chitchat"`.
- **`stream_agent_response()`** — early return for chitchat before any routing LLM call.

### `public/instantReplies.js` *(new)*

- **Mirrors server logic:** `InstantReplies.isInstant()`, `InstantReplies.reply()` with the same phrase sets and domain hints.

### `public/app.js`

- **Instant path:** if `InstantReplies.isInstant(message)`, renders assistant bubble immediately (no typing indicator), then POSTs via non-streaming `sendMessage()` to persist; swaps optimistic IDs for server messages on success.
- **Data questions** — unchanged streaming path.

---

## Major Change 11 — Remove Google OAuth

**Goal:** Drop Google sign-in; password auth only for registered users.

### `api/api_v2.py`

- **Removed** `/auth/google/login`, `/auth/google/callback`, Authlib OAuth client setup, `has_google` checks, and Google-specific session/bootstrap logic.
- **Removed** Google admin promotion paths tied to OAuth provider.
- **Login/signup** — password-only via `auth_identities` with `provider='password'`.

### `api/security.py`

- **`serialize_user()`** — no longer surfaces Google-specific fields; `linked_providers` lists only configured providers (password).

### `api/requirements.txt` / `requirements.txt`

- **Removed Authlib** and other OAuth-only API dependencies from the API install set.

### `example_env.txt`

- **Removed** `GOOGLE_OAUTH_*` variables from the template (Gemini and Drive ingestion Google keys remain for data pipelines).

### `public/index.html`

- **Removed** “Continue with Google” button and Google provider UI.
- **Added** “Continue as Guest” button (`#guest-continue-button`) in its place.

### `public/app.js`

- **Removed** Google OAuth redirect handlers, provider pill rendering, and `google` auth mode branches.

### `public/api.js`

- **Removed** `loginWithGoogle()` and Google callback API helpers.

### `public/styles.css`

- **Removed** Google-branded button styles; **added** `.auth-guest`, `.auth-back-link`, `.header-login-button` styles.

---

## Major Change 12 — Guest Auth Backend

**Goal:** Ephemeral tab-scoped sessions without cookies; guests can chat but never become admin.

### `api/migrations/007_guest_users.sql` *(new)*

- **`users.is_guest BOOLEAN NOT NULL DEFAULT FALSE`** — marks ephemeral guest rows.
- **Index `idx_users_is_guest_created`** — supports cleanup queries by guest flag + age.

### `api/rate_limit.py` *(new)*

- **`RateLimiter`** — fixed-window counter with Redis backend (shared) or thread-safe memory fallback.
- **Redis path** — `SET key 0 EX window NX` + `INCR` pipeline so TTL is always set atomically.
- **Returns `(allowed, retry_after_seconds)`** — API sets `Retry-After` header on 429.

### `api/api_v2.py`

- **`POST /auth/guest`** — creates guest user + `web_sessions` row; returns `session_token`, `csrf_token`, `expires_at` in JSON (no cookies).
- **`_create_guest_user()`** — synthetic `@guest.local` email and `guest-{id}` username with `is_guest=True`.
- **`_create_web_session(..., is_guest=True)`** — shorter TTL from `GUEST_SESSION_HOURS` (default 24h).
- **`before_request`** — reads `Authorization: Bearer` when no session cookie; sets `g.bearer_session` and `g.is_guest`.
- **`_enforce_csrf()`** — for Bearer guests, validates `X-CSRF-Token` against server-stored CSRF hash (cookie double-submit unchanged for registered users).
- **`_touch_session_if_due()`** — debounced `last_seen_at` updates (`SESSION_TOUCH_DEBOUNCE_SECONDS`, default 5 min).
- **`_should_promote_to_admin()`** — returns `False` for guests; guests never promoted via `AUTH_ADMIN_EMAILS`.
- **`_require_admin()`** — rejects `is_guest` with `403 guest_not_allowed`.
- **`auth_me`** — includes top-level `is_guest` boolean.
- **`auth/login`** — accepts optional `guest_session_token` to revoke the old guest session on successful password login (no thread migration).
- **Rate limits** — `guest_create` per IP; `chat` per session on message POST/stream.

### `api/security.py`

- **`serialize_user()`** — adds `"is_guest": bool(row.get("is_guest", False))` to API user payloads.

### `api/test_api_v2.py`

- **`test_guest_bootstrap()`** — verifies `POST /auth/guest` returns Bearer token, CSRF, and `is_guest`.
- **`test_guest_auth_me()`** — Bearer auth on `GET /auth/me`.
- **`test_guest_create_conversation()`** — guest can create threads with CSRF headers.
- **`test_guest_admin_blocked()`** — guest gets `403 guest_not_allowed` on `/admin/stats`.

---

## Major Change 13 — Guest-by-Default Frontend

**Goal:** Open URL → chat immediately as Guest; refresh preserves session; new tab = new guest.

### `public/api.js`

- **`sessionStorage` keys:** `otp_guest_session_token`, `otp_guest_csrf_token`.
- **`storeGuestSession()`, `clearGuestSession()`, `getGuestSessionToken()`, `hasGuestSession()`** — tab-scoped token lifecycle.
- **`applyAuthHeaders()`** — attaches Bearer + CSRF for guests; cookie CSRF for registered users.
- **`createGuestSession()`** — `POST /auth/guest` and stores returned tokens.

### `public/app.js`

- **`state.isGuest`** — tracks guest vs registered session.
- **`ensureSession()`** — on load: if no cookie session, calls `createGuestSession()` and shows app view (auth view hidden by default).
- **`applySessionUser()`** — sets `isGuest`; clears guest tokens when a registered user is detected.
- **`handleAuthFailure()`** — guest 401 → clear tokens, bootstrap fresh guest with toast; registered 401 → auth screen.
- **`ensureActiveThread()`** — lazy thread creation on first message (no thread until user sends).
- **`renderUser()`** — guest shows “Guest” / “Temporary session”; hides admin link; sidebar logout reads “Sign in” for guests.
- **Guest empty-thread copy** — explains tab-scoped temporary sessions.

### `public/index.html`

- **`#app-view` visible by default; `#auth-view` hidden** — chat-first landing experience.

---

## Major Change 14 — Optional Login UX

**Goal:** Password signup/login remains available without forcing auth upfront.

### `public/index.html`

- **`#login-button`** in chat header — visible for guests only.
- **`#auth-back-to-chat`** — “← Back to chat” on auth screen when opened from guest session.

### `public/app.js`

- **`openAuthView()`** — guest opens auth with `authReturnToApp=true` so back-link is shown.
- **`handleBackToChat()`** — returns to app view without logging in.
- **`handleContinueAsGuest()`** — explicit guest bootstrap from auth screen.
- **`handleLogin()`** — passes `guest_session_token` in login payload so server revokes old guest session.
- **`updateAuthChrome()`** — toggles login button and back-link visibility.

### `public/styles.css`

- **`.auth-back-link`, `.header-login-button`, `.auth-guest`** — styling for new auth chrome.

---

## Major Change 15 — Admin Gating for Guests

**Goal:** Admin dashboard and mutating admin APIs require a password-authenticated admin account.

### `api/api_v2.py`

- **`_require_admin()`** — blocks guests (`guest_not_allowed`) and non-admin roles (`admin_required`).
- **`_require_admin_mutation()`** *(production hardening)* — admin check + CSRF for POST/PUT/DELETE admin routes.

### `public/admin.html`

- **`bootstrap()`** — checks `session.data.is_guest` / `user.is_guest`; shows access screen instead of dashboard.
- **`loadAll()`** — handles `403 guest_not_allowed` with guest-specific copy and link back to main app.

### `public/app.js`

- **`renderUser()`** — `#admin-link` hidden when `state.isGuest` (registered admins still see it).

---

## Major Change 16 — Orphan Guest Cleanup

**Goal:** Prevent unbounded growth of guest `users` rows after tabs close.

### `api/cleanup_guest_users.py` *(new)*

- **Finds stale guests** — `is_guest=TRUE`, no `auth_identities`, no active `web_sessions` with recent `last_seen_at` (default inactive 24h).
- **Batch deletes** — removes `interaction_log` rows (no FK cascade), then `users` rows (threads/messages/sessions cascade).
- **`--dry-run`** — reports counts without deleting.
- **Gated by `GUEST_CLEANUP_ENABLED`** — exits 0 with message when disabled.

### `cron_ingest.sh`

- **Added final step** — runs `api/cleanup_guest_users.py` after ingestion (no-op when env flag is false).

### `example_env.txt`

- **Documented:** `GUEST_CLEANUP_ENABLED`, `GUEST_CLEANUP_INACTIVE_HOURS`, `GUEST_CLEANUP_BATCH_SIZE`, `GUEST_CLEANUP_MAX_BATCHES`.

---

## Major Change 17 — Production Security & Hardening

**Goal:** Close security gaps found in the production architecture review.

### `api/api_v2.py`

- **Global error handlers** — `@app.errorhandler(HTTPException)` and `@app.errorhandler(Exception)` return JSON `{error, code}` instead of HTML stack traces.
- **`_verify_log_ownership()`** — `PUT /log` only updates rows owned by the caller's `user_id` or `session_id`; returns `403 log_forbidden` otherwise.
- **`log_interaction()` update path** — double-checks ownership before `UPDATE`.
- **`_validate_message()`** — enforces `MAX_MESSAGE_LENGTH` (default 8000) on all chat endpoints.
- **Content length limits** — `MAX_COMMUNITY_NOTE_LENGTH` on admin knowledge and community notes.
- **Auth rate limits** — `RATE_LIMIT_AUTH_PER_IP` / `RATE_LIMIT_AUTH_WINDOW_SECONDS` on signup and login.
- **Signup `IntegrityError`** — generic conflict message (no raw DB error text).
- **Expanded `/health`** — reports `database`, `redis`, `chroma`, `cache_backend`; status `degraded` when dependencies unhealthy.
- **Streaming parity** — semantic cache short-circuit in stream generator; `correction` SSE event after `apply_post_stream_fallback()`.
- **Community notes CSRF** — `_enforce_csrf()` when `current_user_row` is present.

### `api/rate_limit.py`

- **Redis TTL race fix** — `SET NX EX` + `INCR` pipeline ensures every rate-limit key has a TTL.

### `api/requirements.txt`

- **Added `flask-cors==5.0.1`** — matches `from flask_cors import CORS` in `api_v2.py`.

### `public/api.js`

- **429 handling** — `apiRequest()` and `sendMessageStream()` surface `retry_after` in error messages.
- **Streaming timeout** — `AbortController` with `chatTimeoutMs` (default 120s).

### `public/app.js`

- **429 toasts** — rate-limit errors shown via `showToast()` on chat failures.
- **`eventsPanel` fix** — `elements.eventsPanel = document.getElementById('events-panel')` so `ResizeObserver` attaches for event card layout.

### `example_env.txt`

- **Full production template** — Redis, gunicorn, semantic cache, rate limits, message limits, Gemini timeouts, `PARALLEL_RETRIEVAL`, `FLASK_DEBUG=false`, etc.

### `README.md`

- **Production checklist** — secrets, HTTPS cookies, Redis for multi-worker, gunicorn, cron, health checks, frontend `apiBaseUrl`.
- **Updated quick start** — documents `./start_api.sh` and `./start_redis.sh` alongside dev server.

---

## File Index (Quick Reference)

| File | Role |
|------|------|
| `api/api_v2.py` | Core API: auth, guests, streaming, caches, rate limits, admin, security |
| `api/security.py` | Password hashing, `serialize_user` with `is_guest` |
| `api/rate_limit.py` | **NEW** — Redis/memory rate limiter |
| `api/semantic_cache.py` | **NEW** — embedding similarity cache |
| `api/cleanup_guest_users.py` | **NEW** — orphan guest DB cleanup |
| `api/gunicorn_conf.py` | **NEW** — production server config |
| `api/migrations/007_guest_users.sql` | **NEW** — `users.is_guest` column |
| `api/test_api_v2.py` | Guest + admin smoke tests |
| `api/requirements.txt` | gunicorn, redis, flask-cors |
| `start_api.sh` | **NEW** — start Gunicorn |
| `start_redis.sh` | **NEW** — local Redis for cache |
| `cron_ingest.sh` | flock guard + guest cleanup step |
| `example_env.txt` | Full env template for dev and production |
| `README.md` | Production deployment guide |
| `on_the_porch/unified_chatbot.py` | Orchestrator: parallel RAG, streaming, chitchat, fallbacks |
| `on_the_porch/sql_chat/app4.py` | SQL pipeline: retries, pool, schema cache, events fast path |
| `on_the_porch/rag stuff/retrieval.py` | Chroma singleton, WAL, lock retry, absolute paths |
| `on_the_porch/rag stuff/boston_gov.py` | Safe empty dict returns |
| `on_the_porch/rag stuff/build_vectordb.py` | Path fix |
| `on_the_porch/rag stuff/ingest_rss.py` | Path fix |
| `public/app.js` | Guest-by-default, streaming UX, instant chitchat, auth chrome |
| `public/api.js` | Guest Bearer auth, SSE client, timeouts, 429 handling |
| `public/admin.html` | Guest admin block |
| `public/index.html` | Layout fix, guest/login UI, script tags |
| `public/instantReplies.js` | **NEW** — client-side instant replies |
| `public/config.js` | Streaming + timeout config |
| `public/styles.css` | Chat layout + auth chrome styles |
| `.gitignore` | `.redis/` |

---

## Request Flow (After All Changes)

```
1. Page load → ensureSession()
   ├─ cookie session? → registered user flow
   └─ else POST /auth/guest → Bearer in sessionStorage → app view

2. User sends message
   ├─ [CLIENT] InstantReplies.isInstant? → show bubble, POST /messages (persist)
   └─ else streaming POST /messages/stream
         ├─ semantic cache hit? → pseudo-stream cached answer
         ├─ is_small_talk? → chitchat (no LLM)
         ├─ history/cache gate → answer from context
         ├─ route → sql | rag | hybrid
         ├─ events fast path OR full SQL/RAG pipeline
         ├─ apply_post_stream_fallback() → correction event if enriched
         └─ persist messages + interaction_log

3. Guest closes tab → sessionStorage cleared → new guest on next visit
4. Guest clicks Log In → password auth → guest session revoked, fresh registered session
5. Cron → ingest steps → cleanup_guest_users.py (if enabled)
```

---

## Explicitly NOT Done (Deferred)

| Item | Status |
|------|--------|
| Rename `rag stuff/` → `rag/` | Deferred |
| Structured request-ID logging / per-stage timing | Deferred |
| Gemini → OSS / Hugging Face migration | Discussed only |
| Git commit | Waiting on user request |

---

## How to Run After Changes

```bash
./start_redis.sh                              # terminal 1 (recommended for multi-worker)
./venv/bin/python api/api_v2.py               # dev API (terminal 2)
# OR ./start_api.sh                           # production Gunicorn

cd public && python -m http.server 8000       # frontend (terminal 3)
```

Hard refresh browser (`Cmd+Shift+R`) after frontend changes.

---

*Documents all onboarding work: reliability phases 1–6, guest auth phases 1–6, and production security hardening.*
