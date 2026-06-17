"""
api_v2.py

Authenticated API for the Dorchester community chatbot.
Supports password auth, ephemeral guest sessions, role-based admin access,
and per-user conversation threads backed by MySQL.
"""

from __future__ import annotations

import datetime
import os
import re
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from dotenv import load_dotenv
from flask import Flask, Response, g, has_request_context, jsonify, request, session, stream_with_context
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool

from db_migrations import run_migrations
from rate_limit import RateLimiter
from security import (
    generate_token,
    get_client_ip,
    get_token_secret,
    hash_password,
    hash_token,
    json_dumps,
    json_loads,
    normalize_email,
    normalize_username,
    serialize_user,
    summarize_thread_title,
    utcnow,
    validate_password,
    validate_username,
    verify_password,
)

# Setup paths to import from on_the_porch
_THIS_FILE = Path(__file__).resolve()
_API_DIR = _THIS_FILE.parent
_ROOT_DIR = _API_DIR.parent
load_dotenv(_ROOT_DIR / ".env")
_ON_THE_PORCH_DIR = _ROOT_DIR / "on_the_porch"

if str(_ON_THE_PORCH_DIR) not in sys.path:
    sys.path.insert(0, str(_ON_THE_PORCH_DIR))

_RAG_DIR = _ON_THE_PORCH_DIR / "rag stuff"
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

from unified_chatbot import (  # noqa: E402
    _answer_from_history,
    _bootstrap_env,
    _check_if_needs_new_data,
    _fix_retrieval_vectordb_path,
    _get_llm_client,
    _is_calendar_question,
    _route_question,
    _run_hybrid,
    _run_rag,
    _run_sql,
    build_retrieval_cache,
    build_small_talk_result,
    create_empty_cache,
    is_small_talk,
    apply_post_stream_fallback,
    stream_agent_response,
)
from semantic_cache import SemanticResponseCache  # noqa: E402

try:
    from ingest_community_notes import ingest_community_notes as _ingest_community_notes  # noqa: E402
except Exception as _ingest_import_err:
    _ingest_community_notes = None
    print(f"Warning: could not import ingest_community_notes: {_ingest_import_err}")

try:
    import chromadb as _chromadb
    import retrieval as _retrieval
except Exception as _chroma_import_err:
    _chromadb = None
    _retrieval = None
    print(f"Warning: could not import chromadb/retrieval: {_chroma_import_err}")

_bootstrap_env()
_fix_retrieval_vectordb_path()


def _trigger_ingest() -> None:
    """Run ingest_community_notes in a daemon thread so it doesn't block the response."""
    if _ingest_community_notes is None:
        return

    def _run():
        try:
            stats = _ingest_community_notes()
            print(f"Community notes ingested: {stats}")
        except Exception as exc:
            print(f"Background community notes ingest failed: {exc}")

    threading.Thread(target=_run, daemon=True).start()


def _remove_from_chroma(entry_id: int) -> None:
    """Delete a community note from the Chroma vector DB by its note ID."""
    if _chromadb is None or _retrieval is None:
        return

    def _run():
        try:
            client = _chromadb.PersistentClient(path=str(_retrieval.VECTORDB_DIR))
            collection = client.get_collection("langchain")
            collection.delete(ids=[f"community_note_{entry_id}"])
            print(f"Removed community_note_{entry_id} from Chroma")
        except Exception as exc:
            print(f"Background Chroma removal failed for note {entry_id}: {exc}")

    threading.Thread(target=_run, daemon=True).start()

# The legacy /chat session cache is provided by a pluggable backend defined
# after Config (see `_session_cache`). Set CACHE_BACKEND=redis to share it
# across gunicorn workers; it defaults to a thread-safe in-process cache.


class Config:
    API_VERSION = "v3.0"

    _raw_keys = os.getenv("RETHINKAI_API_KEYS", "").split(",")
    RETHINKAI_API_KEYS = [key.strip() for key in _raw_keys if key.strip()]

    HOST = os.getenv("API_HOST", "127.0.0.1")
    PORT = int(os.getenv("API_PORT", "8888"))
    # Debug mode must be OFF in production (it enables the reloader and an
    # interactive debugger that can execute arbitrary code). Opt in explicitly.
    DEBUG = os.getenv("FLASK_DEBUG", "false").strip().lower() in ("1", "true", "yes")

    # Legacy /chat session cache backend:
    #   "memory" -> per-process, thread-safe, bounded by TTL+LRU (default)
    #   "redis"  -> shared across all gunicorn workers, per-key TTL
    CACHE_BACKEND = os.getenv("CACHE_BACKEND", "memory").strip().lower()
    REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "60"))
    CACHE_MAX_SESSIONS = int(os.getenv("CACHE_MAX_SESSIONS", "100"))
    CACHE_KEY_PREFIX = os.getenv("CACHE_KEY_PREFIX", "rethinkai:chatcache:")

    # Semantic response cache (OFF by default). Answers near-duplicate first-turn
    # questions instantly from a recent cached answer.
    SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "false").strip().lower() in ("1", "true", "yes")
    SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.95"))
    SEMANTIC_CACHE_TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "900"))
    SEMANTIC_CACHE_MAX = int(os.getenv("SEMANTIC_CACHE_MAX", "200"))

    # Response streaming endpoint (OFF by default on the client; the endpoint is
    # always available). See /conversations/<id>/messages/stream.
    STREAMING_ENABLED = os.getenv("STREAMING_ENABLED", "true").strip().lower() in ("1", "true", "yes")

    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "agent-api-secret-2025")
    SESSION_COOKIE_SECURE = os.getenv("FLASK_SESSION_COOKIE_SECURE", "False").lower() == "true"
    AUTH_SESSION_COOKIE_NAME = os.getenv("AUTH_SESSION_COOKIE_NAME", "otp_session")
    AUTH_CSRF_COOKIE_NAME = os.getenv("AUTH_CSRF_COOKIE_NAME", "otp_csrf")
    SESSION_MAX_AGE_DAYS = int(os.getenv("AUTH_SESSION_MAX_AGE_DAYS", "7"))
    SESSION_SAMESITE = os.getenv("AUTH_SESSION_SAMESITE", "Lax")
    GUEST_SESSION_HOURS = max(1, int(os.getenv("GUEST_SESSION_HOURS", "24")))
    SESSION_TOUCH_DEBOUNCE_SECONDS = max(
        0, int(os.getenv("SESSION_TOUCH_DEBOUNCE_SECONDS", "300"))
    )

    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() in (
        "1", "true", "yes",
    )
    RATE_LIMIT_KEY_PREFIX = os.getenv("RATE_LIMIT_KEY_PREFIX", "rethinkai:ratelimit:")
    RATE_LIMIT_GUEST_CREATE_PER_IP = max(
        1, int(os.getenv("RATE_LIMIT_GUEST_CREATE_PER_IP", "30"))
    )
    RATE_LIMIT_GUEST_CREATE_WINDOW_SECONDS = max(
        60, int(os.getenv("RATE_LIMIT_GUEST_CREATE_WINDOW_SECONDS", "3600"))
    )
    RATE_LIMIT_CHAT_PER_SESSION = max(
        1, int(os.getenv("RATE_LIMIT_CHAT_PER_SESSION", "60"))
    )
    RATE_LIMIT_CHAT_WINDOW_SECONDS = max(
        60, int(os.getenv("RATE_LIMIT_CHAT_WINDOW_SECONDS", "3600"))
    )
    RATE_LIMIT_AUTH_PER_IP = max(1, int(os.getenv("RATE_LIMIT_AUTH_PER_IP", "30")))
    RATE_LIMIT_AUTH_WINDOW_SECONDS = max(
        60, int(os.getenv("RATE_LIMIT_AUTH_WINDOW_SECONDS", "900"))
    )

    MAX_MESSAGE_LENGTH = max(500, int(os.getenv("MAX_MESSAGE_LENGTH", "8000")))
    MAX_COMMUNITY_NOTE_LENGTH = max(500, int(os.getenv("MAX_COMMUNITY_NOTE_LENGTH", "4000")))

    APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000")
    API_BASE_URL = os.getenv("API_BASE_URL", f"http://{HOST}:{PORT}")
    _allowed_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    ).split(",")
    ALLOWED_ORIGINS = [origin.strip() for origin in _allowed_origins if origin.strip()]

    MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DB = os.getenv("MYSQL_DB", "rethink_ai_boston")
    # Connection pool tuning. pool_size is capped at 32 by mysql-connector.
    MYSQL_POOL_SIZE = max(1, min(32, int(os.getenv("MYSQL_POOL_SIZE", "10"))))
    MYSQL_CONNECT_TIMEOUT = int(os.getenv("MYSQL_CONNECT_TIMEOUT", "10"))

    LOGIN_WINDOW_MINUTES = int(os.getenv("AUTH_LOGIN_WINDOW_MINUTES", "15"))
    LOGIN_LOCK_THRESHOLD = int(os.getenv("AUTH_LOGIN_LOCK_THRESHOLD", "5"))
    LOGIN_LOCK_MINUTES = int(os.getenv("AUTH_LOGIN_LOCK_MINUTES", "15"))
    _raw_admin_emails = os.getenv("AUTH_ADMIN_EMAILS", "").split(",")
    AUTH_ADMIN_EMAILS = {normalize_email(email) for email in _raw_admin_emails if normalize_email(email)}


DOC_TYPE_DIRS = {
    "policy": urljoin(Config.APP_BASE_URL.rstrip("/") + "/", "Policies/"),
    "transcript": "Data/AI meeting transcripts",
    "calendar_event": "Data/newsletters",
    "boston_gov_answer": "https://www.boston.gov",
}


def _log_exception(prefix: str, exc: Exception) -> None:
    """
    Log exceptions without leaking secrets (e.g., DB usernames/passwords).

    We intentionally avoid printing the full exception string, since some
    connector errors include connection context.
    """
    try:
        errno = getattr(exc, "errno", None)
        suffix = f" (errno={errno})" if errno is not None else ""
        print(f"{prefix}: {exc.__class__.__name__}{suffix}", file=sys.stderr)
    except Exception:
        print(f"{prefix}: <error>", file=sys.stderr)


db_pool = MySQLConnectionPool(
    host=Config.MYSQL_HOST,
    port=Config.MYSQL_PORT,
    user=Config.MYSQL_USER,
    password=Config.MYSQL_PASSWORD,
    database=Config.MYSQL_DB,
    pool_name="api_v2_pool",
    pool_size=Config.MYSQL_POOL_SIZE,
    # Reset session state when a connection is returned to the pool so no
    # leftover state (variables, temp tables) leaks between requests.
    pool_reset_session=True,
    connection_timeout=Config.MYSQL_CONNECT_TIMEOUT,
)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=Config.SECRET_KEY,
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(days=Config.SESSION_MAX_AGE_DAYS),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=Config.SESSION_COOKIE_SECURE,
    SESSION_COOKIE_SAMESITE=Config.SESSION_SAMESITE,
)

CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": Config.ALLOWED_ORIGINS}},
    allow_headers=["Content-Type", "X-CSRF-Token", "RethinkAI-API-Key", "Authorization"],
)


def get_db_connection():
    """Check out a *live* connection from the pool.

    MySQL closes idle connections after ``wait_timeout``. A pooled connection
    that has gone stale raises "MySQL server has gone away" (errno 2006/2013)
    on first use — the classic intermittent 500 that happens precisely when
    traffic is *low* (connections sit idle long enough to be dropped).

    We ``ping(reconnect=True)`` before returning so a dead connection is
    transparently revived. If that slot cannot be revived, we discard it and
    take another from the pool.
    """
    conn = db_pool.get_connection()
    try:
        conn.ping(reconnect=True, attempts=2, delay=1)
        return conn
    except Exception:
        # Could not revive this pooled connection; return it and get a fresh one.
        try:
            conn.close()
        except Exception:
            pass
        conn = db_pool.get_connection()
        conn.ping(reconnect=True, attempts=2, delay=1)
        return conn


def initialize_database() -> None:
    run_migrations(get_db_connection)


initialize_database()

def ensure_interaction_log_columns():
    """Add missing columns to interaction log."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
                       SELECT COLUMN_NAME
                       FROM INFORMATION_SCHEMA.COLUMNS
                       WHERE TABLE_SCHEMA = DATABASE()
                       AND TABLE_NAME = 'interaction_log'
                       """)
        existing = {row["COLUMN_NAME"] for row in cursor.fetchall()}
        additions = [
            ("flagged", "ALTER TABLE interaction_log ADD COLUMN flagged BOOLEAN DEFAULT FALSE"),
            ("flag_reason", "ALTER TABLE interaction_log ADD COLUMN flag_reason VARCHAR(100)"),
            ("flag_details", "ALTER TABLE interaction_log ADD COLUMN flag_details TEXT"),
            ("flagged_at", "ALTER TABLE interaction_log ADD COLUMN flagged_at TIMESTAMP NULL"),
            ("moderator_comment", "ALTER TABLE interaction_log ADD COLUMN moderator_comment TEXT"),
            ("resolved", "ALTER TABLE interaction_log ADD COLUMN resolved BOOLEAN DEFAULT FALSE"),
            ("resolved_at", "ALTER TABLE interaction_log ADD COLUMN resolved_at TIMESTAMP NULL"),
        ]
        for col, sql in additions:
            if col not in existing:
                cursor.execute(sql)
        conn.commit()
        print("✓ interaction_log columns ready")
    except Exception as e:
        _log_exception("Warning: Could not update interaction_log columns", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def ensure_admin_knowledge_table():
    """Create admin_knowledge table if it doesn't exist."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_knowledge (
                id INT AUTO_INCREMENT PRIMARY KEY,
                content TEXT,
                category VARCHAR(100) default 'general',
                expires_at TIMESTAMP NULL,
                source_flag_id INT,
                added_by VARCHAR(255),
                active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        print("✓ admin_knowledge table ready")
    except Exception as e:
        _log_exception("Warning: Could not create admin_knowledge table", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

ensure_interaction_log_columns()
ensure_admin_knowledge_table()

def _bootstrap_admin_users() -> None:
    if not Config.AUTH_ADMIN_EMAILS:
        return

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        placeholders = ", ".join(["%s"] * len(Config.AUTH_ADMIN_EMAILS))
        cursor = conn.cursor()
        cursor.execute(
            f"""
            UPDATE users
            SET role = 'admin'
            WHERE email IN ({placeholders}) AND role <> 'admin'
            """,
            tuple(sorted(Config.AUTH_ADMIN_EMAILS)),
        )
        conn.commit()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


_bootstrap_admin_users()


class _BaseSessionCache:
    """Pluggable store for the legacy /chat retrieval cache."""

    def get(self, session_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def set(self, session_id: str, cache: Dict[str, Any]) -> None:
        raise NotImplementedError


class _MemorySessionCache(_BaseSessionCache):
    """In-process cache. Thread-safe and bounded by TTL + LRU.

    NOTE: this is per-process and is NOT shared across gunicorn workers. It is
    correct for single-process or sticky-session deployments; use the redis
    backend for multi-worker sharing. The lock fixes the previous race where
    concurrent /chat requests mutated a bare module dict.
    """

    def __init__(self, ttl_minutes: int, max_sessions: int) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._ttl_minutes = ttl_minutes
        self._max_sessions = max_sessions

    def get(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            cache = self._data.get(session_id)
            return cache if cache is not None else create_empty_cache()

    def set(self, session_id: str, cache: Dict[str, Any]) -> None:
        with self._lock:
            self._data[session_id] = cache
            self._evict_locked()

    def _evict_locked(self) -> None:
        if len(self._data) <= self._max_sessions:
            return
        now = datetime.datetime.now()
        stale_keys = []
        for sid, cache in self._data.items():
            timestamp = cache.get("timestamp")
            if not timestamp:
                continue
            try:
                cache_time = datetime.datetime.fromisoformat(timestamp)
            except Exception:
                continue
            if (now - cache_time).total_seconds() / 60 > self._ttl_minutes:
                stale_keys.append(sid)
        for sid in stale_keys:
            self._data.pop(sid, None)
        if len(self._data) > self._max_sessions:
            ordered = sorted(self._data.items(), key=lambda kv: kv[1].get("timestamp", ""))
            overflow = len(self._data) - self._max_sessions
            for sid, _ in ordered[:overflow]:
                self._data.pop(sid, None)


class _RedisSessionCache(_BaseSessionCache):
    """Cross-worker cache backed by Redis, with per-key TTL.

    A Redis outage must never break chat: get/set failures are logged and
    degrade to "no cache" (a fresh retrieval) rather than raising.
    """

    def __init__(self, client, ttl_minutes: int, key_prefix: str) -> None:
        self._client = client
        self._ttl_seconds = max(60, ttl_minutes * 60)
        self._key_prefix = key_prefix

    def _key(self, session_id: str) -> str:
        return f"{self._key_prefix}{session_id}"

    def get(self, session_id: str) -> Dict[str, Any]:
        try:
            raw = self._client.get(self._key(session_id))
            if raw:
                return json_loads(raw, create_empty_cache())
        except Exception as exc:  # noqa: BLE001
            _log_exception("Redis cache get failed", exc)
        return create_empty_cache()

    def set(self, session_id: str, cache: Dict[str, Any]) -> None:
        try:
            self._client.set(self._key(session_id), json_dumps(cache), ex=self._ttl_seconds)
        except Exception as exc:  # noqa: BLE001
            _log_exception("Redis cache set failed", exc)


def _build_session_cache() -> _BaseSessionCache:
    """Construct the configured cache backend, falling back to memory on error."""
    if Config.CACHE_BACKEND == "redis":
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(
                Config.REDIS_URL,
                socket_timeout=2,
                socket_connect_timeout=2,
            )
            client.ping()
            print(f"Session cache backend: redis ({Config.REDIS_URL})")
            return _RedisSessionCache(client, Config.CACHE_TTL_MINUTES, Config.CACHE_KEY_PREFIX)
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: redis cache backend unavailable ({exc.__class__.__name__}); "
                "falling back to in-memory cache.",
                file=sys.stderr,
            )
    return _MemorySessionCache(Config.CACHE_TTL_MINUTES, Config.CACHE_MAX_SESSIONS)


_session_cache = _build_session_cache()


def _build_rate_limiter() -> RateLimiter:
    redis_client = None
    if Config.CACHE_BACKEND == "redis":
        try:
            import redis  # type: ignore

            redis_client = redis.Redis.from_url(
                Config.REDIS_URL,
                socket_timeout=2,
                socket_connect_timeout=2,
            )
            redis_client.ping()
        except Exception as exc:  # noqa: BLE001
            _log_exception("Rate limiter redis unavailable; using in-memory fallback", exc)
    return RateLimiter(redis_client, key_prefix=Config.RATE_LIMIT_KEY_PREFIX)


_rate_limiter = _build_rate_limiter()


_semantic_embeddings = None


def _semantic_embed(text: str):
    """Embed a question for the semantic cache, reusing one embeddings object."""
    global _semantic_embeddings
    if _semantic_embeddings is None:
        import retrieval  # available via the path set up by the unified_chatbot import
        _semantic_embeddings = retrieval.GeminiEmbeddings()
    return _semantic_embeddings.embed_query(text)


_semantic_cache = SemanticResponseCache(
    enabled=Config.SEMANTIC_CACHE_ENABLED,
    threshold=Config.SEMANTIC_CACHE_THRESHOLD,
    ttl_seconds=Config.SEMANTIC_CACHE_TTL_SECONDS,
    max_entries=Config.SEMANTIC_CACHE_MAX,
    embed_fn=_semantic_embed,
)
if Config.SEMANTIC_CACHE_ENABLED:
    print(
        f"Semantic response cache: enabled "
        f"(threshold={Config.SEMANTIC_CACHE_THRESHOLD}, ttl={Config.SEMANTIC_CACHE_TTL_SECONDS}s)"
    )

# Words that make a question time-sensitive; such questions are never served
# from (or stored in) the semantic cache because the right answer changes daily.
_SEMANTIC_TIME_WORDS = (
    "today", "tonight", "yesterday", "tomorrow", "this week", "last week",
    "next week", "this month", "last month", "this year", "recent", "recently",
    "latest", "currently", "right now", "this weekend",
)


def _semantic_cache_eligible(message: str, conversation_history: List[Dict[str, str]]) -> bool:
    """Only cache/serve fresh-topic, non-time-sensitive first-turn questions."""
    if not Config.SEMANTIC_CACHE_ENABLED:
        return False
    if conversation_history:
        return False
    if _is_calendar_question(message):
        return False
    lowered = (message or "").lower()
    if any(word in lowered for word in _SEMANTIC_TIME_WORDS):
        return False
    return True


# =============================================================================
# Utility helpers
# =============================================================================

def _cookie_kwargs(*, httponly: bool, expires: Optional[datetime.datetime] = None) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "httponly": httponly,
        "secure": Config.SESSION_COOKIE_SECURE,
        "samesite": Config.SESSION_SAMESITE,
        "path": "/",
    }
    if expires is not None:
        kwargs["expires"] = expires
    return kwargs


def _set_auth_cookies(response, session_token: str, csrf_token: str, expires_at: datetime.datetime) -> None:
    response.set_cookie(
        Config.AUTH_SESSION_COOKIE_NAME,
        session_token,
        **_cookie_kwargs(httponly=True, expires=expires_at),
    )
    response.set_cookie(
        Config.AUTH_CSRF_COOKIE_NAME,
        csrf_token,
        **_cookie_kwargs(httponly=False, expires=expires_at),
    )
    g.csrf_cookie_written = True



def _clear_auth_cookies(response) -> None:
    response.delete_cookie(Config.AUTH_SESSION_COOKIE_NAME, path="/")
    response.delete_cookie(Config.AUTH_CSRF_COOKIE_NAME, path="/")
    g.csrf_cookie_written = True



def _json_error(message: str, status: int, code: Optional[str] = None):
    payload = {"error": message}
    if code:
        payload["code"] = code
    return jsonify(payload), status



@app.errorhandler(HTTPException)
def handle_http_exception(exc: HTTPException):
    return _json_error(exc.description or "Request failed.", exc.code or 500, "http_error")


@app.errorhandler(Exception)
def handle_unhandled_exception(exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    _log_exception("Unhandled server error", exc)
    return _json_error("Internal server error.", 500, "internal_error")


def _provider_names(conn, user_id: str) -> List[str]:
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT provider FROM auth_identities WHERE user_id = %s", (user_id,))
        return [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()



def _fetch_user_by_id(conn, user_id: str) -> Optional[Dict[str, Any]]:
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM users WHERE id = %s LIMIT 1", (user_id,))
        return cursor.fetchone()
    finally:
        cursor.close()



def _fetch_user_by_email(conn, email: str) -> Optional[Dict[str, Any]]:
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM users WHERE email = %s LIMIT 1", (email,))
        return cursor.fetchone()
    finally:
        cursor.close()



def _fetch_user_by_username(conn, username: str) -> Optional[Dict[str, Any]]:
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM users WHERE username = %s LIMIT 1", (username,))
        return cursor.fetchone()
    finally:
        cursor.close()



def _fetch_password_login_row(conn, email: str) -> Optional[Dict[str, Any]]:
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT ai.id AS identity_id, ai.password_hash, ai.last_used_at,
                   u.id, u.email, u.username, u.role, u.status,
                   u.profile_complete, u.created_at, u.updated_at, u.last_login_at
            FROM auth_identities ai
            JOIN users u ON u.id = ai.user_id
            WHERE ai.provider = 'password' AND u.email = %s
            LIMIT 1
            """,
            (email,),
        )
        return cursor.fetchone()
    finally:
        cursor.close()



def _create_user(
    conn,
    email: str,
    username: str,
    *,
    profile_complete: bool,
    is_guest: bool = False,
) -> Dict[str, Any]:
    user_id = str(uuid.uuid4())
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO users (id, email, username, role, status, profile_complete, is_guest)
            VALUES (%s, %s, %s, 'user', 'active', %s, %s)
            """,
            (user_id, email, username, profile_complete, is_guest),
        )
    finally:
        cursor.close()
    return _fetch_user_by_id(conn, user_id)


def _create_guest_user(conn) -> Dict[str, Any]:
    short_id = uuid.uuid4().hex[:8]
    email = f"guest-{uuid.uuid4()}@guest.local"
    username = f"guest-{short_id}"
    return _create_user(conn, email, username, profile_complete=True, is_guest=True)


def _should_promote_to_admin(email: str, *, is_guest: bool = False) -> bool:
    if is_guest:
        return False
    return normalize_email(email) in Config.AUTH_ADMIN_EMAILS


def _promote_user_to_admin_if_configured(
    conn,
    *,
    user_id: str,
    email: str,
    current_role: Optional[str] = None,
    audit_event: Optional[str] = None,
    is_guest: bool = False,
) -> bool:
    if not _should_promote_to_admin(email, is_guest=is_guest):
        return False
    if current_role == "admin":
        return False

    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE users SET role = 'admin' WHERE id = %s AND role <> 'admin'",
            (user_id,),
        )
        updated = cursor.rowcount > 0
    finally:
        cursor.close()

    if updated and audit_event and has_request_context():
        _record_auth_event(
            conn,
            audit_event,
            success=True,
            user_id=user_id,
            details={"email": normalize_email(email), "source": "AUTH_ADMIN_EMAILS"},
        )
    return updated



def _create_auth_identity(
    conn,
    *,
    user_id: str,
    provider: str,
    provider_subject: Optional[str] = None,
    password_hash_value: Optional[str] = None,
) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO auth_identities (id, user_id, provider, provider_subject, password_hash)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (str(uuid.uuid4()), user_id, provider, provider_subject, password_hash_value),
        )
    finally:
        cursor.close()



def _update_user_login_stamp(conn, user_id: str) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE users SET last_login_at = %s WHERE id = %s",
            (utcnow(), user_id),
        )
    finally:
        cursor.close()



def _update_identity_last_used(conn, user_id: str, provider: str) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE auth_identities SET last_used_at = %s WHERE user_id = %s AND provider = %s",
            (utcnow(), user_id, provider),
        )
    finally:
        cursor.close()



def _update_password_hash(conn, identity_id: str, new_password_hash: str) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE auth_identities SET password_hash = %s WHERE id = %s",
            (new_password_hash, identity_id),
        )
    finally:
        cursor.close()



def _create_web_session(conn, user_id: str, *, is_guest: bool = False) -> Dict[str, Any]:
    session_id = str(uuid.uuid4())
    session_token = generate_token(32)
    csrf_token = generate_token(24)
    if is_guest:
        expires_at = utcnow() + datetime.timedelta(hours=Config.GUEST_SESSION_HOURS)
    else:
        expires_at = utcnow() + datetime.timedelta(days=Config.SESSION_MAX_AGE_DAYS)
    secret = get_token_secret()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO web_sessions (
                id, user_id, session_token_hash, csrf_token_hash,
                user_agent, ip_created, last_seen_at, expires_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                session_id,
                user_id,
                hash_token(session_token, secret),
                hash_token(csrf_token, secret),
                request.headers.get("User-Agent", "")[:512],
                get_client_ip(request.headers, request.remote_addr),
                utcnow(),
                expires_at,
            ),
        )
    finally:
        cursor.close()
    return {
        "id": session_id,
        "session_token": session_token,
        "csrf_token": csrf_token,
        "expires_at": expires_at,
        "is_guest": is_guest,
    }



def _revoke_session(conn, session_id: str) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE web_sessions SET revoked_at = %s WHERE id = %s AND revoked_at IS NULL",
            (utcnow(), session_id),
        )
    finally:
        cursor.close()



def _current_session_row(conn, raw_session_token: str) -> Optional[Dict[str, Any]]:
    secret = get_token_secret()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT ws.id AS session_id, ws.user_id, ws.csrf_token_hash,
                   ws.expires_at, ws.revoked_at, ws.last_seen_at,
                   u.id, u.email, u.username, u.role, u.status,
                   u.profile_complete, u.is_guest,
                   u.created_at, u.updated_at, u.last_login_at
            FROM web_sessions ws
            JOIN users u ON u.id = ws.user_id
            WHERE ws.session_token_hash = %s
            LIMIT 1
            """,
            (hash_token(raw_session_token, secret),),
        )
        return cursor.fetchone()
    finally:
        cursor.close()



def _touch_session(conn, session_id: str) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE web_sessions SET last_seen_at = %s WHERE id = %s",
            (utcnow(), session_id),
        )
    finally:
        cursor.close()



def _touch_session_if_due(conn, session_row: Dict[str, Any]) -> bool:
    """Update last_seen_at at most once per debounce window."""
    debounce = Config.SESSION_TOUCH_DEBOUNCE_SECONDS
    if debounce <= 0:
        _touch_session(conn, session_row["session_id"])
        return True

    last_seen = session_row.get("last_seen_at")
    if last_seen is not None:
        elapsed = (utcnow() - last_seen).total_seconds()
        if elapsed < debounce:
            return False

    _touch_session(conn, session_row["session_id"])
    return True



def _activate_web_session(
    conn,
    raw_session_token: str,
    *,
    via_bearer: bool = False,
) -> Optional[Dict[str, Any]]:
    session_row = _current_session_row(conn, raw_session_token)
    if not session_row:
        return None

    if session_row.get("revoked_at") or session_row.get("expires_at") <= utcnow():
        _revoke_session(conn, session_row["session_id"])
        conn.commit()
        return None

    providers = _provider_names(conn, session_row["user_id"])
    promoted = False
    if not session_row.get("is_guest"):
        promoted = _promote_user_to_admin_if_configured(
            conn,
            user_id=session_row["user_id"],
            email=session_row["email"],
            current_role=session_row.get("role"),
            audit_event="admin_role_bootstrap",
            is_guest=False,
        )

    touched = _touch_session_if_due(conn, session_row)
    if touched or promoted:
        conn.commit()

    if promoted:
        session_row = _current_session_row(conn, raw_session_token) or session_row

    g.session_row = session_row
    g.current_user_row = session_row
    g.linked_providers = providers
    g.current_user = serialize_user(session_row, providers)
    g.session_id = session_row["session_id"]
    g.bearer_session = via_bearer
    g.is_guest = bool(session_row.get("is_guest"))
    return session_row



def _record_auth_event(
    conn,
    event_type: str,
    *,
    success: bool,
    user_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO auth_audit_log (user_id, event_type, success, ip_address, user_agent, details_json)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                user_id,
                event_type,
                success,
                get_client_ip(request.headers, request.remote_addr),
                request.headers.get("User-Agent", "")[:512],
                json_dumps(details or {}),
            ),
        )
    finally:
        cursor.close()



def _fetch_login_attempt(conn, email: str, ip_address: str) -> Optional[Dict[str, Any]]:
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT * FROM login_attempts WHERE normalized_email = %s AND ip_address = %s LIMIT 1",
            (email, ip_address),
        )
        return cursor.fetchone()
    finally:
        cursor.close()



def _is_locked(record: Optional[Dict[str, Any]]) -> bool:
    return bool(record and record.get("locked_until") and record["locked_until"] > utcnow())



def _record_failed_login(conn, email: str, ip_address: str) -> Optional[datetime.datetime]:
    now = utcnow()
    record = _fetch_login_attempt(conn, email, ip_address)
    cursor = conn.cursor()
    try:
        if not record:
            failure_count = 1
            locked_until = now + datetime.timedelta(minutes=Config.LOGIN_LOCK_MINUTES) if failure_count >= Config.LOGIN_LOCK_THRESHOLD else None
            cursor.execute(
                """
                INSERT INTO login_attempts (
                    normalized_email, ip_address, failure_count,
                    first_attempt_at, last_attempt_at, locked_until
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (email, ip_address, failure_count, now, now, locked_until),
            )
            return locked_until

        first_attempt = record["first_attempt_at"]
        if first_attempt is None or (now - first_attempt).total_seconds() > Config.LOGIN_WINDOW_MINUTES * 60:
            failure_count = 1
            first_attempt = now
        else:
            failure_count = int(record.get("failure_count", 0)) + 1

        locked_until = now + datetime.timedelta(minutes=Config.LOGIN_LOCK_MINUTES) if failure_count >= Config.LOGIN_LOCK_THRESHOLD else None
        cursor.execute(
            """
            UPDATE login_attempts
            SET failure_count = %s,
                first_attempt_at = %s,
                last_attempt_at = %s,
                locked_until = %s
            WHERE normalized_email = %s AND ip_address = %s
            """,
            (failure_count, first_attempt, now, locked_until, email, ip_address),
        )
        return locked_until
    finally:
        cursor.close()



def _clear_login_attempts(conn, email: str, ip_address: str) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            "DELETE FROM login_attempts WHERE normalized_email = %s AND ip_address = %s",
            (email, ip_address),
        )
    finally:
        cursor.close()



def _enforce_csrf():
    if g.get("api_key_authenticated") and not g.get("current_user_row"):
        return None

    header_token = request.headers.get("X-CSRF-Token", "")
    session_row = g.get("session_row")

    if g.get("bearer_session"):
        if not header_token or not session_row:
            return _json_error("CSRF validation failed.", 403, "csrf_failed")
        expected_hash = session_row.get("csrf_token_hash")
        if expected_hash != hash_token(header_token, get_token_secret()):
            return _json_error("CSRF validation failed.", 403, "csrf_failed")
        return None

    cookie_token = request.cookies.get(Config.AUTH_CSRF_COOKIE_NAME, "")
    if not cookie_token or not header_token or cookie_token != header_token:
        return _json_error("CSRF validation failed.", 403, "csrf_failed")

    if session_row:
        expected_hash = session_row.get("csrf_token_hash")
        if expected_hash != hash_token(cookie_token, get_token_secret()):
            return _json_error("CSRF validation failed.", 403, "csrf_failed")

    return None



def _require_user(*, allow_incomplete: bool = False):
    current_user = g.get("current_user_row")
    if not current_user:
        return _json_error("Authentication required.", 401, "auth_required")
    if current_user.get("status") != "active":
        return _json_error("Account is disabled.", 403, "account_disabled")
    if not allow_incomplete and not current_user.get("profile_complete"):
        return _json_error("Complete your profile before using the app.", 409, "profile_incomplete")
    return None



def _require_admin():
    result = _require_user()
    if result:
        return result
    if g.get("current_user_row", {}).get("is_guest"):
        return _json_error("Admin access required.", 403, "guest_not_allowed")
    if g.get("current_user_row", {}).get("role") != "admin":
        return _json_error("Admin access required.", 403, "admin_required")
    return None


def _require_admin_mutation():
    """Admin check plus CSRF for cookie-authenticated mutating requests."""
    result = _require_admin()
    if result:
        return result
    return _enforce_csrf()


def _validate_message(message: str):
    text = (message or "").strip()
    if not text:
        return _json_error("Message is required.", 400, "missing_message")
    if len(text) > Config.MAX_MESSAGE_LENGTH:
        return _json_error(
            f"Message is too long (max {Config.MAX_MESSAGE_LENGTH} characters).",
            400,
            "message_too_long",
        )
    return None


def _verify_log_ownership(
    conn,
    log_id: int,
    *,
    user_id: Optional[str],
    session_id: str,
) -> bool:
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT user_id, session_id FROM interaction_log WHERE id = %s",
            (log_id,),
        )
        row = cursor.fetchone()
        if not row:
            return False
        if user_id and row.get("user_id") == user_id:
            return True
        if session_id and row.get("session_id") == session_id:
            if row.get("user_id") is None:
                return True
            return bool(user_id and row.get("user_id") == user_id)
        return False
    finally:
        cursor.close()


def _check_redis_health() -> str:
    if Config.CACHE_BACKEND != "redis":
        return "not_configured"
    try:
        import redis  # type: ignore

        client = redis.Redis.from_url(Config.REDIS_URL, socket_connect_timeout=2)
        client.ping()
        return "connected"
    except Exception:
        return "disconnected"


def _check_chroma_health() -> str:
    if _chromadb is None or _retrieval is None:
        return "unavailable"
    try:
        vdb_dir = Path(_retrieval.VECTORDB_DIR)
        if not vdb_dir.exists():
            return "missing"
        client = _chromadb.PersistentClient(path=str(vdb_dir))
        collection = client.get_collection("langchain")
        count = collection.count()
        return "connected" if count >= 0 else "empty"
    except Exception:
        return "degraded"



def _rate_limit_or_none(bucket: str, identifier: str, limit: int, window_seconds: int):
    if not Config.RATE_LIMIT_ENABLED:
        return None
    allowed, retry_after = _rate_limiter.check(bucket, identifier, limit, window_seconds)
    if allowed:
        return None
    response = jsonify(
        {
            "error": "Too many requests. Please try again later.",
            "code": "rate_limited",
            "retry_after": retry_after,
        }
    )
    response.status_code = 429
    response.headers["Retry-After"] = str(retry_after)
    return response



def _require_api_key_or_user():
    if g.get("current_user_row") or g.get("api_key_authenticated"):
        return None
    return _json_error("Authentication required.", 401, "auth_required")



def _serialize_thread(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        "last_message_at": row["last_message_at"].isoformat() if row.get("last_message_at") else None,
        "archived_at": row["archived_at"].isoformat() if row.get("archived_at") else None,
        "deleted_at": row["deleted_at"].isoformat() if row.get("deleted_at") else None,
        "last_message_preview": row.get("last_message_preview"),
    }



def _serialize_message(row: Dict[str, Any]) -> Dict[str, Any]:
    meta = json_loads(row.get("message_meta_json"), {})
    return {
        "id": row["id"],
        "thread_id": row["thread_id"],
        "role": row["role"],
        "content": row["content"],
        "response_mode": row.get("response_mode"),
        "sources": json_loads(row.get("sources_json"), []),
        "model_name": row.get("model_name"),
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "log_id": meta.get("log_id"),
    }



def _fetch_thread(conn, user_id: str, thread_id: str) -> Optional[Dict[str, Any]]:
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT ct.*,
                   (
                       SELECT content FROM conversation_messages cm
                       WHERE cm.thread_id = ct.id
                       ORDER BY cm.created_at DESC,
                                CASE cm.role
                                    WHEN 'assistant' THEN 0
                                    WHEN 'user' THEN 1
                                    ELSE 2
                                END,
                                cm.id DESC
                       LIMIT 1
                   ) AS last_message_preview
            FROM conversation_threads ct
            WHERE ct.id = %s AND ct.user_id = %s AND ct.deleted_at IS NULL
            LIMIT 1
            """,
            (thread_id, user_id),
        )
        return cursor.fetchone()
    finally:
        cursor.close()



def _list_threads(conn, user_id: str) -> List[Dict[str, Any]]:
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT ct.*,
                   (
                       SELECT content FROM conversation_messages cm
                       WHERE cm.thread_id = ct.id
                       ORDER BY cm.created_at DESC,
                                CASE cm.role
                                    WHEN 'assistant' THEN 0
                                    WHEN 'user' THEN 1
                                    ELSE 2
                                END,
                                cm.id DESC
                       LIMIT 1
                   ) AS last_message_preview
            FROM conversation_threads ct
            WHERE ct.user_id = %s AND ct.deleted_at IS NULL
            ORDER BY CASE WHEN ct.archived_at IS NULL THEN 0 ELSE 1 END,
                     ct.last_message_at DESC,
                     ct.created_at DESC
            """,
            (user_id,),
        )
        return cursor.fetchall()
    finally:
        cursor.close()



def _create_thread(conn, user_id: str, title: Optional[str]) -> Dict[str, Any]:
    thread_id = str(uuid.uuid4())
    final_title = title.strip() if title and title.strip() else "New conversation"
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO conversation_threads (id, user_id, title, thread_state_json, last_message_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (thread_id, user_id, final_title, json_dumps(create_empty_cache()), utcnow()),
        )
    finally:
        cursor.close()
    return _fetch_thread(conn, user_id, thread_id)



def _update_thread(conn, thread_id: str, *, title: Optional[str] = None, archived: Optional[bool] = None) -> None:
    updates = []
    values: List[Any] = []
    if title is not None:
        updates.append("title = %s")
        values.append(title)
    if archived is not None:
        updates.append("archived_at = %s")
        values.append(utcnow() if archived else None)
    if not updates:
        return
    values.append(thread_id)
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"UPDATE conversation_threads SET {', '.join(updates)} WHERE id = %s",
            tuple(values),
        )
    finally:
        cursor.close()



def _soft_delete_thread(conn, thread_id: str) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE conversation_threads SET deleted_at = %s WHERE id = %s",
            (utcnow(), thread_id),
        )
    finally:
        cursor.close()



def _fetch_messages(conn, thread_id: str, *, limit: int, before: Optional[str]) -> List[Dict[str, Any]]:
    cursor = conn.cursor(dictionary=True)
    try:
        params: List[Any] = [thread_id]
        where_clause = "WHERE thread_id = %s"
        if before:
            cursor.execute(
                "SELECT created_at FROM conversation_messages WHERE id = %s AND thread_id = %s LIMIT 1",
                (before, thread_id),
            )
            pivot = cursor.fetchone()
            if pivot and pivot.get("created_at"):
                where_clause += " AND created_at < %s"
                params.append(pivot["created_at"])
        params.append(limit)
        cursor.execute(
            f"""
            SELECT *
            FROM conversation_messages
            {where_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            tuple(params),
        )
        rows = cursor.fetchall()
        rows.reverse()
        return rows
    finally:
        cursor.close()



def _fetch_recent_history(conn, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    return _fetch_messages(conn, thread_id, limit=limit, before=None)



def _insert_message(
    conn,
    *,
    thread_id: str,
    user_id: str,
    role: str,
    content: str,
    response_mode: Optional[str] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    model_name: Optional[str] = None,
    message_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    message_id = str(uuid.uuid4())
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO conversation_messages (
                id, thread_id, user_id, role, content, response_mode,
                sources_json, model_name, message_meta_json, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                message_id,
                thread_id,
                user_id,
                role,
                content,
                response_mode,
                json_dumps(sources) if sources is not None else None,
                model_name,
                json_dumps(message_meta) if message_meta is not None else None,
                utcnow(),
            ),
        )
    finally:
        cursor.close()
    return _fetch_messages(conn, thread_id, limit=1, before=None)[-1]



def _update_thread_state(conn, thread_id: str, thread_state: Dict[str, Any]) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE conversation_threads SET thread_state_json = %s, last_message_at = %s WHERE id = %s",
            (json_dumps(thread_state), utcnow(), thread_id),
        )
    finally:
        cursor.close()



def _update_message_meta(conn, message_id: str, message_meta: Dict[str, Any]) -> None:
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE conversation_messages SET message_meta_json = %s WHERE id = %s",
            (json_dumps(message_meta), message_id),
        )
    finally:
        cursor.close()



def extract_sources(mode: str, result: Dict[str, Any]) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []

    def _canonicalize_source_label(raw: str) -> tuple[str, str]:
        value = (raw or "").strip()
        lowered = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
        lowered = re.sub(r"\s+", " ", lowered)
        if lowered in {"dorchester reporter", "dot reporter"} or "dotnews" in lowered:
            return ("dorchester reporter", "Dorchester Reporter")
        return (lowered or value.lower() or "unknown", value or "Unknown")

    if mode == "sql":
        sql_query = result.get("sql", "")
        if sql_query:
            match = re.search(r"FROM\s+`?(\w+)`?", sql_query, re.IGNORECASE)
            if match:
                sources.append({"type": "sql", "table": match.group(1)})

    elif mode == "rag":
        metadata = result.get("metadata", [])
        seen = set()
        boston_entries = [m for m in metadata if m.get("doc_type") == "boston_gov_answer"]
        other_entries = [m for m in metadata if m.get("doc_type") != "boston_gov_answer"]
        prioritized = boston_entries + other_entries
        
        for meta in prioritized[:5]:
            source = meta.get("source", "Unknown")
            doc_type = meta.get("doc_type", "unknown")
            # Dedup by source alone — the same publication ingested via
            # multiple paths (RSS + PDF, etc.) shouldn't show up as
            # separate citations to the user.
            dedupe_key, source_label = _canonicalize_source_label(source)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            link = meta.get("link", "")
            base_dir = DOC_TYPE_DIRS.get(doc_type, "Data")
            if link:
                path = link
            elif base_dir.startswith(("http://", "https://")):
                normalized_source = source_label.replace(" ", "-")
                if normalized_source.endswith(".txt"):
                    normalized_source = normalized_source[:-4] + ".html"
                path = urljoin(base_dir, normalized_source)
            else:
                path = str(Path(base_dir) / source_label)
            sources.append({
                "type": "rag",
                "source": source_label,
                "doc_type": doc_type,
                "path": path,
            })

    elif mode == "hybrid":
        sql_part = result.get("sql", {}) if isinstance(result.get("sql"), dict) else result.get("sql")
        rag_part = result.get("rag", {}) if isinstance(result.get("rag"), dict) else {}
        if isinstance(sql_part, dict):
            sql_query = sql_part.get("sql", "")
            if sql_query:
                match = re.search(r"FROM\s+`?(\w+)`?", sql_query, re.IGNORECASE)
                if match:
                    sources.append({"type": "sql", "table": match.group(1)})
        rag_metadata = rag_part.get("metadata", []) if isinstance(rag_part, dict) else []
        seen = set()
        for meta in rag_metadata[:3]:
            source = meta.get("source", "Unknown")
            doc_type = meta.get("doc_type", "unknown")
            dedupe_key, source_label = _canonicalize_source_label(source)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            link = meta.get("link", "")
            base_dir = DOC_TYPE_DIRS.get(doc_type, "Data")
            sources.append({
                "type": "rag",
                "source": source_label,
                "doc_type": doc_type,
                "path": link or str(Path(base_dir) / source_label),
            })

    return sources



def log_interaction(
    *,
    session_id: str,
    client_query: str,
    app_response: str,
    mode: str = "",
    log_id: Optional[int] = None,
    rating: str = "",
    flag_reason: str = "",
    flag_details: str = "",
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    message_id: Optional[str] = None,
) -> Optional[int]:
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        if log_id:
            cursor.execute(
                "SELECT user_id, session_id FROM interaction_log WHERE id = %s",
                (log_id,),
            )
            existing = cursor.fetchone()
            if not existing:
                return None
            owned = False
            if user_id and existing.get("user_id") == user_id:
                owned = True
            elif session_id and existing.get("session_id") == session_id:
                if existing.get("user_id") is None:
                    owned = True
                elif user_id and existing.get("user_id") == user_id:
                    owned = True
            if not owned:
                return "forbidden"

            update_fields = []
            values: List[Any] = []
            if rating:
                update_fields.append("client_response_rating = %s")
                values.append(rating)
            if flag_reason:
                update_fields.append("flagged = TRUE")
                update_fields.append("flag_reason = %s")
                update_fields.append("flag_details = %s")
                update_fields.append("flagged_at = NOW()")
                values.extend([flag_reason, flag_details])
            if app_response:
                update_fields.append("app_response = %s")
                values.append(app_response)
            if update_fields:
                values.append(log_id)
                cursor.execute(
                    f"UPDATE interaction_log SET {', '.join(update_fields)} WHERE id = %s",
                    tuple(values),
                )
            conn.commit()
            return log_id

        cursor.execute(
            """
            INSERT INTO interaction_log (
                session_id, app_version, client_query, app_response,
                data_selected, user_id, thread_id, message_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                session_id,
                Config.API_VERSION,
                client_query,
                app_response,
                mode,
                user_id,
                thread_id,
                message_id,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    except Exception as exc:
        _log_exception("Error logging interaction", exc)
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()



def _conversation_history_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    history: List[Dict[str, str]] = []
    for row in rows:
        if row["role"] not in {"user", "assistant"}:
            continue
        history.append({"role": row["role"], "content": row["content"]})
    return history[-20:]



def _execute_agent_response(
    message: str,
    *,
    conversation_history: List[Dict[str, str]],
    retrieval_cache: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cache = retrieval_cache or create_empty_cache()
    has_history = bool(conversation_history)
    has_cache = bool(cache and cache.get("mode"))

    if is_small_talk(message):
        return build_small_talk_result(message, cache)

    # Semantic response cache: short-circuit near-duplicate first-turn questions.
    semantic_eligible = _semantic_cache_eligible(message, conversation_history)
    if semantic_eligible:
        cached = _semantic_cache.get(message)
        if cached:
            print(f"  ⚡ Semantic cache hit (similarity={cached.get('_similarity')})")
            return {
                "answer": cached["answer"],
                "mode": cached.get("mode", "semantic_cache"),
                "sources": cached.get("sources", []),
                "result": {"answer": cached["answer"]},
                "retrieval_cache": cache,
            }

    if has_history or has_cache:
        history_check = _check_if_needs_new_data(message, conversation_history, cache)
    else:
        history_check = {"needs_new_data": True, "reason": "No history or cache"}

    if not history_check.get("needs_new_data", True) and (has_history or has_cache):
        answer = _answer_from_history(message, conversation_history, cache)
        return {
            "answer": answer,
            "mode": "history",
            "sources": [],
            "result": {"answer": answer},
            "retrieval_cache": cache,
        }

    plan = _route_question(message)
    mode = plan.get("mode", "hybrid")
    if mode == "sql":
        result = _run_sql(message, conversation_history)
        next_cache = build_retrieval_cache(
            mode="sql",
            question=message,
            answer=result.get("answer", ""),
            sql_result=result.get("result"),
            sql_query=result.get("sql"),
        )
    elif mode == "rag":
        result = _run_rag(message, plan, conversation_history)
        next_cache = build_retrieval_cache(
            mode="rag",
            question=message,
            answer=result.get("answer", ""),
            rag_chunks=result.get("chunks"),
            rag_metadata=result.get("metadata"),
        )
    else:
        result = _run_hybrid(message, plan, conversation_history)
        sql_part = result.get("sql", {}) if isinstance(result.get("sql"), dict) else {}
        rag_part = result.get("rag", {}) if isinstance(result.get("rag"), dict) else {}
        next_cache = build_retrieval_cache(
            mode="hybrid",
            question=message,
            answer=result.get("answer", ""),
            sql_result=sql_part.get("result"),
            sql_query=sql_part.get("sql"),
            rag_chunks=rag_part.get("chunks"),
            rag_metadata=rag_part.get("metadata"),
        )

    answer = result.get("answer", "I couldn't find an answer to your question.")
    sources = extract_sources(mode, result)

    # Store fresh first-turn answers for future near-duplicate questions.
    if semantic_eligible and answer:
        _semantic_cache.put(message, {"answer": answer, "mode": mode, "sources": sources})

    return {
        "answer": answer,
        "mode": mode,
        "sources": sources,
        "result": result,
        "retrieval_cache": next_cache,
    }


# =============================================================================
# Request lifecycle
# =============================================================================
@app.before_request
def before_request_handler():
    if request.method == "OPTIONS":
        return ("", 204)

    g.api_key_authenticated = False
    g.current_user_row = None
    g.current_user = None
    g.linked_providers = []
    g.session_row = None
    g.session_id = None
    g.bearer_session = False
    g.is_guest = False
    g.clear_auth_cookies = False
    g.csrf_cookie_written = False

    provided_api_key = request.headers.get("RethinkAI-API-Key", "")
    if provided_api_key and provided_api_key in Config.RETHINKAI_API_KEYS:
        g.api_key_authenticated = True
        if "session_id" not in session:
            session.permanent = True
            session["session_id"] = str(uuid.uuid4())
        g.session_id = session.get("session_id")

    session_token = request.cookies.get(Config.AUTH_SESSION_COOKIE_NAME, "")
    via_bearer = False
    if not session_token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            session_token = auth_header[7:].strip()
            via_bearer = bool(session_token)

    if not session_token:
        return None

    conn = None
    try:
        conn = get_db_connection()
        activated = _activate_web_session(conn, session_token, via_bearer=via_bearer)
        if not activated:
            if not via_bearer:
                g.clear_auth_cookies = True
    except Exception as exc:
        _log_exception("Warning: session lookup failed", exc)
    finally:
        if conn:
            conn.close()


@app.after_request
def after_request_handler(response):
    if getattr(g, "clear_auth_cookies", False):
        _clear_auth_cookies(response)
    if not request.cookies.get(Config.AUTH_CSRF_COOKIE_NAME) and not getattr(g, "csrf_cookie_written", False):
        response.set_cookie(
            Config.AUTH_CSRF_COOKIE_NAME,
            generate_token(24),
            **_cookie_kwargs(httponly=False),
        )
        g.csrf_cookie_written = True
    return response


# =============================================================================
# Auth endpoints
# =============================================================================
@app.route("/auth/me", methods=["GET"])
def auth_me():
    return jsonify(
        {
            "authenticated": bool(g.get("current_user")),
            "user": g.get("current_user"),
            "is_guest": bool(g.get("is_guest")),
        }
    )


@app.route("/auth/guest", methods=["POST"])
def auth_guest():
    """Create an ephemeral guest user + tab-scoped Bearer session (no cookies)."""
    ip_address = get_client_ip(request.headers, request.remote_addr)
    limited = _rate_limit_or_none(
        "guest_create",
        ip_address,
        Config.RATE_LIMIT_GUEST_CREATE_PER_IP,
        Config.RATE_LIMIT_GUEST_CREATE_WINDOW_SECONDS,
    )
    if limited:
        return limited

    conn = None
    try:
        conn = get_db_connection()
        user = _create_guest_user(conn)
        session_info = _create_web_session(conn, user["id"], is_guest=True)
        _record_auth_event(conn, "guest_bootstrap", success=True, user_id=user["id"])
        conn.commit()

        providers = _provider_names(conn, user["id"])
        fresh_user = _fetch_user_by_id(conn, user["id"])
        return jsonify(
            {
                "user": serialize_user(fresh_user, providers),
                "session_token": session_info["session_token"],
                "csrf_token": session_info["csrf_token"],
                "expires_at": session_info["expires_at"].isoformat(),
            }
        ), 201
    except Exception as exc:
        if conn:
            conn.rollback()
        _log_exception("Error creating guest session", exc)
        return _json_error("Failed to create guest session.", 500, "guest_bootstrap_failed")
    finally:
        if conn:
            conn.close()


@app.route("/auth/signup", methods=["POST"])
def auth_signup():
    csrf_error = _enforce_csrf()
    if csrf_error:
        return csrf_error

    ip_address = get_client_ip(request.headers, request.remote_addr)
    limited = _rate_limit_or_none(
        "auth_signup",
        ip_address,
        Config.RATE_LIMIT_AUTH_PER_IP,
        Config.RATE_LIMIT_AUTH_WINDOW_SECONDS,
    )
    if limited:
        return limited

    payload = request.get_json() or {}
    email = normalize_email(payload.get("email", ""))
    username = normalize_username(payload.get("username", ""))
    password = payload.get("password", "")

    username_error = validate_username(username)
    if username_error:
        return _json_error(username_error, 400, "invalid_username")
    password_error = validate_password(password, username, email)
    if password_error:
        return _json_error(password_error, 400, "invalid_password")
    if not email or "@" not in email:
        return _json_error("A valid email address is required.", 400, "invalid_email")

    conn = None
    try:
        conn = get_db_connection()
        existing_user = _fetch_user_by_email(conn, email)
        if existing_user:
            return _json_error("An account with that email already exists.", 409, "email_exists")
        existing_username = _fetch_user_by_username(conn, username)
        if existing_username:
            return _json_error("That username is already taken.", 409, "username_exists")

        user = _create_user(conn, email, username, profile_complete=True)
        _promote_user_to_admin_if_configured(
            conn,
            user_id=user["id"],
            email=email,
            current_role=user.get("role"),
            audit_event="admin_role_bootstrap",
        )
        _create_auth_identity(
            conn,
            user_id=user["id"],
            provider="password",
            password_hash_value=hash_password(password),
        )
        _update_user_login_stamp(conn, user["id"])
        _update_identity_last_used(conn, user["id"], "password")
        session_info = _create_web_session(conn, user["id"])
        _record_auth_event(conn, "signup_password", success=True, user_id=user["id"], details={"email": email})
        conn.commit()

        providers = _provider_names(conn, user["id"])
        fresh_user = _fetch_user_by_id(conn, user["id"])
        response = jsonify({"user": serialize_user(fresh_user, providers)})
        _set_auth_cookies(response, session_info["session_token"], session_info["csrf_token"], session_info["expires_at"])
        return response, 201
    except RuntimeError as exc:
        if conn:
            conn.rollback()
        return _json_error(str(exc), 500, "dependency_missing")
    except mysql.connector.IntegrityError:
        if conn:
            conn.rollback()
        return _json_error("An account with that email or username already exists.", 409, "signup_conflict")
    except Exception as exc:
        if conn:
            conn.rollback()
        _log_exception("Error in signup", exc)
        return _json_error("Failed to create account.", 500, "signup_failed")
    finally:
        if conn:
            conn.close()


@app.route("/auth/login", methods=["POST"])
def auth_login():
    csrf_error = _enforce_csrf()
    if csrf_error:
        return csrf_error

    payload = request.get_json() or {}
    email = normalize_email(payload.get("email", ""))
    password = payload.get("password", "")
    ip_address = get_client_ip(request.headers, request.remote_addr)

    if not email or not password:
        return _json_error("Email and password are required.", 400, "missing_credentials")

    limited = _rate_limit_or_none(
        "auth_login",
        ip_address,
        Config.RATE_LIMIT_AUTH_PER_IP,
        Config.RATE_LIMIT_AUTH_WINDOW_SECONDS,
    )
    if limited:
        return limited

    conn = None
    try:
        conn = get_db_connection()
        login_attempt = _fetch_login_attempt(conn, email, ip_address)
        if _is_locked(login_attempt):
            return _json_error("Too many failed login attempts. Try again later.", 429, "login_locked")

        login_row = _fetch_password_login_row(conn, email)
        if not login_row or login_row.get("status") != "active":
            locked_until = _record_failed_login(conn, email, ip_address)
            _record_auth_event(conn, "login_password", success=False, details={"email": email, "locked_until": locked_until.isoformat() if locked_until else None})
            conn.commit()
            return _json_error("Invalid email or password.", 401, "invalid_credentials")

        password_ok, replacement_hash = verify_password(login_row["password_hash"], password)
        if not password_ok:
            locked_until = _record_failed_login(conn, email, ip_address)
            _record_auth_event(conn, "login_password", success=False, user_id=login_row["id"], details={"email": email, "locked_until": locked_until.isoformat() if locked_until else None})
            conn.commit()
            return _json_error("Invalid email or password.", 401, "invalid_credentials")

        if replacement_hash:
            _update_password_hash(conn, login_row["identity_id"], replacement_hash)

        guest_session_token = (payload.get("guest_session_token") or "").strip()
        if guest_session_token:
            guest_session = _current_session_row(conn, guest_session_token)
            if guest_session and guest_session.get("is_guest"):
                _revoke_session(conn, guest_session["session_id"])

        _promote_user_to_admin_if_configured(
            conn,
            user_id=login_row["id"],
            email=email,
            current_role=login_row.get("role"),
            audit_event="admin_role_bootstrap",
        )
        _clear_login_attempts(conn, email, ip_address)
        _update_user_login_stamp(conn, login_row["id"])
        _update_identity_last_used(conn, login_row["id"], "password")
        session_info = _create_web_session(conn, login_row["id"])
        _record_auth_event(conn, "login_password", success=True, user_id=login_row["id"], details={"email": email})
        conn.commit()

        providers = _provider_names(conn, login_row["id"])
        user_row = _fetch_user_by_id(conn, login_row["id"])
        response = jsonify({"user": serialize_user(user_row, providers)})
        _set_auth_cookies(response, session_info["session_token"], session_info["csrf_token"], session_info["expires_at"])
        return response
    except RuntimeError as exc:
        if conn:
            conn.rollback()
        return _json_error(str(exc), 500, "dependency_missing")
    except Exception as exc:
        if conn:
            conn.rollback()
        _log_exception("Error in login", exc)
        return _json_error("Failed to log in.", 500, "login_failed")
    finally:
        if conn:
            conn.close()


@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    result = _require_user(allow_incomplete=True)
    if result:
        return result
    csrf_error = _enforce_csrf()
    if csrf_error:
        return csrf_error

    conn = None
    try:
        conn = get_db_connection()
        _revoke_session(conn, g.session_row["session_id"])
        _record_auth_event(conn, "logout", success=True, user_id=g.current_user_row["id"])
        conn.commit()
    except Exception as exc:
        if conn:
            conn.rollback()
        _log_exception("Error in logout", exc)
        return _json_error("Failed to log out.", 500, "logout_failed")
    finally:
        if conn:
            conn.close()

    response = jsonify({"ok": True})
    _clear_auth_cookies(response)
    return response


@app.route("/auth/complete-profile", methods=["POST"])
def auth_complete_profile():
    result = _require_user(allow_incomplete=True)
    if result:
        return result
    csrf_error = _enforce_csrf()
    if csrf_error:
        return csrf_error

    payload = request.get_json() or {}
    username = normalize_username(payload.get("username", ""))
    username_error = validate_username(username)
    if username_error:
        return _json_error(username_error, 400, "invalid_username")

    conn = None
    try:
        conn = get_db_connection()
        existing = _fetch_user_by_username(conn, username)
        if existing and existing["id"] != g.current_user_row["id"]:
            return _json_error("That username is already taken.", 409, "username_exists")

        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE users SET username = %s, profile_complete = TRUE WHERE id = %s",
                (username, g.current_user_row["id"]),
            )
        finally:
            cursor.close()
        _record_auth_event(conn, "complete_profile", success=True, user_id=g.current_user_row["id"], details={"username": username})
        conn.commit()

        user_row = _fetch_user_by_id(conn, g.current_user_row["id"])
        providers = _provider_names(conn, g.current_user_row["id"])
        return jsonify({"user": serialize_user(user_row, providers)})
    except Exception as exc:
        if conn:
            conn.rollback()
        _log_exception("Error completing profile", exc)
        return _json_error("Failed to complete profile.", 500, "profile_update_failed")
    finally:
        if conn:
            conn.close()


# =============================================================================
# Conversation endpoints
# =============================================================================
@app.route("/conversations", methods=["GET"])
def list_conversations():
    result = _require_user()
    if result:
        return result

    conn = None
    try:
        conn = get_db_connection()
        rows = _list_threads(conn, g.current_user_row["id"])
        return jsonify({"threads": [_serialize_thread(row) for row in rows]})
    except Exception as exc:
        _log_exception("Error listing conversations", exc)
        return _json_error("Failed to load conversations.", 500, "conversation_list_failed")
    finally:
        if conn:
            conn.close()


@app.route("/conversations", methods=["POST"])
def create_conversation():
    result = _require_user()
    if result:
        return result
    csrf_error = _enforce_csrf()
    if csrf_error:
        return csrf_error

    payload = request.get_json() or {}
    title = payload.get("title")

    conn = None
    try:
        conn = get_db_connection()
        thread = _create_thread(conn, g.current_user_row["id"], title)
        conn.commit()
        return jsonify({"thread": _serialize_thread(thread)}), 201
    except Exception as exc:
        if conn:
            conn.rollback()
        _log_exception("Error creating conversation", exc)
        return _json_error("Failed to create conversation.", 500, "conversation_create_failed")
    finally:
        if conn:
            conn.close()


@app.route("/conversations/<thread_id>", methods=["PATCH"])
def update_conversation(thread_id: str):
    result = _require_user()
    if result:
        return result
    csrf_error = _enforce_csrf()
    if csrf_error:
        return csrf_error

    payload = request.get_json() or {}
    title = payload.get("title")
    archived = payload.get("archived") if "archived" in payload else None

    conn = None
    try:
        conn = get_db_connection()
        thread = _fetch_thread(conn, g.current_user_row["id"], thread_id)
        if not thread:
            return _json_error("Conversation not found.", 404, "conversation_not_found")
        final_title = None
        if title is not None:
            final_title = title.strip()
            if not final_title:
                return _json_error("Title cannot be empty.", 400, "invalid_title")
        _update_thread(conn, thread_id, title=final_title, archived=bool(archived) if archived is not None else None)
        conn.commit()
        refreshed = _fetch_thread(conn, g.current_user_row["id"], thread_id)
        return jsonify({"thread": _serialize_thread(refreshed)})
    except Exception as exc:
        if conn:
            conn.rollback()
        _log_exception("Error updating conversation", exc)
        return _json_error("Failed to update conversation.", 500, "conversation_update_failed")
    finally:
        if conn:
            conn.close()


@app.route("/conversations/<thread_id>", methods=["DELETE"])
def delete_conversation(thread_id: str):
    result = _require_user()
    if result:
        return result
    csrf_error = _enforce_csrf()
    if csrf_error:
        return csrf_error

    conn = None
    try:
        conn = get_db_connection()
        thread = _fetch_thread(conn, g.current_user_row["id"], thread_id)
        if not thread:
            return _json_error("Conversation not found.", 404, "conversation_not_found")
        _soft_delete_thread(conn, thread_id)
        conn.commit()
        return jsonify({"ok": True})
    except Exception as exc:
        if conn:
            conn.rollback()
        _log_exception("Error deleting conversation", exc)
        return _json_error("Failed to delete conversation.", 500, "conversation_delete_failed")
    finally:
        if conn:
            conn.close()


@app.route("/conversations/<thread_id>/messages", methods=["GET"])
def get_conversation_messages(thread_id: str):
    result = _require_user()
    if result:
        return result

    limit = max(1, min(request.args.get("limit", 50, type=int), 100))
    before = request.args.get("before")

    conn = None
    try:
        conn = get_db_connection()
        thread = _fetch_thread(conn, g.current_user_row["id"], thread_id)
        if not thread:
            return _json_error("Conversation not found.", 404, "conversation_not_found")
        messages = _fetch_messages(conn, thread_id, limit=limit, before=before)
        return jsonify({
            "thread": _serialize_thread(thread),
            "messages": [_serialize_message(row) for row in messages],
        })
    except Exception as exc:
        _log_exception("Error fetching messages", exc)
        return _json_error("Failed to load messages.", 500, "conversation_messages_failed")
    finally:
        if conn:
            conn.close()


@app.route("/conversations/<thread_id>/messages", methods=["POST"])
def post_conversation_message(thread_id: str):
    result = _require_user()
    if result:
        return result
    limited = _rate_limit_or_none(
        "chat",
        g.session_id or g.current_user_row["id"],
        Config.RATE_LIMIT_CHAT_PER_SESSION,
        Config.RATE_LIMIT_CHAT_WINDOW_SECONDS,
    )
    if limited:
        return limited
    csrf_error = _enforce_csrf()
    if csrf_error:
        return csrf_error

    payload = request.get_json() or {}
    message = (payload.get("message") or "").strip()
    message_error = _validate_message(message)
    if message_error:
        return message_error

    conn = None
    try:
        conn = get_db_connection()
        thread = _fetch_thread(conn, g.current_user_row["id"], thread_id)
        if not thread:
            return _json_error("Conversation not found.", 404, "conversation_not_found")

        history_rows = _fetch_recent_history(conn, thread_id, limit=20)
        conversation_history = _conversation_history_from_rows(history_rows)
        thread_state = json_loads(thread.get("thread_state_json"), create_empty_cache())
        conn.close()
        conn = None

        # Run the expensive agent path outside any open DB transaction so
        # long LLM/sql execution does not hold row locks on the conversation.
        agent_result = _execute_agent_response(
            message,
            conversation_history=conversation_history,
            retrieval_cache=thread_state,
        )
        answer = agent_result["answer"]
        mode = agent_result["mode"]
        sources = agent_result["sources"]
        next_cache = agent_result["retrieval_cache"]

        conn = get_db_connection()
        thread = _fetch_thread(conn, g.current_user_row["id"], thread_id)
        if not thread:
            return _json_error("Conversation not found.", 404, "conversation_not_found")

        user_message_row = _insert_message(
            conn,
            thread_id=thread_id,
            user_id=g.current_user_row["id"],
            role="user",
            content=message,
        )

        assistant_message_row = _insert_message(
            conn,
            thread_id=thread_id,
            user_id=g.current_user_row["id"],
            role="assistant",
            content=answer,
            response_mode=mode,
            sources=sources,
            model_name=os.getenv("GEMINI_MODEL", ""),
        )

        if thread["title"] == "New conversation" and len(history_rows) == 0:
            _update_thread(conn, thread_id, title=summarize_thread_title(message))
        _update_thread_state(conn, thread_id, next_cache)

        log_id = log_interaction(
            session_id=g.session_row["session_id"],
            client_query=message,
            app_response=answer,
            mode=mode,
            user_id=g.current_user_row["id"],
            thread_id=thread_id,
            message_id=assistant_message_row["id"],
        )
        _update_message_meta(conn, assistant_message_row["id"], {"log_id": log_id})
        conn.commit()

        refreshed_thread = _fetch_thread(conn, g.current_user_row["id"], thread_id)
        assistant_message_row["message_meta_json"] = json_dumps({"log_id": log_id})

        return jsonify(
            {
                "thread": _serialize_thread(refreshed_thread),
                "user_message": _serialize_message(user_message_row),
                "assistant_message": _serialize_message(assistant_message_row),
            }
        ), 201
    except Exception as exc:
        if conn:
            conn.rollback()
        _log_exception("Error posting conversation message", exc)
        return _json_error("Failed to send message.", 500, "conversation_message_failed")
    finally:
        if conn:
            conn.close()


@app.route("/conversations/<thread_id>/messages/stream", methods=["POST"])
def post_conversation_message_stream(thread_id: str):
    """Streamed variant of POST /conversations/<id>/messages (Server-Sent Events).

    Emits `data: {"type":"delta","text":...}` events as the answer is produced,
    then a final `data: {"type":"final", ...}` with the persisted thread/messages
    (or `{"type":"error", ...}`). The non-streaming endpoint remains the default.
    """
    if not Config.STREAMING_ENABLED:
        return _json_error("Streaming is disabled.", 404, "streaming_disabled")

    result = _require_user()
    if result:
        return result
    limited = _rate_limit_or_none(
        "chat",
        g.session_id or g.current_user_row["id"],
        Config.RATE_LIMIT_CHAT_PER_SESSION,
        Config.RATE_LIMIT_CHAT_WINDOW_SECONDS,
    )
    if limited:
        return limited
    csrf_error = _enforce_csrf()
    if csrf_error:
        return csrf_error

    payload = request.get_json() or {}
    message = (payload.get("message") or "").strip()
    message_error = _validate_message(message)
    if message_error:
        return message_error

    # Load context up front, then release the DB before the long LLM work.
    conn = None
    try:
        conn = get_db_connection()
        thread = _fetch_thread(conn, g.current_user_row["id"], thread_id)
        if not thread:
            return _json_error("Conversation not found.", 404, "conversation_not_found")
        history_rows = _fetch_recent_history(conn, thread_id, limit=20)
        conversation_history = _conversation_history_from_rows(history_rows)
        thread_state = json_loads(thread.get("thread_state_json"), create_empty_cache())
    except Exception as exc:
        _log_exception("Error preparing streamed message", exc)
        return _json_error("Failed to send message.", 500, "conversation_message_failed")
    finally:
        if conn:
            conn.close()

    # Capture request-scoped values so the generator does not depend on the
    # request context once streaming begins.
    user_id = g.current_user_row["id"]
    session_id = g.session_row["session_id"]
    is_first_message = len(history_rows) == 0
    existing_title = thread["title"]

    def _sse(obj: Dict[str, Any]) -> str:
        return f"data: {json_dumps(obj)}\n\n"

    def _generate():
        full_answer = ""
        mode = "hybrid"
        result_payload: Dict[str, Any] = {}
        next_cache = thread_state
        semantic_eligible = _semantic_cache_eligible(message, conversation_history)

        try:
            cached = _semantic_cache.get(message) if semantic_eligible else None
            if cached:
                full_answer = cached.get("answer", "")
                mode = cached.get("mode", "semantic_cache")
                result_payload = {"answer": full_answer}
                for i, word in enumerate(full_answer.split()):
                    yield _sse({"type": "delta", "text": word if i == 0 else f" {word}"})
            else:
                for kind, data in stream_agent_response(message, conversation_history, thread_state):
                    if kind == "delta":
                        yield _sse({"type": "delta", "text": data})
                    elif kind == "final":
                        full_answer = data.get("answer", "")
                        mode = data.get("mode", "hybrid")
                        result_payload = data.get("result", {})
                        next_cache = data.get("retrieval_cache", thread_state)
        except Exception as exc:  # noqa: BLE001
            _log_exception("Error during streamed generation", exc)
            yield _sse({"type": "error", "error": "Generation failed."})
            return

        if full_answer:
            enriched = apply_post_stream_fallback(message, full_answer, mode)
            if enriched != full_answer:
                yield _sse({"type": "correction", "text": enriched})
                full_answer = enriched
            if semantic_eligible and mode != "semantic_cache":
                sources_for_cache = extract_sources(mode, result_payload)
                _semantic_cache.put(
                    message,
                    {"answer": full_answer, "mode": mode, "sources": sources_for_cache},
                )

        conn2 = None
        try:
            sources = extract_sources(mode, result_payload)
            conn2 = get_db_connection()
            thread2 = _fetch_thread(conn2, user_id, thread_id)
            if not thread2:
                yield _sse({"type": "error", "error": "Conversation not found."})
                return
            user_message_row = _insert_message(
                conn2, thread_id=thread_id, user_id=user_id, role="user", content=message,
            )
            assistant_message_row = _insert_message(
                conn2, thread_id=thread_id, user_id=user_id, role="assistant",
                content=full_answer, response_mode=mode, sources=sources,
                model_name=os.getenv("GEMINI_MODEL", ""),
            )
            if existing_title == "New conversation" and is_first_message:
                _update_thread(conn2, thread_id, title=summarize_thread_title(message))
            _update_thread_state(conn2, thread_id, next_cache)
            log_id = log_interaction(
                session_id=session_id, client_query=message, app_response=full_answer,
                mode=mode, user_id=user_id, thread_id=thread_id,
                message_id=assistant_message_row["id"],
            )
            _update_message_meta(conn2, assistant_message_row["id"], {"log_id": log_id})
            conn2.commit()
            refreshed_thread = _fetch_thread(conn2, user_id, thread_id)
            assistant_message_row["message_meta_json"] = json_dumps({"log_id": log_id})
            yield _sse({
                "type": "final",
                "answer": full_answer,
                "mode": mode,
                "sources": sources,
                "thread": _serialize_thread(refreshed_thread),
                "user_message": _serialize_message(user_message_row),
                "assistant_message": _serialize_message(assistant_message_row),
            })
        except Exception as exc:  # noqa: BLE001
            if conn2:
                try:
                    conn2.rollback()
                except Exception:
                    pass
            _log_exception("Error persisting streamed message", exc)
            yield _sse({"type": "error", "error": "Failed to save message."})
        finally:
            if conn2:
                conn2.close()

    return Response(
        stream_with_context(_generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# Legacy compatibility endpoints
# =============================================================================
@app.route("/chat", methods=["POST"])
def chat():
    result = _require_api_key_or_user()
    if result:
        return result

    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    conversation_history = data.get("conversation_history", [])
    message_error = _validate_message(message)
    if message_error:
        return message_error

    session_id = g.get("session_id") or str(uuid.uuid4())
    if g.get("api_key_authenticated") and "session_id" not in session:
        session.permanent = True
        session["session_id"] = session_id
    elif g.get("api_key_authenticated"):
        session_id = session.get("session_id")

    retrieval_cache = _session_cache.get(session_id)

    try:
        agent_result = _execute_agent_response(
            message,
            conversation_history=conversation_history,
            retrieval_cache=retrieval_cache,
        )
        _session_cache.set(session_id, agent_result["retrieval_cache"])
        log_id = log_interaction(
            session_id=session_id,
            client_query=message,
            app_response=agent_result["answer"],
            mode=agent_result["mode"],
            user_id=g.current_user_row["id"] if g.get("current_user_row") else None,
        )
        return jsonify(
            {
                "session_id": session_id,
                "response": agent_result["answer"],
                "sources": agent_result["sources"],
                "mode": agent_result["mode"],
                "log_id": log_id,
            }
        )
    except Exception as exc:
        _log_exception("Error in /chat", exc)
        return _json_error("Internal server error.", 500, "chat_failed")


@app.route("/log", methods=["POST", "PUT"])
def log_endpoint():
    result = _require_api_key_or_user()
    if result:
        return result
    if request.method in {"POST", "PUT"} and g.get("current_user_row"):
        csrf_error = _enforce_csrf()
        if csrf_error:
            return csrf_error

    data = request.get_json() or {}
    session_id = g.get("session_id") or (g.session_row["session_id"] if g.get("session_row") else str(uuid.uuid4()))

    if request.method == "POST":
        client_query = data.get("client_query", "")
        app_response = data.get("app_response", "")
        mode = data.get("mode", "")
        if not client_query:
            return _json_error("client_query is required", 400, "missing_client_query")
        log_id = log_interaction(
            session_id=session_id,
            client_query=client_query,
            app_response=app_response,
            mode=mode,
            user_id=g.current_user_row["id"] if g.get("current_user_row") else None,
            thread_id=data.get("thread_id"),
            message_id=data.get("message_id"),
        )
        if not log_id:
            return _json_error("Failed to create log entry", 500, "log_create_failed")
        return jsonify({"log_id": log_id, "message": "Log entry created"}), 201

    log_id = data.get("log_id")
    if not log_id:
        return _json_error("log_id is required", 400, "missing_log_id")
    owner_user_id = g.current_user_row["id"] if g.get("current_user_row") else None
    conn = None
    try:
        conn = get_db_connection()
        if not _verify_log_ownership(
            conn,
            int(log_id),
            user_id=owner_user_id,
            session_id=session_id,
        ):
            return _json_error("Not allowed to update this log entry.", 403, "log_forbidden")
    finally:
        if conn:
            conn.close()

    updated_id = log_interaction(
        session_id=session_id,
        client_query="",
        app_response="",
        log_id=log_id,
        rating=data.get("client_response_rating", ""),
        flag_reason=data.get("flag_reason", ""),
        flag_details=data.get("flag_details", ""),
        user_id=owner_user_id,
    )
    if updated_id == "forbidden":
        return _json_error("Not allowed to update this log entry.", 403, "log_forbidden")
    if not updated_id:
        return _json_error("Failed to update log entry", 500, "log_update_failed")
    return jsonify({"log_id": updated_id, "message": "Log entry updated"})


@app.route("/events", methods=["GET"])
def events():
    result = _require_api_key_or_user()
    if result:
        return result

    limit = max(1, min(request.args.get("limit", 10, type=int), 100))
    days_ahead = max(1, min(request.args.get("days_ahead", 7, type=int), 365))

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, event_name, event_date, start_date, end_date,
                   start_time, end_time, raw_text, source_pdf
            FROM weekly_events
            WHERE start_date >= CURDATE()
              AND start_date <= DATE_ADD(CURDATE(), INTERVAL %s DAY)
            ORDER BY start_date ASC, start_time ASC
            LIMIT %s
            """,
            (days_ahead, limit),
        )
        rows = cursor.fetchall()
        events_list = []
        for row in rows:
            events_list.append(
                {
                    "id": row["id"],
                    "event_name": row["event_name"],
                    "event_date": row["event_date"],
                    "start_date": str(row["start_date"]) if row["start_date"] else None,
                    "end_date": str(row["end_date"]) if row["end_date"] else None,
                    "start_time": str(row["start_time"]) if row["start_time"] else None,
                    "end_time": str(row["end_time"]) if row["end_time"] else None,
                    "description": row["raw_text"],
                    "source": row["source_pdf"],
                }
            )
        return jsonify({"events": events_list, "total": len(events_list)})
    except mysql.connector.Error as exc:
        if getattr(exc, "errno", None) == 1146:
            return jsonify({"events": [], "total": 0})
        _log_exception("Error in /events", exc)
        return _json_error("Failed to fetch events.", 500, "events_failed")
    except Exception as exc:
        _log_exception("Error in /events", exc)
        return _json_error("Failed to fetch events.", 500, "events_failed")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route("/health", methods=["GET"])
def health():
    status = {
        "status": "ok",
        "version": Config.API_VERSION,
        "cache_backend": Config.CACHE_BACKEND,
    }
    try:
        conn = get_db_connection()
        conn.close()
        status["database"] = "connected"
    except Exception:
        status["database"] = "disconnected"
        status["status"] = "degraded"

    status["redis"] = _check_redis_health()
    if status["redis"] == "disconnected" and Config.CACHE_BACKEND == "redis":
        status["status"] = "degraded"

    status["chroma"] = _check_chroma_health()
    if status["chroma"] in {"missing", "unavailable", "degraded"}:
        status["status"] = "degraded"

    return jsonify(status)


# =============================================================================
# Admin endpoints
# =============================================================================
@app.route("/admin/stats", methods=["GET"])
def admin_stats():
    result = _require_admin()
    if result:
        return result

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) AS cnt FROM interaction_log")
        total = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) AS cnt FROM interaction_log WHERE flagged = TRUE")
        total_flagged = cursor.fetchone()["cnt"]

        cursor.execute(
            """
            SELECT COUNT(*) AS cnt FROM interaction_log
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            """
        )
        this_week = cursor.fetchone()["cnt"]

        cursor.execute(
            """
            SELECT COUNT(*) AS cnt FROM interaction_log
            WHERE app_response LIKE %s
               OR app_response LIKE %s
               OR app_response LIKE %s
            """,
            ("%No results found%", "%couldn't find%", "%no data%"),
        )
        no_results = cursor.fetchone()["cnt"]

        cursor.execute(
            """
            SELECT data_selected AS mode, COUNT(*) AS cnt
            FROM interaction_log
            WHERE data_selected IS NOT NULL AND data_selected != ''
            GROUP BY data_selected
            ORDER BY cnt DESC
            """
        )
        mode_breakdown = {row["mode"]: row["cnt"] for row in cursor.fetchall()}

        cursor.execute(
            """
            SELECT flag_reason, COUNT(*) AS cnt
            FROM interaction_log
            WHERE flagged = TRUE AND flag_reason IS NOT NULL
            GROUP BY flag_reason
            ORDER BY cnt DESC
            """
        )
        flag_reasons = {row["flag_reason"]: row["cnt"] for row in cursor.fetchall()}

        return jsonify(
            {
                "total_interactions": total,
                "total_flagged": total_flagged,
                "interactions_this_week": this_week,
                "no_result_count": no_results,
                "mode_breakdown": mode_breakdown,
                "flag_reasons": flag_reasons,
            }
        )
    except Exception as exc:
        return _json_error(str(exc), 500, "admin_stats_failed")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route("/admin/flags", methods=["GET"])
def admin_flags():
    result = _require_admin()
    if result:
        return result

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, session_id, user_id, thread_id, client_query, app_response,
                   data_selected, flag_reason, flag_details, flagged_at, created_at, moderator_comment,
                   resolved, resolved_at
            FROM interaction_log
            WHERE flagged = TRUE
            ORDER BY flagged_at DESC
            LIMIT 200
            """
        )
        rows = cursor.fetchall()
        for row in rows:
            for key in ("flagged_at", "created_at", "resolved_at"):
                if row.get(key) and hasattr(row[key], "isoformat"):
                    row[key] = row[key].isoformat()
        return jsonify({"flags": rows})
    except Exception as exc:
        return _json_error(str(exc), 500, "admin_flags_failed")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route("/admin/interactions", methods=["GET"])
def admin_interactions():
    result = _require_admin()
    if result:
        return result

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, session_id, user_id, thread_id, client_query, app_response,
                   data_selected, flagged, created_at
            FROM interaction_log
            ORDER BY created_at DESC
            LIMIT 50
            """
        )
        rows = cursor.fetchall()
        for row in rows:
            if row.get("created_at") and hasattr(row["created_at"], "isoformat"):
                row["created_at"] = row["created_at"].isoformat()
            if row.get("app_response") and len(row["app_response"]) > 200:
                row["app_response"] = row["app_response"][:200] + "..."
        return jsonify({"interactions": rows})
    except Exception as exc:
        return _json_error(str(exc), 500, "admin_interactions_failed")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route("/admin/no-results", methods=["GET"])
def admin_no_results():
    result = _require_admin()
    if result:
        return result

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, user_id, thread_id, client_query, data_selected, created_at
            FROM interaction_log
            WHERE app_response LIKE %s
               OR app_response LIKE %s
               OR app_response LIKE %s
               OR app_response LIKE %s
            ORDER BY created_at DESC
            LIMIT 100
            """,
            ("%No results found%", "%couldn't find%", "%no data%", "%I couldn't find%"),
        )
        rows = cursor.fetchall()
        for row in rows:
            if row.get("created_at") and hasattr(row["created_at"], "isoformat"):
                row["created_at"] = row["created_at"].isoformat()
        return jsonify({"no_results": rows})
    except Exception as exc:
        return _json_error(str(exc), 500, "admin_no_results_failed")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.route("/admin/knowledge", methods=["GET"])
def admin_get_knowledge():
    result = _require_admin()
    if result:
        return result
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, content, category, expires_at, added_by, active, created_at
            FROM admin_knowledge
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        for row in rows:
            for key in ("expires_at", "created_at"):
                if row.get(key) and hasattr(row[key], "isoformat"):
                    row[key] = row[key].isoformat()
        return jsonify({"knowledge": rows})
    except Exception as e:
        return _json_error(str(e), 500, "admin_knowledge_failed")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


@app.route("/admin/knowledge", methods=["POST"])
def admin_add_knowledge():
    result = _require_admin_mutation()
    if result:
        return result
    data = request.get_json() or {}
    content = data.get("content", "").strip()
    if not content:
        return _json_error("content is required", 400, "missing_content")
    if len(content) > Config.MAX_COMMUNITY_NOTE_LENGTH:
        return _json_error(
            f"Content is too long (max {Config.MAX_COMMUNITY_NOTE_LENGTH} characters).",
            400,
            "content_too_long",
        )
    expires_at = data.get("expires_at") or (
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
    ).strftime("%Y-%m-%d")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            INSERT INTO admin_knowledge (content, category, expires_at, source_flag_id, added_by)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            content,
            data.get("category", "general"),
            expires_at,
            data.get("source_flag_id"),
            g.current_user_row["username"] if g.get("current_user_row") else "admin",
        ))
        conn.commit()
        _trigger_ingest()
        return jsonify({"id": cursor.lastrowid, "message": "Knowledge entry added"}), 201
    except Exception as e:
        return _json_error(str(e), 500, "admin_knowledge_add_failed")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


@app.route("/admin/knowledge/<int:entry_id>", methods=["PUT"])
def admin_edit_knowledge(entry_id):
    result = _require_admin_mutation()
    if result:
        return result
    data = request.get_json() or {}
    content = data.get("content", "").strip()
    if not content:
        return _json_error("content is required", 400, "admin_knowledge_edit_invalid")
    if len(content) > Config.MAX_COMMUNITY_NOTE_LENGTH:
        return _json_error(
            f"Content is too long (max {Config.MAX_COMMUNITY_NOTE_LENGTH} characters).",
            400,
            "content_too_long",
        )
    category = data.get("category", "general")
    expires_at = data.get("expires_at") or None
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "UPDATE admin_knowledge SET content = %s, category = %s, expires_at = %s WHERE id = %s",
            (content, category, expires_at, entry_id),
        )
        conn.commit()
        _trigger_ingest()
        return jsonify({"message": "Note updated"})
    except Exception as e:
        return _json_error(str(e), 500, "admin_knowledge_edit_failed")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


@app.route("/admin/knowledge/<int:entry_id>", methods=["DELETE"])
def admin_deactivate_knowledge(entry_id):
    result = _require_admin_mutation()
    if result:
        return result
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("UPDATE admin_knowledge SET active = FALSE WHERE id = %s", (entry_id,))
        conn.commit()
        _remove_from_chroma(entry_id)
        return jsonify({"message": "Entry deactivated"})
    except Exception as e:
        return _json_error(str(e), 500, "admin_knowledge_deactivate_failed")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


@app.route("/admin/flags/<int:flag_id>/comment", methods=["PUT"])
def admin_comment_flag(flag_id):
    result = _require_admin_mutation()
    if result:
        return result
    data = request.get_json() or {}
    comment = data.get("moderator_comment", "").strip()
    resolved = data.get("resolved", False)
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            UPDATE interaction_log
            SET moderator_comment = %s,
                resolved = %s,
                resolved_at = %s
            WHERE id = %s
        """, (
            comment,
            resolved,
            datetime.datetime.utcnow() if resolved else None,
            flag_id,
        ))
        conn.commit()
        return jsonify({"message": "Flag updated"})
    except Exception as e:
        return _json_error(str(e), 500, "admin_flag_comment_failed")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()

@app.route("/community/notes/chat", methods=["POST"])
def community_notes_chat():
    result = _require_api_key_or_user()
    if result:
        return result
    if g.get("current_user_row"):
        csrf_error = _enforce_csrf()
        if csrf_error:
            return csrf_error
    data = request.get_json() or {}
    messages = data.get("messages", [])
    if not messages:
        return _json_error("messages required", 400, "missing_messages")
    try:
        client = _get_llm_client()
        model = client.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
        system = (
            "You are helping a Dorchester community member report a local issue or share "
            "neighborhood information for a community chatbot.\n\n"
            "Ask short friendly follow-up questions ONE AT A TIME to gather useful details. "
            "Focus on: exact location (street, intersection, landmark), time/date, severity, "
            "who is affected, and any safety considerations.\n\n"
            "After 3-5 exchanges when you have enough detail, respond with EXACTLY this format "
            "and nothing else:\n"
            "READY:\n"
            "What: <one sentence describing the issue>\n"
            "Where: <specific location>\n"
            "When: <time or date>\n"
            "Details: <any extra context>\n\n"
            "Do not ask more than 5 questions. Keep questions short and friendly. "
            "Never mention AI, databases, or that you are compiling a note."
        )
        history = "\n".join([
            f"{'Community member' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in messages
        ])
        resp = model.generate_content(
            system + "\n\nConversation:\n" + history + "\n\nAssistant:",
            generation_config={"temperature": 0.4}
        )
        try:
            text = (resp.text or "").strip()
        except (ValueError, AttributeError):
            # gemini-2.5-pro returns thinking parts alongside the text part;
            # resp.text raises ValueError when multiple parts exist.
            parts = resp.candidates[0].content.parts
            text_parts = [p.text for p in parts if getattr(p, "text", None)]
            text = (text_parts[-1] if text_parts else "").strip()
        is_ready = text.startswith("READY:")
        compiled = text[6:].strip() if is_ready else None
        return jsonify({"reply": text if not is_ready else None, "compiled": compiled, "ready": is_ready})
    except Exception as e:
        return _json_error(str(e), 500, "community_chat_failed")


@app.route("/community/notes", methods=["POST"])
def community_add_note():
    result = _require_api_key_or_user()
    if result:
        return result
    if g.get("current_user_row"):
        csrf_error = _enforce_csrf()
        if csrf_error:
            return csrf_error
    data = request.get_json() or {}
    content = data.get("content", "").strip()
    if not content:
        return _json_error("content is required", 400, "missing_content")
    if len(content) > Config.MAX_COMMUNITY_NOTE_LENGTH:
        return _json_error(
            f"Content is too long (max {Config.MAX_COMMUNITY_NOTE_LENGTH} characters).",
            400,
            "content_too_long",
        )
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            INSERT INTO admin_knowledge (content, category, added_by, active)
            VALUES (%s, %s, %s, FALSE)
        """, (
            content,
            data.get("category", "community"),
            g.current_user_row["username"] if g.get("current_user_row") else "anonymous",
        ))
        conn.commit()
        return jsonify({"message": "Note submitted for review"}), 201
    except Exception as e:
        return _json_error(str(e), 500, "community_note_failed")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


@app.route("/admin/knowledge/pending", methods=["GET"])
def admin_get_pending():
    result = _require_admin()
    if result:
        return result
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, content, category, added_by, created_at
            FROM admin_knowledge
            WHERE active = FALSE
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        for row in rows:
            if row.get("created_at") and hasattr(row["created_at"], "isoformat"):
                row["created_at"] = row["created_at"].isoformat()
        return jsonify({"pending": rows})
    except Exception as e:
        return _json_error(str(e), 500, "pending_failed")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


@app.route("/admin/knowledge/<int:entry_id>/approve", methods=["PUT"])
def admin_approve_note(entry_id):
    result = _require_admin_mutation()
    if result:
        return result
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("UPDATE admin_knowledge SET active = TRUE WHERE id = %s", (entry_id,))
        conn.commit()
        _trigger_ingest()
        return jsonify({"message": "Note approved"})
    except Exception as e:
        return _json_error(str(e), 500, "approve_failed")
    finally:
        if cursor: cursor.close()
        if conn: conn.close()


if __name__ == "__main__":
    print(f"\n🚀 Agent API {Config.API_VERSION}")
    print(f"   Host: {Config.HOST}:{Config.PORT}")
    print(f"   Auth Modes: user sessions{' + API keys' if Config.RETHINKAI_API_KEYS else ''}")
    print(f"   Debug: {Config.DEBUG}")
    print("   NOTE: this is the Flask dev server. For production use gunicorn:")
    print("         ./start_api.sh   (or: gunicorn -c api/gunicorn_conf.py api_v2:app)")
    print()
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, threaded=True)
