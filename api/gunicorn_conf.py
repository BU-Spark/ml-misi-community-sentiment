"""Gunicorn configuration for the RethinkAI API (production server).

Run with:
    gunicorn -c api/gunicorn_conf.py api_v2:app
or simply:
    ./start_api.sh

Every setting can be overridden with an environment variable so the same config
works across machines without edits.
"""

import os
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_API_DIR = _THIS_FILE.parent
_ROOT_DIR = _API_DIR.parent

# Run from the repo root, but make `import api_v2` (and its sibling modules)
# importable by putting the api/ directory on the path.
chdir = str(_ROOT_DIR)
pythonpath = str(_API_DIR)

# --- Networking -------------------------------------------------------------
bind = os.getenv(
    "GUNICORN_BIND",
    f"{os.getenv('API_HOST', '127.0.0.1')}:{os.getenv('API_PORT', '8888')}",
)

# --- Concurrency ------------------------------------------------------------
# gthread: each worker is a process running a pool of threads. This suits this
# app well because requests are I/O-bound (waiting on Gemini / MySQL / Chroma),
# so threads give cheap concurrency while a handful of processes use all cores.
worker_class = os.getenv("GUNICORN_WORKER_CLASS", "gthread")
workers = int(os.getenv("GUNICORN_WORKERS", "3"))
threads = int(os.getenv("GUNICORN_THREADS", "8"))

# --- Timeouts ---------------------------------------------------------------
# A chat request can legitimately take a while (several LLM calls). Keep the
# worker timeout generous so a slow-but-valid request is not killed, but finite
# so a truly hung worker is recycled instead of wedged forever.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "5"))

# --- Memory hygiene ---------------------------------------------------------
# Periodically recycle workers to bound any slow leaks (LLM/vector clients).
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "100"))

# --- Worker initialization --------------------------------------------------
# Do NOT preload the app. Each worker must build its own MySQL connection pool
# and Chroma client AFTER forking; sharing those across forked processes
# corrupts pooled sockets and SQLite handles.
preload_app = False

# --- Logging ----------------------------------------------------------------
accesslog = os.getenv("GUNICORN_ACCESS_LOG", "-")
errorlog = os.getenv("GUNICORN_ERROR_LOG", "-")
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
