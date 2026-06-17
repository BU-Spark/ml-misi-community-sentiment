#!/usr/bin/env bash
# Start the RethinkAI API under gunicorn (production server).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Prefer a project virtualenv if present, else fall back to PATH.
if [ -x ".venv/bin/gunicorn" ]; then
  GUNICORN=".venv/bin/gunicorn"
elif [ -x ".venv/Scripts/gunicorn.exe" ]; then
  GUNICORN=".venv/Scripts/gunicorn.exe"
elif command -v gunicorn >/dev/null 2>&1; then
  GUNICORN="$(command -v gunicorn)"
else
  echo "ERROR: gunicorn not found. Install it (pip install gunicorn) or activate your venv." >&2
  exit 1
fi

if [ ! -f ".env" ]; then
  echo "WARNING: no .env at $ROOT_DIR/.env; relying on the process environment." >&2
fi

exec "$GUNICORN" -c "$ROOT_DIR/api/gunicorn_conf.py" api_v2:app
