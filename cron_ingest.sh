#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/cron_ingest.log"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*" | tee -a "$LOG_FILE"
}

if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON=".venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  PYTHON="$(command -v python)"
fi

if [ ! -f ".env" ]; then
  log "ERROR: Missing .env at $ROOT_DIR/.env"
  exit 1
fi

# Prevent overlapping ingestion runs. If a run is slow, the next scheduled run
# must not start and double-write to the vector store / MySQL tables.
LOCK_FILE="$LOG_DIR/cron_ingest.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    log "Another ingestion run is already in progress; exiting."
    exit 0
  fi
else
  log "WARNING: 'flock' not available; skipping overlap protection."
fi

run_step() {
  local label="$1"
  shift
  log "START: $label"
  "$@" 2>&1 | tee -a "$LOG_FILE"
  log "DONE: $label"
}

log "Cron ingestion job started"
log "Using Python: $PYTHON"

run_step \
  "Boston Open Data sync (311 & 911)" \
  "$PYTHON" "on_the_porch/data_ingestion/boston_data_sync/boston_data_sync.py"

run_step \
  "DotNews newsletter ingestion" \
  "$PYTHON" "on_the_porch/data_ingestion/run_dotnews_ingest_v2.py"

run_step \
  "Tribe events ingestion" \
  "$PYTHON" "on_the_porch/data_ingestion/tribe_events_ingester.py" "all"

run_step \
  "RSS ingestion" \
  "$PYTHON" "on_the_porch/rag stuff/ingest_rss_wayback.py"

run_step \
  "Google Drive ingestion" \
  "$PYTHON" "on_the_porch/data_ingestion/google_drive_to_vectordb.py"

run_step \
  "Orphan guest user cleanup" \
  "$PYTHON" "api/cleanup_guest_users.py"

log "Cron ingestion job finished successfully"
