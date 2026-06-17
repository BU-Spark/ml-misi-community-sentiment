#!/usr/bin/env bash
# Run a PROJECT-LOCAL Redis server for the session cache.
#
# This is intentionally NOT a global "brew services" login daemon. It runs in
# the foreground, bound to localhost, stores its data under ./.redis, and lives
# and dies with this terminal — the closest thing to "Redis scoped to this
# project" alongside your virtualenv. Run it in its own terminal:
#
#     ./start_redis.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v redis-server >/dev/null 2>&1; then
  REDIS_SERVER="$(command -v redis-server)"
elif [ -x /opt/homebrew/opt/redis/bin/redis-server ]; then
  REDIS_SERVER="/opt/homebrew/opt/redis/bin/redis-server"
elif [ -x /usr/local/opt/redis/bin/redis-server ]; then
  REDIS_SERVER="/usr/local/opt/redis/bin/redis-server"
else
  echo "ERROR: redis-server not found. Install it first: brew install redis" >&2
  exit 1
fi

DATA_DIR="$ROOT_DIR/.redis"
mkdir -p "$DATA_DIR"

echo "Starting project-local Redis on 127.0.0.1:${REDIS_PORT:-6379} (data: $DATA_DIR)"
# --save "" + --appendonly no: this is a cache, so we don't need persistence.
exec "$REDIS_SERVER" \
  --bind 127.0.0.1 \
  --port "${REDIS_PORT:-6379}" \
  --dir "$DATA_DIR" \
  --save "" \
  --appendonly no
