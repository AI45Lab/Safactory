#!/usr/bin/env bash

# Increase file descriptor limit for high concurrency
ulimit -n 65536 2>/dev/null || echo "Warning: Could not set ulimit -n 65536 (current: $(ulimit -n))"

set -euo pipefail

# Load environment variables
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/env.sh"

export PYTHONPATH="${PYTHONPATH:-}:/mnt/shared-storage-user/chenxinquan/Safactory"
export AIEVOBOX_ROOT="${AIEVOBOX_ROOT:-/mnt/shared-storage-user/chenxinquan/Safactory}"
export AIEVOBOX_DB_URL="${AIEVOBOX_DB_URL:-sqlite://${SCRIPT_DIR}/rl.db}"
export ROLLBUF_HOST="${ROLLBUF_HOST:-0.0.0.0}"
export ROLLBUF_PORT="${ROLLBUF_PORT:-18889}"

echo "Starting Buffer Server..."
echo "  Host: ${ROLLBUF_HOST}"
echo "  Port: ${ROLLBUF_PORT}"
echo "  DB URL: ${AIEVOBOX_DB_URL}"

PYTHON_CMD=(python3)

if [[ "${DEBUG_BUFFER_SERVER:-0}" == "1" ]]; then
  PYTHON_CMD=(
    python3 -m debugpy
    --listen "0.0.0.0:${DEBUGPY_BUFFER_PORT:-5678}"
    --configure-subProcess True
  )
  if [[ "${DEBUGPY_WAIT_FOR_CLIENT:-1}" == "1" ]]; then
    PYTHON_CMD+=(--wait-for-client)
  fi
fi

"${PYTHON_CMD[@]}" "${AIEVOBOX_ROOT}/rl/buffer_server.py"
