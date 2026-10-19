#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$PROJECT_DIR/output/backend"
PID_FILE="$STATE_DIR/backend.pid"
EXPECTED_COMMAND="$PROJECT_DIR/.venv/bin/python -m translate_video.web"
mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/control.lock"
flock 9

if [[ ! -f "$PID_FILE" ]]; then
  echo "Không có backend do start-backend.sh quản lý."
  exit 0
fi

backend_pid="$(cat "$PID_FILE")"
owns_process() {
  [[ "$backend_pid" =~ ^[1-9][0-9]*$ ]] &&
    [[ "$(ps -p "$backend_pid" -o args= 2>/dev/null || true)" == "$EXPECTED_COMMAND" ]]
}

if ! owns_process; then
  rm -f -- "$PID_FILE"
  echo "Backend đã dừng. Đã xóa PID cũ."
  exit 0
fi

echo "Đang dừng backend (PID $backend_pid)…"
kill -TERM "$backend_pid"
for ((attempt=0; attempt<120; attempt++)); do
  if ! owns_process; then
    rm -f -- "$PID_FILE"
    echo "Đã tắt backend."
    exit 0
  fi
  sleep .25
done

# A model load can block graceful shutdown. Kill only the verified owned process.
if owns_process; then
  kill -KILL "$backend_pid"
  echo "Đã buộc tắt backend sau 30 giây chờ."
fi
rm -f -- "$PID_FILE"
