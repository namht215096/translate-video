#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="$PROJECT_DIR/output/backend"
PID_FILE="$STATE_DIR/backend.pid"
LOG_FILE="$STATE_DIR/backend.log"
PYTHON="$PROJECT_DIR/.venv/bin/python"
EXPECTED_COMMAND="$PYTHON -m translate_video.web"
mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/control.lock"
flock 9

owns_process() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]] &&
    [[ "$(ps -p "$1" -o args= 2>/dev/null || true)" == "$EXPECTED_COMMAND" ]]
}

if [[ -f "$PID_FILE" ]]; then
  backend_pid="$(cat "$PID_FILE")"
  if owns_process "$backend_pid"; then
    echo "Backend đã chạy (PID $backend_pid): http://127.0.0.1:8000"
    exit 0
  fi
  rm -f -- "$PID_FILE"
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Không tìm thấy .venv/bin/python. Hãy cài môi trường theo README.md." >&2
  exit 1
fi

if ! "$PYTHON" - <<'PY'
import socket
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", 8000))
    except OSError:
        raise SystemExit(1)
PY
then
  echo "Cổng 8000 đang được dùng. Hãy dừng backend cũ trước." >&2
  exit 1
fi

cd -- "$PROJECT_DIR"
# Close the control lock in the child so stopping it never waits on this lock.
nohup setsid "$PYTHON" -m translate_video.web >"$LOG_FILE" 2>&1 < /dev/null 9>&- &
backend_pid=$!
echo "$backend_pid" >"$PID_FILE"

for ((attempt=0; attempt<100; attempt++)); do
  if ! kill -0 "$backend_pid" 2>/dev/null; then
    rm -f -- "$PID_FILE"
    echo "Backend không khởi động được. Xem log: $LOG_FILE" >&2
    tail -n 15 "$LOG_FILE" >&2
    exit 1
  fi
  if "$PYTHON" - <<'PY' 2>/dev/null
import urllib.request
with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=.3) as response:
    assert response.status == 200
PY
  then
    echo "Đã bật backend (PID $backend_pid): http://127.0.0.1:8000"
    echo "Đợi trang báo Model đã sẵn sàng trước khi phát. Có thể đóng terminal này."
    echo "Log: $LOG_FILE"
    exit 0
  fi
  sleep .1
done

echo "Backend đang khởi động (PID $backend_pid). Xem log: $LOG_FILE"
