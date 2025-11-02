#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/.venv"
PY="$VENV/bin/python"
LOG="$DIR/server.log"
PIDFILE="$DIR/server.pid"

usage() {
  echo "Usage: $0 {start|stop|status|run} [port]"
  exit 1
}

if [ ! -x "$PY" ]; then
  echo "Warning: virtualenv python not found at $PY"
  echo "Make sure you've created the venv or adjust the script."
fi

cmd="$1"

case "$cmd" in
  start)
    PORT=${2:-5000}
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" >/dev/null 2>&1; then
      echo "Already running (pid $(cat $PIDFILE)). Use $0 stop to stop it."
      exit 0
    fi
  echo "Starting server (background). Logs: $LOG"
  # Suppress all warnings in this development run to avoid noisy urllib3 LibreSSL warnings
  export PYTHONWARNINGS="ignore"
  nohup "$PY" "$DIR/app.py" >"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    sleep 0.3
    echo "Started pid $(cat $PIDFILE)"
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then
      PID=$(cat "$PIDFILE")
      echo "Stopping pid $PID"
      kill "$PID" || true
      rm -f "$PIDFILE"
      echo "Stopped"
    else
      echo "Not running"
    fi
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" >/dev/null 2>&1; then
      echo "Running pid $(cat $PIDFILE)"
    else
      echo "Not running"
    fi
    echo "--- Last 50 log lines ($LOG) ---"
    tail -n 50 "$LOG" 2>/dev/null || true
    ;;
  run)
    "$PY" "$DIR/app.py"
    ;;
  *)
    usage
    ;;
esac
