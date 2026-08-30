#!/usr/bin/env bash
# Start the mock traffic API with uvicorn.
set -euo pipefail

cd "$(dirname "$0")"

# Prefer the project venv, then a system uvicorn.
if [[ -x .venv/bin/uvicorn ]]; then
  UVICORN=".venv/bin/uvicorn"
elif command -v uvicorn >/dev/null 2>&1; then
  UVICORN="uvicorn"
else
  echo "uvicorn not found. Run: python -m pip install fastapi uvicorn" >&2
  exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo "Starting traffic API at http://${HOST}:${PORT} (Ctrl+C to stop)"
exec "$UVICORN" api:app --host "$HOST" --port "$PORT" --reload
