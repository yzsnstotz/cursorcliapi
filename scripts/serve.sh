#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${CODEX_GATEWAY_ENV_FILE:-$ROOT_DIR/.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  echo "Create it via: cp $ROOT_DIR/.env.example $ENV_FILE" >&2
  exit 1
fi

HOST="${1:-127.0.0.1}"
PORT="${2:-8000}"

exec uv run --env-file "$ENV_FILE" uvicorn main:app --host "$HOST" --port "$PORT"
