#!/bin/bash
set -e
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x "venv/bin/python3" ]; then PYTHON_BIN="venv/bin/python3"; fi
"$PYTHON_BIN" -m compiler &
COMPILER_PID=$!
trap 'kill "$COMPILER_PID" 2>/dev/null || true' EXIT INT TERM
cd web
if [ ! -d node_modules ]; then
  if [ -f package-lock.json ]; then npm ci; else npm install; fi
fi
npm run dev -- --host 127.0.0.1
