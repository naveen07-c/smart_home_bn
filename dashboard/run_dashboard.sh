#!/usr/bin/env bash
#
# run_dashboard.sh — One-command launcher for the Smart Home Decision Network
# live dashboard (FastAPI + D3.js).
#
# What it does:
#   1. Resolves the project root and the Python virtualenv (creates one if absent)
#   2. Ensures every required package is installed (incl. websockets for /ws)
#   3. Verifies the trained model and the test dataset exist
#   4. Starts uvicorn on http://0.0.0.0:8000 and prints the dashboard URL
#
# Usage:
#   ./run_dashboard.sh                      # default host 0.0.0.0, port 8000
#   HOST=127.0.0.1 PORT=9000 ./run_dashboard.sh
#   RELOAD=1 ./run_dashboard.sh             # auto-reload on code changes
#   ./run_dashboard.sh --log-level debug    # extra args are passed to uvicorn
#
set -euo pipefail

# ----------------------------------------------------------------------------
# Locate directories (script lives in <project>/dashboard/)
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-0}"

MODEL_FILE="$PROJECT_ROOT/outputs/models/bayesian_network.pkl"
DATA_FILE="$PROJECT_ROOT/data/processed/test.csv"

log()  { printf '\033[1;34m[run_dashboard]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[run_dashboard]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[run_dashboard] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------
# 1. Python interpreter + virtualenv
# ----------------------------------------------------------------------------
PY="${VENV_DIR}/bin/python"
if ! command -v python3 >/dev/null 2>&1; then
    die "python3 not found on PATH. Install Python 3.10+ and retry."
fi

if [ ! -x "$PY" ]; then
    warn "Virtualenv not found at $VENV_DIR — creating it..."
    python3 -m venv "$VENV_DIR"
fi
log "Using virtualenv: $VENV_DIR ($("$PY" --version))"

PIP="${VENV_DIR}/bin/pip"

# ----------------------------------------------------------------------------
# 2. Dependencies — install only what is missing
# ----------------------------------------------------------------------------
CORE_IMPORTS="fastapi, uvicorn, pandas, numpy, pgmpy, networkx"

if "$PY" -c "import $CORE_IMPORTS, websockets" >/dev/null 2>&1; then
    log "All required packages already installed."
else
    log "Installing/updating Python dependencies (this may take a minute)..."
    # Pipeline deps from the project's requirements.txt
    "$PIP" install --quiet --upgrade pip
    "$PIP" install --quiet -r "$PROJECT_ROOT/requirements.txt"
    # Dashboard-specific deps — uvicorn[standard] + websockets give the /ws
    # WebSocket support that requirements.txt does not include.
    "$PIP" install --quiet fastapi "uvicorn[standard]" websockets
    log "Dependencies ready."
fi

# ----------------------------------------------------------------------------
# 3. Runtime artifacts — trained model + test data
# ----------------------------------------------------------------------------
[ -f "$MODEL_FILE" ] || die "Trained model not found: $MODEL_FILE
  → Run the full pipeline first:  cd $PROJECT_ROOT && python main.py"
[ -f "$DATA_FILE" ]  || die "Test dataset not found: $DATA_FILE
  → Run the full pipeline first:  cd $PROJECT_ROOT && python main.py"
log "Model + dataset found (test.csv: $(du -h "$DATA_FILE" | cut -f1))."

# ----------------------------------------------------------------------------
# 4. Launch the site
# ----------------------------------------------------------------------------
EXTRA_ARGS=("$@")
RELOAD_FLAG=()
if [ "$RELOAD" = "1" ]; then
    RELOAD_FLAG=(--reload)
fi

log "Starting dashboard: http://localhost:$PORT  (Ctrl+C to stop)"
cd "$SCRIPT_DIR"
exec "$PY" -m uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    "${RELOAD_FLAG[@]}" \
    "${EXTRA_ARGS[@]}"
