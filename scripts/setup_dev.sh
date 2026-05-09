#!/usr/bin/env bash
# One-shot dev environment setup for Algo Bot.
#
# Usage:  ./scripts/setup_dev.sh
#
# Idempotent: re-running just verifies things.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Auto-detect: prefer 3.11, fall back to 3.12 (both supported).
if [[ -z "${PY_BIN:-}" ]]; then
    if command -v python3.11 >/dev/null 2>&1; then
        PY_BIN="python3.11"
    elif command -v python3.12 >/dev/null 2>&1; then
        PY_BIN="python3.12"
    else
        PY_BIN="python3"
    fi
fi
VENV_DIR=".venv"

echo "==> Project root: $PROJECT_ROOT"

if ! command -v "$PY_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PY_BIN not found. Install Python 3.11 first (brew install python@3.11)." >&2
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "==> Creating venv at $VENV_DIR"
    "$PY_BIN" -m venv "$VENV_DIR"
else
    echo "==> Reusing existing venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
python -m pip install -U pip wheel setuptools

echo "==> Installing dependencies (this can take a few minutes)"
pip install -r requirements.txt

echo "==> Installing pre-commit hooks"
pre-commit install

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "==> Created .env from template — fill in your API keys before going to demo/live."
fi

echo "==> Running smoke tests"
pytest -q || {
    echo "Smoke tests failed. Please investigate before continuing."
    exit 1
}

echo "==> Done. Activate with:  source $VENV_DIR/bin/activate"
