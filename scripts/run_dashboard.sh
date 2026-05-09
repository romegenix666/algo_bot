#!/usr/bin/env bash
# Start the Streamlit dashboard.
#
# Usage:  ./scripts/run_dashboard.sh
# Then open http://localhost:8501 in your browser.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec streamlit run src/monitor/dashboard.py "$@"
