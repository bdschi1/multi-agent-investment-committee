#!/usr/bin/env bash
# ============================================================
# Multi-Agent Investment Committee (MAIC) — Run Script
# ============================================================
# Usage:
#   ./run.sh              Setup environment and launch Gradio app
#   ./run.sh setup        Install dependencies only
#   ./run.sh app          Launch Gradio UI only
#   ./run.sh api          Launch FastAPI server only
#   ./run.sh test         Run test suite
#   ./run.sh help         Show this help message
# ============================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python3"

# ── Colors ─────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}═══════════════════════════════════════════════${NC}"; \
            echo -e "${BOLD}${CYAN}  $*${NC}"; \
            echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════${NC}\n"; }

# ── Find Python 3.9+ ──────────────────────────────────────
find_python() {
    local candidates=(python3.13 python3.12 python3.11 python3.10 python3.9 python3)
    for candidate in "${candidates[@]}"; do
        if command -v "${candidate}" &>/dev/null; then
            local ver major minor
            ver=$("${candidate}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if (( major >= 3 && minor >= 9 )); then
                SYSTEM_PYTHON="${candidate}"
                return 0
            fi
        fi
    done
    return 1
}

# ── Setup ──────────────────────────────────────────────────
setup() {
    header "Setting up Multi-Agent Investment Committee"

    if ! find_python; then
        error "Python 3.9+ not found. Install Python and retry."
        exit 1
    fi
    success "Python detected (${SYSTEM_PYTHON})"

    if [[ ! -d "${VENV_DIR}" ]]; then
        info "Creating virtual environment..."
        "${SYSTEM_PYTHON}" -m venv "${VENV_DIR}"
        success "Virtual environment created"
    elif ! "${PYTHON}" -c "import pip" &>/dev/null 2>&1; then
        warn "Existing venv is broken — recreating..."
        rm -rf "${VENV_DIR}"
        "${SYSTEM_PYTHON}" -m venv "${VENV_DIR}"
        success "Virtual environment recreated"
    else
        success "Virtual environment already exists"
    fi

    info "Upgrading pip..."
    "${PYTHON}" -m pip install --upgrade pip --quiet

    info "Installing dependencies..."
    "${PYTHON}" -m pip install -e ".[dev]" --quiet
    success "All dependencies installed"

    if [[ ! -f "${REPO_DIR}/.env" ]] && [[ -f "${REPO_DIR}/.env.example" ]]; then
        warn "No .env file found — copying from .env.example"
        cp "${REPO_DIR}/.env.example" "${REPO_DIR}/.env"
        warn "Edit .env and add your API keys before running"
    fi

    echo ""
    success "Setup complete!"
}

# ── Gradio App ─────────────────────────────────────────────
run_app() {
    header "Launching MAIC (Gradio UI)"

    if [[ ! -f "${PYTHON}" ]]; then
        error "Virtual environment not found. Run './run.sh setup' first."
        exit 1
    fi

    info "Starting Gradio app on http://localhost:7860 ..."
    info "Press Ctrl+C to stop"
    echo ""
    "${PYTHON}" "${REPO_DIR}/app.py"
}

# ── FastAPI Server ─────────────────────────────────────────
run_api() {
    header "Launching MAIC (FastAPI)"

    if [[ ! -f "${PYTHON}" ]]; then
        error "Virtual environment not found. Run './run.sh setup' first."
        exit 1
    fi

    info "Starting FastAPI server on http://localhost:8000 ..."
    info "Press Ctrl+C to stop"
    echo ""
    "${PYTHON}" -m uvicorn api.main:app --reload
}

# ── Tests ──────────────────────────────────────────────────
run_tests() {
    header "Running tests"

    if [[ ! -f "${PYTHON}" ]]; then
        error "Virtual environment not found. Run './run.sh setup' first."
        exit 1
    fi

    "${PYTHON}" -m pytest tests/ -v
}

# ── Help ───────────────────────────────────────────────────
show_help() {
    cat <<'HELP'

Multi-Agent Investment Committee — Run Script
────────────────────────────────────────────────

USAGE
    ./run.sh [command]

COMMANDS
    (none)      Setup environment and launch Gradio UI
    setup       Create venv and install dependencies
    app         Launch Gradio UI only (skip setup)
    api         Launch FastAPI REST server only
    test        Run the test suite
    help        Show this help message

EXAMPLES
    ./run.sh            # Setup + launch Gradio UI
    ./run.sh setup      # Install deps only
    ./run.sh app        # Launch Gradio UI (localhost:7860)
    ./run.sh api        # Launch FastAPI server (localhost:8000)
    ./run.sh test       # Run tests

HELP
}

# ── Main ───────────────────────────────────────────────────
main() {
    local command="${1:-}"
    case "${command}" in
        setup)      setup ;;
        app)        run_app ;;
        api)        run_api ;;
        test|tests) run_tests ;;
        help|--help|-h) show_help ;;
        "")         setup; run_app ;;
        *)          error "Unknown command: ${command}"; show_help; exit 1 ;;
    esac
}

main "$@"
