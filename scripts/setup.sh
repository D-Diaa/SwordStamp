#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_COMPARISONS=0

die() {
    printf 'setup error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

while (( $# )); do
    case "$1" in
        --comparisons) INSTALL_COMPARISONS=1 ;;
        *) die "unknown option: $1 (supported: --comparisons)" ;;
    esac
    shift
done

require_command uv

cd "$REPO_ROOT"
printf '%s\n' '==> Installing the frozen Python 3.11 environment'
uv sync --frozen --python 3.11

printf '%s\n' '==> Installing NLTK Punkt resources'
uv run --frozen python scripts/install_punkt.py

if (( INSTALL_COMPARISONS )); then
    PMARK_VENV="$REPO_ROOT/comparisons/pmark/.venv"
    PMARK_PYTHON="$PMARK_VENV/bin/python"
    if [[ ! -x "$PMARK_PYTHON" ]]; then
        printf '%s\n' '==> Creating the isolated PMark Python 3.11 environment'
        uv venv "$PMARK_VENV" --python 3.11
    fi
    printf '%s\n' '==> Installing the pinned PMark integration dependencies'
    uv pip install \
        --python "$PMARK_PYTHON" \
        --index https://download.pytorch.org/whl/cu124 \
        --requirements "$REPO_ROOT/requirements/pmark.txt"
    "$PMARK_PYTHON" "$REPO_ROOT/scripts/install_punkt.py"
fi

printf '%s\n' '==> Verifying the anonymous artifact bundle'
check_args=()
(( INSTALL_COMPARISONS )) && check_args=(--comparisons)
uv run --frozen python scripts/check_artifact.py "${check_args[@]}"

printf '%s\n' 'Setup complete.'
printf '%s\n' 'Comparison setup is opt-in: bash scripts/setup.sh --comparisons.'
printf '%s\n' 'Scheduler installation is explicit: inspect scripts/install_bundled_scheduler.sh --print-plan.'
exit 0
