#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PMARK_COMMIT="5a8ad3008fdc2c58e607de2f60941c2d1911deb9"
SAMARK_COMMIT="9160b8025fda05be9a36e02abf0450c1249693cc"

die() {
    printf 'setup error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

verify_submodule() {
    local path="$1"
    local expected="$2"
    local actual

    [[ -e "$REPO_ROOT/$path/.git" ]] || die "$path is not initialized"
    actual="$(git -C "$REPO_ROOT/$path" rev-parse HEAD)"
    [[ "$actual" == "$expected" ]] || die "$path is at $actual, expected $expected"
}

require_command git
require_command uv

cd "$REPO_ROOT"
printf '%s\n' '==> Initializing pinned comparison submodules'
git submodule sync --recursive
git submodule update --init --recursive
verify_submodule external/pmark "$PMARK_COMMIT"
verify_submodule external/SAMark "$SAMARK_COMMIT"

printf '%s\n' '==> Installing the frozen SwordStamp environment'
uv sync --frozen --python 3.11

printf '%s\n' '==> Installing NLTK Punkt resources'
uv run --frozen python scripts/install_punkt.py

PMARK_VENV="$REPO_ROOT/external/pmark/.venv"
PMARK_PYTHON="$PMARK_VENV/bin/python"
if [[ ! -x "$PMARK_PYTHON" ]]; then
    printf '%s\n' '==> Creating the dedicated PMark Python 3.11 environment'
    uv venv "$PMARK_VENV" --python 3.11
fi

printf '%s\n' '==> Installing the pinned minimal PMark dependency set'
uv pip install \
    --python "$PMARK_PYTHON" \
    --index https://download.pytorch.org/whl/cu124 \
    --requirements "$REPO_ROOT/requirements/pmark.txt"
"$PMARK_PYTHON" "$REPO_ROOT/scripts/install_punkt.py"

printf '%s\n' '==> Verifying release metadata and local installation'
uv run --frozen python scripts/check_artifact.py

printf '%s\n' 'Setup complete.'
printf '%s\n' 'The GPU scheduler is intentionally separate; see scripts/install_gpu_scheduler.sh.'
exit 0
