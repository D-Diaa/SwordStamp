#!/bin/bash
# Attack worker. Env: BASE, CONFIG, optional SET.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
uv run python -m attacks "$BASE" --config "$CONFIG" ${SET:-}
