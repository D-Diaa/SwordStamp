#!/bin/bash
# Generation worker. Env: BASE, CONFIG, optional SET.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONUNBUFFERED=1
uv run python -u -m watermarking.generate "$BASE" --config "$CONFIG" ${SET:-}
