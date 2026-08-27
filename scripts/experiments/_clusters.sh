#!/bin/bash
# KMeans center worker. Env: BASE, CONFIG, OUTPUT, optional SET and RESTARTS.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONUNBUFFERED=1
uv run python -u -m watermarking.generate_clusters "$BASE" \
  --config "$CONFIG" --output "$OUTPUT" --restarts "${RESTARTS:-3}" ${SET:-}
