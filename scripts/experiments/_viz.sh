#!/bin/bash
# CPU visualization worker. Env: DIR.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/../.."
uv run python "$HERE/detection_summary.py" "$DIR"
