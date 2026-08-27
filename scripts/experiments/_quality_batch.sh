#!/bin/bash
# Batched quality worker. Env: DIRS, CONFIG, optional SET, FORCE, and CPU_ONLY.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Qwen3-32B needs the larger vLLM allocation for the judge KV cache.
: "${SEMSTAMP__RUNTIME__VLLM_UTILIZATION:=0.9}"
export SEMSTAMP__RUNTIME__VLLM_UTILIZATION
# shellcheck disable=SC2086  # DIRS and SET intentionally word-split into args
uv run python -m quality.batch $DIRS --config "$CONFIG" ${SET:-} \
  ${FORCE:+--force} ${CPU_ONLY:+--cpu-only}
