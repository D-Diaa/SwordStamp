#!/bin/bash
# Online PMark generation worker. The input dataset determines the document
# count; N is the per-sentence candidate budget.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PMARK_DIR="$REPO/external/pmark"
# shellcheck disable=SC1091
source "$PMARK_DIR/.venv/bin/activate"
cd "$PMARK_DIR"

args=(
    --num_samples "${N:-64}"
    --log_dir "$LOG_DIR"
    --data_path "$REPO/$BASE"
    --model_path "$MODEL"
    --tokenizer_path "$MODEL"
    --embedder_path "$EMBEDDER"
    --backend vllm
    --pivot rand
    --median_method hd
    --msig "${MSIG:-4}"
    --temperature "${TEMPERATURE:-0.9}"
    --top_p "${TOP_P:-0.9}"
    --repetition_penalty "${REPETITION_PENALTY:-1.05}"
    --max_new_sentences "${MAX_NEW_SENTENCES:-12}"
)

python pmark.py "${args[@]}"
