#!/bin/bash
# Online PMark detection worker using its analytical null.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PMARK_DIR="$REPO/comparisons/pmark"
# shellcheck disable=SC1091
source "$PMARK_DIR/.venv/bin/activate"
cd "$PMARK_DIR"

args=(
    --mode hf
    --hf_dataset    "$SAVE_DIR"
    --output_dir    "$SAVE_DIR"
    --embedder_name "$EMBEDDER"
    --num_samples   "${N:-64}"
    --pivot rand
    --median_method hd
    --msig "${MSIG:-4}"
    --soft_k "${SOFT_K:-150}"
    --soft_delta "${SOFT_DELTA:-0.001}"
    --temperature "${TEMPERATURE:-0.9}"
    --top_p "${TOP_P:-0.9}"
    --repetition_penalty "${REPETITION_PENALTY:-1.05}"
    --model_name "$MODEL"
    --tokenizer_name "$MODEL"
    --backend vllm
)

python detect.py "${args[@]}"
