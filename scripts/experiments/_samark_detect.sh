#!/bin/bash
# SAMark detection worker. Env: HF_DATASET, EMBEDDER, MSIG, and optional overrides.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

# Limit BLAS threads so concurrent detector jobs do not contend.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

uv run python comparisons/samark/samark_detect.py \
    --mode hf \
    --hf_dataset "$HF_DATASET" \
    ${OUTPUT_DIR:+--output_dir "$OUTPUT_DIR"} \
    --embedder_path "$EMBEDDER" \
    --device "${DEVICE:-cuda}" \
    --msig "${MSIG:-2}" \
    --transform "${TRANSFORM:-none}" \
    --transform_param "${TRANSFORM_PARAM:-30.0}" \
    --human_text "${HUMAN_TEXT:-data/c4-human-def}" \
    --human_n "${HUMAN_N:-1024}" \
    --human_max_sentences "${HUMAN_MAX_SENTENCES:-12}" \
    --fpr_target "${FPR_TARGET:-0.01}"
