#!/bin/bash
# SAMark generation worker. Env: LOG_DIR, BASE, MODEL, EMBEDDER, MSIG, and tuning values.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

# Spawn vLLM workers because SAMark initializes CUDA first.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

uv run python external/SAMark/samark_gen.py \
    --model_path "$MODEL" \
    --embedder_path "$EMBEDDER" \
    --data_path "$REPO/$BASE" \
    --log_dir "$LOG_DIR" \
    --msig "${MSIG:-2}" \
    --num_samples "${NUM_SAMPLES:-64}" \
    --flag_scope "${FLAG_SCOPE:-run}" \
    --temperature "${TEMPERATURE:-0.9}" \
    --top_p "${TOP_P:-0.9}" \
    --repetition_penalty "${REPETITION_PENALTY:-1.05}" \
    --epsilon "${EPSILON:-80}" \
    --ngram_overlap_thresh "${NGRAM_THRESH:-0.5}" \
    --semantic_sim_thresh "${SEM_THRESH:-0.85}" \
    --diversity_weight "${DIV_WEIGHT:-0.3}" \
    --novelty_weight "${NOV_WEIGHT:-0.15}" \
    --cossim_transform "${COSSIM_TRANSFORM:-tanh}" \
    --cossim_transform_param "${COSSIM_TRANSFORM_PARAM:-30.0}"
