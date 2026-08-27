#!/bin/bash
set -e

# SAMark full pipeline (Token-level=None only)
# Usage:
#   bash SAMark/run_pipeline.sh MODEL_PATH EMBEDDER_PATH DATA_NAME START END [EPS] [REF_DIR] [RUN_LLM_JUDGE]

MODEL_PATH="${1:-../models/Mistral-Small-3.1-24B-Base-2503}"
EMBEDDER_PATH="${2:-../models/all-mpnet-base-v2}"
DATA_NAME="${3:-booksum}"
START="${4:-0}"
END="${5:-5}"
EPS="${6:-80}"
REF_DIR="${7:-}"
RUN_LLM_JUDGE="${8:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

GPU_LIST="0"
read -a GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

MSIG=2
NUM_SAMPLES=64
NGRAM_OVERLAP_THRESH=0.5
SEMANTIC_SIM_THRESH=0.85
DIVERSITY_WEIGHT=0.3
NOVELTY_WEIGHT=0.15
COSSIM_TRANSFORM="tanh"
COSSIM_TRANSFORM_PARAM=30.0
HUMAN_TEXT="${HUMAN_TEXT:-data/c4-human-def}"
HUMAN_N="${HUMAN_N:-1024}"

DOC_P_ATTACKS=("Doc-P I" "Doc-P II")
PROMPT_ATTACK=1
TRANS_ATTACK_NAME="Doc-T(GPT)"
TRANS_ATTACK_RATIO=0.0
TRANS_PROMPT_ATTACK=1

MODEL_NAME=$(basename "$MODEL_PATH")
LOG_DIR="./logs/${DATA_NAME}_logs/${MODEL_NAME}_log"
SUFFIX="softmax-ep${EPS}_div${DIVERSITY_WEIGHT}_nov${NOVELTY_WEIGHT}_sim${SEMANTIC_SIM_THRESH}_op${NGRAM_OVERLAP_THRESH}_sample${NUM_SAMPLES}_tanh${COSSIM_TRANSFORM_PARAM%.0}"
GEN_LOG_DIR="${LOG_DIR}/samark_${SUFFIX}"

TOTAL_SAMPLES=$((END - START))
CHUNK_SIZE=$(((TOTAL_SAMPLES + NUM_GPUS - 1) / NUM_GPUS))


echo "============================================================"
echo " SAMark Full Pipeline"
echo "============================================================"
echo " Model              : ${MODEL_PATH}"
echo " Embedder           : ${EMBEDDER_PATH}"
echo " Data               : ${DATA_NAME}"
echo " Range              : ${START} - ${END}"
echo " Cossim transform   : ${COSSIM_TRANSFORM}(${COSSIM_TRANSFORM_PARAM})"
echo " Output             : ${GEN_LOG_DIR}"
echo "============================================================"

run_worker() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3
    local worker_id=$4

    export CUDA_VISIBLE_DEVICES=$gpu_id
    local prefix="[Worker $worker_id | GPU $gpu_id]"

    echo "$prefix Generate: ${start_idx} to ${end_idx}"
    python "${SCRIPT_DIR}/samark_gen.py" \
        --model_path "${MODEL_PATH}" \
        --embedder_path "${EMBEDDER_PATH}" \
        --data_name "${DATA_NAME}" \
        --log_dir "${GEN_LOG_DIR}" \
        --msig "${MSIG}" \
        --num_samples "${NUM_SAMPLES}" \
        --epsilon "${EPS}" \
        --ngram_overlap_thresh "${NGRAM_OVERLAP_THRESH}" \
        --semantic_sim_thresh "${SEMANTIC_SIM_THRESH}" \
        --diversity_weight "${DIVERSITY_WEIGHT}" \
        --novelty_weight "${NOVELTY_WEIGHT}" \
        --cossim_transform "${COSSIM_TRANSFORM}" \
        --cossim_transform_param "${COSSIM_TRANSFORM_PARAM}" \
        --start "${start_idx}" \
        --end "${end_idx}"

    echo "$prefix Detect baseline: ${start_idx} to ${end_idx}"
    python "${SCRIPT_DIR}/samark_detect.py" \
        --embedder_path "${EMBEDDER_PATH}" \
        --log_dir "${GEN_LOG_DIR}" \
        --msig "${MSIG}" \
        --human_text "${HUMAN_TEXT}" \
        --human_n "${HUMAN_N}" \
        --transform none \
        --detect_subdir detect \
        --start "${start_idx}" \
        --end "${end_idx}"

    echo "$prefix Detect tanh: ${start_idx} to ${end_idx}"
    python "${SCRIPT_DIR}/samark_detect.py" \
        --embedder_path "${EMBEDDER_PATH}" \
        --log_dir "${GEN_LOG_DIR}" \
        --msig "${MSIG}" \
        --human_text "${HUMAN_TEXT}" \
        --human_n "${HUMAN_N}" \
        --transform tanh \
        --transform_param "${COSSIM_TRANSFORM_PARAM}" \
        --detect_subdir detect_tanh${COSSIM_TRANSFORM_PARAM%.0} \
        --start "${start_idx}" \
        --end "${end_idx}"

    for ATTACK_NAME in "${DOC_P_ATTACKS[@]}"; do
        python "${SCRIPT_DIR}/attack_gpt.py" \
            --log_dir "${LOG_DIR}" \
            --sub_dir "samark_${SUFFIX}" \
            --start "${start_idx}" \
            --end "${end_idx}" \
            --attack_name "${ATTACK_NAME}" \
            --prompt_attack "${PROMPT_ATTACK}"
    done

    python "${SCRIPT_DIR}/attack_trans.py" \
        --log_dir "${LOG_DIR}" \
        --sub_dir "samark_${SUFFIX}" \
        --start "${start_idx}" \
        --end "${end_idx}" \
        --attack_name "${TRANS_ATTACK_NAME}" \
        --ratio "${TRANS_ATTACK_RATIO}" \
        --prompt_attack "${TRANS_PROMPT_ATTACK}"

    for ATTACK_NAME in "${DOC_P_ATTACKS[@]}"; do
        ATTACK_SUB="${ATTACK_NAME}"
        python "${SCRIPT_DIR}/samark_detect.py" \
            --embedder_path "${EMBEDDER_PATH}" \
            --log_dir "${GEN_LOG_DIR}" \
            --sub_dir "attack/${ATTACK_SUB}" \
            --msig "${MSIG}" \
            --human_text "${HUMAN_TEXT}" \
            --human_n "${HUMAN_N}" \
            --transform none \
            --detect_subdir detect \
            --start "${start_idx}" \
            --end "${end_idx}"

        python "${SCRIPT_DIR}/samark_detect.py" \
            --embedder_path "${EMBEDDER_PATH}" \
            --log_dir "${GEN_LOG_DIR}" \
            --sub_dir "attack/${ATTACK_SUB}" \
            --msig "${MSIG}" \
            --human_text "${HUMAN_TEXT}" \
            --human_n "${HUMAN_N}" \
            --transform tanh \
            --transform_param "${COSSIM_TRANSFORM_PARAM}" \
            --detect_subdir detect_tanh${COSSIM_TRANSFORM_PARAM%.0} \
            --start "${start_idx}" \
            --end "${end_idx}"
    done

    TRANS_ATTACK_SUB="${TRANS_ATTACK_NAME}${TRANS_ATTACK_RATIO}-${TRANS_PROMPT_ATTACK}"
    python "${SCRIPT_DIR}/samark_detect.py" \
        --embedder_path "${EMBEDDER_PATH}" \
        --log_dir "${GEN_LOG_DIR}" \
        --sub_dir "attack/${TRANS_ATTACK_SUB}" \
        --msig "${MSIG}" \
        --human_text "${HUMAN_TEXT}" \
        --human_n "${HUMAN_N}" \
        --transform none \
        --detect_subdir detect \
        --start "${start_idx}" \
        --end "${end_idx}"

    python "${SCRIPT_DIR}/samark_detect.py" \
        --embedder_path "${EMBEDDER_PATH}" \
        --log_dir "${GEN_LOG_DIR}" \
        --sub_dir "attack/${TRANS_ATTACK_SUB}" \
        --msig "${MSIG}" \
        --human_text "${HUMAN_TEXT}" \
        --human_n "${HUMAN_N}" \
        --transform tanh \
        --transform_param "${COSSIM_TRANSFORM_PARAM}" \
        --detect_subdir detect_tanh${COSSIM_TRANSFORM_PARAM%.0} \
        --start "${start_idx}" \
        --end "${end_idx}"

    echo "$prefix Done all steps: ${start_idx} to ${end_idx}"
}

PIDS=()
for (( i=0; i<NUM_GPUS; i++ )); do
    CHUNK_START=$(( START + i * CHUNK_SIZE ))
    CHUNK_END=$(( CHUNK_START + CHUNK_SIZE ))
    if (( CHUNK_END > END )); then CHUNK_END=$END; fi
    if (( CHUNK_START >= END )); then break; fi
    run_worker "${GPUS[$i]}" "$CHUNK_START" "$CHUNK_END" "$i" &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid"; done

for ATTACK_NAME in "${DOC_P_ATTACKS[@]}"; do
    ATTACK_SUB="${ATTACK_NAME}"
    python "${SCRIPT_DIR}/hierarchical_tpr.py" \
        --root_dir "${GEN_LOG_DIR}" \
        --attack_name "${ATTACK_SUB}" \
        --num_samples "${TOTAL_SAMPLES}" \
        --start "${START}"
    python "${SCRIPT_DIR}/hierarchical_tpr.py" \
        --root_dir "${GEN_LOG_DIR}" \
        --attack_name "${ATTACK_SUB}" \
        --num_samples "${TOTAL_SAMPLES}" \
        --start "${START}" \
        --detect_suffix "_tanh${COSSIM_TRANSFORM_PARAM%.0}"
done

TRANS_ATTACK_SUB="${TRANS_ATTACK_NAME}${TRANS_ATTACK_RATIO}-${TRANS_PROMPT_ATTACK}"
python "${SCRIPT_DIR}/hierarchical_tpr.py" \
    --root_dir "${GEN_LOG_DIR}" \
    --attack_name "${TRANS_ATTACK_SUB}" \
    --num_samples "${TOTAL_SAMPLES}" \
    --start "${START}"
python "${SCRIPT_DIR}/hierarchical_tpr.py" \
    --root_dir "${GEN_LOG_DIR}" \
    --attack_name "${TRANS_ATTACK_SUB}" \
    --num_samples "${TOTAL_SAMPLES}" \
    --start "${START}" \
    --detect_suffix "_tanh${COSSIM_TRANSFORM_PARAM%.0}"

python "${SCRIPT_DIR}/diversity_eval.py" \
    --log_dir "${GEN_LOG_DIR}" \
    --start "${START}" \
    --end "${END}" \
    --text_key "generated_text"
python "${SCRIPT_DIR}/diversity_eval.py" \
    --log_dir "${GEN_LOG_DIR}" \
    --start "${START}" \
    --end "${END}" \
    --text_key "original_text"

if [[ -z "${REF_DIR}" && ( "${RUN_LLM_JUDGE}" == "1" ) ]]; then
    REF_DIR="${LOG_DIR}/unwatermarked_ref"
    echo "Generating unwatermarked reference at ${REF_DIR}"
    python "${SCRIPT_DIR}/samark_gen_unwatermarked.py" \
        --model_path "${MODEL_PATH}" \
        --data_name "${DATA_NAME}" \
        --log_dir "${REF_DIR}" \
        --start "${START}" \
        --end "${END}"
fi

if [[ -n "${REF_DIR}" ]]; then
    python "${SCRIPT_DIR}/jsd_eval.py" \
        --target_dir "${GEN_LOG_DIR}" \
        --ref_dir "${REF_DIR}" \
        --start "${START}" \
        --end "${END}" \
        --text_key "generated_text"
fi

if [[ "${RUN_LLM_JUDGE}" == "1" && -n "${REF_DIR}" ]]; then
    python "${SCRIPT_DIR}/llm_judge_eval.py" \
        --log_dir "${GEN_LOG_DIR}" \
        --ref_dir "${REF_DIR}" \
        --start "${START}" \
        --end "${END}" \
        --text_key "generated_text" \
        --model "gpt-3.5-turbo" \
        --num_workers 4 \
        --dataset "${DATA_NAME}"
fi

echo "============================================================"
echo " SAMark pipeline complete"
echo " Results: ${GEN_LOG_DIR}"
echo " Detect: ${GEN_LOG_DIR}/detect"
echo " Detect tanh: ${GEN_LOG_DIR}/detect_tanh${COSSIM_TRANSFORM_PARAM%.0}"
echo "============================================================"
