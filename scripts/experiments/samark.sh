#!/bin/bash
# Run the paper's single flags-run SAMark comparison through shared attacks.
set -euo pipefail

W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
J="$W/scripts/experiments"
cd "$W"

BASE="${BASE:-data/c4-val-def-256}"
MODEL="meta-llama/Llama-3.1-8B"
EMBEDDER="sentence-transformers/all-mpnet-base-v2"
MSIG=2
FLAG_SCOPE=run
NUM_SAMPLES=64
TEMPERATURE=0.9
TOP_P=0.9
REPETITION_PENALTY=1.05
EPSILON=80
NGRAM_THRESH=0.5
SEM_THRESH=0.85
DIV_WEIGHT=0.3
NOV_WEIGHT=0.15
TRANSFORM=tanh
TRANSFORM_PARAM=30.0
# Match the human-null corpus used by other detectors.
HUMAN_TEXT="data/c4-human-def"
HUMAN_N=1024
HUMAN_MAX_SENTENCES=12
FPR_TARGET=0.01
JUDGE="Qwen/Qwen3-32B"
CORPUS="data/c4-human-def"
ATTACKS_FILTER="${ATTACKS_FILTER:-}"
PRIORITY="${PRIORITY:-10}"

# shellcheck source=scripts/experiments/_lib.sh
source "$J/_lib.sh"   # -> enqueue/after_args/state/..., DRY_RUN, FORCE_*, QUALITY_BATCH
parse_force_flags "$@"
force_env=(); [[ "$FORCE_EVAL" == 1 ]] && force_env=(--env "FORCE=1")

ORACLE_KS=""
source "$J/attacks.sh"
mapfile -t ATTACKS < <(attacks_for_cell samark sentence fixed | filter_attacks)

NJOBS=0; NSKIP=0
echo "=== SAMark paper cell  base=$BASE flags=run msig=$MSIG N=$NUM_SAMPLES  force[gen=$FORCE_GEN att=$FORCE_ATT det=$FORCE_DET eval=$FORCE_EVAL]  quality_batch=$QUALITY_BATCH  dry_run=$DRY_RUN ==="

  SBASE="$W/$BASE/samark/flags-$FLAG_SCOPE/msig$MSIG/rejection/sentence-nltk"
  SAMARK_HF="$SBASE/watermarked"
  echo "--- msig=$MSIG  root=${SBASE#$W/}  prio=$PRIORITY ---"

  gen_ex=$(has_glob "$SAMARK_HF/*.arrow")
  do_gen=$(( FORCE_GEN || gen_ex == 0 ))
  gen_dep=""
  if (( do_gen )); then
    gen_dep=$(enqueue "$J/_samark_gen.sh" --workdir "$W" \
      --env "LOG_DIR=$SAMARK_HF" --env "BASE=$BASE" --env "MODEL=$MODEL" \
      --env "EMBEDDER=$EMBEDDER" --env "MSIG=$MSIG" --env "NUM_SAMPLES=$NUM_SAMPLES" \
      --env "FLAG_SCOPE=$FLAG_SCOPE" --env "TEMPERATURE=$TEMPERATURE" \
      --env "TOP_P=$TOP_P" --env "REPETITION_PENALTY=$REPETITION_PENALTY" \
      --env "EPSILON=$EPSILON" --env "NGRAM_THRESH=$NGRAM_THRESH" \
      --env "SEM_THRESH=$SEM_THRESH" --env "DIV_WEIGHT=$DIV_WEIGHT" \
      --env "NOV_WEIGHT=$NOV_WEIGHT" --env "COSSIM_TRANSFORM=$TRANSFORM" \
      --env "COSSIM_TRANSFORM_PARAM=$TRANSFORM_PARAM"); NJOBS=$((NJOBS+1))
  else NSKIP=$((NSKIP+1)); fi

  # Compute the unattacked ROC and TPR from the independent human null.
  wd_ex=$(is_file "$SAMARK_HF/detect/results.csv")
  do_wmdet=$(( do_gen || FORCE_DET || wd_ex == 0 ))
  if (( do_wmdet )); then
    enqueue "$J/_samark_detect.sh" --workdir "$W" \
      $(after_args "$gen_dep") \
      --env "HF_DATASET=$SAMARK_HF" --env "EMBEDDER=$EMBEDDER" --env "MSIG=$MSIG" \
      --env "TRANSFORM=$TRANSFORM" --env "TRANSFORM_PARAM=$TRANSFORM_PARAM" \
      --env "HUMAN_TEXT=$HUMAN_TEXT" --env "HUMAN_N=$HUMAN_N" \
      --env "HUMAN_MAX_SENTENCES=$HUMAN_MAX_SENTENCES" \
      --env "FPR_TARGET=$FPR_TARGET" >/dev/null
    NJOBS=$((NJOBS+1))
  else NSKIP=$((NSKIP+1)); fi

  # Evaluate unattacked watermarked text against the base corpus.
  wq_ex=$(is_file "$SAMARK_HF/eval_quality.csv")
  do_wmqua=$(( do_gen || FORCE_EVAL || wq_ex == 0 ))
  if (( do_wmqua )); then
    enqueue "$J/_quality_batch.sh" --workdir "$W" $(after_args "$gen_dep") \
      --env "DIRS=$SAMARK_HF" --env "CONFIG=config/presets/sentence_none.yaml" \
      --env "SET=--set quality.reference=$BASE --set quality.judge_model=$JUDGE --set quality.corpus=$CORPUS --set quality.judge_len_prompt=100" \
      "${force_env[@]}" >/dev/null
    NJOBS=$((NJOBS+1))
  else NSKIP=$((NSKIP+1)); fi
  printf '  watermarked: gen=%s detect=%s quality=%s\n' \
    "$(state "$do_gen")" \
    "$(state "$do_wmdet")" \
    "$(state "$do_wmqua")"

  # Attack-quality batch accumulators for the single paper cell.
  # Probes batch separately: they run quality without the judges.
  QB_DIRS=(); QB_DEPS=(); QB_PROBE_DIRS=(); QB_PROBE_DEPS=()

  for entry in "${ATTACKS[@]}"; do
    label=${entry%%|*}; atkset=${entry#*|}
    SAVE_DIR="$SBASE/$label"

    ax=$(has_glob "$SAVE_DIR/data*.arrow")
    dx=$(is_file "$SAVE_DIR/detect/results.csv")
    qx=$(is_file "$SAVE_DIR/eval_quality.csv")
    do_att=$(( do_gen || FORCE_ATT  || ax == 0 ))
    do_det=$(( do_att || FORCE_DET  || dx == 0 ))
    do_qua=$(( do_att || FORCE_EVAL || qx == 0 ))

    atk_enq=enqueue
    # Probe atoms and their deterministic quality bundle are both CPU-only.
    qua_enq=enqueue; qua_env=()
    qjudge="--set quality.judge_model=$JUDGE"; qlane=main
    if attack_is_probe "$label"; then
      atk_enq=enqueue_cpu; qua_enq=enqueue_cpu; qjudge=""; qlane=probe
      qua_env=(--env "CPU_ONLY=1")
    fi

    atk_id=""; atk_dep=""
    if (( do_att )); then
      atk_id=$($atk_enq "$J/_attack.sh" --workdir "$W" $(after_args "$gen_dep") \
                --env "BASE=$SAMARK_HF" --env "CONFIG=config/presets/sentence_none.yaml" \
                --env "SET=--set io.output_dir=$SAVE_DIR $atkset"); NJOBS=$((NJOBS+1)); atk_dep="$atk_id"
    else NSKIP=$((NSKIP+1)); fi

    if (( do_det )); then
      enqueue "$J/_samark_detect.sh" --workdir "$W" \
        $(after_args "$atk_dep") \
        --env "HF_DATASET=$SAVE_DIR" --env "EMBEDDER=$EMBEDDER" --env "MSIG=$MSIG" \
        --env "TRANSFORM=$TRANSFORM" --env "TRANSFORM_PARAM=$TRANSFORM_PARAM" \
        --env "HUMAN_TEXT=$HUMAN_TEXT" --env "HUMAN_N=$HUMAN_N" \
        --env "HUMAN_MAX_SENTENCES=$HUMAN_MAX_SENTENCES" \
        --env "FPR_TARGET=$FPR_TARGET" >/dev/null
      NJOBS=$((NJOBS+1))
    else NSKIP=$((NSKIP+1)); fi

    if [[ "$QUALITY_BATCH" == off ]]; then
      if (( do_qua )); then
        $qua_enq "$J/_quality_batch.sh" --workdir "$W" $(after_args "$atk_dep") \
          --env "DIRS=$SAVE_DIR" --env "CONFIG=config/presets/sentence_none.yaml" \
          --env "SET=--set quality.reference=$SAMARK_HF $qjudge --set quality.corpus=$CORPUS --set quality.judge_len_prompt=100" \
          "${qua_env[@]}" "${force_env[@]}" >/dev/null
        NJOBS=$((NJOBS+1)); qua_state=run
      else NSKIP=$((NSKIP+1)); qua_state=skip; fi
    else
      # batch mode: collect this attack dir (+ its attack dep) for one per-msig quality job
      if (( do_qua )); then
        if [[ "$qlane" == probe ]]; then
          QB_PROBE_DIRS+=("$SAVE_DIR"); [[ -n "$atk_dep" ]] && QB_PROBE_DEPS+=("$atk_dep")
        else
          QB_DIRS+=("$SAVE_DIR"); [[ -n "$atk_dep" ]] && QB_DEPS+=("$atk_dep")
        fi
        qua_state=batch
      else NSKIP=$((NSKIP+1)); qua_state=skip; fi
    fi

    printf '  %-14s atk=%s det=%s qua=%s\n' "$label" \
      "$(state "$do_att")" \
      "$(state "$do_det")" \
      "$qua_state"
  done

  # Run one full quality job and one CPU-only probe job for the paper cell.
  if [[ "$QUALITY_BATCH" != off ]]; then
    if (( ${#QB_DIRS[@]} > 0 )); then
      enqueue "$J/_quality_batch.sh" --workdir "$W" \
        $(after_args ${QB_DEPS[@]+"${QB_DEPS[@]}"}) \
        --env "DIRS=${QB_DIRS[*]}" --env "CONFIG=config/presets/sentence_none.yaml" \
        --env "SET=--set quality.reference=$SAMARK_HF --set quality.judge_model=$JUDGE --set quality.corpus=$CORPUS --set quality.judge_len_prompt=100" \
        "${force_env[@]}" >/dev/null
      NJOBS=$((NJOBS+1))
      echo "  quality-batch[$QUALITY_BATCH]: ${#QB_DIRS[@]} attack dirs (deps=${#QB_DEPS[@]})"
    fi
    if (( ${#QB_PROBE_DIRS[@]} > 0 )); then
      enqueue_cpu "$J/_quality_batch.sh" --workdir "$W" \
        $(after_args ${QB_PROBE_DEPS[@]+"${QB_PROBE_DEPS[@]}"}) \
        --env "DIRS=${QB_PROBE_DIRS[*]}" --env "CONFIG=config/presets/sentence_none.yaml" \
        --env "SET=--set quality.reference=$SAMARK_HF --set quality.corpus=$CORPUS --set quality.judge_len_prompt=100" \
        --env "CPU_ONLY=1" "${force_env[@]}" >/dev/null
      NJOBS=$((NJOBS+1))
      echo "  quality-batch[$QUALITY_BATCH,cpu-only]: ${#QB_PROBE_DIRS[@]} probe dirs (deps=${#QB_PROBE_DEPS[@]})"
    fi
  fi

print_footer "paper cell: flags-run SAMark, msig=$MSIG, N=$NUM_SAMPLES    attacks: ${#ATTACKS[@]}    quality_batch: $QUALITY_BATCH    enqueued: $NJOBS    skipped(present): $NSKIP"
