#!/bin/bash
# Run the paper's online-PMark comparison through the shared attack stages.
set -euo pipefail

W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
J="$W/scripts/experiments"
cd "$W"

BASE="${BASE:-data/c4-val-def-256}"
MODEL="meta-llama/Llama-3.1-8B"
EMBEDDER="sentence-transformers/all-mpnet-base-v2"
N=64
MSIG=4
TEMPERATURE=0.9
TOP_P=0.9
REPETITION_PENALTY=1.05
MAX_NEW_SENTENCES=12
SOFT_K=150
SOFT_DELTA=0.001
JUDGE="Qwen/Qwen3-32B"
CORPUS="data/c4-human-def"
ATTACKS_FILTER="${ATTACKS_FILTER:-}"   # regex; only attack labels matching it run (empty=all)

# shellcheck source=scripts/experiments/_lib.sh
source "$J/_lib.sh"   # -> enqueue/after_args/state/..., DRY_RUN, FORCE_*, QUALITY_BATCH
parse_force_flags "$@"
force_env=(); [[ "$FORCE_EVAL" == 1 ]] && force_env=(--env "FORCE=1")

PRIORITY="${PRIORITY:-10}"

ORACLE_KS=""
source "$J/attacks.sh"
mapfile -t ATTACKS < <(attacks_for_cell pmark sentence context | filter_attacks)

NJOBS=0; NSKIP=0
echo "=== PMark paper cell  base=$BASE mode=online N=$N  force[gen=$FORCE_GEN att=$FORCE_ATT det=$FORCE_DET eval=$FORCE_EVAL]  quality_batch=$QUALITY_BATCH  dry_run=$DRY_RUN ==="
  PBASE="$W/$BASE/pmark/online/rejection/sentence-nltk"
  PMARK_HF="$PBASE/watermarked"
  echo "--- root=${PBASE#$W/}  priority=$PRIORITY ---"

  gen_ex=$(has_glob "$PMARK_HF/*.arrow")
  do_gen=$(( FORCE_GEN || gen_ex == 0 ))
  gen_id=""; gen_dep=""
  if (( do_gen )); then
    gen_id=$(enqueue "$J/_pmark_gen.sh" --workdir "$W" \
              --env "LOG_DIR=$PMARK_HF" --env "BASE=$BASE" \
              --env "MODEL=$MODEL" --env "EMBEDDER=$EMBEDDER" --env "MSIG=$MSIG" \
              --env "N=$N" --env "TEMPERATURE=$TEMPERATURE" --env "TOP_P=$TOP_P" \
              --env "REPETITION_PENALTY=$REPETITION_PENALTY" \
              --env "MAX_NEW_SENTENCES=$MAX_NEW_SENTENCES"); NJOBS=$((NJOBS+1))
    gen_dep="$gen_id"
  else NSKIP=$((NSKIP+1)); fi

  # Score the unattacked cell once with PMark's online analytical null.
  wd_ex=$(is_file "$PMARK_HF/results.csv")
  do_wmdet=$(( do_gen || FORCE_DET || wd_ex == 0 ))
  wmdet_dep=""
  if (( do_wmdet )); then
    wmdet_dep=$(enqueue "$J/_pmark_detect.sh" --workdir "$W" $(after_args "$gen_dep") \
      --env "SAVE_DIR=$PMARK_HF" --env "MODEL=$MODEL" \
      --env "EMBEDDER=$EMBEDDER" --env "MSIG=$MSIG" --env "N=$N" \
      --env "TEMPERATURE=$TEMPERATURE" --env "TOP_P=$TOP_P" \
      --env "REPETITION_PENALTY=$REPETITION_PENALTY" --env "SOFT_K=$SOFT_K" \
      --env "SOFT_DELTA=$SOFT_DELTA")
    NJOBS=$((NJOBS+1))
  else NSKIP=$((NSKIP+1)); fi

  # Evaluate unattacked watermarked text against the base corpus.
  wq_ex=$(is_file "$PMARK_HF/eval_quality.csv")
  do_wmqua=$(( do_gen || FORCE_EVAL || wq_ex == 0 ))
  wmqua_dep=""
  if (( do_wmqua )); then
    wmqua_id=$(enqueue "$J/_quality_batch.sh" --workdir "$W" $(after_args "$gen_dep") \
      --env "DIRS=$PMARK_HF" --env "CONFIG=config/presets/sentence_none.yaml" \
      --env "SET=--set quality.reference=$BASE --set quality.judge_model=$JUDGE --set quality.corpus=$CORPUS --set quality.judge_len_prompt=100" \
      "${force_env[@]}")
    NJOBS=$((NJOBS+1)); wmqua_dep="$wmqua_id"
  else NSKIP=$((NSKIP+1)); fi

  # Attack-quality batch accumulators for the single online cell.
  # Probes batch separately: they run quality without the judges.
  QB_DIRS=(); QB_DEPS=(); QB_PROBE_DIRS=(); QB_PROBE_DEPS=()

  # Per-dir viz figure for the watermarked dir.
  wv_ex=$(is_file "$PMARK_HF/detection_summary.png")
  do_wmviz=$(( do_wmqua || wv_ex == 0 ))
  [[ "$QUALITY_BATCH" != off ]] && do_wmviz=0   # per-dir viz disabled under batching
  if (( do_wmviz )); then
    enqueue_cpu "$J/_viz.sh" --workdir "$W" $(after_args "$wmdet_dep" "$wmqua_dep") \
      --env "DIR=$PMARK_HF" >/dev/null; NJOBS=$((NJOBS+1))
  else NSKIP=$((NSKIP+1)); fi
  printf '  watermarked: gen=%s det=%s quality=%s viz=%s\n' \
    "$(state "$do_gen")" \
    "$(state "$do_wmdet")" \
    "$(state "$do_wmqua")" \
    "$(state "$do_wmviz")"

  for entry in "${ATTACKS[@]}"; do
    label=${entry%%|*}; atkset=${entry#*|}
    SAVE_DIR="$PBASE/$label"

    ax=$(has_glob "$SAVE_DIR/data*.arrow")
    dx=$(is_file "$SAVE_DIR/results.csv")
    qx=$(is_file "$SAVE_DIR/eval_quality.csv")
    do_att=$(( do_gen || FORCE_ATT  || ax == 0 ))
    do_det=$(( do_att || FORCE_DET  || dx == 0 ))
    do_qua=$(( do_att || FORCE_EVAL || qx == 0 ))
    # Probe atoms and their deterministic quality bundle are both CPU-only.
    atk_enq=enqueue; qua_enq=enqueue; qua_env=()
    qjudge="--set quality.judge_model=$JUDGE"; qlane=main
    if attack_is_probe "$label"; then
      atk_enq=enqueue_cpu; qua_enq=enqueue_cpu; qjudge=""; qlane=probe
      qua_env=(--env "CPU_ONLY=1")
    fi

    atk_id=""; atk_dep=""
    if (( do_att )); then
      atk_id=$($atk_enq "$J/_attack.sh" --workdir "$W" $(after_args "$gen_dep") \
                --env "BASE=$PMARK_HF" --env "CONFIG=config/presets/sentence_none.yaml" \
                --env "SET=--set io.output_dir=$SAVE_DIR $atkset"); NJOBS=$((NJOBS+1)); atk_dep="$atk_id"
    else NSKIP=$((NSKIP+1)); fi

    det_id=""; det_dep=""
    if (( do_det )); then
      det_id=$(enqueue "$J/_pmark_detect.sh" --workdir "$W" $(after_args "$atk_dep" "$wmdet_dep") \
        --env "SAVE_DIR=$SAVE_DIR" --env "MODEL=$MODEL" \
        --env "EMBEDDER=$EMBEDDER" --env "MSIG=$MSIG" --env "N=$N" \
        --env "TEMPERATURE=$TEMPERATURE" --env "TOP_P=$TOP_P" \
        --env "REPETITION_PENALTY=$REPETITION_PENALTY" --env "SOFT_K=$SOFT_K" \
        --env "SOFT_DELTA=$SOFT_DELTA")
      NJOBS=$((NJOBS+1)); det_dep="$det_id"
    else NSKIP=$((NSKIP+1)); fi

    if [[ "$QUALITY_BATCH" == off ]]; then
      qua_id=""; qua_dep=""
      if (( do_qua )); then
        qua_id=$($qua_enq "$J/_quality_batch.sh" --workdir "$W" $(after_args "$atk_dep") \
          --env "DIRS=$SAVE_DIR" --env "CONFIG=config/presets/sentence_none.yaml" \
          --env "SET=--set quality.reference=$PMARK_HF $qjudge --set quality.corpus=$CORPUS --set quality.judge_len_prompt=100" \
          "${qua_env[@]}" "${force_env[@]}")
        NJOBS=$((NJOBS+1)); qua_dep="$qua_id"
      else NSKIP=$((NSKIP+1)); fi

      # Per-dir viz figure, after this attack's detect AND quality (fresh results).
      vx=$(is_file "$SAVE_DIR/detection_summary.png")
      do_viz=$(( do_det || do_qua || vx == 0 ))
      if (( do_viz )); then
        enqueue_cpu "$J/_viz.sh" --workdir "$W" $(after_args "$det_dep" "$qua_dep") \
          --env "DIR=$SAVE_DIR" >/dev/null; NJOBS=$((NJOBS+1))
      else NSKIP=$((NSKIP+1)); fi
      qua_state=$(state "$do_qua")
      viz_state=$(state "$do_viz")
    else
      # Batch mode collects attack directories for one quality job; viz is off.
      if (( do_qua )); then
        if [[ "$qlane" == probe ]]; then
          QB_PROBE_DIRS+=("$SAVE_DIR"); [[ -n "$atk_dep" ]] && QB_PROBE_DEPS+=("$atk_dep")
        else
          QB_DIRS+=("$SAVE_DIR"); [[ -n "$atk_dep" ]] && QB_DEPS+=("$atk_dep")
        fi
        qua_state=batch
      else NSKIP=$((NSKIP+1)); qua_state=skip; fi
      viz_state=off
    fi

    printf '  %-14s atk=%s det=%s qua=%s viz=%s\n' "$label" \
      "$(state "$do_att")" \
      "$(state "$do_det")" \
      "$qua_state" "$viz_state"
  done

  # Run one full quality job and one CPU-only probe job for the paper cell.
  if [[ "$QUALITY_BATCH" != off ]]; then
    if (( ${#QB_DIRS[@]} > 0 )); then
      enqueue "$J/_quality_batch.sh" --workdir "$W" \
        $(after_args ${QB_DEPS[@]+"${QB_DEPS[@]}"}) \
        --env "DIRS=${QB_DIRS[*]}" --env "CONFIG=config/presets/sentence_none.yaml" \
        --env "SET=--set quality.reference=$PMARK_HF --set quality.judge_model=$JUDGE --set quality.corpus=$CORPUS --set quality.judge_len_prompt=100" \
        "${force_env[@]}" >/dev/null
      NJOBS=$((NJOBS+1))
      echo "  quality-batch[$QUALITY_BATCH]: ${#QB_DIRS[@]} attack dirs (deps=${#QB_DEPS[@]})"
    fi
    if (( ${#QB_PROBE_DIRS[@]} > 0 )); then
      enqueue_cpu "$J/_quality_batch.sh" --workdir "$W" \
        $(after_args ${QB_PROBE_DEPS[@]+"${QB_PROBE_DEPS[@]}"}) \
        --env "DIRS=${QB_PROBE_DIRS[*]}" --env "CONFIG=config/presets/sentence_none.yaml" \
        --env "SET=--set quality.reference=$PMARK_HF --set quality.corpus=$CORPUS --set quality.judge_len_prompt=100" \
        --env "CPU_ONLY=1" "${force_env[@]}" >/dev/null
      NJOBS=$((NJOBS+1))
      echo "  quality-batch[$QUALITY_BATCH,cpu-only]: ${#QB_PROBE_DIRS[@]} probe dirs (deps=${#QB_PROBE_DEPS[@]})"
    fi
  fi

print_footer "paper cell: online PMark, N=$N    attacks: ${#ATTACKS[@]}    quality_batch: $QUALITY_BATCH    enqueued: $NJOBS    skipped(present): $NSKIP"
