#!/bin/bash
# Run the exact ten-cell SemStamp/k-SemStamp paper ladder plus its baseline.
set -euo pipefail

W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
J="$W/scripts/experiments"
cd "$W"

BASE="${BASE:-data/c4-val-def-256}"
JUDGE="Qwen/Qwen3-32B"
CORPUS="data/c4-human-def"
ATTACKS_FILTER="${ATTACKS_FILTER:-}"
PRIORITY="${PRIORITY:-0}"
AFTER="${AFTER:-}"
TASK_IDS_FILE="${TASK_IDS_FILE:-}"

# shellcheck source=scripts/experiments/_lib.sh
source "$J/_lib.sh"   # -> enqueue/after_args/state/..., DRY_RUN, FORCE_*, QUALITY_BATCH
parse_force_flags "$@"
force_env=(); [[ "$FORCE_EVAL" == 1 ]] && force_env=(--env "FORCE=1")

JQ="--set quality.judge_model=$JUDGE --set quality.corpus=$CORPUS"
# Probe atoms run only the CPU quality bundle: deterministic diversity,
# BLEU/ROUGE, and anchor channels. They never reserve a GPU or load a judge.
JQ_PROBE="--set quality.corpus=$CORPUS"

# Detector access is the paper's explicit K={4,8,16,32,64} upper bound.
# attacks.sh remains the shared attack-suite definition; only this runner opts
# the oracle cells in.
PAPER_ORACLE_KS=$(uv run python "$J/paper.py" oracle-ks)
ORACLE_KS=""

# shellcheck source=scripts/experiments/attacks.sh
source "$J/attacks.sh"   # -> ATTACKS=( "label|--set ..." ... )

PAPER_RUNG_TSV=$(uv run python "$J/paper.py" rungs)
mapfile -t PAPER_ROWS <<< "$PAPER_RUNG_TSV"
if (( ${#PAPER_ROWS[@]} != 10 )); then
  echo "paper registry must contain exactly 10 watermark rungs" >&2
  exit 2
fi

NJOBS=0; NSKIP=0; NCELLS=0
# Accumulate all-run quality inputs; quality settings are preset-invariant.
# Probes are batched separately because they run without the judges.
QB_ALL_DIRS=(); QB_ALL_DEPS=(); QB_ALL_CONFIG=""
QB_ALL_PROBE_DIRS=(); QB_ALL_PROBE_DEPS=()
echo "=== SwordStamp paper ladder  base=$BASE  force[gen=$FORCE_GEN att=$FORCE_ATT det=$FORCE_DET eval=$FORCE_EVAL]  quality_batch=$QUALITY_BATCH  dry_run=$DRY_RUN ==="

for row in "${PAPER_ROWS[@]}"; do
    IFS=$'\t' read -r RUNG fam mask samp segmentation CONFIG NUM_CANDIDATES <<< "$row"
    [[ -f "$W/$CONFIG" ]] || { echo "missing paper preset: $CONFIG" >&2; exit 2; }
    case "$segmentation" in
      sentence-*) SEG=sentence ;;
      semspan-*) SEG=semspan ;;
      *) echo "unsupported paper segmentation: $segmentation" >&2; exit 2 ;;
    esac
    GEN_SET="--set generation.num_candidates=$NUM_CANDIDATES"
    NCELLS=$((NCELLS + 1))
    echo "--- rung $RUNG: $fam/$mask/$samp/$segmentation ($CONFIG) ---"

    # EDA-D is reported only for k-SemStamp and k-SwordStamp.
    case "$RUNG" in
      kbase|kspan) ORACLE_KS="$PAPER_ORACLE_KS" ;;
      *) ORACLE_KS="" ;;
    esac
    mapfile -t sel < <(
      attacks_for_cell "$fam" "$SEG" "$mask" | filter_attacks
    )

    # One planner process resolves the cell + every attack dir and reports existence.
    declare -A ATT_EX DET_EX QUA_EX ATT_DIR
    ATT_EX=(); DET_EX=(); QUA_EX=(); ATT_DIR=()
    wm_dir=""; gen_ex=0; detwm_ex=0; quawm_ex=0
    while IFS=$'\t' read -r kind c1 c2 c3 c4 c5; do
      case "$kind" in
        CELL)   wm_dir="$c1"; gen_ex="$c2"; detwm_ex="$c3"; quawm_ex="$c4" ;;
        ATTACK) ATT_DIR["$c1"]="$c2"; ATT_EX["$c1"]="$c3"; DET_EX["$c1"]="$c4"; QUA_EX["$c1"]="$c5" ;;
      esac
    done < <(
      for entry in "${sel[@]}"; do printf '%s\t%s\n' "${entry%%|*}" "${entry#*|}"; done \
        | uv run python "$J/_plan.py" "$BASE" "$CONFIG" $GEN_SET
    )

    # viz presence (detection_summary figure) is checked directly off the dir.
    vizwm_ex=$(is_file "$wm_dir/detection_summary.png")

    do_gen=$((   FORCE_GEN  || gen_ex   == 0 ))
    do_detwm=$(( do_gen || FORCE_DET  || detwm_ex == 0 ))
    do_quawm=$(( do_gen || FORCE_EVAL || quawm_ex == 0 ))
    do_vizwm=$(( do_detwm || do_quawm || vizwm_ex == 0 ))
    [[ "$QUALITY_BATCH" != off ]] && do_vizwm=0   # per-dir viz disabled under batching
    # per-cell attack-quality batch accumulators (QUALITY_BATCH=cell)
    QB_DIRS=(); QB_DEPS=(); QB_PROBE_DIRS=(); QB_PROBE_DEPS=()

    gen_id=""; detwm_id=""; quawm_id=""
    if (( do_gen )); then
      gen_id=$(enqueue "$J/_gen.sh" --workdir "$W" \
                 --env "BASE=$BASE" --env "CONFIG=$CONFIG" --env "SET=$GEN_SET"); NJOBS=$((NJOBS+1))
    else NSKIP=$((NSKIP+1)); fi
    gen_dep=""; (( do_gen )) && gen_dep="$gen_id"

    if (( do_detwm )); then
      detwm_id=$(enqueue "$J/_detect.sh" --workdir "$W" $(after_args "$gen_dep") \
                   --env "BASE=$BASE" --env "CONFIG=$CONFIG" --env "SET=$GEN_SET"); NJOBS=$((NJOBS+1))
    else NSKIP=$((NSKIP+1)); fi
    detwm_dep=""; (( do_detwm )) && detwm_dep="$detwm_id"

    if (( do_quawm )); then
      quawm_id=$(enqueue "$J/_quality_batch.sh" --workdir "$W" $(after_args "$gen_dep") \
                   --env "DIRS=$wm_dir" --env "CONFIG=$CONFIG" \
                   --env "SET=$JQ --set quality.reference=$BASE $GEN_SET" "${force_env[@]}"); NJOBS=$((NJOBS+1))
    else NSKIP=$((NSKIP+1)); fi
    quawm_dep=""; (( do_quawm )) && quawm_dep="$quawm_id"

    if (( do_vizwm )); then
      enqueue_cpu "$J/_viz.sh" --workdir "$W" $(after_args "$detwm_dep" "$quawm_dep") \
        --env "DIR=$wm_dir" >/dev/null; NJOBS=$((NJOBS+1))
    else NSKIP=$((NSKIP+1)); fi
    printf '  watermarked: gen=%s detect=%s quality=%s viz=%s\n' \
      "$(state "$do_gen")" "$(state "$do_detwm")" \
      "$(state "$do_quawm")" "$(state "$do_vizwm")"

    for entry in "${sel[@]}"; do
      label=${entry%%|*}; atkset=${entry#*|}
      adir="${ATT_DIR[$label]:-}"
      ax="${ATT_EX[$label]:-0}"; dx="${DET_EX[$label]:-0}"; qx="${QUA_EX[$label]:-0}"
      vx=$(is_file "$adir/detection_summary.png")

      do_att=$((   do_gen || FORCE_ATT  || ax == 0 ))
      do_detatt=$((do_att || FORCE_DET  || dx == 0 ))
      do_quaatt=$((do_att || FORCE_EVAL || qx == 0 ))
      do_vizatt=$((do_detatt || do_quaatt || vx == 0 ))
      # Probes use a separate CPU-only quality lane.
      qset="$JQ"; qlane=main; qua_enq=enqueue; qua_env=()
      if attack_is_probe "$label"; then
        qset="$JQ_PROBE"; qlane=probe; qua_enq=enqueue_cpu
        qua_env=(--env "CPU_ONLY=1")
      fi

      # Probe atoms are pure string edits and need no GPU.
      atk_enq=enqueue
      if attack_is_probe "$label"; then atk_enq=enqueue_cpu; fi

      atk_id=""; det_id=""; qua_id=""
      if (( do_att )); then
        atk_id=$($atk_enq "$J/_attack.sh" --workdir "$W" $(after_args "$gen_dep") \
                   --env "BASE=$BASE" --env "CONFIG=$CONFIG" --env "SET=$atkset $GEN_SET"); NJOBS=$((NJOBS+1))
      else NSKIP=$((NSKIP+1)); fi
      atk_dep=""; (( do_att )) && atk_dep="$atk_id"

      if (( do_detatt )); then
        det_id=$(enqueue "$J/_detect.sh" --workdir "$W" $(after_args "$atk_dep" "$detwm_dep") \
          --env "BASE=$BASE" --env "CONFIG=$CONFIG" --env "SET=$atkset --set io.target=attack $GEN_SET"); NJOBS=$((NJOBS+1))
      else NSKIP=$((NSKIP+1)); fi
      det_dep=""; (( do_detatt )) && det_dep="$det_id"

      if [[ "$QUALITY_BATCH" == off ]]; then
        if (( do_quaatt )); then
          qua_id=$($qua_enq "$J/_quality_batch.sh" --workdir "$W" $(after_args "$atk_dep") \
            --env "DIRS=$adir" --env "CONFIG=$CONFIG" --env "SET=$qset $GEN_SET" \
            "${qua_env[@]}" "${force_env[@]}"); NJOBS=$((NJOBS+1))
        else NSKIP=$((NSKIP+1)); fi
        qua_dep=""; (( do_quaatt )) && qua_dep="$qua_id"
        if (( do_vizatt )); then
          enqueue_cpu "$J/_viz.sh" --workdir "$W" $(after_args "$det_dep" "$qua_dep") \
            --env "DIR=$adir" >/dev/null; NJOBS=$((NJOBS+1))
        else NSKIP=$((NSKIP+1)); fi
        qua_state=$(state "$do_quaatt")
        viz_state=$(state "$do_vizatt")
      else
        # Batch quality inputs and disable per-directory plots.
        if (( do_quaatt )); then
          if [[ "$QUALITY_BATCH" == all ]]; then
            if [[ "$qlane" == probe ]]; then
              QB_ALL_PROBE_DIRS+=("$adir"); [[ -n "$atk_dep" ]] && QB_ALL_PROBE_DEPS+=("$atk_dep")
            else
              QB_ALL_DIRS+=("$adir"); [[ -n "$atk_dep" ]] && QB_ALL_DEPS+=("$atk_dep")
            fi
            [[ -z "$QB_ALL_CONFIG" ]] && QB_ALL_CONFIG="$CONFIG"
          elif [[ "$qlane" == probe ]]; then
            QB_PROBE_DIRS+=("$adir"); [[ -n "$atk_dep" ]] && QB_PROBE_DEPS+=("$atk_dep")
          else
            QB_DIRS+=("$adir"); [[ -n "$atk_dep" ]] && QB_DEPS+=("$atk_dep")
          fi
          qua_state=batch
        else NSKIP=$((NSKIP+1)); qua_state=skip; fi
        viz_state=off
      fi

      printf '  %-18s atk=%s det=%s qua=%s viz=%s\n' "$label" \
        "$(state "$do_att")" "$(state "$do_detatt")" \
        "$qua_state" "$viz_state"
    done

    # Run one full quality job and one CPU-only probe job per cell.
    if [[ "$QUALITY_BATCH" == cell ]]; then
      if (( ${#QB_DIRS[@]} > 0 )); then
        enqueue "$J/_quality_batch.sh" --workdir "$W" \
          $(after_args ${QB_DEPS[@]+"${QB_DEPS[@]}"}) \
          --env "DIRS=${QB_DIRS[*]}" --env "CONFIG=$CONFIG" --env "SET=$JQ $GEN_SET" "${force_env[@]}" >/dev/null
        NJOBS=$((NJOBS+1))
        echo "  quality-batch[cell]: ${#QB_DIRS[@]} attack dirs (deps=${#QB_DEPS[@]})"
      fi
      if (( ${#QB_PROBE_DIRS[@]} > 0 )); then
        enqueue_cpu "$J/_quality_batch.sh" --workdir "$W" \
          $(after_args ${QB_PROBE_DEPS[@]+"${QB_PROBE_DEPS[@]}"}) \
          --env "DIRS=${QB_PROBE_DIRS[*]}" --env "CONFIG=$CONFIG" \
          --env "SET=$JQ_PROBE $GEN_SET" --env "CPU_ONLY=1" \
          "${force_env[@]}" >/dev/null
        NJOBS=$((NJOBS+1))
        echo "  quality-batch[cell,cpu-only]: ${#QB_PROBE_DIRS[@]} probe dirs (deps=${#QB_PROBE_DEPS[@]})"
      fi
    fi
    unset ATT_EX DET_EX QUA_EX ATT_DIR
done

# Generate the single sentence-segmented no-watermark quality baseline.
CONFIG_NONE=$(uv run python "$J/paper.py" baseline)
if [[ ! -f "$W/$CONFIG_NONE" ]]; then
  echo "missing paper baseline preset: $CONFIG_NONE" >&2
  exit 2
else
    NCELLS=$((NCELLS + 1))
    none_dir=$(uv run python -m config.paths watermark-dir "$BASE" --config "$CONFIG_NONE")
    echo "--- none baseline ($CONFIG_NONE) -> $none_dir ---"
    ng_ex=$(has_glob "$none_dir/data*.arrow")
    nq_ex=$(is_file "$none_dir/eval_quality.csv")
    nv_ex=$(is_file "$none_dir/detection_summary.png")
    do_ng=$(( FORCE_GEN || ng_ex == 0 ))
    do_nq=$(( do_ng || FORCE_EVAL || nq_ex == 0 ))
    do_nv=$(( do_nq || nv_ex == 0 ))
    [[ "$QUALITY_BATCH" != off ]] && do_nv=0   # per-dir viz disabled under batching
    ng_id=""; ng_dep=""
    if (( do_ng )); then
      ng_id=$(enqueue "$J/_gen.sh" --workdir "$W" \
                --env "BASE=$BASE" --env "CONFIG=$CONFIG_NONE" --env "SET="); NJOBS=$((NJOBS+1)); ng_dep="$ng_id"
    else NSKIP=$((NSKIP+1)); fi
    nq_id=""; nq_dep=""
    if (( do_nq )); then
      nq_id=$(enqueue "$J/_quality_batch.sh" --workdir "$W" $(after_args "$ng_dep") \
                --env "DIRS=$none_dir" --env "CONFIG=$CONFIG_NONE" \
                --env "SET=$JQ --set quality.reference=$BASE" "${force_env[@]}"); NJOBS=$((NJOBS+1)); nq_dep="$nq_id"
    else NSKIP=$((NSKIP+1)); fi
    if (( do_nv )); then
      enqueue_cpu "$J/_viz.sh" --workdir "$W" $(after_args "$nq_dep") --env "DIR=$none_dir" >/dev/null; NJOBS=$((NJOBS+1))
    else NSKIP=$((NSKIP+1)); fi
    printf '  none: gen=%s quality=%s viz=%s (no detection — no watermark)\n' \
      "$(state "$do_ng")" "$(state "$do_nq")" "$(state "$do_nv")"
fi

# Run one full quality job for ordinary attacks and one CPU-only probe job.
if [[ "$QUALITY_BATCH" == all ]]; then
  if (( ${#QB_ALL_DIRS[@]} > 0 )); then
    enqueue "$J/_quality_batch.sh" --workdir "$W" \
      $(after_args ${QB_ALL_DEPS[@]+"${QB_ALL_DEPS[@]}"}) \
      --env "DIRS=${QB_ALL_DIRS[*]}" --env "CONFIG=$QB_ALL_CONFIG" --env "SET=$JQ $GEN_SET" "${force_env[@]}" >/dev/null
    NJOBS=$((NJOBS+1))
    echo "quality-batch[all]: ${#QB_ALL_DIRS[@]} attack dirs (deps=${#QB_ALL_DEPS[@]})"
  fi
  if (( ${#QB_ALL_PROBE_DIRS[@]} > 0 )); then
    enqueue_cpu "$J/_quality_batch.sh" --workdir "$W" \
      $(after_args ${QB_ALL_PROBE_DEPS[@]+"${QB_ALL_PROBE_DEPS[@]}"}) \
      --env "DIRS=${QB_ALL_PROBE_DIRS[*]}" --env "CONFIG=$QB_ALL_CONFIG" \
      --env "SET=$JQ_PROBE $GEN_SET" --env "CPU_ONLY=1" \
      "${force_env[@]}" >/dev/null
    NJOBS=$((NJOBS+1))
    echo "quality-batch[all,cpu-only]: ${#QB_ALL_PROBE_DIRS[@]} probe dirs (deps=${#QB_ALL_PROBE_DEPS[@]})"
  fi
fi

print_footer "paper cells: $NCELLS (10 rungs + baseline)${ATTACKS_FILTER:+    attack filter: $ATTACKS_FILTER}    quality_batch: $QUALITY_BATCH    enqueued: $NJOBS    skipped(present): $NSKIP"
