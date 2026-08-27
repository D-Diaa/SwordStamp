# Shared orchestrator helpers for swordstamp.sh, pmark.sh, and samark.sh.
#
# Source this after the script's own defaults; it fills in the knobs every
# orchestrator shares and brings in the GPU policy. Reads DRY_RUN, PRIORITY,
# AFTER, TASK_IDS_FILE, ATTACKS_FILTER, UV_NO_SYNC,
# SEMSTAMP__RUNTIME__VLLM_UTILIZATION, and VLLM_MAX_MODEL_LEN.

# shellcheck source=scripts/experiments/_gpu_policy.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_gpu_policy.sh"   # -> needs_whole_gpu()

DRY_RUN="${DRY_RUN:-0}"
FORCE_GEN="${FORCE_GEN:-0}"; FORCE_ATT="${FORCE_ATT:-0}"
FORCE_DET="${FORCE_DET:-0}"; FORCE_EVAL="${FORCE_EVAL:-0}"
# Per-directory quality jobs, one model-reusing job per cell, or one per run.
# Each orchestrator decides what "cell" means for its own grid.
QUALITY_BATCH="${QUALITY_BATCH:-cell}"
case "$QUALITY_BATCH" in off|cell|all) ;; *) echo "bad QUALITY_BATCH=$QUALITY_BATCH (off|cell|all)" >&2; exit 2 ;; esac

# Knobs are env vars; the command line carries only the force flags.
parse_force_flags() {
  local a
  for a in "$@"; do
    case "$a" in
      --force-gen)  FORCE_GEN=1 ;;
      --force-att)  FORCE_ATT=1 ;;
      --force-det)  FORCE_DET=1 ;;
      --force-eval) FORCE_EVAL=1 ;;
      *) echo "unknown arg: $a (knobs are env vars; flags are --force-{gen,att,det,eval})" >&2; exit 2 ;;
    esac
  done
}

# Optionally record task IDs for an external completion barrier.
if [[ -n "${TASK_IDS_FILE:-}" && "$DRY_RUN" != 1 ]]; then
  mkdir -p "$(dirname "$TASK_IDS_FILE")"
  : > "$TASK_IDS_FILE"
fi
record_task_id() {
  [[ -z "${TASK_IDS_FILE:-}" ]] || printf '%s\n' "$1" >> "$TASK_IDS_FILE"
}

# Enqueue one worker and echo its task id. `enqueue` takes a GPU slot — a whole
# GPU for vLLM workers, per _gpu_policy.sh — and forwards the vLLM env;
# `enqueue_cpu` reserves nothing. Both honor DRY_RUN, the global AFTER barrier,
# PRIORITY, and TASK_IDS_FILE.
_enqueue() {
  local kind="$1"; shift
  if [[ "$DRY_RUN" == 1 ]]; then echo "DRYRUN-$RANDOM"; return; fi
  local extra=() task_id
  [[ -z "${AFTER:-}" ]] || extra+=(--after "$AFTER")
  if [[ "$kind" == cpu ]]; then
    extra+=(--cpu)
  else
    if needs_whole_gpu "$@"; then extra+=(--exclusive); fi
    [[ -z "${SEMSTAMP__RUNTIME__VLLM_UTILIZATION:-}" ]] || \
      extra+=(--env "SEMSTAMP__RUNTIME__VLLM_UTILIZATION=$SEMSTAMP__RUNTIME__VLLM_UTILIZATION")
    [[ -z "${VLLM_MAX_MODEL_LEN:-}" ]] || extra+=(--env "VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN")
  fi
  task_id=$(gpu-enqueue "$@" "${extra[@]}" \
    --env "UV_NO_SYNC=${UV_NO_SYNC:-1}" --priority "${PRIORITY:-0}") || return
  record_task_id "$task_id"
  echo "$task_id"
}
enqueue()     { _enqueue gpu "$@"; }
enqueue_cpu() { _enqueue cpu "$@"; }

# Emit "--after a,b" for the non-empty dep ids; nothing if all empty.
after_args() {
    local deps=() d
    for d in "$@"; do [[ -n "$d" ]] && deps+=("$d"); done
    (( ${#deps[@]} )) || return 0
    local IFS=,; echo "--after ${deps[*]}"
}

has_glob() { compgen -G "$1" >/dev/null 2>&1 && echo 1 || echo 0; }   # 1 if glob matches
is_file()  { [[ -f "$1" ]] && echo 1 || echo 0; }
state()    { (( $1 )) && echo run || echo skip; }   # stage decision, for the progress lines

# Drop "label|--set ..." entries on stdin whose label ATTACKS_FILTER rejects.
filter_attacks() {
  local entry
  while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    [[ -n "${ATTACKS_FILTER:-}" && ! "${entry%%|*}" =~ $ATTACKS_FILTER ]] && continue
    printf '%s\n' "$entry"
  done
}

print_footer() {   # $1 = the run's one-line tally
  echo ""
  echo "==========================================================="
  echo "$1"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY RUN — nothing was enqueued. Re-run without DRY_RUN=1 to submit."
  else
    echo "Submitted. Monitor with: gpu-status"
  fi
  echo "==========================================================="
}
