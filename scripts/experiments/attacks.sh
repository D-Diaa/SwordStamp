# Canonical attack matrix; each entry is "label|<--set tokens>".

# Adaptive templates using the paper's base paraphraser.
SEMQ="--set attack.paraphraser=adaptive --set attack.custom_model=Qwen/Qwen2.5-3B-Instruct --set attack.prompt_style=standard --set attack.surrogate_model=BAAI/bge-base-en-v1.5 --set attack.surrogate_tag=bge"
# Bag-anchor attack using base Qwen and the blind BGE surrogate.
BAGQ="--set attack.paraphraser=adaptive --set attack.custom_model=Qwen/Qwen2.5-3B-Instruct --set attack.prompt_style=standard --set attack.surrogate_model=BAAI/bge-base-en-v1.5 --set attack.surrogate_tag=bge --set attack.anchor=bag --set attack.bag_agg=min"
# Detector-aware upper bound. The oracle wraps the defender's own score function
# and inherits its segmentation, so it needs no surrogate: it bounds what any
# attacker holding the detector could achieve at a given candidate budget.
ORACLE="--set attack.paraphraser=oracle --set attack.custom_model=Qwen/Qwen2.5-3B-Instruct --set attack.prompt_style=standard"
ATTACKS=(
  # Passive paraphrasers and bigram-diversity variants.
  "pegasus|--set attack.paraphraser=pegasus --set attack.num_beams=25 --set attack.do_sample=False --set attack.batch_size=8"
  "parrot|--set attack.paraphraser=parrot --set attack.batch_size=8"
  "pegasus-bigram|--set attack.paraphraser=pegasus-bigram --set attack.num_beams=25 --set attack.do_sample=False --set attack.batch_size=8"
  "parrot-bigram|--set attack.paraphraser=parrot-bigram --set attack.batch_size=8"
)

# Sweep DIPPER lexical and order diversity on separate axes.
_dipper() {
  local L="$1" O="$2"
  ATTACKS+=("dipper-l$L-o$O|--set attack.paraphraser=dipper --set attack.dipper_lex=$L --set attack.dipper_order=$O --set attack.batch_size=8")
}
# The 60 points on each axis sat between neighbours that already bracket the
# trend, so they are dropped: five runs still span both axes end to end.
for O in 20 40 80; do
  _dipper 20 "$O"
done
for L in 40 80; do  # 20 already covered by the order sweep above
  _dipper "$L" 20
done

# Controlled probes reported in the paper: reorder and synonym substitution
# trace five-point response curves, while the split and merge unit-count
# controls use three levels.
PROBE_MAIN_RATIOS="${PROBE_MAIN_RATIOS:-0.1 0.25 0.5 0.75 1.0}"
PROBE_COUNT_RATIOS="${PROBE_COUNT_RATIOS:-0.2 0.5 1.0}"
PROBE_ATTACKS=()
_probe_ratio() {
  local label="$1" name="$2" ratios="$3" r
  for r in $ratios; do
    PROBE_ATTACKS+=("probe-$label-r$r|--set attack.paraphraser=$name --set attack.word_edit_ratio=$r")
  done
}
_probe_ratio reorder  controlled_reorder    "$PROBE_MAIN_RATIOS"
_probe_ratio split    split_midpoint        "$PROBE_COUNT_RATIOS"
_probe_ratio merge    merge_adjacent        "$PROBE_COUNT_RATIOS"
_probe_ratio syn      synonym_substitution  "$PROBE_MAIN_RATIOS"

# Probe attacks and their CPU quality bundle run without a GPU.
attack_is_probe() { [[ "$1" == probe-* ]]; }

# Full and representative adaptive-search budgets. K_FULL drives the attacker
# arm adapted to the cell; K_SUB drives arms that drop one adaptation and only
# need to bracket the observed peak.
K_FULL="1 2 4 8 16 32 64"
K_SUB="4 16 64"
# The SwordStamp runner supplies oracle budgets only for the two k-means cells
# reported under detector access. Comparison runners leave this empty.
ORACLE_KS="${ORACLE_KS:-}"

# Emit one adaptive arm at one budget.
#
# Each defender lever has an attacker counterpart: a stationary mask (fixed*) is
# answered by the bag anchor, span-level evidence by matched segmentation. Only
# the arm carrying every counterpart the cell calls for is "matched" and earns
# the dense K_FULL sweep; an arm that drops one counterpart is an ablation and
# only needs K_SUB to bracket the peak. Against a sentence defender there is no
# segmentation to match, so the base entry is already the matched one.
_emit_adaptive() {
  local label="$1" tmpl="$2" K="$3" seg="$4" matched="$5"
  if [[ "$matched" == 1 && "$seg" != sentence ]]; then
    printf '%s\n' "$label-K$K-$seg|$tmpl --set attack.num_candidates=$K --set segmentation.attacker_type=$seg"
  else
    printf '%s\n' "$label-K$K|$tmpl --set attack.num_candidates=$K"
  fi
}

# Expand exactly the attacks reported for one paper cell.
attacks_for_cell() {
  local family="$1" seg="$2" mask="$3" entry K
  for entry in "${ATTACKS[@]}"; do printf '%s\n' "$entry"; done

  # Controlled probes are reported across the SwordStamp ladders only.
  if [[ "$family" == lsh || "$family" == kmeans ]]; then
    for entry in "${PROBE_ATTACKS[@]}"; do printf '%s\n' "$entry"; done
  fi

  # Adaptive positional sweep. Under `context` the positional anchor *is* the
  # matched attacker, so it draws the dense sweep; under `fixed*` the bag anchor
  # below takes that role and positional becomes the anchor ablation.
  # `fixed*` covers fixed and fixed_diverse: both are constant-key deployment
  # targets and must draw the same attack set to stay comparable.
  if [[ "$mask" == fixed* ]]; then
    for K in $K_SUB; do
      _emit_adaptive adp "$SEMQ" "$K" "$seg" 0
    done
  else
    for K in $K_FULL; do _emit_adaptive adp "$SEMQ" "$K" "$seg" 1; done
  fi

  # Detector-aware upper bound. The oracle already runs on the defender's
  # segmentation, so it takes no matched-attacker variant.
  for K in $ORACLE_KS; do
    printf '%s\n' "oracle-K$K|$ORACLE --set attack.num_candidates=$K"
  done

  if [[ "$mask" == fixed* ]]; then
    # Fixed cells (constant-key deployment target): the bag anchor is the
    # mask-matched attacker and gets the dense sweep; dropping matched
    # segmentation leaves an ablation that only needs the subsample.
    for K in $K_FULL; do
      _emit_adaptive adpbag "$BAGQ" "$K" "$seg" 1
    done
  fi
}
