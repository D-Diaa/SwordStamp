# Attacks

The attack CLI reads the config-derived watermarked dataset and writes a sibling
dataset containing `text` (watermarked source) and `para_text` (attacked output):

```bash
uv run python -m attacks DATA_PATH --config PRESET \
  --set attack.paraphraser=dipper
```

`config.paths.attack_spec` owns output naming; do not construct attack leaves by
hand. `io.output_dir` is available for an explicit noncanonical destination.

## Package layout

Implementations are grouped by how they transform text:

| Package | Attacks | Boundary behavior |
| --- | --- | --- |
| `sentence_level/` | PEGASUS, Parrot, OpenAI | Rewrite one sentence at a time; optional bigram-diversity selection. |
| `simple/` | Back translation, word deletion, WordNet substitution, contextual substitution | Apply translation or direct lexical edits. |
| `simple/structural.py` | Single-channel probe atoms | Edit segment boundaries, order, or content volume directly. |
| `paraphrasing/` | Custom chat rewrites, DIPPER, adaptive and oracle search | Rewrite documents or search model-generated paraphrases. |

Shared runtime/result contracts live in `base.py`; common sentence and diversity
helpers live in `utils.py`. Every implementation exposes `run_attack(texts,
AttackConfig)`. CLI aliases needing another function (`custom_sent` and the
word-edit variants) are mapped centrally in `attacks.__main__`.

## Configured attack names

| `attack.paraphraser` | Implementation |
| --- | --- |
| `parrot`, `parrot-bigram` | Parrot sentence rewrites; optional bigram selection. |
| `pegasus`, `pegasus-bigram` | PEGASUS sentence rewrites; optional bigram selection. |
| `openai`, `openai-bigram` | OpenAI sentence rewrites; optional bigram selection. |
| `word_deletion` | Random word deletion at `word_edit_ratio`. |
| `synonym_substitution` | WordNet synonym substitution. |
| `context_synonym` | Contextual masked-LM substitution. |
| `back_translation` | Translation through `back_translation_lang` and back to English. |
| `custom` | Whole-document chat-model rewrite using a prompt style. |
| `custom_sent` | Sentence-by-sentence custom chat rewrite. |
| `dipper` | DIPPER rewrite with lexical/order controls and prior-sentence context. |
| `adaptive` | Best-of-K rewrite maximizing displacement under one surrogate encoder. |
| `oracle` | Best-of-K rewrite keeping the closest candidate that lands in the defender's red region. |
| `sentence_deletion` | Drop whole sentences at `word_edit_ratio`; survivors intact. |
| `merge_adjacent` | Join adjacent sentence pairs; unit count falls. |
| `split_midpoint` | Cut sentences at their midpoint; unit count rises. |
| `clause_migrate` | Move a tail clause to the next sentence; unit count fixed. |
| `permute_sentences` | Permute whole sentences; units and wording untouched. |
| `controlled_reorder` | Reorder whole sentences to target a specified fraction of all possible pair inversions. |
| `boundary_exchange` | Move selected boundaries to preceding-sentence midpoints, one-for-one, while preserving words and word order. |
| `uniform_rechunk` | Re-cut into `rechunk_words`-word chunks; no lexical change. |
| `random_content_sub` | Substitute content words drawn from other documents. |
| `sentence_insertion` | Splice unwatermarked sentences in from other documents. |

## Probe atoms

`simple/structural.py` implements single-channel probes for channel-to-z
attribution. The model-based attack grid is nearly collinear — every condition
traces one "attack strength" axis, which leaves per-channel effects on the
detection z-score unidentified. Each atom moves one channel and pins the rest,
so the design matrix reaches full rank.

The default battery separates rewording, reordering, and resegmentation while
holding lexical content fixed except on the deliberate synonym axis.
`merge_adjacent` and `split_midpoint` move the unit count in opposite directions
at the same resegmentation sign.
`controlled_reorder` interprets `attack.word_edit_ratio` as a target inversion
fraction: zero is the original order and one is complete sentence reversal.
Unlike `permute_sentences`, its maximum strength spans the full
`anchor_reorder` range rather than approaching the 0.5 expectation of a random
permutation.
`boundary_exchange` is the count-neutral resegmentation control: it moves each
selected boundary to the midpoint of its preceding sentence. Together,
`merge_adjacent`, `split_midpoint`, and `boundary_exchange` distinguish boundary
damage from decreasing, increasing, and fixed intended unit counts.
`synonym_substitution` supplies the lexical-rewording axis with little order or
boundary displacement.

The default sweep uses ratios `{0.1,0.25,0.5,0.75,1.0}` for controlled reorder,
boundary exchange, and synonym substitution, and `{0.2,0.5,1.0}` for split and
merge: 21 attacked conditions plus the shared clean baseline.

`scripts/experiments/attacks.sh` expands these into a `probe-`-prefixed grid
appended to every cell, and `attack_is_probe` marks them for special routing in
the SwordStamp runner:

- **CPU only.** The atoms are string edits, so they enqueue without a GPU slot.
- **CPU-only quality.** Deterministic diversity, BLEU, ROUGE, and the anchor
  channels run; causal-LM, MAUVE, embedding, BERTScore, and judge phases do not.
- **Separate batch lane.** Probes accumulate into their own CPU scheduler job
  (`quality-batch[...,cpu-only]`).
- **Detection runs normally.** That is the entire point of the battery.

The public config field remains `attack.paraphraser` so existing resolved
configs and result paths remain reproducible even though the package is now
named `attacks`.

## Segmentation behavior

- Sentence-level attacks always operate on `sentence` units and inherit the
  watermark's configured sentence backend. Explicit attacker overrides do not
  affect them.
- Adaptive attacks use `segmentation.attacker_type` and
  `segmentation.attacker_backend` when set, otherwise `sentence-nltk`.
- The oracle attack ignores the attacker overrides and uses the defender's
  `segmentation.type`/`backend`, matching what its detector will resegment.
- For non-sentence watermark arms, the experiment scheduler can emit adaptive
  variants whose attacker type matches the watermark type.

Every boundary call goes through the `segmentation` package.

## Adaptive attack

`attack.custom_model` is required. A single `attack.surrogate_model` embeds
source anchors and candidate units. `attack.anchor=positional` compares output
unit *i* to source unit *i* (reusing the last anchor for extra units);
`attack.anchor=bag` compares against all source anchors and reduces with
`attack.bag_agg={min,mean}`. `attack.num_candidates` is the best-of-K budget.

When `segmentation.attacker_type=semspan`, the surrogate model also chooses the
attacker's semantic-span boundaries. The attacker never borrows the defender's
watermark encoder; detection independently resegments with
`watermark.embedder`.

## Oracle attack

The informed upper bound on the adaptive attacker. `oracle` is given the
defender's encoder *and* its green/red partition, so it evaluates detection
exactly instead of approximating it with a surrogate. Per unit it keeps the
candidate closest to the source unit, in the defender's own embedding space,
among those the detector would score as a miss.

Selection is best-of-N only: one wave per unit, no rejection loop and no second
wave. When no candidate escapes, the attacker falls back to the adaptive
attacker's objective and commits the most displaced candidate, which is the one
sitting shallowest inside the green region. Yellow is not a third region here —
detection counts it exactly as it counts green, so the attacker does too.

It shares `adaptive`'s sampling pipeline through
`attacks.paraphrasing.adaptive.unitwise_rewrite` and differs only in the
ranking objective, so a paired `adaptive`/`oracle` run at the same
`attack.num_candidates` prices exactly what the partition knowledge is worth.

`attack.custom_model` is required. The partition comes from the run's own
`watermark` section, which `attacks.__main__` passes through, so generation,
attack, and detection cannot drift apart. Segmentation is the defender's
`segmentation.type`/`backend`: an attacker holding the encoder also knows where
the detector cuts, and `segmentation.attacker_*` is ignored.

Deliberate simplifications, each a lower bound on what this attacker could do:

- Greedy, no lookahead over the context-dependent mask chain.
- No attacker-side margin; landing exactly on a hyperplane is measure-zero
  under sampling.
- No calibration toward the null. The attack minimizes hits rather than
  targeting a hit rate of `watermark.lmbd`, so its z-scores run *below* the
  human null and a two-sided test would flag them.

## Configuration highlights

| Field | Default | Meaning |
| --- | ---: | --- |
| `attack.paraphraser` | `parrot` | Attack implementation. |
| `attack.custom_model` | `null` | Required for `custom`, `custom_sent`, `adaptive`, and `oracle`. |
| `attack.prompt_style` | `standard` | Key in `paraphrasing/custom_prompts.yaml`. |
| `attack.backend` | `vllm` | `vllm` or `hf` for custom/adaptive models. |
| `attack.surrogate_model` | `BAAI/bge-base-en-v1.5` | Adaptive encoder and optional semspan boundary encoder. |
| `attack.num_candidates` | `32` | Adaptive and oracle search budget. |
| `attack.dipper_lex` / `dipper_order` | `60` / `0` | DIPPER diversity controls. |
| `attack.word_edit_ratio` | `0.3` | Word-edit fraction. |
| `attack.back_translation_lang` | `zh` | Pivot language. |
| `runtime.vllm_utilization` | `0.9` | Shared local vLLM memory fraction. |

DIPPER leaves include both diversity controls. Adaptive leaves include model,
prompt, K, bag/surrogate tags, and nondefault attacker segmentation where
applicable. These naming rules are centralized in `config/paths.py`.
