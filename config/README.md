# Configuration

`config/schema.py` is the canonical typed schema. Its frozen dataclasses supply
defaults and reject unknown keys and invalid values. `config/paper.py` is the
separate, authoritative registry for the paper's datasets, ten additive
watermark cells, two comparison cells, candidate budget, and oracle budgets.

## Resolution

Every application CLI accepts repeatable `--config` and `--set` arguments.
Values resolve in this order:

```text
schema defaults < YAML files (left to right) < SEMSTAMP__SECTION__KEY < --set
```

```bash
uv run python -m watermarking.generate data/c4-val-def-256 \
  --config config/presets/sentence_lsh_fixed_best-of-n.yaml
```

Environment keys always use the layered double-underscore form. For example:

```bash
SEMSTAMP__RUNTIME__VLLM_UTILIZATION=0.85 \
  uv run python -m watermarking.generate DATA_PATH --config PRESET
```

YAML scalar coercion is also used for `--set`, so booleans and numbers remain
typed.

## Paper defaults

- watermarked candidate budget: `generation.num_candidates=64`;
- no-watermark preset candidate budget: `1`;
- semantic-span policy: spaCy sentence boundaries, maximum 15 content words,
  comparison window 5;
- vLLM memory utilization: `runtime.vllm_utilization=0.9` unless a preset binds
  a smaller value;
- semantic-span encoder batching: `runtime.semcut_batch_size=512`.

`generation.max_active_docs=None` lets the sampler choose the document pool
from the candidate budget. `runtime.semcut_batch_size` affects throughput only
and is excluded from experiment and cache identity.

## Sections

| Section | Purpose | Important fields |
| --- | --- | --- |
| `io` | Base input and target selection | `data_path`, `target`, `output_dir` |
| `watermark` | Shared generation/detection key | `mode`, `sp_dim`, `lmbd`, `margin`, `secret_message`, `embedder`, `cc_path`, `hash_key` |
| `generation` | LM and candidate sampling | `model`, `backend`, `sampling_method`, `num_candidates`, `max_active_docs`, token and decoding settings |
| `segmentation` | Defender and adaptive-attacker boundaries | `type`, `backend`, `attacker_type`, `attacker_backend`, `semcut_max_words`, `semcut_window` |
| `detection` | Detector inputs | `mode`, `human_text`, `wm_data_path` |
| `attack` | Attack selection | paraphraser, model, surrogate, anchor, budget, and attack-specific controls |
| `quality` | Metrics and judges | reference/corpus paths, judge settings, and similarity models |
| `runtime` | Shared local runtime | `vllm_utilization`, `semcut_batch_size` |

## Derived paths

The positional `DATA_PATH` is always the base prompt corpus. The path layer
derives:

```text
watermark_dir = DATA_PATH/{lsh|kmeans}/{context|fixed|fixed_diverse}/{sampling}/{segmentation}/candidates-64/watermarked
attack_dir    = a sibling of watermarked named by config.paths.attack_spec
target_dir    = watermark_dir or attack_dir according to io.target
```

Every watermarked run includes `candidates-N`; there is no historical implicit
candidate budget. Semantic spans include their full boundary identity, for
example `semspan-spacy-max15-win5`. The generation-only baseline uses
`DATA_PATH/none/sentence-nltk`. `io.output_dir` is an explicit override.

```bash
uv run python -m config.paths watermark-dir DATA_PATH --config PRESET
uv run python -m config.paths attack-dir DATA_PATH --config PRESET \
  --set attack.paraphraser=dipper
uv run python -m config.paths target-dir DATA_PATH --config PRESET \
  --set io.target=attack
```

Generation and attacks persist `resolved_config.yaml`. Detection must reuse the
same watermark, segmentation, and generation-relevant settings.

## Presets

Only the additive paper ladder is retained for each partition family:

1. context mask, rejection sampling, sentence units;
2. context mask, best-of-N, sentence units;
3. fixed mask, best-of-N, sentence units;
4. fixed-diverse mask, best-of-N, sentence units;
5. fixed-diverse mask, best-of-N, `semspan-spacy-max15-win5`.

The same five rungs exist for LSH and KMeans. The only additional preset is the
sentence no-watermark baseline. KMeans presets bind the packaged centers:

- `artifacts/centers/cc_sentence_nltk_k8.pt`;
- `artifacts/centers/cc_semspan_spacy_max15_win5_k8.pt`.

`semspan` has no independent encoder setting. Defender boundaries use
`watermark.embedder`; adaptive-attacker boundaries use `attack.surrogate_model`.

## Entry points

```bash
uv run python -m watermarking.generate DATA_PATH --config PRESET
uv run python -m watermarking.detect DATA_PATH --config PRESET
uv run python -m watermarking.generate_clusters DATA_PATH --config PRESET
uv run python -m attacks DATA_PATH --config PRESET
uv run python -m quality DATA_PATH --config PRESET
uv run python -m quality.batch DIR [DIR ...] --config PRESET
```
