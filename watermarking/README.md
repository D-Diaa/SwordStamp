# Watermarking

This package owns semantic watermark primitives, score factories, generation,
detection, and KMeans-center construction. Candidate sampling itself lives in
`sampling/` and is watermark-agnostic.

## Modes and candidate scores

| Mode | Partition | Key |
| --- | --- | --- |
| `lsh` | Random-hyperplane bins | Previous committed unit |
| `lsh_fixed` | Random-hyperplane bins | `watermark.secret_message` |
| `lsh_fixed_diverse` | Random-hyperplane bins | Fixed key; previous committed region is yellow |
| `kmeans` | KMeans clusters | Previous committed unit |
| `kmeans_fixed` | KMeans clusters | `watermark.secret_message` |
| `kmeans_fixed_diverse` | KMeans clusters | Fixed key; previous committed region is yellow |
| `none` | No watermark | Generation only |

Each generation score function returns one structured score per candidate:

```python
CandidateScore(region=Region.GREEN | Region.YELLOW | Region.RED, depth=depth)
```

Depth is the nonnegative distance inside the assigned region. Margin is external
to scoring: rejection accepts the first green candidate with
`depth > watermark.margin`; on exhaustion it commits the last yellow candidate,
or the last candidate if no yellow exists. Best-of-n ranks green above yellow
above red, preferring deeper green/yellow candidates and shallower red candidates.
`lmbd` selects the green partition and defines the detector's Bernoulli null; it
does not scale depth.

`create_lsh_score_fn` and `create_kmeans_score_fn` implement this contract;
`setup_lsh_mode` and `setup_kmeans_mode` load the embedder/key artifacts.
Their shared callback is
`score_fn(predecessor: Unit, candidates: list[Unit], unit_idx: int)`. Context-keyed
modes derive the next mask directly from `predecessor.normalized`; fixed-diverse
modes derive their yellow region from the same value. Scoring does not segment the
accumulated document.

For fixed-diverse scoring, only a predecessor region already inside the fixed
mask can be yellow. If rejection commits a red fallback, a following candidate in
that same out-of-mask region remains red; it is never promoted to the yellow tier.
Fixed and no-watermark modes ignore the predecessor but retain the same callback
shape.

## Generation

```bash
uv run python -m watermarking.generate DATA_PATH --config PRESET [--set ...]
```

Generation:

1. extracts an identical `sentence-nltk` prompt for every segmentation arm;
2. builds the HF or vLLM sampler;
3. generates candidate waves and truncates them with the configured segmentation;
4. segments each source prompt once to initialize its predecessor, then carries
   the exact selected `Unit` between score calls;
5. applies rejection or best-of-n using region, depth, and the external margin;
6. writes the HF dataset, `resolved_config.yaml`, and `generation_stats.json`.

`generation.do_sample` defaults to true and is passed to the model generation
configuration. Watermarked generation defaults to 64 candidates; the retained
no-watermark preset explicitly uses one draw.
`max_active_docs=None` lets the pooled sampler target about 512 simultaneous
candidate sequences. vLLM memory defaults to 0.9 through the shared runtime
resolver.

Generated internal `Unit` metadata is used for contract checks; persisted
experiment output retains the text column and summary statistics.

## Detection

```bash
uv run python -m watermarking.detect DATA_PATH --config PRESET
uv run python -m watermarking.detect DATA_PATH --config PRESET --set io.target=attack
```

Detection resegments the stored text with the same type/backend, reconstructs the
green mask from the context or fixed secret, counts green units, and computes the
binomial z-score using `watermark.lmbd`. Outputs include aggregate CSVs and
per-document z-score arrays. `io.target` selects watermarked versus attacked
derived paths; `detection.wm_data_path` can supply an explicit comparison input.

For `semspan`, generation, KMeans-center construction, and detection reuse the
watermark runtime's one `watermark.embedder` instance for both semantic boundaries
and watermark embeddings. Human-score cache provenance includes that encoder and
the maximum-span and comparison-window settings.
Adaptive attacks may use a different surrogate-bound semantic-span policy, but defender
detection never inherits it.

Generation and detection must match on mode, embedder, `sp_dim`, `lmbd`, margin,
hash key, segmentation, and fixed secret. KMeans additionally requires the exact
same center file.

## KMeans centers

```bash
uv run python -m watermarking.generate_clusters DATA_PATH \
  --config config/presets/sentence_kmeans_context_rejection.yaml
```

The builder segments and embeds the corpus at the target granularity, runs three
cosine-KMeans restarts by default, keeps the lowest-inertia result, and writes
`DATA_PATH/cc_{segmentation-identity}_k{k}.pt` unless `--output` is supplied.
For semantic spans that identity includes the maximum-span and comparison-window
settings; embedding caches use the same policy identity. Paper presets bind the
two verified files under `artifacts/centers/` rather than a live data directory.

## Modules

- `primitives.py`: LSH/KMeans masks, signed-depth geometry, embedding caches,
  cosine KMeans, prompt extraction.
- `scoring.py`: active region-and-depth score factories and mode setup.
- `generate.py`: dataset generation and statistics.
- `detect.py`: LSH/KMeans detection, z-scores, ROC summaries.
- `generate_clusters.py`: segmentation-matched KMeans centers.

Tests cover structured score semantics, detector math, key symmetry, partition
statistics, segmentation agreement, and CUDA end-to-end generation/detection.
