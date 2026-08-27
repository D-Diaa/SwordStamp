# Sampling

`sampling` is the shared HF/vLLM candidate-wave engine. A caller supplies an
explicit region-and-depth score; watermark-specific geometry stays in
`watermarking.scoring`, while `attacks.paraphrasing.adaptive` represents every
candidate as green and uses displacement as depth.

```python
score_fn(...) -> list[CandidateScore]
CandidateScore(region=Region.GREEN | Region.YELLOW | Region.RED, depth=float)
accept_fn(scores: list[CandidateScore]) -> list[bool]  # optional
```

Depth is nonnegative and margin-free. The public generation methods take an
external `margin`; default rejection acceptance requires a green candidate with
`depth > margin`. Numeric score callbacks are rejected.

The score function receives the exact preceding `Unit`, one candidate wave, and
the zero-based generated-unit position. Before continuation, the sampler uses its
configured segmenter once per source prompt and supplies the final prompt unit at
index zero. After each step it retains the selected `Unit` itself and supplies it
on the next step; it never recovers a predecessor by segmenting the growing
document. This also applies to an unaccepted rejection fallback: selected means
committed, not necessarily green.

An empty or unsegmentable prompt gets an explicit synthetic `Unit` whose display
is the original prompt and whose normalized value is the normal text
normalization. No later step retries prompt or accumulated-document segmentation.
`accept_fn` sees the whole score list, so pool-relative criteria are possible.

## Selection

- `rejection`: choose the first accepted candidate; if none passes, use the last
  yellow candidate, or the last nonempty candidate when no yellow exists, and
  mark it unaccepted.
- `best-of-n`: rank green above yellow above red; prefer greater depth within
  green/yellow and smaller depth within red. Acceptance only annotates whether
  that winner passed `accept_fn`.

`generate_with_rejects` uses acceptance partitions for rejection mode and
disjoint top/bottom rank partitions for best-of-n. It has no in-tree caller since the
preference collector was removed; it is retained as a public sampler API.

## Candidate waves and document pool

Candidates are decoded in `chunk_tokens` increments. After each chunk, completed
sentence/EOS candidates are finalized; remaining requests continue until every
candidate completes or the token budget is exhausted. Each result is truncated
to exactly one configured `Unit` through `segmentation.first_units`.

`generate_batched_continuation` maintains a refilling pool of active documents.
One raw generation wave contains candidates for all active documents, and one
batched segmentation pass truncates them. `max_active=None` targets roughly 512
concurrent candidate sequences (eight documents at the paper's N=64, up to 512
at N=1).
Predecessors, score functions, indices, and unit caps are isolated per document.

## Public methods

All are implemented by `BaseSampler` and inherited by both backends:

- `generate`: one selected unit plus score/acceptance/pool counts.
- `generate_with_rejects`: accepted/rejected `GeneratedCandidate` partitions.
- `scored_candidate_wave`: raw scored candidates with duplicates preserved.
- `generate_continuation`: one document, implemented as a pool of one.
- `generate_batched_continuation`: pooled multi-document generation.
- `generate_raw`: backend primitive implemented by HF and vLLM subclasses.

`create_sampler` accepts the backend, model, candidate/chunk budgets, optional
LoRA adapter/strength, and a bound `Segmenter` (plus segmentation identity for
ordinary sentence boundaries). An unbound `semspan` policy is rejected
before either backend loads its generator. HF supports
device, micro-batch size, and attention implementation. vLLM supports memory
utilization, tensor parallelism, max model length, and prefix caching.

The shared vLLM utilization resolver defaults to 0.9. Native vLLM LoRA is used
at adapter strength 1; other nonzero strengths are merged to a temporary model,
and strength 0 skips the adapter.

There is no pipeline CLI. `uv run python -m sampling` runs the contract examples;
use `uv run python -m watermarking.generate` for experiment generation.
