# Quality evaluation

The quality pipeline evaluates clean watermarked text or attacked `para_text`,
writes scalar aggregates to `eval_quality.csv`, dense per-sample arrays to
`eval_quality_per_sample.npz`, and ragged/object artifacts to
`eval_quality_per_sample_aux.npz`.

```bash
uv run python -m quality DATA_PATH --config PRESET \
  --set quality.corpus=data/c4-human-def

uv run python -m quality DATA_PATH --config PRESET \
  --set io.target=attack --set attack.paraphraser=dipper \
  --set quality.corpus=data/c4-human-def
```

The input directory and default column/reference are derived by `quality.io` and
`config.paths`. Watermark-stage evaluation normally skips per-pair metrics;
attack-stage evaluation pairs each `para_text` with its row-matched watermarked
`text`. Attack generation itself lives in the top-level `attacks` package;
quality consumes its common dataset contract and does not import an individual
attack implementation.

## Metrics

| Group | Outputs |
| --- | --- |
| Causal-LM fluency | Corpus-level `gen_ppl` from next-token perplexity (documents capped at 2048 tokens). |
| Repetition/diversity | tokenizer-based `rep_2`, `rep_3`, `rep_4`; deterministic `sent_dup_pct`, `distinct_2`, `4g_repeat_pct`; bigram/trigram entropy. |
| Semantic distribution | semantic entropy and MAUVE. |
| Pairwise preservation | BERTScore P/R/F1, EmbeddingGemma cosine similarity, BLEU, ROUGE-1/2/L. |
| Structural edits | patience-anchor rewording, novel bigrams, reorder, merge, split, resegmentation, anchor coverage/count. |
| LLM judges | generation-quality rubric for clean watermark outputs; pairwise faithfulness rubric for attacks. |

Most metrics emit a mean, 95% confidence-interval half-width, median, and
`*_per_sample` values where defined.

For the deterministic metrics, SD uses the repository's NLTK sentence backend
followed by trimming and lowercasing; D-2 and 4g use lowercased whitespace
tokens over the whole output, including n-grams that cross sentence boundaries.
Zero denominators map to zero. SD and 4g are percentages, while D-2 is a ratio
in `[0, 1]`.

The active perplexity metric is causal-LM perplexity. `gen_ppl` exponentiates
the token-weighted mean cross-entropy over the corpus; it does not average
per-document perplexities. Each document contributes its predicted-token count
(tokenized length minus one), and failed or prediction-free documents are
excluded. Per-document perplexities remain available in
`gen_ppl_per_sample` for diagnostics. The corresponding evidence is stored as
`gen_ppl_total_nll_per_sample` and `gen_ppl_scored_tokens_per_sample`, with
their sums in `gen_ppl_total_nll` and `gen_ppl_scored_tokens`. `gen_ppl` is
always derived as `exp(gen_ppl_total_nll / gen_ppl_scored_tokens)`, including
when compiler cells pool multiple dataset shards. Repetition is gathered by
`evaluate_ngrams`; causal-LM perplexity is the only perplexity implementation
in the artifact.

## Semantic entropy

Semantic entropy clusters model-derived representations using either a corpus
fit or `quality.load_kmeans_path`. `quality.sem_ent_mode` is one of
`last_token`, `last_mean_pooling`, or `all_mean_pooling`. The evaluator may write
model-derived caches into the configured corpus/output directory, so point
exploratory runs at deliberate locations.

## LLM judge

`quality.judge_model` selects the judge. Names beginning with OpenAI model
prefixes use the OpenAI API; other names use one reusable local vLLM engine.
Rubrics live in `metrics/rubrics.yaml` and return normalized weighted scores plus
dimension-level per-sample arrays. `quality.judge_repeats` defaults to three
independent judgments per prompt. Every configured judgment must be valid, and
each criterion is mean-reduced across the three judge runs before the rubric's
valence and weights are applied. A local decode without a complete final JSON decision is
retried from the original prompt with a fresh seed up to three additional
times; a repetition that still fails makes that sample `NaN` without aborting
the directory. Failed reasoning is never included in a retry. The NPZ sidecar
retains normalized per-criterion repetition arrays (`N x judge_repeats`) plus
attempt counts, seeds, token counts, finish reasons, errors, and truncation counts so
judge dispersion and failures remain auditable.

Evaluation selects the rubric by stage: clean watermark directories run
the intrinsic generation-quality rubric, while attack directories run the
pairwise preservation rubric. Local repetitions are submitted to vLLM as one
packed request queue while retaining an independent seed for every draw.

Local judges share `runtime.vllm_utilization` (default 0.9) with generation and
attacks. `SEMSTAMP__RUNTIME__VLLM_UTILIZATION`, `VLLM_TENSOR_PARALLEL_SIZE`,
and `VLLM_MAX_MODEL_LEN` are scheduler/runtime overrides. Local judge engines
default to a 16,384-token model length and an 8,192-token output cap. Only the
candidate/generated input text is truncated, and only for local judges, to its
first 8,192 judge-tokenizer tokens; rubric instructions, the original prompt,
and the pairwise reference are preserved. The per-request output allowance is
reduced when necessary to fit the rendered prompt in the 16K context. OpenAI
input handling is unchanged and does not use this local truncation path.

Qwen3-family judges run with thinking explicitly enabled and their recommended
reasoning parser. The judge scores only the final response after `</think>`;
JSON-like provisional scores inside the reasoning trace are ignored, and a
decode that never closes its reasoning block is retried as unparseable.
Qwen3.5/3.6 multimodal checkpoints additionally use vLLM's language-model-only
mode because these rubrics contain no image or video input. The text-only
Qwen3-32B checkpoint uses the same final-answer parsing without that no-op mode.
Rendered local prompts and their exact token IDs are cached across repetitions
and retries. Empty generations remain deterministically scored zero but report
zero valid judge repetitions because no model request was made.

## Single-directory and batch execution

Both entry points use the same phase-ordered engine. `python -m quality` derives
one target directory from the application config; `python -m quality.batch
DIR...` accepts one or more explicit derived directories. The engine releases
the causal LM before loading MAUVE and later the local judge, evaluates multiple
directories while loading each heavy model once, isolates per-directory
failures, and saves each directory as soon as its final judge evaluation
completes. `--cpu-only` runs the deterministic diversity metrics, BLEU, ROUGE,
and structural anchors without GPU phases or a judge.

## Public orchestration API

- `evaluate_model_quality_metrics`: causal perplexity, semantic entropy,
  tokenizer repetition/entropy, deterministic diversity, and MAUVE.
- `evaluate_pair_quality_metrics`: BERTScore, embedding similarity, BLEU,
  ROUGE, and anchor structure, or a stable NaN schema when skipped.
- `evaluate_llm_judge_metrics`: requested intrinsic and pairwise rubrics with
  one reusable judge pipe.
- `evaluate_quality`: composes all three for library callers.

Metric modules are lazy-exported so optional heavyweight dependencies load only
when their metric runs.
