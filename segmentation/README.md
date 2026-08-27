# Segmentation

This package is the only sentence/semantic-span boundary API used by generation,
detection, attacks, and quality analysis.

```python
from sentence_transformers import SentenceTransformer
from segmentation import Segmenter, segment, first_unit

sentences = segment(text, type="sentence", backend="nltk")
first = first_unit(text, type="sentence", backend="nltk")

encoder = SentenceTransformer("example/encoder")
semspan = Segmenter.from_sentence_transformer(
    "semspan", "nltk", encoder, "example/encoder",
    semcut_max_words=15, semcut_window=5,
)
wave = semspan.first_units(candidate_texts)
```

## Unit contract

`Unit.display` preserves the exact source span, including boundary whitespace,
so concatenating all displays reconstructs the input. `Unit.normalized` is the
trimmed, lowercased scoring/embedding text. Keep display text for output and
normalized text for hashes, clusters, and surrogate scoring.

Inter-unit whitespace belongs to the following unit. This keeps word-boundary
continuations aligned with causal-LM tokenizers whose tokens commonly include
the leading space of a word; generation prompts must not end in a detached
space token.

## Supported types

| Type | Boundary strategy |
| --- | --- |
| `sentence` | Sentence units from the selected `nltk` or `spacy` backend. |
| `semspan` | Bounded semantic spans cut inside boundaries supplied by the selected sentence backend; requires an explicitly bound encoder. |

`first_units` batches candidate waves. `semspan` deliberately has no
process-global/default encoder: production callers construct a bound `Segmenter`,
and that object owns both scalar and batched boundaries.

`_semspan.py` is intentionally separate from `_spacy.py`. The latter only
produces spaCy sentence units; semantic spans may wrap either `nltk` or `spacy`
sentence boundaries and choose their internal cuts with the bound embedding
encoder. The static spaCy English stop-word list used by semspan is not a reason
to couple the two boundary implementations.

The semantic-span boundary policy is explicit: `semcut_max_words` bounds a span
in content words, and `semcut_window` selects the number of tokens compared on
each side of a gap while guarding both sides of every split. The paper defaults
are 15 and 5. Encoder batch size is runtime-only and cannot change boundary or cache
identity.

Watermark generation, center construction, and detection bind semantic spans to
`watermark.embedder`. Adaptive attacks bind attacker semantic spans to the
already-loaded `attack.surrogate_model`. Detection of attacked text always uses
the defender-bound segmenter.

## Generation boundary helpers

- `normalize_generated_whitespace` collapses horizontal decoded whitespace.
- `display_with_boundary_space` inserts only the separator needed between
  accepted context and a new display span.
- `insert_missing_punctuation` repairs nonempty lines without terminal marks.
- `check_split_idempotent` is a diagnostic reconstruction check.
- `segmentation_kwargs` serializes boundary provenance, including the semantic
  encoder and complete boundary-policy identity for semantic spans.

Generation prompts are cut at the default `sentence-nltk` boundary for every
arm. Candidate stopping still uses sentence boundaries, while accepted units are
truncated and scored at the configured sentence or semantic-span granularity.

Run the smoke test with `uv run python -m segmentation`. Regression coverage is
in `test_segmentation.py` and `test_semspan_backend.py`.
