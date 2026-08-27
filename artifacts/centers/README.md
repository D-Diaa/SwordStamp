# k-SemStamp centers

These are the two eight-centroid tensors used by the paper's additive
k-SemStamp ladder. They were fit with three restarts over embeddings from the
8,192-document held-out C4 `realnewslike` center split; the lowest-inertia
restart was retained.

| File | Segmentation | SHA-256 |
| --- | --- | --- |
| `cc_sentence_nltk_k8.pt` | `sentence-nltk` | `31e3acdb5471fd508e6c1b44df07d29b6e6340c13190a3843bbdae60d4e214f5` |
| `cc_semspan_spacy_max15_win5_k8.pt` | `semspan-spacy-max15-win5` | `0b838b81865e303ba84f5e2d79134f22137801aadad35e9d5ee3c45dbb0f925d` |

To regenerate them after preparing the C4 splits:

```bash
uv run python -m watermarking.generate_clusters data/c4-center-8192 \
  --config config/presets/sentence_kmeans_context_rejection.yaml \
  --output artifacts/centers/cc_sentence_nltk_k8.pt

uv run python -m watermarking.generate_clusters data/c4-center-8192 \
  --config config/presets/semspan_kmeans_fixed_diverse_best-of-n.yaml \
  --output artifacts/centers/cc_semspan_spacy_max15_win5_k8.pt
```

The committed tensors let evaluators run generation and detection without
repeating the center-fitting stage. `SHA256SUMS` provides a machine-checkable
copy of the hashes above.
