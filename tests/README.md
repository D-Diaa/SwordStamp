# Test suite

Run the complete suite through the managed environment:

```bash
uv run python -m unittest discover -s tests -v
```

CUDA/model-dependent tests skip themselves when prerequisites are unavailable.
For GPU work, use the scheduler so it supplies `CUDA_VISIBLE_DEVICES`; integration
tests use locally cached `Qwen/Qwen2.5-3B-Instruct` weights. Set
`SWORDSTAMP_TEST_EMBEDDER` to a local watermark-encoder checkpoint when needed.

Focused examples:

```bash
uv run python -m unittest discover -s tests -p 'test_segmentation.py' -v
uv run python -m unittest discover -s tests -p 'test_hf_sampler.py' -v
uv run python -m unittest discover -s tests -p 'test_detection_utils.py' -v
uv run python -m unittest discover -s tests -p 'test_attack_utils.py' -v
```

## Coverage map

| Area | Files |
| --- | --- |
| Configuration and paths | `test_config.py`, `test_paths.py` |
| Exact paper registry and dry-run orchestration | `test_paper_registry.py` |
| Deterministic C4 preparation | `test_prepare_c4.py` |
| Release pins, submodules, dependencies, and setup | `test_release_metadata.py` |
| Sentence/semantic-span contracts | `test_segmentation.py`, `test_semspan_backend.py` |
| Sampler pool/selection | `test_hf_sampler.py` |
| Scalar generation scores | `test_generator_score_fns.py` |
| Detector math | `test_detection_utils.py` |
| CUDA watermark geometry | `test_lsh_primitives.py`, `test_kmeans_primitives.py`, `test_invariants.py` |
| Adaptive surrogate/anchors | `test_surrogate_scorer.py`, `test_anchor_structure.py` |
| Shared attack helpers | `test_attack_utils.py` |
| End-to-end generation/detection | `test_integration.py` |
| Paper result compilation and Matplotlib rendering | `test_compile_results.py`, `test_visualization.py` |

## Important invariants

- Every boundary comes from `segmentation`; `Unit.display` reconstruction and
  `Unit.normalized` scoring remain stable under regeneration.
- Generation and detection use identical keys, partitions, segmentation, and
  cluster centers.
- Generation scores carry an explicit region and nonnegative boundary depth;
  margin is supplied to selection. Tests exercise active `compute_lsh_margins`
  and `compute_kmeans_margins` directly.
- Rejection selects the first margin-clearing green and otherwise the last yellow
  (or last candidate when no yellow exists); best-of-n ranks by region and depth.
  The pooled driver preserves
  document ordering, refills active slots, and carries exact predecessor units in
  isolated per-document state.
- Adaptive scoring uses one encoder and preserves positional versus bag-anchor
  semantics.

Tests prefer real code paths and small controlled embeddings. Fakes replace only
expensive model generation or embedding calls needed to isolate a contract; they
do not reimplement the selection/detection algorithms under test.
