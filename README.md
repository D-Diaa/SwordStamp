# SwordStamp

SwordStamp adds order-robust detection and semantic-span units to
embedding-based semantic watermarks. This artifact contains the exact paper
configuration, generation and detection code, the unchanged attack suite,
quality evaluation, and deterministic result extraction and plotting.

## Headline results

At 5% false-positive rate and a 90% content-preservation requirement, the
scheme-specific embedding displacement attack (EDA-S) achieves:

| Watermark | EDA-S attack success rate |
| --- | ---: |
| SemStamp | 47.9% |
| k-SemStamp | 33.3% |
| PMark | 46.1% |
| SAMark | 32.6% |
| **SwordStamp** | **18.7%** |
| **k-SwordStamp** | **10.8%** |

Relative to their published baselines, SwordStamp and k-SwordStamp reduce
EDA-S success by 29.2 and 22.5 percentage points, at clean-fidelity costs of
1.8 and 4.4 points. Their clean detection rates are 94.9% and 97.0%. Under the
stronger detector-access EDA-D stress test, maximum quality-gated success is
39.7% on k-SwordStamp versus 65.5% on k-SemStamp.

![Strongest no-box attack success across content-preservation requirements](docs/figures/teaser.png)

![Clean fidelity versus strongest no-box attack success](docs/figures/fidelity-robustness.png)

These results cover English news continuations, one provider model, two
provider encoders, and one EDA paraphraser/surrogate pair.

## Plot a compiled bundle

`visualization` consumes only the compiled CSV/Parquet bundle. It fails on
missing paper cells rather than drawing a partial figure.

```bash
uv sync --frozen --only-group plot --no-install-project
uv run --no-sync python -m visualization extract \
  --bundle results/paper --output results/paper
uv run --no-sync python -m visualization render \
  --bundle results/paper --output results/paper
```

This writes auditable tables under `results/paper/tables/` and seven figures in
PNG and PDF under `results/paper/figures/`. The committed projection preserves
every field consumed by the extractors, removes raw C4 text, and keeps each file
below the anonymous mirror's 8 MB limit.

## Embedding displacement attack (EDA)

The embedding displacement attack (EDA) is an adaptive whole-text paraphrasing
attack. At each output unit it samples `K` continuations and keeps the candidate
with the largest cosine displacement from its source anchor under a surrogate
encoder. Because every candidate continues the full rewrite prefix, one
objective can reword content, reorder it, and split or merge detector units.

No-box EDA sees the marked text and the public watermark design. It does not
query the detector or generator and receives no secret key, provider encoder,
or quality oracle.

| Variant | Anchor and unit | Configuration |
| --- | --- | --- |
| EDA-P | positional sentence | `adaptive`, `anchor=positional`, attacker unit `sentence` |
| EDA-S | target's public design | positional sentences for SemStamp, k-SemStamp, and PMark; bag/sentence for SAMark; bag/semspan for both SwordStamp variants |
| EDA-D | provider detector | `oracle`; inherits the defender encoder, partition, and segmentation |

For bag anchoring, the paper uses `attack.bag_agg=min`: maximize distance from
the nearest source anchor. Reported no-box runs use
`Qwen/Qwen2.5-3B-Instruct`, `BAAI/bge-base-en-v1.5`,
`K={1,2,4,8,16,32,64}`, temperature 1.0, top-p 0.95, at most 96 tokens per
candidate, and at most 512 rewrite tokens.

Other watermark designers can call EDA directly:

```python
from attacks.paraphrasing.adaptive import adaptive_attack_paraphrase

attacked = adaptive_attack_paraphrase(
    texts,
    base_model="Qwen/Qwen2.5-3B-Instruct",
    surrogate_model="BAAI/bge-base-en-v1.5",
    num_candidates=32,
    anchor="positional",       # EDA-P; use "bag" for an order-robust design
    bag_agg="min",
    segmentation_type="sentence",
    segmentation_backend="nltk",
)
```

For the paper's SwordStamp-aware EDA-S, set `anchor="bag"`,
`segmentation_type="semspan"`, `semcut_max_words=15`, and
`semcut_window=5`. Within this repository the equivalent CLI is:

```bash
uv run python -m attacks DATA_PATH --config PRESET \
  --set attack.paraphraser=adaptive \
  --set attack.custom_model=Qwen/Qwen2.5-3B-Instruct \
  --set attack.surrogate_model=BAAI/bge-base-en-v1.5 \
  --set attack.num_candidates=32 \
  --set attack.anchor=bag --set attack.bag_agg=min \
  --set segmentation.attacker_type=semspan \
  --set segmentation.attacker_backend=nltk
```

`DATA_PATH` is always the base corpus and `PRESET` identifies the watermarked
cell. See `attacks/README.md` for all retained attacks and configuration fields.

![Surrogate displacement versus provider-encoder displacement](docs/figures/encoder-transfer.png)

## Exact paper matrix

`config/paper.py` is the single source of truth. For each of LSH and k-means,
`scripts/experiments/swordstamp.sh` runs this five-rung additive ladder with 64
provider candidates:

1. context-dependent rejection over sentences;
2. best-of-64 selection;
3. a fixed valid set;
4. diversity-aware ranking; and
5. spaCy semantic spans with maximum 15 words and comparison window 5.

The first-party grid contains only these ten cells and one no-watermark
baseline. The comparison grid adds only online PMark and flags-run SAMark.
EDA-D budgets are `K={4,8,16,32,64}`.

## Complete experiment flow

The complete order is data preparation → generation → attacks → detection
→ quality → compilation → extraction → rendering. Plot reproduction is
complete in this branch. Public [PMark](https://github.com/PMark-repo/PMark)
and [SAMark](https://github.com/Z1zs/SAMark) integration snapshots are included
under `comparisons/`; only their paper cells enter the bundle. Local changes
are limited to Hugging Face input/output for the shared attack/quality pipeline,
common operating-point calibration, and two path cleanups; watermark algorithms
are unchanged.

First-party runs require Git, [uv](https://docs.astral.sh/uv/), Python 3.11,
CUDA, the pinned public models, access to the gated provider checkpoint, and
scratch storage. The bundled scheduler is installed only by the explicit,
offline command below; setup never changes an existing scheduler.

```bash
git clone ANONYMOUS_REPOSITORY_URL
cd SwordStamp
bash scripts/setup.sh
uv run --frozen python scripts/check_artifact.py
```

1. Prepare the deterministic, disjoint C4 partitions:

   ```bash
   uv run --frozen python scripts/prepare_c4.py --output-dir data
   ```

2. Inspect and explicitly install the bundled scheduler, then initialize only
   the GPU indices allocated to you. An exact installation is a no-op, even if
   running; running or conflicting non-exact installations are refused.

   ```bash
   bash scripts/install_bundled_scheduler.sh --print-plan
   bash scripts/install_bundled_scheduler.sh --yes
   export PATH="$HOME/.local/bin:$PATH"
   gpu-scheduler init --gpus 0,1
   gpu-scheduler start
   ```

3. Preview, then submit the exact three-runner matrix on every physical shard:

   ```bash
   for shard in c4-val-def-256 c4-val-def-256b c4-val-def-512; do
     BASE="data/$shard" DRY_RUN=1 bash scripts/experiments/swordstamp.sh
     BASE="data/$shard" DRY_RUN=1 bash scripts/experiments/pmark.sh
     BASE="data/$shard" DRY_RUN=1 bash scripts/experiments/samark.sh
   done
   # Remove DRY_RUN=1 after inspecting the exact DAGs.
   ```

4. Compilation requires raw outputs for all 13 paper cells. If a complete raw
   corpus is supplied, compile, extract, and render it with:

   ```bash
   uv run --frozen python -m visualization all \
     --bundle results/paper --output results/paper
   ```

Install PMark's isolated environment with `bash scripts/setup.sh --comparisons`
before full comparison runs. For plot-only review, use the committed bundle and
the commands under “Plot a compiled bundle.”
The scheduler supplies `CUDA_VISIBLE_DEVICES`; do not select GPUs by probing
`nvidia-smi`. Full generation is a multi-day GPU workload.
`ARTIFACT.md` gives the reviewer checklist; `scripts/experiments/README.md`
documents resume and force behavior.

## Layout and license

- `config/`, `segmentation/`, `sampling/`, `watermarking/`: SwordStamp.
- `attacks/`: EDA and the retained attack suite.
- `quality/`: fidelity and content-preservation evaluation.
- `visualization/`: paper-only compilation, tables, and Matplotlib figures.
- `scripts/experiments/`: scheduler-backed exact SwordStamp workflow.
- `comparisons/`: public PMark and SAMark evaluation source snapshots.
- `tools/gpu_scheduler/`: anonymous, pinned scheduler runtime (Apache-2.0).
- `results/paper/`: anonymous plot-only compiled evidence.

First-party code is released under the MIT License. The comparison sources do
not publish a license file; other inputs retain their terms. See
`comparisons/README.md` and `THIRD_PARTY.md`.
