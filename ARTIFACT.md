# Artifact evaluation

## Fast path: paper plots

Requirements: `uv`, Python 3.11, CPU, and about 2 GB of free space for the
lightweight environment. No GPU, model, dataset, or scheduler is needed.

```bash
uv sync --frozen --only-group plot --no-install-project
(cd results/paper && sha256sum --check SHA256SUMS)
uv run --no-sync python -m visualization extract
uv run --no-sync python -m visualization render
```

Expected output:

- `visualization extract`: 107 files under `results/paper/tables/`;
- `visualization render`: 14 nonempty files under `results/paper/figures/`
  (seven PNG and seven PDF);
- no missing-cell warning or partial output.

The committed bundle contains 407 pooled cells, 1,199 shard-level cells,
416,768 document rows, and 154,326 aligned encoder-transfer rows. Raw generated
text is not included. `results/paper/manifest.json` records the exact released
projection and checksums.

## Functional code path

The CPU suite validates configuration, the exact 10-rung-plus-baseline matrix,
generation/detection math, segmentation, EDA scoring, attacks, quality metrics,
compilation, and rendering:

```bash
bash scripts/setup.sh
uv run --frozen python scripts/check_artifact.py
MPLCONFIGDIR=/tmp/swordstamp-matplotlib \
  uv run --frozen python -m unittest discover -s tests -v
DRY_RUN=1 bash scripts/experiments/swordstamp.sh
```

CUDA/model tests skip with an explicit reason when their inputs are absent.

## Full first-party experiment path

1. Prepare the deterministic disjoint C4 partitions:

   ```bash
   uv run --frozen python scripts/prepare_c4.py --output-dir data
   ```

2. Cache the public `AbeHou/SemStamp-c4-sbert` checkpoint at the revision in
   `artifacts/revisions.yaml`; `scripts/check_artifact.py --models` reports its
   local status without downloading it.

3. After setup, inspect and explicitly install the bundled scheduler without
   network access. An exact install is a no-op; running or conflicting
   non-exact installs are refused. Then initialize only your allocated GPUs:

   ```bash
   bash scripts/install_bundled_scheduler.sh --print-plan
   bash scripts/install_bundled_scheduler.sh --yes
   export PATH="$HOME/.local/bin:$PATH"
   gpu-scheduler init --gpus 0,1
   gpu-scheduler start
   ```

4. For each physical shard, preview and then run:

   ```bash
   BASE=data/c4-val-def-256 DRY_RUN=1 \
     bash scripts/experiments/swordstamp.sh
   BASE=data/c4-val-def-256 \
     bash scripts/experiments/swordstamp.sh
   ```

   Repeat for `c4-val-def-256b` and `c4-val-def-512`.

5. Compilation requires raw outputs for all 13 paper cells. If a complete raw
   corpus is supplied, compile and plot it with:

   ```bash
   uv run --frozen python -m visualization all \
     --bundle results/paper --output results/paper
   ```

Run `bash scripts/setup.sh --comparisons` once for PMark's isolated environment,
then run `pmark.sh` and `samark.sh` for every shard before step 5. Their vendored
public evaluation snapshots are described in `comparisons/README.md`.

Practical resources are one 24 GB CUDA GPU for provider generation and EDA,
40 GB recommended for large paraphrasers, and an 80 GB GPU or equivalent
aggregate VRAM for the quality judge. Full generation is a multi-day workload
and needs hundreds of GB for models and intermediate data.

## Review scope

- Exact generation budget: 64 candidates.
- Exact semantic-span policy: spaCy, maximum 15 words, comparison window 5.
- Exact EDA-D budgets: 4, 8, 16, 32, and 64.
- Prompt shards: 256, 256, and 512 documents; human null: 1,024 documents.
- PMark rows/runners are online/HD-only; SAMark rows/runners are flags-run only.

First-party code is MIT licensed. The bundled scheduler retains Apache-2.0.
The comparison repositories publish no license file; C4, model checkpoints,
and comparison results retain their original terms. See `THIRD_PARTY.md`.
