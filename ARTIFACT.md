# Artifact Evaluation Guide

This guide separates a short functional evaluation from the full paper
reproduction. Exact immutable inputs are recorded in
`artifacts/revisions.yaml`; generated datasets and model checkpoints are not
redistributed.

## 1. Scope

The artifact supports these paper claims:

| Claim or output | Primary commands |
| --- | --- |
| Exact additive LSH and k-means ladders, N=64 | `python scripts/experiments/paper.py rungs`; `DRY_RUN=1 bash scripts/experiments/swordstamp.sh` |
| Clean detection, fidelity, and provider cost | `swordstamp.sh`, then `python -m visualization compile` and `extract` |
| Score retention under rewording, reordering, and resegmentation | `swordstamp.sh`, then `python -m visualization all` |
| No-box attack comparison | `swordstamp.sh`, `pmark.sh`, `samark.sh`, then `visualization all` |
| Detector-access EDA on k-SemStamp and k-SwordStamp | the oracle cells in `swordstamp.sh`, then `visualization all` |
| PMark online and SAMark flags-run comparison | `pmark.sh` and `samark.sh` |
| Paper tables and figures from an existing bundle | `visualization extract` and `visualization render` |
| Per-directory detector sanity plots | `scripts/experiments/detection_summary.py` or `_viz.sh` |

The retained attack package exposes additional attacks that are not paper
cells. Their presence does not expand the paper protocol in `config/paper.py`.

## 2. Functional path

Expected evaluator time is 5–15 minutes after dependencies are available; a
first dependency install can take substantially longer and uses several GB.
This path needs no GPU and downloads no checkpoints or C4 content.

```bash
bash scripts/setup.sh
uv run --frozen python scripts/check_artifact.py
uv run --frozen python scripts/experiments/paper.py rungs

MPLCONFIGDIR=/tmp/swordstamp-matplotlib \
  uv run --frozen \
  python -m unittest discover -s tests -v

DRY_RUN=1 bash scripts/experiments/swordstamp.sh
DRY_RUN=1 bash scripts/experiments/pmark.sh
DRY_RUN=1 bash scripts/experiments/samark.sh
```

Success means that submodule and center hashes pass, the CPU suite passes, GPU
tests either pass or skip explicitly, and each dry run lists only the paper
cells.

With a supplied compiled bundle, also run:

```bash
uv run --frozen python -m visualization extract \
  --bundle results/paper --output results/paper
uv run --frozen python -m visualization render \
  --bundle results/paper --output results/paper
```

## 3. Full path

Full reproduction consists of five stages:

1. prepare and verify the C4 partitions;
2. verify or download every recorded model revision;
3. configure the pinned GPU scheduler;
4. run the three orchestrators on each physical shard; and
5. compile, extract, and render the pooled bundle.

### Data

```bash
uv run --frozen python scripts/prepare_c4.py --output-dir data
```

The source is `allenai/c4`, configuration `realnewslike`, train split, revision
`1588ec454efa1a09f29cd18ddd04fe05fc8653a2`. From the first 10,240 unique,
nonempty texts, the script uses a seed-42 SHA-256 order and partitions documents
in this order:

| Output | Documents | Purpose |
| --- | ---: | --- |
| `data/c4-center-8192` | 8,192 | k-means center fitting |
| `data/c4-val-def-256` | 256 | evaluation shard 1 |
| `data/c4-val-def-256b` | 256 | evaluation shard 2 |
| `data/c4-val-def-512` | 512 | evaluation shard 3 |
| `data/c4-human-def` | 1,024 | disjoint empirical detector null |

`data/c4_manifest.json` records ordered text digests and asserts content
disjointness. The preparation script refuses existing or symlinked targets so
it cannot overwrite a live experiment store.

### Models

Run `uv run python scripts/check_artifact.py --models` to compare the Hugging
Face cache with the exact commits in `artifacts/revisions.yaml`. Use
`--require-models` on a fully provisioned host. The full paper path requires:

- `meta-llama/Llama-3.1-8B` for provider generation and perplexity;
- `AbeHou/SemStamp-c4-sbert` for SwordStamp and SemStamp regions;
- `sentence-transformers/all-mpnet-base-v2` for PMark and SAMark;
- `Qwen/Qwen2.5-3B-Instruct` and `BAAI/bge-base-en-v1.5` for no-box EDA;
- `kalpeshk2011/dipper-paraphraser-xxl` and `google/t5-v1_1-xxl` for DIPPER;
- `Qwen/Qwen3-32B` for the three-run quality judges; and
- the quality models listed with `scope: paper-quality` or `scope: paper` in
  the revision manifest.

The Llama checkpoint is gated. Accept its upstream terms and authenticate with
Hugging Face before downloading. Checkpoints remain governed by their upstream
licenses and can require hundreds of GB in aggregate.

### Hardware and runtime

| Stage | Minimum practical resource | Guidance |
| --- | --- | --- |
| Functional checks / bundle rendering | CPU, 16 GB RAM | No CUDA or model download required. |
| Llama-3.1-8B generation and EDA | one CUDA GPU with at least 24 GB VRAM | More VRAM permits larger active batches. |
| DIPPER and comparison generation | one CUDA GPU with at least 40 GB VRAM recommended | The scheduler marks exclusive vLLM jobs where needed. |
| Qwen3-32B judges | one 80 GB GPU, or adapt vLLM tensor parallelism to equivalent aggregate VRAM | The paper uses local vLLM, three judge repetitions. |
| Full corpus | hundreds of GB of local model/data/output storage | Keep `data/` and `results/` on scratch-backed storage. |

Runtime depends strongly on GPU type, vLLM version, and queue concurrency. The
full matrix contains 1,024 provider documents, a 1,024-document null, two
five-stage ladders, two comparison systems, controlled edits, and attack-effort
grids; budget multi-day scheduler access rather than an interactive AE slot.
Always use `DRY_RUN=1` first to record the submitted DAG and job count.

### Scheduler

```bash
bash scripts/install_gpu_scheduler.sh --print-plan
bash scripts/install_gpu_scheduler.sh --yes
gpu-scheduler init --gpus 0,1,...
gpu-scheduler start
uv run --frozen python scripts/check_artifact.py --scheduler
```

The script clones `git@github.com:D-Diaa/gpu_scheduler.git` at
`a86c356d13c8b5fad0097e0847e11dcda4579f3d`. Its upstream installer writes
`~/.gpu-scheduler` and `~/.local/bin`; if a dispatcher is already running, it
stops and restarts it while replacing installed files. The wrapper deliberately
does not select GPUs, initialize the pool, or start a new dispatcher.

### Experiment commands

For each of `data/c4-val-def-256`, `data/c4-val-def-256b`, and
`data/c4-val-def-512`, preview and then run:

```bash
BASE=data/c4-val-def-256 DRY_RUN=1 bash scripts/experiments/swordstamp.sh
BASE=data/c4-val-def-256 DRY_RUN=1 bash scripts/experiments/pmark.sh
BASE=data/c4-val-def-256 DRY_RUN=1 bash scripts/experiments/samark.sh

BASE=data/c4-val-def-256 bash scripts/experiments/swordstamp.sh
BASE=data/c4-val-def-256 bash scripts/experiments/pmark.sh
BASE=data/c4-val-def-256 bash scripts/experiments/samark.sh
```

Substitute the other two shard paths. Re-running is resumable: completed stage
outputs are skipped unless the documented force flag is used. Never choose a
GPU by probing `nvidia-smi`; the scheduler injects `CUDA_VISIBLE_DEVICES`.

### Compile and render

```bash
uv run --frozen python -m visualization compile \
  --bundle results/paper
uv run --frozen python -m visualization extract \
  --bundle results/paper --output results/paper
uv run --frozen python -m visualization render \
  --bundle results/paper --output results/paper
```

The equivalent one-shot command is:

```bash
uv run --frozen python -m visualization all \
  --bundle results/paper --output results/paper
```

## 4. Expected outputs and validation

The compiled bundle contains:

- `manifest.json`: consumed paths, configuration filters, counts, and methods;
- `cell_summary.csv`: pooled paper cells;
- `dataset_summary.csv`: one row per physical shard;
- `per_sample.parquet`: paired document-level evidence; and
- encoder-transfer Parquet and provenance files when that analysis is enabled.

Extraction writes deterministic table fragments and plot data under the chosen
output root. Rendering writes nonempty `.png` and `.pdf` figures. Required-cell
coverage, deterministic bootstrap settings, and source fingerprints are checked
by the visualization code and tests.

The two committed center tensors can be checked independently:

```bash
cd artifacts/centers
sha256sum --check SHA256SUMS
```

## 5. Exact external pins

| Component | Revision |
| --- | --- |
| PMark `codex/fair-evaluation` | `5a8ad3008fdc2c58e607de2f60941c2d1911deb9` |
| SAMark `codex/fair-evaluation` | `9160b8025fda05be9a36e02abf0450c1249693cc` |
| gpu_scheduler | `a86c356d13c8b5fad0097e0847e11dcda4579f3d` |
| authoritative paper source used for this artifact | `8c1ac5cd6d9cfcd76c4051a2457854e03576bbfd` |

The PMark environment is intentionally separate and pinned in
`requirements/pmark.txt`. SAMark runs in the root frozen uv environment.

## 6. Distribution and licensing

Generated text, C4 content, and model checkpoints are excluded because of size
and upstream terms. PMark and SAMark are gitlinks, not copied working trees.

SwordStamp is released under the MIT License. PMark and SAMark ship no license
file at their pinned revisions; they remain separate gitlinks rather than
redistributed source trees. See `THIRD_PARTY.md` before using those comparisons.
