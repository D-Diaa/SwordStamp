# Paper experiment pipelines

These scheduler-backed runners reproduce only the final paper experiment
surface. `config/paper.py` is the authoritative registry; the shell-facing
`paper.py` command prints the same registry as TSV or line-oriented values.

```bash
uv run python scripts/experiments/paper.py rungs
uv run python scripts/experiments/paper.py datasets
uv run python scripts/experiments/paper.py oracle-ks
```

Prepare the five mutually content-disjoint C4 corpora before running:

```bash
uv run python scripts/prepare_c4.py --output-dir data
```

The source is `allenai/c4`, configuration `realnewslike`, at the revision
recorded in `scripts/prepare_c4.py`. The command writes 8,192 center-training
documents, prompt shards of 256, 256, and 512 documents in that order, a 1,024
document human null, and `data/c4_manifest.json`. It refuses to overwrite an
existing corpus or write through a symlink.

## Watermark ladder

`swordstamp.sh` runs five ordered additive rungs for LSH and the same five for
KMeans, followed by one shared sentence no-watermark baseline:

| Rung | Mask | Sampling | Segmentation |
| --- | --- | --- | --- |
| base | context | rejection | sentence-NLTK |
| bestn | context | best-of-N | sentence-NLTK |
| fixed | fixed | best-of-N | sentence-NLTK |
| diverse | fixed-diverse | best-of-N | sentence-NLTK |
| span | fixed-diverse | best-of-N | semantic-span, spaCy, max 15/window 5 |

Every watermarked cell uses exactly 64 provider candidates. There are no
Cartesian matrix variables or arbitrary preset overrides. Run the three prompt
shards separately:

```bash
for dataset in c4-val-def-256 c4-val-def-256b c4-val-def-512; do
  BASE="data/$dataset" bash scripts/experiments/swordstamp.sh
done
```

Preview first when checking a scheduler installation:

```bash
DRY_RUN=1 ATTACKS_FILTER='^$' bash scripts/experiments/swordstamp.sh
```

`attacks.sh` schedules exactly the paper attacks: Pegasus and Parrot with and
without 3%-tolerance bigram selection; DIPPER at
`(L,O)={(20,20),(20,40),(20,80),(40,20),(80,20)}`; the 16 controlled reorder,
synonym, split, and merge
conditions on the two ladders; and BGE-surrogate EDA-P/EDA-S at the reported
budgets. EDA-D uses K = 4, 8, 16, 32, and 64 only on `kbase` and `kspan`.
Other implementations retained in `attacks/` are library functionality and are
not scheduled or compiled as paper experiments.

## Comparison cells

The public snapshots under `comparisons/` expose only one scheduled paper cell
per system: online/HD PMark and flags-run SAMark, both with N=64. Install the
PMark environment with `bash scripts/setup.sh --comparisons`, then preview:

```bash
DRY_RUN=1 ATTACKS_FILTER='^$' bash scripts/experiments/pmark.sh
DRY_RUN=1 ATTACKS_FILTER='^$' bash scripts/experiments/samark.sh
```

The workers bridge comparison output into the shared attack and quality stages.
The strict compiler requires these two cells plus the 11 first-party cells.

## Scheduling and resume behavior

The local GPU scheduler supplies `gpu-enqueue`, injects
`CUDA_VISIBLE_DEVICES`, and owns GPU selection. `_lib.sh` forwards only the
canonical layered runtime override
`SEMSTAMP__RUNTIME__VLLM_UTILIZATION`. `VLLM_MAX_MODEL_LEN`, `PRIORITY`,
`AFTER`, and `TASK_IDS_FILE` are operational scheduling controls.

`QUALITY_BATCH` is `off`, `cell`, or `all`. `ATTACKS_FILTER` restricts attack
labels without changing the declared watermark cells. Existing canonical
outputs are skipped; the force flags rerun a stage and its required downstream
work:

| Flag | Effect |
| --- | --- |
| `--force-gen` | regenerate and rerun downstream stages |
| `--force-att` | rerun attacks, detection, and attack quality |
| `--force-det` | rerun detection only |
| `--force-eval` | rerun quality only |

## Workers and paper outputs

- `_gen.sh`, `_attack.sh`, and `_detect.sh` run one native stage.
- `_pmark_*` and `_samark_*` run the two public comparison snapshots.
- `_quality_batch.sh` evaluates explicit output directories.
- `_plan.py` derives paths and checks completion.
- `_viz.sh` invokes the retained per-directory `detection_summary` report.

The installable visualization package consumes the same `config.paper`
registry and the extractor logic supplied with the paper source:

```bash
uv run python -m visualization compile --help
uv run python -m visualization extract --help
uv run python -m visualization render --help
uv run python -m visualization all --help
```

Use `--bundle` and `--output` to choose the compiled input and generated table
or Matplotlib output locations.
