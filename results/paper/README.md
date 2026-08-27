# Compiled paper bundle

This review bundle is the output of the strict paper compiler, projected to the
columns consumed by `python -m visualization extract`. It reproduces every
paper table and Matplotlib figure without model weights, GPUs, raw generated
text, or executing comparison-system source code. Its rows cover only the two
five-rung additive ladders, the unwatermarked baseline, PMark online, SAMark
flags-run, and the attacks and transfer measurements reported in the paper.

```bash
uv sync --frozen --only-group plot --no-install-project
uv run --no-sync python -m visualization extract
uv run --no-sync python -m visualization render
```

`cell_summary.csv` and `dataset_summary.csv` retain compiled aggregate evidence.
`per_sample.parquet` retains paired document measurements used by the
bootstraps. `unit_encoder_transfer.parquet` retains only paired displacement
values; source and rewritten C4 sentences were removed because plots do not
consume them. The provider-encoder identifier is part of the public paper
configuration and is retained. Numerical values, grouping keys, and
document/sample identifiers are unchanged.

Every file is below anonymous.4open.science's 8 MB limit. Check the committed
inputs with `(cd results/paper && sha256sum --check SHA256SUMS)`.
