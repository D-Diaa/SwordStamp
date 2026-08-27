# Third-party inputs

SwordStamp's first-party code is MIT licensed. That license does not change the
terms of datasets, models, services, or comparison implementations.

## Public comparison implementations

`comparisons/pmark/` is derived from the public
[PMark repository](https://github.com/PMark-repo/PMark), and
`comparisons/samark/` from the public
[SAMark repository](https://github.com/Z1zs/SAMark). Exact upstream bases,
bundled file hashes, exclusions, calibration adapters, and local input-only
changes are recorded in `comparisons/README.md` and `artifacts/revisions.yaml`.
Neither pinned upstream tree contains a `LICENSE`, `COPYING`, or `NOTICE` file.
The root MIT License does not cover these directories; consult the original
authors and repository terms before reuse or redistribution.

Six SAMark-bundled [MarkLLM](https://github.com/THU-BPM/MarkLLM) utilities
retain their individual Apache-2.0 copyright headers. The full license is in
`comparisons/LICENSES/MarkLLM-APACHE-2.0.txt` and applies only to those
header-marked files, as enumerated in `comparisons/README.md`.

## Bundled GPU scheduler

`tools/gpu_scheduler/` is the pinned runtime snapshot used for the experiments.
`scripts/install_bundled_scheduler.sh` installs it only after explicit approval,
without cloning or downloading. Exact installs are unchanged; running or
conflicting non-exact installs are refused. Its runtime files are unmodified and
retain the included Apache-2.0 license; origin metadata and the repository
revision are withheld for review.

## Data and models

- C4 content is not redistributed. `scripts/prepare_c4.py` downloads the pinned
  `allenai/c4` revision and records document hashes.
- Model checkpoints are not redistributed and retain their repository terms.
- The provider checkpoint is gated and requires separate authorization.
- The public `AbeHou/SemStamp-c4-sbert` provider encoder is used at the exact
  revision recorded in `artifacts/revisions.yaml`.
- Optional OpenAI attacks use a remote service whose model revision is outside
  this artifact's control.

## Python dependencies

`uv.lock` records exact resolved packages. The plot-only reviewer path installs
only Matplotlib, NumPy, PyArrow, and their small runtime dependencies. The full
environment includes CUDA/model packages and optional attack dependencies under
their respective licenses.
