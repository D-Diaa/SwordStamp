# Third-Party Software and Assets

This inventory highlights bundled or revision-sensitive upstream work. It is
not legal advice and does not replace the license text of any dependency.

## Git submodules

| Component | Original source; integration pin | License status in pinned tree |
| --- | --- | --- |
| PMark | [PMark-repo/PMark](https://github.com/PMark-repo/PMark); `git@github.com:D-Diaa/PMark.git` at `5a8ad3008fdc2c58e607de2f60941c2d1911deb9` | No `LICENSE`, `COPYING`, or `NOTICE` file is present. Resolve permission before public redistribution. |
| SAMark | [Z1zs/SAMark](https://github.com/Z1zs/SAMark); `git@github.com:D-Diaa/SAMark.git` at `9160b8025fda05be9a36e02abf0450c1249693cc` | No `LICENSE`, `COPYING`, or `NOTICE` file is present. Resolve permission before public redistribution. |

The gitlinks pin the tested fair-evaluation forks. The optional `branch` entry
in `.gitmodules` is informational; reproducibility comes from the gitlink
commit, not the branch head.

Those integration branches preserve the watermark algorithms and only adapt
dataset/result I/O and common false-positive calibration for this artifact.

## GPU scheduler

`scripts/install_gpu_scheduler.sh` retrieves
`git@github.com:D-Diaa/gpu_scheduler.git` at
`a86c356d13c8b5fad0097e0847e11dcda4579f3d`. That repository declares the
Apache License 2.0. It is installed into the evaluator's user environment and
is not vendored into SwordStamp.

## Direct-source Python dependency

The Parrot paraphraser is installed from
`PrithivirajDamodaran/Parrot_Paraphraser` at commit
`03084c54b64019ba5fa0b620b9c70ad81123e458`. Its transitive models are fetched
from Hugging Face and remain governed by their model-card terms.

## Adapted implementations

Source comments identify routines adapted from SemStamp, k-SemStamp, MarkLLM,
PMark, SAMark, DIPPER, and Parrot. Corresponding scholarly references are in
the paper. Keep those attribution comments when redistributing modified code.

## Models and data

`artifacts/revisions.yaml` records identifiers and content revisions; it does
not redistribute checkpoint or dataset contents. In particular:

- C4 content remains under the terms published with `allenai/c4`;
- Llama checkpoints require acceptance of Meta's terms and Hugging Face access;
- all other Hugging Face checkpoints retain their repository-specific terms;
- OpenAI-backed optional attacks use an external service and cannot pin a
  provider-side model revision; and
- the optional Helsinki-NLP back-translation and remote Parrot generation-code
  paths are explicitly recorded as unverified external inputs rather than
  paper dependencies.

## Root project license

SwordStamp's first-party code is released under the MIT License in `LICENSE`.
That grant does not override the separate terms of the components above.
