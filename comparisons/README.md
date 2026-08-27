# Public comparison implementations

These ordinary source snapshots make the review artifact self-contained. They
come from the authors' public repositories, not from SwordStamp:

| System | Original public repository | Upstream base | Bundled source aggregate |
| --- | --- | --- | --- |
| PMark | [PMark-repo/PMark](https://github.com/PMark-repo/PMark) | `75140fe7142f88c51ba50984d9eb07f44f5961b3` | `ee623782f772e72aa9b566e58a73b316e75a54e004eed796292f28f51f5f1f03` |
| SAMark | [Z1zs/SAMark](https://github.com/Z1zs/SAMark) | `e098bd347ba936c3e9335db6f24632fd0c319141` | `e0b4cc27a82269371b8475bd6061b4964f8447656323b134cbdbd13a3866ae26` |

The imported integration snapshot contains every tracked source/configuration
file. We omit only the tracked C4/BookSum demonstration dataset
files and PMark's tracked Python bytecode. Those are inputs or generated caches,
not source; the BookSum Arrow file also exceeds anonymous.4open.science's 8 MB
per-file limit. No model, generated text, log, or repository history is copied.

The integration patches preserve the watermark algorithms while adapting their
experiment boundaries. PMark gains arbitrary Hugging Face dataset input,
Hugging Face output, explicit sampler settings, and standard result tables;
online/HD detection retains PMark's analytical normal null. SAMark gains the
same dataset bridge and emits raw scores, then calibrates 1% and 5% operating
points and AUROC against the shared 1,024-document human null. This empirical
calibration replaces SAMark's fixed `z=3` binary decision so every system is
reported at the same measured false-positive rates; it does not change the raw
SAMark score. Early-stopped generated rows are retained by both exporters.

SAMark also replaces one host-local OPT tokenizer default with the equivalent
public `facebook/opt-1.3b` identifier and makes one exporter docstring path
repository-independent. `artifacts/revisions.yaml` records file counts and
aggregate checksums, and `scripts/check_artifact.py` verifies them.
Against the public upstream bases, the integration diff is five PMark files
(+635/-204 lines) and seven SAMark files (+462/-191); the manifest lists every
changed path.

SwordStamp uses only the paper cells: online/HD PMark and flags-run SAMark, each
with 64 candidates. The workers feed their Hugging Face outputs to SwordStamp's
unchanged attack suite and shared quality pipeline. The comparison repositories'
own attack/evaluation utilities remain present for source completeness but are
not scheduled by the SwordStamp runners.

Neither original repository publishes a `LICENSE`, `COPYING`, or `NOTICE` file
in the pinned tree. The root MIT License therefore does not grant rights to
`comparisons/pmark/` or `comparisons/samark/`; consult the original authors and
repository terms before redistribution or reuse.

Six utilities bundled by SAMark retain their original `Copyright 2024 THU-BPM
MarkLLM` Apache-2.0 headers: `evaluation/tools/{oracle,
success_rate_calculator,text_editor,text_quality_analyzer}.py`,
`exceptions/exceptions.py`, and `utils/openai_utils.py`. Their complete license
is preserved at `comparisons/LICENSES/MarkLLM-APACHE-2.0.txt`. That license does
not imply a license for the remaining PMark or SAMark files.
