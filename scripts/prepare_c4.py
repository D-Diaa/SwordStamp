#!/usr/bin/env python3
"""Prepare every C4 corpus used by the paper artifact.

The source revision, selection policy, seed, and output sizes are deliberately
constants.  The only argument chooses a new output directory; existing outputs
are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from datasets import Dataset, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paper import (  # noqa: E402
    CENTER_DATASET,
    HUMAN_NULL_DATASET,
    PAPER_CORPORA,
)


DATASET_ID = "allenai/c4"
DATASET_CONFIG = "realnewslike"
DATASET_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
SOURCE_SPLIT = "train"
SEED = 42

OUTPUT_SIZES = PAPER_CORPORA
TOTAL_DOCUMENTS = sum(size for _, size in OUTPUT_SIZES)


def _digest(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _selection_key(source_index: int) -> bytes:
    """Return the fully specified, seeded ordering key for one source row."""
    return hashlib.sha256(f"{SEED}:{source_index}".encode("ascii")).digest()


def _source_texts() -> tuple[list[str], int]:
    stream = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=SOURCE_SPLIT,
        revision=DATASET_REVISION,
        streaming=True,
    )
    texts: list[str] = []
    seen: set[str] = set()
    rows_scanned = 0
    for row in stream:
        rows_scanned += 1
        text = row.get("text")
        if not isinstance(text, str) or not text.strip() or text in seen:
            continue
        seen.add(text)
        texts.append(text)
        if len(texts) == TOTAL_DOCUMENTS:
            return texts, rows_scanned
    raise RuntimeError(
        f"{DATASET_ID}/{DATASET_CONFIG}:{SOURCE_SPLIT} ended after "
        f"{len(texts)} usable unique documents; need {TOTAL_DOCUMENTS}"
    )


def prepare(output_dir: Path) -> dict[str, object]:
    if output_dir.is_symlink():
        raise FileExistsError(
            f"refusing to write through symlinked output directory: {output_dir}"
        )

    target_names = [name for name, _ in OUTPUT_SIZES]
    targets = [output_dir / name for name in target_names]
    manifest_path = output_dir / "c4_manifest.json"
    occupied = [path for path in [*targets, manifest_path] if path.exists()]
    if occupied:
        formatted = "\n  ".join(map(str, occupied))
        raise FileExistsError(f"refusing to overwrite existing outputs:\n  {formatted}")

    texts, rows_scanned = _source_texts()
    order = sorted(range(TOTAL_DOCUMENTS), key=_selection_key)

    partitions: dict[str, list[str]] = {}
    offset = 0
    for name, size in OUTPUT_SIZES:
        partitions[name] = [texts[index] for index in order[offset : offset + size]]
        offset += size

    names = tuple(partitions)
    for index, left in enumerate(names):
        left_texts = set(partitions[left])
        for right in names[index + 1 :]:
            if not left_texts.isdisjoint(partitions[right]):
                raise AssertionError(f"C4 outputs are not content-disjoint: {left}, {right}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-c4-", dir=output_dir.parent))
    output_records: dict[str, dict[str, object]] = {}
    manifest: dict[str, object] = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "revision": DATASET_REVISION,
        "source_split": SOURCE_SPLIT,
        "source_rows_scanned": rows_scanned,
        "selection": {
            "source_filter": (
                f"first {TOTAL_DOCUMENTS} unique, nonempty text values in source order"
            ),
            "ordering": (
                "ascending SHA-256 of ASCII '<seed>:<zero-based-index-in-filtered-list>'"
            ),
            "partition_order": [name for name, _ in OUTPUT_SIZES],
            "ordered_text_digest": (
                "SHA-256 over each UTF-8 text prefixed by its 8-byte big-endian length"
            ),
        },
        "seed": SEED,
        "total_documents": TOTAL_DOCUMENTS,
        "pairwise_content_disjoint": True,
        "outputs": output_records,
    }

    try:
        offset = 0
        for name, size in OUTPUT_SIZES:
            selected = partitions[name]
            Dataset.from_dict({"text": selected}).save_to_disk(staging / name)
            output_records[name] = {
                "offset": offset,
                "rows": size,
                "role": "human-null" if name == HUMAN_NULL_DATASET else (
                    "center-training" if name == CENTER_DATASET else "prompt-shard"
                ),
                "ordered_text_sha256": _digest(selected),
            }
            offset += size

        staged_manifest = staging / manifest_path.name
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        raced = [path for path in [*targets, manifest_path] if path.exists()]
        if raced:
            formatted = "\n  ".join(map(str, raced))
            raise FileExistsError(f"outputs appeared while preparing data:\n  {formatted}")
        for name in [*target_names, manifest_path.name]:
            os.replace(staging / name, output_dir / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="new corpus directory (default: data)",
    )
    args = parser.parse_args()

    manifest = prepare(args.output_dir)
    outputs = manifest["outputs"]
    assert isinstance(outputs, dict)
    for name, details in outputs.items():
        assert isinstance(name, str) and isinstance(details, dict)
        print(f"{args.output_dir / name}: {details['rows']} documents")
    print(f"manifest: {args.output_dir / 'c4_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
