"""Unit-level encoder-transfer measurements for landed adaptive attacks.

This module deliberately does not generate or attack text.  It segments the
marked and selected attacked documents already stored on disk, fixes a
positional sentence alignment, and evaluates that same alignment with the
attacker surrogate and the watermark provider encoder.
"""
from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from datasets import load_from_disk

from config.paper import PROVIDER_CANDIDATES
from segmentation import segment


K_SWEEP = (1, 2, 4, 8, 16, 32, 64)
SURROGATE_ENCODER = "BAAI/bge-base-en-v1.5"
SEMSTAMP_ENCODER = "AbeHou/SemStamp-c4-sbert"
PMARK_ENCODER = "sentence-transformers/all-mpnet-base-v2"
OUTPUT_STEM = "unit_encoder_transfer"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TransferCell:
    """One physical adaptive-attack shard included in the transfer table."""

    dataset: str
    family: str
    mask: str
    sampling: str
    segmentation: str
    attack: str
    K: int
    provider_encoder: str
    directory: str


def discover_transfer_cells(
    root: str | Path,
    datasets: Sequence[str],
) -> list[TransferCell]:
    """Find selected sentence-context adaptive arms at the requested budgets."""
    root = Path(root).resolve()
    cells: list[TransferCell] = []
    for dataset in datasets:
        pmark_base = (
            root / "data" / dataset / "pmark" / "online" / "rejection" /
            "sentence-nltk"
        )
        for budget in K_SWEEP:
            attack = (
                f"adaptive-Qwen2.5-3B-Instruct-standard-K{budget}-min-surr=bge"
            )
            for family in ("lsh",):
                base = (
                    root / "data" / dataset / family / "context" / "rejection" /
                    "sentence-nltk" / f"candidates-{PROVIDER_CANDIDATES}"
                )
                directory = base / attack
                if (directory / "dataset_info.json").exists():
                    cells.append(TransferCell(
                        dataset=dataset, family=family, mask="context",
                        sampling="rejection", segmentation="sentence-nltk",
                        attack=attack, K=budget,
                        provider_encoder=SEMSTAMP_ENCODER,
                        directory=str(directory),
                    ))

            attack = f"adp-K{budget}"
            directory = pmark_base / attack
            if (directory / "dataset_info.json").exists():
                cells.append(TransferCell(
                    dataset=dataset, family="pmark", mask="online",
                    sampling="rejection", segmentation="sentence-nltk",
                    attack=attack, K=budget, provider_encoder=PMARK_ENCODER,
                    directory=str(directory),
                ))
    return cells


def align_landed_sentences(cells: Iterable[TransferCell]) -> pd.DataFrame:
    """Return every positional sentence pair retained in landed attack text.

    Alignment is fixed before either encoder is evaluated.  Pairing sentence
    index ``i`` with sentence index ``i`` avoids introducing an encoder into the
    alignment algorithm; unmatched sentences are reported through the count
    columns and are not silently represented as pairs.
    """
    rows: list[dict] = []
    for cell in cells:
        dataset = load_from_disk(cell.directory)
        missing = {"text", "para_text"} - set(dataset.column_names)
        if missing:
            raise ValueError(f"{cell.directory} lacks columns: {sorted(missing)}")
        for document_id, example in enumerate(dataset):
            marked = segment(example["text"], type="sentence", backend="nltk")
            attacked = segment(example["para_text"], type="sentence", backend="nltk")
            aligned_count = min(len(marked), len(attacked))
            counts = dict(
                marked_sentence_count=len(marked),
                attacked_sentence_count=len(attacked),
                aligned_sentence_count=aligned_count,
            )
            for sentence_index in range(aligned_count):
                rows.append(dict(
                    dataset=cell.dataset,
                    family=cell.family,
                    mask=cell.mask,
                    sampling=cell.sampling,
                    segmentation=cell.segmentation,
                    attack=cell.attack,
                    K=float(cell.K),
                    document_id=document_id,
                    sample_id=document_id,
                    sentence_index=sentence_index,
                    marked_sentence=marked[sentence_index].normalized,
                    attacked_sentence=attacked[sentence_index].normalized,
                    provider_encoder=cell.provider_encoder,
                    **counts,
                ))
    return pd.DataFrame(rows)


def _encode_unique(model, rows: pd.DataFrame, indices: np.ndarray, batch_size: int) -> np.ndarray:
    """Encode unique sentence strings once and return pair displacements."""
    marked = rows.loc[indices, "marked_sentence"].tolist()
    attacked = rows.loc[indices, "attacked_sentence"].tolist()
    unique = list(dict.fromkeys(marked + attacked))
    lookup = {text: index for index, text in enumerate(unique)}
    embeddings = np.asarray(model.encode(
        unique,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ))
    marked_index = np.fromiter((lookup[text] for text in marked), dtype=np.int64)
    attacked_index = np.fromiter((lookup[text] for text in attacked), dtype=np.int64)
    cosine = np.einsum(
        "ij,ij->i", embeddings[marked_index], embeddings[attacked_index], optimize=True,
    )
    return np.maximum(0.0, 1.0 - cosine)


def add_encoder_displacements(
    rows: pd.DataFrame,
    *,
    device: str = "cpu",
    batch_size: int = 128,
) -> pd.DataFrame:
    """Evaluate one fixed alignment under BGE and each provider encoder."""
    if rows.empty:
        rows = rows.copy()
        rows["surrogate_encoder"] = pd.Series(dtype=str)
        rows["surrogate_cosine_displacement"] = pd.Series(dtype=float)
        rows["provider_cosine_displacement"] = pd.Series(dtype=float)
        return rows

    # Import lazily: cached bundles do not need to load torch/model machinery.
    from sentence_transformers import SentenceTransformer

    result = rows.copy()
    result["surrogate_encoder"] = SURROGATE_ENCODER
    all_indices = result.index.to_numpy()
    print(f"encoder transfer: loading surrogate {SURROGATE_ENCODER} on {device}")
    model = SentenceTransformer(SURROGATE_ENCODER, device=device)
    result["surrogate_cosine_displacement"] = _encode_unique(
        model, result, all_indices, batch_size,
    )
    del model
    gc.collect()

    result["provider_cosine_displacement"] = np.nan
    for encoder in result["provider_encoder"].drop_duplicates():
        indices = result.index[result["provider_encoder"] == encoder].to_numpy()
        print(f"encoder transfer: loading provider {encoder} on {device}")
        model = SentenceTransformer(encoder, device=device)
        result.loc[indices, "provider_cosine_displacement"] = _encode_unique(
            model, result, indices, batch_size,
        )
        del model
        gc.collect()
    return result


def _source_fingerprint(cells: Sequence[TransferCell]) -> dict:
    sources = []
    for cell in cells:
        directory = Path(cell.directory)
        files = sorted(directory.glob("*.arrow")) + [directory / "dataset_info.json"]
        sources.append(dict(
            cell=asdict(cell),
            files=[dict(
                name=path.name,
                size=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns,
            ) for path in files],
        ))
    payload = dict(
        schema_version=SCHEMA_VERSION,
        surrogate_encoder=SURROGATE_ENCODER,
        alignment="positional sentence index after sentence-nltk segmentation",
        distance="max(0, 1 - cosine) over normalized sentence embeddings",
        sources=sources,
    )
    encoded = json.dumps(payload, sort_keys=True).encode()
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def build_transfer_table(
    root: str | Path,
    datasets: Sequence[str],
    out: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 128,
    force: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Build or reuse the unit table and return it with its provenance."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    parquet_path = out / f"{OUTPUT_STEM}.parquet"
    manifest_path = out / f"{OUTPUT_STEM}.manifest.json"
    cells = discover_transfer_cells(root, datasets)
    fingerprint = _source_fingerprint(cells)

    cached = None
    if manifest_path.exists():
        try:
            cached = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            cached = None
    if (
        not force and parquet_path.exists() and cached is not None
        and cached.get("source_fingerprint", {}).get("sha256") == fingerprint["sha256"]
    ):
        print(f"encoder transfer: reusing {parquet_path}")
        frame = pd.read_parquet(parquet_path)
        return frame, cached

    aligned = align_landed_sentences(cells)
    frame = add_encoder_displacements(aligned, device=device, batch_size=batch_size)
    frame.to_parquet(parquet_path, index=False)

    by_cell = []
    if not frame.empty:
        for identity, group in frame.groupby(
            ["dataset", "family", "mask", "sampling", "segmentation", "attack", "K"],
            sort=True,
        ):
            by_cell.append(dict(zip(
                ["dataset", "family", "mask", "sampling", "segmentation", "attack", "K"],
                identity,
            ), rows=len(group), documents=int(group["document_id"].nunique())))
    manifest = dict(
        generated_by="visualization.compile_results",
        source_fingerprint=fingerprint,
        rows=len(frame),
        cells=len(cells),
        cell_counts=by_cell,
        sentence_index_base=0,
        text_columns_note=(
            "marked_sentence and attacked_sentence are normalized Unit.normalized strings "
            "and are the exact inputs to both encoders"
        ),
        retained_output_limitation=(
            "Attack datasets persist only the final selected rewrite at each K, not all K "
            "candidate rewrites. This table contains every aligned pair recoverable without "
            "rerunning attacks."
        ),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return frame, manifest
