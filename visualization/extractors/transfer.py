"""Build fig:transfer: displacement under the provider encoder, given the surrogate's.

Reads unit_encoder_transfer.parquet, one row per aligned marked/attacked sentence
pair with its cosine displacement measured under both encoders from the same
alignment, so the encoder is the only thing that varies.

The figure tests one assumption: a candidate that moves far under the
attacker's surrogate also moves far under the provider's encoder.  That is a
statement about how the paired measurements move together, so we bin pairs by
surrogate displacement and report the provider displacement within each bin.
A marginal summary would not do: the marginal distribution is bimodal, with a
near-copy mode and a fully-displaced mode, so neither its mean nor its median
describes a typical pair.  Conditioning removes that problem, since each bin
holds pairs the surrogate scored alike.

Per scheme, emits <scheme>-band.dat: one row per bin with the bin's mean
surrogate displacement, and the median and quartiles of provider displacement
over the pairs in it.

Run through ``python -m visualization extract``.
"""

from __future__ import annotations

import collections
import statistics

from .common import (
    DATA_ROOT,
    bundle_path,
    prune_unwritten,
    read_manifest,
    read_parquet_rows,
    write_manifest,
    write_table,
)

OUT_DIR = DATA_ROOT / "transfer"
SOURCE = "unit_encoder_transfer.parquet"


def require_source_freshness(bundle) -> None:
    """Require the transfer table to have been produced by this compilation."""
    rows = read_manifest(bundle).get("counts", {}).get(
        "unit_encoder_transfer_rows", None)
    if rows:
        return
    raise SystemExit(
        f"{SOURCE} has no rows in this bundle's manifest; recompile the paper bundle"
    )


# Equal-width bins over the observed surrogate range.  Equal-width rather than
# equal-count, so a bin's horizontal position means the same thing on both
# curves and the two schemes stay comparable; MIN_BIN then drops the sparse
# tail instead of letting a handful of pairs set a quartile.
BINS = 16
MIN_BIN = 30

# (key, scheme, mask) -> the curve that scheme contributes.
SCHEMES = [
    ("semstamp", "SemStamp", "context"),
    ("pmark", "PMark", "online"),
]


def band(points: list[tuple[float, float]]) -> list[list[float]]:
    """Return one row per populated bin: mean x, then median and quartiles of y."""
    top = max(x for x, _ in points)
    width = top / BINS
    binned: dict[int, list[tuple[float, float]]] = collections.defaultdict(list)
    for x, y in points:
        binned[min(BINS - 1, int(x / width))].append((x, y))

    rows = []
    for index in sorted(binned):
        cell = binned[index]
        if len(cell) < MIN_BIN:
            continue
        ys = sorted(y for _, y in cell)
        # Quartiles by position, so a bin of any size yields the same summary
        # that a reader would compute from the pairs themselves.
        lower = statistics.median(ys[: len(ys) // 2])
        upper = statistics.median(ys[(len(ys) + 1) // 2:])
        rows.append([
            statistics.mean(x for x, _ in cell),
            statistics.median(ys),
            lower,
            upper,
            float(len(cell)),
        ])
    return rows


def main() -> None:
    bundle = bundle_path()
    path = bundle / SOURCE
    if not path.exists():
        raise SystemExit(
            f"{path} is missing; the transfer figure needs the unit-level "
            "encoder table, not just per_sample.parquet"
        )
    require_source_freshness(bundle)

    rows = read_parquet_rows(path)

    grouped: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    documents: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    encoders: dict[str, tuple[str, str]] = {}
    for row in rows:
        # The oracle holds the provider's encoder, so its displacement is
        # trivially "transferred" and would collapse this comparison.  Filtered
        # even though today's source table carries no oracle rows.
        if row["attack_family"] != "adaptive" or row["surrogate"] == "detector":
            continue
        for key, scheme, mask in SCHEMES:
            if row["scheme"] == scheme and row["mask"] == mask:
                grouped[key].append((
                    float(row["surrogate_cosine_displacement"]),
                    float(row["provider_cosine_displacement"]),
                ))
                documents[key].add((row["dataset"], row["document_id"]))
                encoders[key] = (row["surrogate_encoder"], row["provider_encoder"])
                break

    notes: list[str] = []
    written = set()
    pair_counts: dict[str, int] = {}
    for key, scheme, _ in SCHEMES:
        points = grouped.get(key, [])
        if not points:
            notes.append(f"MISSING  {key}: no adaptive rows in this bundle")
            continue
        table = band(points)
        path = OUT_DIR / f"{key}-band.dat"
        write_table(path, ["x", "y", "lo", "hi", "n"], table)
        written.add(path)
        pair_counts[key] = len(points)
        notes.append(
            f"{key:9s} n={len(points):<6} pairs from {len(documents[key])} "
            f"documents, bins={len(table)}/{BINS} (>= {MIN_BIN} pairs each), "
            f"y median {table[0][1]:.3f}..{table[-1][1]:.3f} over "
            f"x {table[0][0]:.3f}..{table[-1][0]:.3f}"
        )
        if key in encoders:
            notes.append(f"{key:9s} surrogate={encoders[key][0]} "
                         f"provider={encoders[key][1]}")

    missing = sorted({key for key, _, _ in SCHEMES} - set(pair_counts))
    if missing:
        raise SystemExit(
            "cannot emit the transfer figure; missing " + ", ".join(missing)
        )

    def grouped_integer(value: int) -> str:
        return r"\,".join(f"{value:,}".split(","))

    # The caption states one document count for the figure, so the two curves
    # must rest on the same corpus; a bundle that measured one scheme on fewer
    # documents would otherwise print a count that is true of neither.
    covered = {key: len(docs) for key, docs in documents.items()}
    if len(set(covered.values())) != 1:
        raise SystemExit(
            "the transfer curves cover different document sets, so the figure "
            "cannot name one count: " + ", ".join(
                f"{key}={count}" for key, count in sorted(covered.items()))
        )

    constants = [
        r"% Generated by python -m visualization extract -- do not edit.",
        rf"\newcommand{{\resultTransferDocs}}"
        rf"{{{grouped_integer(next(iter(covered.values())))}}}",
    ]
    constants_path = OUT_DIR / "constants.tex"
    constants_path.write_text("\n".join(constants) + "\n")

    removed = prune_unwritten(OUT_DIR, written)
    for path in removed:
        notes.append(f"removed stale table {path.name}")

    write_manifest(OUT_DIR / "MANIFEST.txt", bundle, notes,
                   uncertainty="interquartile range of provider displacement "
                               "within each surrogate-displacement bin")
    print(f"transfer: wrote {len(written)} tables to {OUT_DIR}")
    for note in notes:
        if note.startswith(("MISSING", "removed")):
            print("  " + note)


if __name__ == "__main__":
    main()
