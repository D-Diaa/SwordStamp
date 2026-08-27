"""Build the teaser: best no-box attacker against every scheme.

One curve per scheme: at each quality bar, the maximum ASR@5% over every
non-oracle attack and setting.  The oracle is excluded because it is not a
no-box attacker.  A scheme with no data in the bundle is skipped and named in
the manifest, and the figure tolerates a missing .dat.

Emits tables/pgfplots/teaser/<key>.dat with columns bar, y, em, ep, n.
"""

from __future__ import annotations

import collections
from pathlib import Path

from .common import (
    BARS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    DATA_ROOT,
    FRONTIER_SCHEMES,
    bar_value,
    bundle_path,
    document_id,
    num,
    prune_unwritten,
    read_cell_summary,
    read_per_sample,
    resample_indices,
    write_manifest,
    write_shared_macros,
    write_table,
)

OUT_DIR = DATA_ROOT / "teaser"

# The strongest observed no-box attacker, whatever design it was aimed at: this
# figure reports what a scheme concedes.  Configurations and the bar grid come
# from common, shared with cross_scheme so the two cannot drift.

# The oracle holds the provider's encoder, so it is not a no-box attacker.
EXCLUDED_FAMILIES = {"oracle"}

# A curve is comparable only if the scheme faced the same class of attacker;
# otherwise a coverage gap looks like a robustness difference.
REQUIRED_FAMILIES = {"adaptive", "dipper"}


def bootstrap_max_ci(success: list[list[float]]) -> tuple[float, float]:
    """Percentile interval for the maximum ASR over the evaluated attacks.

    ``success[attack][document]`` holds the binary outcome at one quality bar.
    Every resample draws document indices once, reuses that draw for every
    attack, then recomputes the maximum over attack settings.  Resampling a
    document as a block preserves the dependence among attacks evaluated on
    that document.

    The observed maximum remains the plotted point.  The bootstrap mean is
    higher when several attacks are close, because selecting a winner is
    optimistic; it is not a bias-corrected point estimate and is not plotted.
    These intervals describe the envelope, not a fixed selected attack, and
    overlap is not a test of the gap between two scheme curves.
    """
    if not success or not success[0]:
        return float("nan"), float("nan")

    attack_count = len(success)
    document_count = len(success[0])
    if any(len(row) != document_count for row in success):
        raise ValueError("attacks in a teaser cell do not share a document set")

    draws: list[float] = []
    for document_indices in resample_indices(document_count):
        counts = [
            sum(success[attack][document] for document in document_indices)
            for attack in range(attack_count)
        ]
        draws.append(max(counts) / document_count)

    draws.sort()
    tail = (1.0 - CI_LEVEL) / 2.0
    return (
        draws[int(tail * (BOOTSTRAP_RESAMPLES - 1))],
        draws[int((1.0 - tail) * (BOOTSTRAP_RESAMPLES - 1))],
    )


def main() -> None:
    bundle = bundle_path()
    rows = read_per_sample(bundle)
    summary = read_cell_summary(bundle)

    notes: list[str] = []
    written: set[Path] = set()

    for config in FRONTIER_SCHEMES:
        key, label = config.key, config.label
        cells = [
            r for r in rows
            if config.matches(r)
            and r["stage"] == "attack"
            and r["attack_family"] not in EXCLUDED_FAMILIES
            # No judge score, no success.  Drops the probe atoms and stops an
            # unjudged family from satisfying the coverage guard below.
            and num(r, "llm_judge") == num(r, "llm_judge")
        ]
        if not cells:
            raise SystemExit(f"{key} ({label}): no judged no-box attack rows")

        present = {r["attack_family"] for r in cells}
        absent = REQUIRED_FAMILIES - present
        if absent:
            raise SystemExit(
                f"{key}: incomplete no-box coverage; missing "
                + ", ".join(sorted(absent))
            )

        # Documents are the bootstrap unit, so success is indexed
        # (attack x document); a document with no row counts as a failure.
        documents = sorted({document_id(r) for r in cells})
        doc_index = {d: i for i, d in enumerate(documents)}
        attacks = sorted({r["attack"] for r in cells})
        attack_index = {a: i for i, a in enumerate(attacks)}

        evaded = collections.defaultdict(bool)
        judged: dict[tuple[int, int], float] = {}
        for r in cells:
            slot = (attack_index[r["attack"]], doc_index[document_id(r)])
            evaded[slot] = num(r, "evades_at5") >= 0.5
            judged[slot] = num(r, "llm_judge")

        table = []
        owners = []
        for bar in BARS:
            threshold = bar_value(bar)
            success = [[0.0] * len(documents) for _ in attacks]
            for slot, did_evade in evaded.items():
                quality = judged.get(slot, float("nan"))
                if did_evade and quality == quality and quality >= threshold:
                    success[slot[0]][slot[1]] = 1.0

            counts = [int(sum(row)) for row in success]
            best = max(range(len(counts)), key=lambda i: counts[i])
            value = counts[best] / len(documents)
            lo, hi = bootstrap_max_ci(success)
            table.append([
                threshold, value,
                max(0.0, value - lo), max(0.0, hi - value),
                float(len(documents)),
            ])
            winner = attacks[best]
            owners.append(f"{bar}:" + next(
                r["attack_family"] for r in cells if r["attack"] == winner
            ))

        path = OUT_DIR / f"{key}.dat"
        write_table(path, ["bar", "y", "em", "ep", "n"], table)
        written.add(path)
        notes.append(
            f"{key:14s} {len(table)} bars, ASR {table[0][1]:.3f}..{table[-1][1]:.3f}, "
            f"n={len(documents)}, {len(attacks)} attacks pooled"
        )
        notes.append(f"{'':14s} envelope owner: {' '.join(owners)}")

    for path in prune_unwritten(OUT_DIR, written):
        notes.append(f"PRUNED   {path.name}: not part of the paper surface")
    write_manifest(
        OUT_DIR / "MANIFEST.txt",
        bundle,
        notes,
        uncertainty=(
            f"{BOOTSTRAP_RESAMPLES} document bootstrap resamples, seed "
            f"{BOOTSTRAP_SEED}, {int(CI_LEVEL * 100)}% percentile; "
            "maximum recomputed across attack settings in every resample"
        ),
    )
    # Document-wide constants for the preamble.  Written from here because
    # Every extractor imports common, but only one need trigger the write.
    grid = write_shared_macros(summary)
    print(f"teaser: wrote {len(written)}/{len(FRONTIER_SCHEMES)} scheme curves to {OUT_DIR}")
    print(f"        and shared LaTeX constants to {grid}")
    for note in notes:
        if note.startswith(("MISSING", "PRUNED")):
            print("  " + note)


if __name__ == "__main__":
    main()
