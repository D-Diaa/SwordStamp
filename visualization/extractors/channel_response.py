"""Build the channel-response plot tables."""

from __future__ import annotations

import collections
from pathlib import Path

from .common import (
    DATA_ROOT,
    APPENDIX_LADDER,
    LADDER,
    bootstrap_ci,
    bundle_path,
    mean,
    num,
    prune_unwritten,
    read_per_sample,
    write_manifest,
    write_table,
    z_retention,
)

OUT_DIR = DATA_ROOT / "channel_response"

PANELS = [
    ("reword", ("synonym_substitution",), "anchor_reword"),
    ("reorder", ("controlled_reorder",), "anchor_reorder"),
    ("reseg", ("split_midpoint", "merge_adjacent"), "anchor_reseg"),
]

RUNGS = LADDER + APPENDIX_LADDER


def probe_ratio(attack: str) -> float:
    """Pull the configured intensity out of an attack name like foo-ratio=0.25."""
    return float(attack.split("ratio=")[1])


def main() -> None:
    bundle = bundle_path()
    rows = read_per_sample(bundle)

    grouped: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in rows:
        if row["stage"] != "attack":
            continue
        for rung in RUNGS:
            if rung.matches(row):
                grouped[(rung.key, row["attack"])].append(row)
                break

    notes: list[str] = []
    written: set = set()
    for panel, atoms, strength_col in PANELS:
        for rung in RUNGS:
            attacks = [
                attack for (key, attack) in grouped
                if key == rung.key and any(attack.startswith(atom) for atom in atoms)
            ]
            if not attacks:
                raise SystemExit(
                    f"{panel} {rung.key}: missing required "
                    f"{' + '.join(atoms)} probe cells"
                )

            ratios = sorted({probe_ratio(attack) for attack in attacks})

            table = [[0.0, 100.0, 0.0, 0.0, 0.0, 0.0]]

            for ratio in ratios:
                ratio_attacks = [
                    attack for attack in attacks if probe_ratio(attack) == ratio
                ]
                expected = len(atoms)
                if len(ratio_attacks) != expected:
                    raise SystemExit(
                        f"{panel} {rung.key} ratio={ratio}: "
                        f"expected {expected} probes, found {len(ratio_attacks)}"
                    )
                cell = [
                    row for attack in ratio_attacks
                    for row in grouped[(rung.key, attack)]
                ]
                pairs = [
                    (num(r, "z"), num(r, "z_wm"))
                    for r in cell
                    if num(r, "z") == num(r, "z") and num(r, "z_wm") == num(r, "z_wm")
                ]
                if not pairs:
                    raise SystemExit(
                        f"{panel} {rung.key} ratio={ratio}: no z pairs"
                    )
                point = z_retention(pairs)
                lo, hi = bootstrap_ci(pairs)
                strengths = [num(r, strength_col) for r in cell]
                strengths = [strength for strength in strengths if strength == strength]
                if not strengths:
                    raise SystemExit(
                        f"{panel} {rung.key} ratio={ratio}: no channel values"
                    )
                strength = mean(strengths)
                table.append([
                    strength,
                    point,
                    max(0.0, point - lo),
                    max(0.0, hi - point),
                    ratio,
                    float(len(pairs)),
                ])

            path = OUT_DIR / f"{panel}-{rung.key}.dat"
            write_table(path, ["x", "y", "em", "ep", "ratio", "n"], table)
            written.add(path)
            span = f"{table[1][0]:.3f}..{table[-1][0]:.3f}"
            notes.append(
                f"{panel:8s} {rung.key:4s} {len(table) - 1} points, "
                f"strength {span}, retention "
                f"{table[1][1]:.1f}..{table[-1][1]:.1f}%"
            )

    for path in prune_unwritten(OUT_DIR, written):
        notes.append(f"PRUNED   {path.name}: no source rows in this bundle")
    write_manifest(OUT_DIR / "MANIFEST.txt", bundle, notes)
    print(f"channel_response: wrote {len(list(OUT_DIR.glob('*.dat')))} tables "
          f"to {OUT_DIR.relative_to(Path.cwd()) if OUT_DIR.is_relative_to(Path.cwd()) else OUT_DIR}")
    for note in notes:
        if note.startswith(("MISSING", "PRUNED")):
            print("  " + note)


if __name__ == "__main__":
    main()
