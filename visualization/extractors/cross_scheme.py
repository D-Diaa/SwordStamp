"""Build fig:cross-scheme: quality-bar robustness across watermark schemes.

One panel per scheme, one envelope per attack family, each reporting the
strongest observed setting within its family at each bar.  The adaptive series
carries only the configuration's matched attacker, because this figure
attributes; the teaser and tab:costs deliberately do the opposite.  Every panel
whose scheme fixes its valid set also carries the base positional attacker as a
fixed comparison, without which the matched-attacker rule would be unfalsifiable.

Emits <scheme>-<family>.dat (bar, y, em, ep, n) and <scheme>-<family>-labels.dat
(bar, y, label), the latter carrying one row per change of owning setting.

Run through ``python -m visualization extract``.
"""

from __future__ import annotations

import collections
import functools
import math
from dataclasses import dataclass

import numpy as np

from .common import (
    BARS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    DATA_ROOT,
    FRONTIER_SCHEMES,
    POSITIONAL_ANCHOR,
    bar_value,
    bundle_path,
    document_id,
    is_matched_attacker,
    matched_anchor,
    num,
    prune_unwritten,
    read_per_sample,
    resample_indices,
    write_manifest,
    write_table,
)

OUT_DIR = DATA_ROOT / "cross_scheme"
PROSE_QUALITY_BAR = 0.90
# The paper compares the adaptive attack with Dipper at its primary quality
# bar.  Lower bars remain descriptive figure results and need not preserve the
# same ordering as the primary-bar claim.
DIPPER_PROSE_QUALITY_BARS = (PROSE_QUALITY_BAR,)

# The published adaptive attack: positional anchor, sentence segmentation.
# Matched for every context-masked scheme, and what SwordStamp's two switches
# are aimed at, so those panels carry it as a named comparison.
BASE_ATTACKER = (POSITIONAL_ANCHOR, "sentence-nltk")

# Only a scheme that moved off the positional anchor has anything to show;
# elsewhere the two series would be the same rows drawn twice.  SAMark belongs
# here for the same reason the SwordStamp designs do: its run-scoped flags make
# position uninformative, so its matched attacker is the bag anchor, and the
# positional attacker is a distinct comparison rather than a relabel.
#
# The macro stem each panel's prose numbers are named with.  Panels and stems
# come from one mapping so a panel that gains the comparison cannot silently
# fail to publish it.
PROSE_STEMS = {
    "swordstamp": "resultSword",
    "kswordstamp": "resultKSword",
    "samark": "resultSamark",
}
BASE_ATTACKER_SCHEMES = frozenset(PROSE_STEMS)

# The two series the prose contrasts on those panels, and the macro suffix each
# carries.  \edas is the configuration the target's own published rules select
# (\S\ref{subsec:scheme-attack}); \edap is the positional, sentence-anchored
# variant of \S\ref{sec:attacks}, run unchanged.  Both numbers are read off this
# figure rather than tab:costs, whose adaptive column maximizes over every
# attacker and so does not name the scheme's own configuration on a panel where
# the positional one matches or beats it.
PROSE_SERIES = {
    "adaptive": "EdasAsr",
    "adaptive-base": "EdapAsr",
}


@dataclass(frozen=True)
class Series:
    """One plotted trajectory within a panel.

    ``sources`` names the bundle attack families it draws from, and the two
    optional scheme and attacker qualifiers narrow that further. ``required``
    marks a series whose absence makes a panel's no-box coverage incomplete.
    """

    key: str
    sources: frozenset[str]
    label: str
    required: bool
    # Restrict to these scheme keys, for an attack not defined against every
    # scheme.
    schemes: frozenset[str] | None = None
    # (anchor, segmentation) an adaptive attack must match.  None takes the
    # configuration's own; a fixed pair names one attacker across panels.
    attacker: tuple[str, str] | None = None

    def cells(
        self, rows: list[dict], scheme_key: str, anchor: str, segmentation: str
    ) -> list[dict]:
        """Return this series' rows out of one scheme's judged attack rows."""
        if self.schemes is not None and scheme_key not in self.schemes:
            return []
        if self.attacker is not None:
            anchor, segmentation = self.attacker
        return [
            row for row in rows
            if row["attack_family"] in self.sources
            and is_matched_attacker(row, anchor, segmentation)
        ]


# Separate trajectories, not one envelope: a bound is stated at an effort, and
# maximizing over efforts would hide how much of it the effort buys.
FAMILIES = [
    Series("sentence", frozenset({"pegasus", "parrot"}),
           "sentence-level controls", True),
    Series("dipper", frozenset({"dipper"}), "Dipper", True),
    Series("adaptive", frozenset({"adaptive"}), "adaptive", True),
    Series("adaptive-base", frozenset({"adaptive"}), "adaptive, base", False,
           schemes=BASE_ATTACKER_SCHEMES, attacker=BASE_ATTACKER),
]

# Dipper sweeps one knob at a time and holds the other at this value, so the
# raised knob names the setting.
DIPPER_BASE = 20


def short_label(row: dict) -> str:
    """Return the compact figure label for the attack setting a row carries.

    The label is read from the bundle's typed columns rather than parsed out of
    the free-form attack name, so a run that renames its attacks or adds a
    setting still labels correctly.  The vocabulary matches the one the earlier
    channel-signature figures use, so a setting reads the same wherever it
    appears in the paper.
    """
    family = row.get("attack_family")
    if family in {"pegasus", "parrot"}:
        stem = "Peg" if family == "pegasus" else "Par"
        # The bigram variants differ only in a threshold the figure ignores.
        return f"{stem}+B" if "bigram" in str(row.get("attack_setting")) else stem
    if family == "dipper":
        lexical = int(num(row, "dipper_lex"))
        order = int(num(row, "dipper_order"))
        if order > DIPPER_BASE:
            return f"O{order}"
        if lexical > DIPPER_BASE:
            return f"L{lexical}"
        # The shared base is named on the order sweep.
        return f"O{DIPPER_BASE}"
    if family == "adaptive":
        # The budget alone names it: a series pins anchor and segmentation.
        return f"K{int(num(row, 'K'))}"
    return str(row.get("attack"))


@functools.lru_cache(maxsize=None)
def resample_multiplicities(document_count: int) -> np.ndarray:
    """Return shared bootstrap draws as a resample-by-document count matrix."""
    indices = np.asarray(resample_indices(document_count), dtype=np.intp)
    counts = np.zeros(indices.shape, dtype=np.float64)
    rows = np.arange(indices.shape[0], dtype=np.intp)[:, None]
    np.add.at(counts, (rows, indices), 1.0)
    return counts


def bootstrap_max_cis(
    successes: list[list[list[float]]],
) -> list[tuple[float, float]]:
    """Return maximum-ASR percentile intervals for every quality bar.

    ``successes[bar][attack][document]`` holds the binary outcome.  Every
    resample draws document indices once, reuses that draw for every bar and
    attack, then reselects the maximum attack setting within each bar.
    Resampling a document as a block preserves dependence among attacks.

    The observed maximum remains the plotted point.  These intervals describe
    the family envelope, not a fixed selected attack, and overlap is not a test
    of the gap between two plotted families or schemes.
    """
    if not successes or not successes[0] or not successes[0][0]:
        return [(float("nan"), float("nan")) for _ in successes]

    success_array = np.asarray(successes, dtype=np.float64)
    if success_array.ndim != 3:
        raise ValueError(
            "cross-scheme success data must be bar by attack by document"
        )
    bar_count, attack_count, document_count = success_array.shape

    flattened = success_array.reshape(bar_count * attack_count, document_count)
    counts = resample_multiplicities(document_count) @ flattened.T
    counts = counts.reshape(BOOTSTRAP_RESAMPLES, bar_count, attack_count)
    draws = np.sort(np.max(counts, axis=2) / document_count, axis=0)
    tail = (1.0 - CI_LEVEL) / 2.0
    low_index = int(tail * (BOOTSTRAP_RESAMPLES - 1))
    high_index = int((1.0 - tail) * (BOOTSTRAP_RESAMPLES - 1))
    return [
        (float(draws[low_index, bar]), float(draws[high_index, bar]))
        for bar in range(bar_count)
    ]


def family_table(
    cells: list[dict],
) -> tuple[list[list[float]], list[str], list[list]]:
    """Return the envelope, the selected attack per bar, and the label rows."""
    documents = sorted({document_id(r) for r in cells})
    doc_index = {document: i for i, document in enumerate(documents)}
    attacks = sorted({r["attack"] for r in cells})
    attack_index = {attack: i for i, attack in enumerate(attacks)}

    evaded = collections.defaultdict(bool)
    judged: dict[tuple[int, int], float] = {}
    # One row per attack: rows of an attack differ only in their document.
    exemplar: dict[str, dict] = {}
    for row in cells:
        slot = (attack_index[row["attack"]], doc_index[document_id(row)])
        evaded[slot] = num(row, "evades_at5") >= 0.5
        judged[slot] = num(row, "llm_judge")
        exemplar.setdefault(row["attack"], row)

    bar_successes: list[list[list[float]]] = []
    for bar in BARS:
        threshold = bar_value(bar)
        successes = [[0.0] * len(documents) for _ in attacks]
        for slot, did_evade in evaded.items():
            quality = judged.get(slot, float("nan"))
            if did_evade and quality == quality and quality >= threshold:
                successes[slot[0]][slot[1]] = 1.0
        bar_successes.append(successes)

    intervals = bootstrap_max_cis(bar_successes)
    table: list[list[float]] = []
    winners: list[str] = []
    owners: list[str] = []
    for bar, successes, (lo, hi) in zip(BARS, bar_successes, intervals):
        threshold = bar_value(bar)
        counts = [int(sum(row)) for row in successes]
        best = max(range(len(counts)), key=lambda index: counts[index])
        value = counts[best] / len(documents)
        table.append([
            threshold,
            value,
            max(0.0, value - lo),
            max(0.0, hi - value),
            float(len(documents)),
        ])
        winners.append(f"{bar}:{attacks[best]}")
        owners.append(attacks[best])

    labels: list[list] = []
    previous_label = None
    previous_value = None
    for point, owner in zip(table, owners):
        # Ties are broken arbitrarily above, so relabeling on one would assert
        # a difference the data does not carry.  The earlier label stands.
        if point[1] == previous_value:
            continue
        label = short_label(exemplar[owner])
        if label != previous_label:
            labels.append([point[0], point[1], label])
        previous_label = label
        previous_value = point[1]
    return table, winners, labels


def main() -> None:
    bundle = bundle_path()
    rows = read_per_sample(bundle)

    notes: list[str] = []
    written = 0
    written_paths: set = set()
    prose_values: dict[tuple[str, str], float] = {}
    series_tables: dict[tuple[str, str], list[list[float]]] = {}
    for config in FRONTIER_SCHEMES:
        key, label = config.key, config.label
        anchor = matched_anchor(config.mask)
        # Unfiltered by attacker: the anchor and segmentation rules belong to
        # the series.  Rows no series claims are reported in the manifest.
        scheme_rows: list[dict] = []
        saw_config = False
        for row in rows:
            if not config.matches(row):
                continue
            saw_config = True
            if row["stage"] != "attack":
                continue
            # Probe atoms carry no quality score and cannot be an ASR success.
            if num(row, "llm_judge") != num(row, "llm_judge"):
                continue
            scheme_rows.append(row)
        if not scheme_rows:
            # The bundle is the single source of truth: a scheme it does not
            # carry has no tables, whether it was dropped from the run or merely
            # left out of this compilation's family selection.  Keeping the
            # previous compilation's files would put numbers in the paper that
            # the named bundle cannot reproduce.  The note distinguishes the two
            # causes, because only one of them is fixed by recompiling.
            cause = (
                "no judged attack rows" if saw_config
                else "not compiled into this bundle (check selected_families)"
            )
            # No deletion here: a file this run did not write is removed by the
            # prune pass at the end, which also reports it.
            notes.append(f"MISSING  {key:14s} ({label}): {cause}")
            continue

        missing: list[str] = []
        claimed: set[str] = set()
        for series in FAMILIES:
            cells = series.cells(scheme_rows, key, anchor, config.segmentation)
            if not cells:
                if series.required:
                    missing.append(series.key)
                continue
            claimed.update(row["attack"] for row in cells)

            path = OUT_DIR / f"{key}-{series.key}.dat"
            label_path = OUT_DIR / f"{key}-{series.key}-labels.dat"
            table, winners, labels = family_table(cells)
            series_tables[(key, series.key)] = table
            write_table(path, ["bar", "y", "em", "ep", "n"], table)
            write_table(label_path, ["bar", "y", "label"], labels)
            written_paths.update({path, label_path})
            written += 1
            if key in PROSE_STEMS and series.key in PROSE_SERIES:
                point = min(table, key=lambda row: abs(row[0] - PROSE_QUALITY_BAR))
                if not math.isclose(point[0], PROSE_QUALITY_BAR, abs_tol=1e-12):
                    raise SystemExit(
                        f"{key}: no {series.key} point at "
                        f"q={PROSE_QUALITY_BAR:.2f}"
                    )
                prose_values[(key, series.key)] = point[1]
            notes.append(
                f"{key:14s} {series.key:10s} {len(table)} bars, "
                f"ASR {table[0][1]:.3f}..{table[-1][1]:.3f}, "
                f"n={int(table[0][4])}, {len(set(row['attack'] for row in cells))} attacks"
            )
            notes.append(f"{'':14s} {series.label:22s} owner: {' '.join(winners)}")

        # What no series drew, split by why.  An adaptive attack left over is an
        # attacker aimed at another design, which the anchor rule explains; an
        # attack of any other family is one no series names at all, and calling
        # that an anchor mismatch would state a cause the data does not carry.
        unclaimed = [row for row in scheme_rows if row["attack"] not in claimed]
        dropped = {
            row["attack"] for row in unclaimed
            if row["attack_family"] == "adaptive"
        }
        unused = {
            row["attack_family"] for row in unclaimed
            if row["attack_family"] != "adaptive"
        }
        if dropped:
            notes.append(
                f"{key:14s} unmatched  dropped, {anchor} anchor is the matched "
                f"attacker: {', '.join(sorted(dropped))}"
            )
        if unused:
            notes.append(
                f"{key:14s} unplotted  no series names these families: "
                f"{', '.join(sorted(unused))}"
            )
        if missing:
            raise SystemExit(
                f"{key} ({label}): incomplete required attack coverage; missing "
                + ", ".join(missing)
            )

    required = {
        (key, series_key): f"{stem}{suffix}"
        for key, stem in PROSE_STEMS.items()
        for series_key, suffix in PROSE_SERIES.items()
    }
    missing_prose = sorted(
        macro for slot, macro in required.items() if slot not in prose_values
    )
    if missing_prose:
        raise SystemExit(
            "cannot emit cross-scheme prose results; missing "
            + ", ".join(missing_prose)
        )

    for config in FRONTIER_SCHEMES:
        adaptive = series_tables.get((config.key, "adaptive"))
        dipper = series_tables.get((config.key, "dipper"))
        if adaptive is None or dipper is None:
            raise SystemExit(
                f"cannot compare adaptive and Dipper on {config.key}"
            )
        if len(adaptive) != len(dipper):
            raise SystemExit(
                f"adaptive and Dipper bars differ on {config.key}"
            )
        for quality_bar in DIPPER_PROSE_QUALITY_BARS:
            adaptive_point = min(
                adaptive, key=lambda row: abs(row[0] - quality_bar)
            )
            dipper_point = min(
                dipper, key=lambda row: abs(row[0] - quality_bar)
            )
            if not (
                math.isclose(adaptive_point[0], quality_bar, abs_tol=1e-12)
                and math.isclose(dipper_point[0], quality_bar, abs_tol=1e-12)
            ):
                raise SystemExit(
                    f"adaptive and Dipper lack q={quality_bar:.2f} on {config.key}"
                )
            if adaptive_point[1] <= dipper_point[1]:
                raise SystemExit(
                    "adaptive does not exceed Dipper at "
                    f"q={quality_bar:.2f} on {config.key}"
                )

    constants_path = OUT_DIR / "constants.tex"
    constants = [
        r"% Generated by python -m visualization extract -- do not edit.",
    ]
    constants.extend(
        rf"\newcommand{{\{macro}}}{{{100.0 * prose_values[slot]:.1f}\%}}"
        for slot, macro in required.items()
    )
    constants.append(
        r"\newcommand{\resultAdaptiveDipperQualityRequirements}{"
        + " and ".join(
            f"{100.0 * bar:.0f}\\%" for bar in DIPPER_PROSE_QUALITY_BARS
        )
        + "}"
    )
    constants_path.write_text("\n".join(constants) + "\n")
    written_paths.add(constants_path)

    for path in prune_unwritten(OUT_DIR, written_paths):
        notes.append(f"PRUNED   {path.name}: no source rows in this bundle")
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
    print(f"cross_scheme: wrote {written} tables to {OUT_DIR}")
    for note in notes:
        if note.startswith("MISSING"):
            print("  " + note)


if __name__ == "__main__":
    main()
