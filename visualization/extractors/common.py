"""Shared helpers for turning a compiled results bundle into pgfplots tables.

A bundle holds manifest.json, cell_summary.csv (one row per config/attack cell),
and per_sample.parquet (one row per document, paired z and z_wm). Extraction
writes deterministic PGFPlots data and LaTeX fragments under results/paper/tables.
"""

from __future__ import annotations

import csv
import functools
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as parquet
from config.paper import (
    COMPARISONS,
    LADDERS,
    MAIN_SEMSPAN,
    PROVIDER_CANDIDATES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "paper"
DEFAULT_BUNDLE = DEFAULT_OUTPUT_ROOT
TABLE_ROOT = DEFAULT_OUTPUT_ROOT / "tables"
DATA_ROOT = TABLE_ROOT / "pgfplots"


def configure(bundle: Path | str | None = None, output: Path | str | None = None) -> None:
    """Configure one extraction run before importing extractor modules."""
    global DEFAULT_BUNDLE, DEFAULT_OUTPUT_ROOT, TABLE_ROOT, DATA_ROOT
    if bundle is not None:
        DEFAULT_BUNDLE = Path(bundle).resolve()
    if output is not None:
        DEFAULT_OUTPUT_ROOT = Path(output).resolve()
    TABLE_ROOT = DEFAULT_OUTPUT_ROOT / "tables"
    DATA_ROOT = TABLE_ROOT / "pgfplots"


def output_root() -> Path:
    """Return the configured root for tables and figures."""
    return DEFAULT_OUTPUT_ROOT

# Fixed seed: a rebuild with unchanged inputs must produce byte-identical .dat.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260812
CI_LEVEL = 0.95


@dataclass(frozen=True)
class Rung:
    key: str            # id used in filenames and pgfplots series names
    scheme: str
    mask: str
    sampling: str
    segmentation: str
    label: str          # human-readable, manifest only
    # The bundle can contain multiple provider budgets, so the four identity
    # columns alone are not unique. Every paper scheme uses exactly N=64.
    num_candidates: float | None = None

    def matches(self, row: dict) -> bool:
        if (
            row["scheme"] != self.scheme
            or row["mask"] != self.mask
            or row["sampling"] != self.sampling
            or row["segmentation"] != self.segmentation
        ):
            return False
        if self.num_candidates is None:
            return True
        budget = num(row, "generation_num_candidates")
        # Only duplicated human-null rows omit the provider budget.
        if budget != budget:
            return row.get("stage") == "human"
        return budget == self.num_candidates


# The fair-comparison contract: every provider scheme draws from N=64.
NPROV = float(PROVIDER_CANDIDATES)
SPAN_MAIN = MAIN_SEMSPAN

FAMILIES = {"lsh": ("SemStamp", ""), "kmeans": ("k-SemStamp", "k")}

# Macro names, never expansions: commands.tex owns the spelling.
SCHEME_MACRO = {"SemStamp": "semstamp", "k-SemStamp": "ksemstamp"}
FINAL_MACRO = {"SemStamp": "ourstamp", "k-SemStamp": "kourstamp"}


def ladder(family: str) -> list[Rung]:
    """Return the canonical five-rung paper ladder for ``family``."""
    return [
        Rung(
            rung.key,
            rung.scheme,
            rung.mask,
            rung.sampling,
            rung.segmentation,
            rung.key,
            float(rung.num_candidates),
        )
        for rung in LADDERS[family]
    ]


# One family carries the main-text channel figure; ten series in four panels
# would destroy the closure diagonal.  The other goes to the appendix.
MAIN_FAMILY = "kmeans"
APPENDIX_FAMILY = next(f for f in FAMILIES if f != MAIN_FAMILY)

LADDER = ladder(MAIN_FAMILY)
APPENDIX_LADDER = ladder(APPENDIX_FAMILY)

# Shared by the teaser and cross-scheme figures, which keeps them synchronized.
# SAMark's flags-run is the scope the bundle has attack runs for.
FRONTIER_SCHEMES = [
    Rung("semstamp", "SemStamp", "context", "rejection", "sentence-nltk",
         "SemStamp", NPROV),
    Rung("ksemstamp", "k-SemStamp", "context", "rejection", "sentence-nltk",
         "k-SemStamp", NPROV),
    Rung(COMPARISONS[0].key, "PMark", COMPARISONS[0].mask,
         COMPARISONS[0].sampling, COMPARISONS[0].segmentation, "PMark (online)",
         float(COMPARISONS[0].num_candidates)),
    Rung(COMPARISONS[1].key, "SAMark", COMPARISONS[1].mask,
         COMPARISONS[1].sampling, COMPARISONS[1].segmentation, "SAMark",
         float(COMPARISONS[1].num_candidates)),
    Rung("swordstamp", "SemStamp", "fixed_diverse", "best-of-n",
         SPAN_MAIN, "SwordStamp", NPROV),
    Rung("kswordstamp", "k-SemStamp", "fixed_diverse", "best-of-n",
         SPAN_MAIN, "k-SwordStamp", NPROV),
]

# Quality-bar grid for every ASR envelope; the figure axes come from the
# generated \qbarmin and \qbarmax.
BARS = [f"q{value}" for value in range(65, 96, 5)]


def bar_value(bar: str) -> float:
    """Return the quality bar a ``q<percent>`` key names."""
    return float(bar[1:]) / 100.0


def corpus_counts(rows: list[dict]) -> tuple[int, int]:
    """Return the prompt-set size and the human-null size."""
    marked = [row for row in rows if row["stage"] == "watermark"]
    null = {int(row["n_samples"]) for row in rows if row["stage"] == "human"}
    if not marked or not null:
        raise SystemExit("bundle has no watermark or no human rows to count")
    if len(null) > 1:
        raise SystemExit(
            f"human null splits disagree: {sorted(null)}; the calibration claim "
            "in the setup names one number and cannot name several"
        )
    return max(int(row["n_samples"]) for row in marked), null.pop()


def write_shared_macros(rows: list[dict] | None = None) -> Path:
    """Emit the LaTeX macros this module owns, so no copy can go stale.

    Without ``rows`` the two corpus macros are omitted and the build stops on
    them, which is intended: the document should not cite a size the bundle
    did not supply.
    """
    values = [bar_value(bar) for bar in BARS]
    # Pad the limits: a marker exactly on xmin has its error-bar cap clipped.
    pad = 0.02
    axis_min = values[0] - pad
    axis_max = values[-1] + pad
    # Ticks on round tenths; labels at 0.05 spacing collide in this panel.
    span = int(round((values[-1] - values[0]) * 10))
    ticks = [values[0] + 0.1 * i for i in range(span + 1)]

    main_scheme, main_prefix = FAMILIES[MAIN_FAMILY]
    alt_scheme, alt_prefix = FAMILIES[APPENDIX_FAMILY]

    path = DATA_ROOT / "constants.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("% Generated by python -m visualization extract -- do not edit.\n")
        fh.write(rf"\newcommand{{\nprov}}{{{NPROV:g}}}" "\n")
        fh.write(rf"\newcommand{{\qbarmin}}{{{100.0 * values[0]:.0f}\%}}" "\n")
        fh.write(rf"\newcommand{{\qbarmax}}{{{100.0 * values[-1]:.0f}\%}}" "\n")
        fh.write(rf"\newcommand{{\qbarstep}}{{{100.0 * (values[1] - values[0]):.0f}\%}}" "\n")
        fh.write(rf"\newcommand{{\qbaraxismin}}{{{axis_min:.2f}}}" "\n")
        fh.write(rf"\newcommand{{\qbaraxismax}}{{{axis_max:.2f}}}" "\n")
        fh.write(r"\newcommand{\qbarticks}{"
                 + ",".join(f"{value:.2f}" for value in ticks) + "}\n")
        fh.write(rf"\newcommand{{\rungprefix}}{{{main_prefix}}}" "\n")
        fh.write(rf"\newcommand{{\ladderbase}}{{\{SCHEME_MACRO[main_scheme]}}}" "\n")
        fh.write(rf"\newcommand{{\ladderfinal}}{{\{FINAL_MACRO[main_scheme]}}}" "\n")
        fh.write(rf"\newcommand{{\altrungprefix}}{{{alt_prefix}}}" "\n")
        fh.write(rf"\newcommand{{\altladderbase}}{{\{SCHEME_MACRO[alt_scheme]}}}" "\n")
        fh.write(rf"\newcommand{{\altladderfinal}}{{\{FINAL_MACRO[alt_scheme]}}}" "\n")
        if rows is not None:
            prompts, null = corpus_counts(rows)
            fh.write(rf"\newcommand{{\ndocs}}{{{prompts:,d}}}".replace(",", r"\,")
                     + "\n")
            fh.write(rf"\newcommand{{\nnull}}{{{null:,d}}}".replace(",", r"\,")
                     + "\n")
    return path


# Every configuration reported by the final paper.
ALL_RUNGS = LADDER + APPENDIX_LADDER


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def bundle_path() -> Path:
    """Return the configured compiled-results bundle."""
    path = DEFAULT_BUNDLE
    if not path.is_dir():
        raise SystemExit(f"no results bundle at {path}")
    return path


def read_manifest(bundle: Path) -> dict:
    with open(bundle / "manifest.json") as fh:
        return json.load(fh)


def read_parquet_rows(path: Path) -> list[dict]:
    """Read a required Parquet table as native Python row dictionaries."""
    if not path.is_file():
        raise SystemExit(f"required Parquet result table is missing: {path}")
    return parquet.read_table(path).to_pylist()


def read_per_sample(bundle: Path) -> list[dict]:
    return read_parquet_rows(bundle / "per_sample.parquet")


def read_cell_summary(bundle: Path) -> list[dict]:
    with open(bundle / "cell_summary.csv") as fh:
        return list(csv.DictReader(fh))


def num(row: dict, key: str) -> float:
    """Parse a result field, mapping blanks and NaN text to float('nan')."""
    value = row.get(key, "")
    if value is None or value == "" or value == "nan":
        return float("nan")
    return float(value)


def document_id(row: dict) -> tuple[str, str]:
    """Return a document identity that remains unique after shard pooling."""
    # CSV supplied both fields as strings. Preserve that stable ordering when
    # Parquet exposes sample_id as an integer, so fixed-seed bootstrap draws do
    # not change merely because the storage format changed.
    return str(row["dataset"]), str(row["sample_id"])


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def z_retention(pairs: list[tuple[float, float]]) -> float:
    """Normalized z retention as a percentage, per eq:z-retention.

    ``pairs`` holds (attacked z, clean z).  This is the ratio of corpus sums,
    which is what the paper defines and is stable when a clean score is near
    zero, not the mean of per-document ratios.
    """
    num_sum = sum(p[0] for p in pairs)
    den_sum = sum(p[1] for p in pairs)
    if den_sum == 0:
        return float("nan")
    return 100.0 * num_sum / den_sum


@functools.lru_cache(maxsize=None)
def resample_indices(
    n: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[tuple[int, ...], ...]:
    """Return the document index draws every bootstrap in the paper shares.

    Every call re-seeds from the same constant, so caching is a pure speedup
    with byte-identical output.  Sharing one draw is also correct: compared
    quantities are resampled over the same documents.
    """
    rng = random.Random(seed)
    return tuple(
        tuple(rng.randrange(n) for _ in range(n))
        for _ in range(resamples)
    )


def bootstrap_ci(
    pairs: list[tuple[float, float]],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap interval for z_retention, resampling documents."""
    if len(pairs) < 2:
        point = z_retention(pairs)
        return point, point
    n = len(pairs)
    draws = [
        z_retention([pairs[index] for index in indices])
        for indices in resample_indices(n, resamples, seed)
    ]
    draws.sort()
    tail = (1.0 - CI_LEVEL) / 2.0
    lo = draws[int(tail * (resamples - 1))]
    hi = draws[int((1.0 - tail) * (resamples - 1))]
    return lo, hi


def mean(values: list[float]) -> float:
    finite = [v for v in values if v == v]
    return sum(finite) / len(finite) if finite else float("nan")


# --------------------------------------------------------------------------
# Attacker matching
# --------------------------------------------------------------------------

# A matched attacker matches both public switches of the design
# (\Cref{alg:matched}): the anchor and the segmentation.  Matching one and not
# the other aims the attacker at a different design.
POSITIONAL_ANCHOR = "positional"
BAG_ANCHOR = "bag"

# Derived from the mask, not declared per scheme: a fixed valid set makes
# position uninformative however the scheme fixes it.  An unknown mask raises.
MATCHED_ANCHOR_BY_MASK = {
    "context": POSITIONAL_ANCHOR,
    "online": POSITIONAL_ANCHOR,
    "fixed": BAG_ANCHOR,
    "fixed_diverse": BAG_ANCHOR,
    "flags-run": BAG_ANCHOR,
}


def matched_anchor(mask: str) -> str:
    """Return the anchor the configuration's own design forces on an attacker."""
    try:
        return MATCHED_ANCHOR_BY_MASK[mask]
    except KeyError:
        raise SystemExit(
            f"no matched attacker anchor defined for mask '{mask}'; add it to "
            "MATCHED_ANCHOR_BY_MASK once the design decides whether position "
            "carries information"
        ) from None

# Provider and attacker may reach one segmentation through different backends.
SEGMENTATION_BACKENDS = {"nltk", "spacy"}


@functools.lru_cache(maxsize=None)
def segmentation_signature(value: str) -> tuple[str, ...]:
    """Return a segmentation name reduced to its backend-independent parts."""
    return tuple(
        part for part in str(value).split("-")
        if part not in SEGMENTATION_BACKENDS
    )


def is_matched_attacker(row: dict, anchor: str, segmentation: str) -> bool:
    """True unless the row is an adaptive attack aimed at a different design.

    A maximum over every attacker in the bundle would credit a scheme's
    robustness to an attacker its design tells the attacker not to use, and
    would reward configurations that happen to have more variants run against
    them.  Non-adaptive families have no switch to match and pass through.
    """
    if row.get("attack_family") != "adaptive":
        return True
    if row.get("anchor") != anchor:
        return False
    return (
        segmentation_signature(row.get("attacker_seg"))
        == segmentation_signature(segmentation)
    )


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def write_table(path: Path, columns: list[str], rows: list[list]) -> None:
    """Write a whitespace-delimited pgfplots table.

    Named columns on line one; no comment lines, since pgfplots parses these
    under TeX catcodes.  Provenance goes in the sibling MANIFEST.txt.  A string
    cell is written verbatim for symbolic columns and must carry no whitespace.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(" ".join(columns) + "\n")
        for row in rows:
            cells = []
            for value in row:
                if isinstance(value, str):
                    if not value or any(c.isspace() for c in value):
                        raise ValueError(
                            f"{path.name}: symbolic cell {value!r} is empty or "
                            "carries whitespace, which breaks column alignment"
                        )
                    cells.append(value)
                    continue
                # Snap denormal residue so an invariant cell writes 0.
                cells.append(f"{0.0 if abs(value) < 1e-9 else value:.6g}")
            fh.write(" ".join(cells) + "\n")


def prune_unwritten(out_dir: Path, written: set[Path]) -> list[Path]:
    """Delete .dat files in ``out_dir`` that this run did not write.

    A renamed series would otherwise leave a file no figure loads today but a
    later rename could pick up, carrying numbers from a forgotten bundle.
    """
    removed = []
    for path in sorted(out_dir.glob("*.dat")):
        if path not in written:
            path.unlink()
            removed.append(path)
    return removed


def write_manifest(
    path: Path,
    bundle: Path,
    lines: list[str],
    uncertainty: str | None = None,
) -> None:
    """Record which bundle a directory of .dat files came from.

    ``uncertainty`` overrides the default bootstrap header for figures that
    report a different interval.
    """
    manifest = read_manifest(bundle)
    counts = manifest.get("counts", {})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"bundle: {bundle.name}\n")
        fh.write(f"datasets: {', '.join(manifest.get('datasets', []))}\n")
        fh.write(f"cells: {counts.get('cells', '?')}, "
                 f"per-sample rows: {counts.get('per_sample_rows', '?')}\n")
        if uncertainty is None:
            fh.write(f"bootstrap: {BOOTSTRAP_RESAMPLES} resamples, "
                     f"seed {BOOTSTRAP_SEED}, {int(CI_LEVEL * 100)}% percentile\n")
        else:
            fh.write(f"uncertainty: {uncertainty}\n")
        fh.write("\n")
        for line in lines:
            fh.write(line + "\n")
