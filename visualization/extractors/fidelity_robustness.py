"""Build the fidelity-versus-ASR teaser at the primary operating point.

Each point pairs clean reference-free fidelity with the strongest no-box ASR at
the 5% false-positive rate and 90% content-preservation bar. The no-box envelope
and its bootstrap interval use the same logic as the existing
ASR-versus-quality-bar teaser.

Emits tables/pgfplots/fidelity_robustness/points.dat and constants.tex.
"""

from __future__ import annotations

import collections

from .teaser import (
    EXCLUDED_FAMILIES,
    REQUIRED_FAMILIES,
    bootstrap_max_ci,
)
from .common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CI_LEVEL,
    DATA_ROOT,
    FRONTIER_SCHEMES,
    bundle_path,
    document_id,
    num,
    read_cell_summary,
    read_per_sample,
    write_manifest,
    write_table,
)

OUT_DIR = DATA_ROOT / "fidelity_robustness"
QUALITY_BAR = 0.90


def clean_row(rows: list[dict], config) -> dict:
    """Return the unique clean marked row for one plotted configuration."""
    matches = [
        row for row in rows
        if config.matches(row)
        and row["stage"] == "watermark"
        and row["attack"] == "none"
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"{config.key}: expected one clean marked row, found {len(matches)}"
        )
    return matches[0]


def unwatermarked_row(rows: list[dict]) -> dict:
    """Return the unique clean no-watermark quality baseline."""
    matches = [
        row for row in rows
        if row["family"] == "none"
        and row["stage"] == "watermark"
        and row["attack"] == "none"
    ]
    if len(matches) != 1:
        raise SystemExit(
            "expected one clean no-watermark row, found "
            f"{len(matches)}"
        )
    return matches[0]


def strongest_no_box_asr(rows: list[dict], config) -> tuple[
    float, float, float, int, str
]:
    """Return strongest ASR, interval, document count, and winning attack."""
    cells = [
        row for row in rows
        if config.matches(row)
        and row["stage"] == "attack"
        and row["attack_family"] not in EXCLUDED_FAMILIES
        and num(row, "llm_judge") == num(row, "llm_judge")
    ]
    if not cells:
        raise SystemExit(f"{config.key}: no judged no-box attack rows")
    present = {row["attack_family"] for row in cells}
    absent = REQUIRED_FAMILIES - present
    if absent:
        raise SystemExit(
            f"{config.key}: incomparable attack coverage; missing "
            + ", ".join(sorted(absent))
        )

    documents = sorted({document_id(row) for row in cells})
    document_index = {document: index for index, document in enumerate(documents)}
    attacks = sorted({row["attack"] for row in cells})
    attack_index = {attack: index for index, attack in enumerate(attacks)}

    success = [[0.0] * len(documents) for _ in attacks]
    for row in cells:
        if (num(row, "evades_at5") >= 0.5
                and num(row, "llm_judge") >= QUALITY_BAR):
            success[attack_index[row["attack"]]][
                document_index[document_id(row)]
            ] = 1.0

    counts = [int(sum(outcomes)) for outcomes in success]
    best = max(range(len(counts)), key=lambda index: counts[index])
    point = counts[best] / len(documents)
    lo, hi = bootstrap_max_ci(success)
    return point, lo, hi, len(documents), attacks[best]


def point_row(point_id: int, config, summary_rows: list[dict],
              sample_rows: list[dict]) -> tuple[list[float], str]:
    """Return one fidelity-ASR point and its manifest note."""
    clean = clean_row(summary_rows, config)
    fidelity = num(clean, "llm_quality")
    fidelity_ci = num(clean, "llm_quality_ci")
    if fidelity != fidelity or fidelity_ci != fidelity_ci:
        raise SystemExit(f"{config.key}: missing clean fidelity or interval")

    asr, asr_lo, asr_hi, document_count, winner = strongest_no_box_asr(
        sample_rows, config
    )
    row = [
        float(point_id),
        100.0 * fidelity,
        100.0 * fidelity_ci,
        100.0 * fidelity_ci,
        100.0 * asr,
        100.0 * (asr - asr_lo),
        100.0 * (asr_hi - asr),
        float(document_count),
    ]
    note = (
        f"{point_id} {config.key:14s} fidelity={100.0 * fidelity:.1f}% "
        f"+/-{100.0 * fidelity_ci:.1f}; strongest ASR={100.0 * asr:.1f}% "
        f"({winner}); n={document_count}"
    )
    return row, note


def main() -> None:
    bundle = bundle_path()
    summary_rows = read_cell_summary(bundle)
    sample_rows = read_per_sample(bundle)

    table = []
    notes = []
    for point_id, config in enumerate(FRONTIER_SCHEMES):
        row, note = point_row(point_id, config, summary_rows, sample_rows)
        table.append(row)
        notes.append("schemes " + note)

    baseline = unwatermarked_row(summary_rows)
    baseline_fidelity = num(baseline, "llm_quality")
    baseline_ci = num(baseline, "llm_quality_ci")
    baseline_n = num(baseline, "n_samples")
    if (baseline_fidelity != baseline_fidelity
            or baseline_ci != baseline_ci
            or baseline_n != baseline_n):
        raise SystemExit("no-watermark row is missing fidelity, interval, or n")
    baseline_mean = 100.0 * baseline_fidelity
    baseline_low = 100.0 * (baseline_fidelity - baseline_ci)
    baseline_high = 100.0 * (baseline_fidelity + baseline_ci)

    write_table(
        OUT_DIR / "points.dat",
        ["id", "fidelity", "fem", "fep", "asr", "aem", "aep", "n"],
        table,
    )
    constants = [
        "% Generated by python -m visualization extract -- do not edit.",
        rf"\newcommand{{\unwatermarkedfidelitymean}}{{{baseline_mean:.6f}}}",
        rf"\newcommand{{\unwatermarkedfidelitylow}}{{{baseline_low:.6f}}}",
        rf"\newcommand{{\unwatermarkedfidelityhigh}}{{{baseline_high:.6f}}}",
        rf"\newcommand{{\resultUnwatermarkedFidelity}}{{{baseline_mean:.1f}\%}}",
        rf"\newcommand{{\resultUnwatermarkedFidelityLow}}{{{baseline_low:.1f}\%}}",
        rf"\newcommand{{\resultUnwatermarkedFidelityHigh}}{{{baseline_high:.1f}\%}}",
    ]
    (OUT_DIR / "constants.tex").write_text("\n".join(constants) + "\n")
    notes.append(
        "unwatermarked fidelity="
        f"{baseline_mean:.1f}% [{baseline_low:.1f}, {baseline_high:.1f}]; "
        f"n={int(baseline_n)}"
    )
    write_manifest(
        OUT_DIR / "MANIFEST.txt",
        bundle,
        notes,
        uncertainty=(
            "horizontal bars and shaded baseline: 95% Student-t intervals "
            "from clean fidelity; vertical bars: "
            f"{BOOTSTRAP_RESAMPLES} document bootstrap resamples, seed "
            f"{BOOTSTRAP_SEED}, {int(CI_LEVEL * 100)}% percentile, with the "
            "strongest no-box attack recomputed in every resample"
        ),
    )
    print(
        f"fidelity-ASR: wrote {len(table)} points to "
        f"{OUT_DIR.relative_to(DATA_ROOT.parent.parent)}"
    )


if __name__ == "__main__":
    main()
