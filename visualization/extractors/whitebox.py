"""Build white-box metric tables for the k-means and LSH pairs."""

from __future__ import annotations

from pathlib import Path

from .common import (
    BOOTSTRAP_RESAMPLES,
    CI_LEVEL,
    DATA_ROOT,
    FRONTIER_SCHEMES,
    bundle_path,
    document_id,
    mean,
    num,
    prune_unwritten,
    read_manifest,
    read_per_sample,
    resample_indices,
    write_manifest,
    write_table,
)

OUT_DIR = DATA_ROOT / "whitebox"
QUALITY_BAR = 0.90
WHITEBOX_KEYS = ("ksemstamp", "kswordstamp")


def percentile_ci(values: list[float], conditional: bool = False) -> tuple[float, float]:
    """Return a paired-document percentile interval for a scalar statistic.

    ``conditional`` computes the statistic over finite values only. This lets
    the quality-given-evasion series retain paired-document resampling while
    excluding documents on which the attack did not evade.
    """
    point = mean(values)
    if len(values) < 2:
        return point, point
    draws = []
    for indices in resample_indices(len(values)):
        sample = [values[index] for index in indices]
        value = mean(sample) if conditional else sum(sample) / len(sample)
        if value == value:
            draws.append(value)
    if not draws:
        return point, point
    draws.sort()
    tail = (1.0 - CI_LEVEL) / 2.0
    return (
        draws[int(tail * (len(draws) - 1))],
        draws[int((1.0 - tail) * (len(draws) - 1))],
    )


def main() -> None:
    bundle = bundle_path()
    manifest = read_manifest(bundle)
    oracle_ks = manifest.get("selected_oracle_ks")
    if not oracle_ks:
        raise SystemExit("bundle manifest has no selected_oracle_ks")
    efforts = tuple(f"K{int(value)}" for value in sorted(oracle_ks))
    rows = read_per_sample(bundle)
    configs = {
        config.key: config
        for config in FRONTIER_SCHEMES
        if config.key in WHITEBOX_KEYS
    }

    notes: list[str] = []
    written: set[Path] = set()
    prose_asr: dict[str, list[float]] = {}
    prose_evasion: dict[str, list[float]] = {}
    prose_quality_given_evasion: dict[str, list[float]] = {}
    for key in WHITEBOX_KEYS:
        config = configs.get(key)
        if config is None:
            notes.append(f"MISSING  {key}: configuration is not a frontier scheme")
            continue
        cells = [
            row for row in rows
            if config.matches(row)
            and row["stage"] == "attack"
            and row["attack_family"] == "oracle"
            and num(row, "llm_judge") == num(row, "llm_judge")
        ]
        if not cells:
            notes.append(f"MISSING  {key}: no judged white-box rows")
            continue

        metrics: dict[str, list[list[float]]] = {
            "evasion": [],
            "asr": [],
            "quality-pass-given-evasion": [],
        }
        for effort in efforts:
            effort_rows = [row for row in cells if row["attack"].endswith(f"-{effort}")]
            if not effort_rows:
                notes.append(f"MISSING  {key} {effort}: no rows")
                continue
            by_document = {document_id(row): row for row in effort_rows}
            if len(by_document) != len(effort_rows):
                raise SystemExit(f"{key} {effort}: multiple white-box rows per document")

            cell = [by_document[document] for document in sorted(by_document)]
            evasion = [1.0 if num(row, "evades_at5") >= 0.5 else 0.0 for row in cell]
            quality_pass = [
                1.0 if num(row, "llm_judge") >= QUALITY_BAR else 0.0
                for row in cell
            ]
            asr = [
                1.0 if did_evade and did_pass else 0.0
                for did_evade, did_pass in zip(evasion, quality_pass)
            ]
            quality_pass_given_evasion = [
                did_pass if did_evade else float("nan")
                for did_evade, did_pass in zip(evasion, quality_pass)
            ]
            values_by_metric = {
                "evasion": (evasion, False),
                "asr": (asr, False),
                "quality-pass-given-evasion": (
                    quality_pass_given_evasion,
                    True,
                ),
            }
            for metric, (values, conditional) in values_by_metric.items():
                point = mean(values)
                lo, hi = percentile_ci(values, conditional)
                metrics[metric].append([
                    float(effort[1:]),
                    point,
                    max(0.0, point - lo),
                    max(0.0, hi - point),
                    float(sum(value == value for value in values)),
                ])
            notes.append(
                f"{key:10s} {effort:3s} n={len(cell)}, evasion={mean(evasion):.3f}, "
                f"quality>={QUALITY_BAR:.2f}={mean(quality_pass):.3f}, "
                f"quality>={QUALITY_BAR:.2f}|evasion="
                f"{mean(quality_pass_given_evasion):.3f}, "
                f"ASR@{QUALITY_BAR:.2f}={mean(asr):.3f}"
            )

        for metric, table in metrics.items():
            if not table:
                continue
            path = OUT_DIR / f"{metric}-{key}.dat"
            write_table(path, ["K", "y", "em", "ep", "n"], table)
            written.add(path)
            if metric == "asr":
                prose_asr[key] = [row[1] for row in table]
            elif metric == "evasion":
                prose_evasion[key] = [row[1] for row in table]
            elif metric == "quality-pass-given-evasion":
                prose_quality_given_evasion[key] = [row[1] for row in table]

    required = {
        "ksemstamp": "resultWhiteboxKSemAsr",
        "kswordstamp": "resultWhiteboxKSwordAsr",
    }
    missing_prose = sorted(set(required) - set(prose_asr))
    if missing_prose:
        raise SystemExit(
            "cannot emit white-box prose results; missing "
            + ", ".join(missing_prose)
        )
    missing_evasion = sorted(set(required) - set(prose_evasion))
    if missing_evasion:
        raise SystemExit(
            "cannot emit white-box evasion results; missing "
            + ", ".join(missing_evasion)
        )
    missing_quality_given_evasion = sorted(
        set(required) - set(prose_quality_given_evasion)
    )
    if missing_quality_given_evasion:
        raise SystemExit(
            "cannot emit white-box conditional-quality results; missing "
            + ", ".join(missing_quality_given_evasion)
        )
    constants_path = OUT_DIR / "constants.tex"
    constants = [
        r"% Generated by python -m visualization extract -- do not edit.",
    ]
    for key, stem in required.items():
        low, high = min(prose_asr[key]), max(prose_asr[key])
        constants.extend([
            rf"\newcommand{{\{stem}Min}}{{{100.0 * low:.1f}\%}}",
            rf"\newcommand{{\{stem}Max}}{{{100.0 * high:.1f}\%}}",
            rf"\newcommand{{\{stem}Range}}{{\{stem}Min--\{stem}Max}}",
        ])
    constants.extend([
        rf"\newcommand{{\resultWhiteboxKSemEvasionLowBudget}}{{{100.0 * prose_evasion['ksemstamp'][0]:.1f}\%}}",
        rf"\newcommand{{\resultWhiteboxKSwordEvasionLowBudget}}{{{100.0 * prose_evasion['kswordstamp'][0]:.1f}\%}}",
        rf"\newcommand{{\resultWhiteboxKSwordEvasionHighBudget}}{{{100.0 * prose_evasion['kswordstamp'][-1]:.1f}\%}}",
        rf"\newcommand{{\resultWhiteboxKSwordEvadingQualityPassHighBudget}}{{{100.0 * prose_quality_given_evasion['kswordstamp'][-1]:.1f}\%}}",
    ])
    constants_path.write_text("\n".join(constants) + "\n")
    written.add(constants_path)

    for path in prune_unwritten(OUT_DIR, written):
        notes.append(f"PRUNED   {path.name}: no source rows in this bundle")
    write_manifest(
        OUT_DIR / "MANIFEST.txt",
        bundle,
        notes,
        uncertainty=(
            f"{BOOTSTRAP_RESAMPLES} paired-document bootstrap resamples, "
            f"{int(CI_LEVEL * 100)}% percentile"
        ),
    )
    destination = (
        OUT_DIR.relative_to(Path.cwd())
        if OUT_DIR.is_relative_to(Path.cwd())
        else OUT_DIR
    )
    print(f"whitebox: wrote {len(written)} tables to {destination}")
    for note in notes:
        if note.startswith(("MISSING", "PRUNED")):
            print("  " + note)


if __name__ == "__main__":
    main()
