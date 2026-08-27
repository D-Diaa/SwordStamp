r"""Build tab:costs: clean performance and prices.

Base schemes, then one five-stage ladder per partition family.  Every value
comes from a clean watermark or human row, never an attacked cell, except the
ASR columns.  PMark online uses its analytic threshold; everything else uses
its empirical human null.

Emits tables/costs.tex as macros, since a bare \input inside a tabular leaves
a cell open. Run through ``python -m visualization extract``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from config.paper import HUMAN_NULL_DOCUMENTS

from .common import (
    BAG_ANCHOR,
    FAMILIES,
    FRONTIER_SCHEMES,
    POSITIONAL_ANCHOR,
    is_matched_attacker,
    TABLE_ROOT,
    Rung,
    bundle_path,
    document_id,
    ladder,
    matched_anchor,
    num,
    read_cell_summary,
    read_per_sample,
    segmentation_signature,
    write_manifest,
)

OUT_DIR = TABLE_ROOT
ASR_QUALITY_BARS = (0.90,)
PRIMARY_FPR = 5
PUBLISHED_SORT_METRIC = "adaptive_at1_q90"

# One no-box column per attacker class, ordered by strength.  Families match
# extract_cross_scheme's series, and together they
# cover every judged no-box row in the bundle: the remaining family is the
# probe atoms, which carry no judge score by design.
ATTACKER_COLUMNS = (
    ("sentence", frozenset({"pegasus", "parrot"})),
    ("dipper", frozenset({"dipper"})),
    ("adaptive", frozenset({"adaptive"})),
)
# Rows are Rungs: same identity and matching, candidate budget included.
Configuration = Rung


@dataclass(frozen=True)
class Metric:
    key: str
    source: str
    higher_is_better: bool
    formatting: str
    places: int
    ci_key: str | None = None
    emphasize: bool = True


# The base schemes we do not build on.  \semstamp and \ksemstamp are absent
# because each opens its own ladder block instead.  Keyed off FRONTIER_SCHEMES
# so detector choices recorded there move this table with the figures.
PUBLISHED_LABELS = {
    "pmark-online": r"\pmark\ (online)",
    "samark": r"\samark",
}
PUBLISHED = [
    replace(scheme, label=PUBLISHED_LABELS[scheme.key])
    for scheme in FRONTIER_SCHEMES
    if scheme.key in PUBLISHED_LABELS
]


# Row labels by key suffix; identities come from common so a component
# change lands in one place.  Macro names are shared with the figure legends.
RUNG_LABELS = {
    "bestn": r"\quad\rungaddbestn",
    "fixed": r"\quad\rungaddfixed",
    "diverse": r"\quad\rungadddiverse",
}

def _ladder_block(family: str, base: str) -> list[Rung]:
    """Return one family's five-stage component sequence."""
    prefix = FAMILIES[family][1]
    rows = []
    for rung in ladder(family):
        suffix = rung.key[len(prefix):]
        if suffix == "base":
            label = base
        elif suffix == "span":
            label = r"\quad\rungfinaltableentry"
        else:
            label = RUNG_LABELS[suffix]
        rows.append(replace(rung, label=label))
    return rows


LSH_LADDER = _ladder_block("lsh", r"\semstamp")
KMEANS_LADDER = _ladder_block("kmeans", r"\ksemstamp")
LADDER_CONFIGS = LSH_LADDER + KMEANS_LADDER


METRICS = [
    Metric("TPR_at1", "marked", True, "proportion", 1),
    Metric("TPR_at5", "marked", True, "proportion", 1),
    Metric("thr1", "null", False, "decimal", 2, emphasize=False),
    Metric("thr5", "null", False, "decimal", 2, emphasize=False),
    *(
        Metric(f"{name}_at{fpr}_q{int(round(bar * 100))}", name, False,
               "proportion", 1)
        for name, _ in ATTACKER_COLUMNS
        for bar in ASR_QUALITY_BARS
        for fpr in (1, 5)
    ),
    Metric("llm_quality", "marked", True, "proportion", 1,
           ci_key="llm_quality_ci"),
    Metric("provider_draws_per_sentence", "marked", False, "decimal", 1),
    Metric("gen_ppl_corpus", "marked", False, "decimal", 2),
    Metric("sem_ent", "marked", True, "decimal", 2),
    Metric("rep_3", "marked", False, "proportion", 1),
    Metric("sent_dup_pct", "marked", False, "percent_points", 1),
]

# Leading columns pair the two FPR points; the rest get one column each.
# Derived from METRICS so adding a quality bar cannot shift the columns after
# it.  Emphasis stays metric-wise inside a pair.
# One ASR pair per no-box attacker column per quality bar.
_PAIRED = 2 + len(ATTACKER_COLUMNS) * len(ASR_QUALITY_BARS)
DISPLAY_COLUMNS = (
    [(2 * i, 2 * i + 1) for i in range(_PAIRED)]
    + [(i,) for i in range(2 * _PAIRED, len(METRICS))]
)

METRIC_INDEX = {metric.key: index for index, metric in enumerate(METRICS)}


def result_command(name: str, value: float, *, percent: bool = False,
                   places: int = 1) -> str:
    """Format one generated prose-facing result command."""
    if math.isnan(value):
        raise SystemExit(f"cannot emit \\{name}: source value is missing")
    scaled = value * 100.0 if percent else value
    unit = r"\%" if percent else ""
    return rf"\newcommand{{\{name}}}{{{scaled:.{places}f}{unit}}}"

def present(rows: list[dict], configuration: Configuration) -> bool:
    """True when the bundle carries a clean marked row for this configuration.

    A family absent from the bundle is a coverage fact, not an error; the table
    names what has not landed.  An ambiguous match still fails in require_one.
    """
    return any(
        configuration.matches(row)
        and row["stage"] == "watermark"
        and row["attack"] == "none"
        for row in rows
    )


def require_one(rows: list[dict], configuration: Configuration, stage: str) -> dict:
    """Return the unique source row, failing rather than silently choosing."""
    candidates = [
        row for row in rows
        if configuration.matches(row)
        and row["stage"] == stage
        and row["attack"] == "none"
    ]
    if len(candidates) != 1:
        raise SystemExit(
            f"{configuration.key}: expected one {stage} row with attack=none, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def judged_attack_rows(rows: list[dict], configuration: Configuration,
                       families: frozenset[str],
                       scheme_specific: bool = True) -> list[dict]:
    """Return the configuration's judged attack rows in ``families``.

    With ``scheme_specific``, the adaptive family is narrowed to \\seda, the
    configuration a row's own published anchor rule and segmenter select, so an
    adaptive attack aimed at a different design cannot win the column maximum.
    Other families carry no such switch and pass through either way.  The
    positional-\\eda macros pass False, because their whole job is to read the
    arm the scheme's own rules do not select.

    Split out of highest_asr, which runs once per operating point, so the
    per-sample table is scanned once per configuration and column.
    """
    anchor = matched_anchor(configuration.mask)
    return [
        row for row in rows
        if configuration.matches(row)
        and row["stage"] == "attack"
        and row["attack_family"] in families
        and num(row, "llm_judge") == num(row, "llm_judge")
        and (not scheme_specific
             or is_matched_attacker(row, anchor, configuration.segmentation))
    ]


def highest_asr(cells: list[dict], configuration: Configuration, fpr: int,
                quality_bar: float) -> tuple[float, str]:
    """Return the strongest observed ASR over ``cells`` at ``quality_bar``.

    Every setting judged_attack_rows kept counts, with a missing document
    counted as a failure, so a column reports what a row concedes to the
    strongest effort setting of that attacker class.
    """
    detection_field = f"detected_at{fpr}"
    documents = sorted({document_id(row) for row in cells})
    attacks = sorted({row["attack"] for row in cells})
    document_index = {document: index for index, document in enumerate(documents)}
    attack_index = {attack: index for index, attack in enumerate(attacks)}
    successes = [[0.0] * len(documents) for _ in attacks]
    for row in cells:
        if (num(row, detection_field) < 0.5
                and num(row, "llm_judge") >= quality_bar):
            successes[attack_index[row["attack"]]][document_index[document_id(row)]] = 1.0

    counts = [int(sum(success)) for success in successes]
    best = max(range(len(counts)), key=lambda index: counts[index])
    return counts[best] / len(documents), attacks[best]


# --------------------------------------------------------------------------
# Prose macros that name one attack
# --------------------------------------------------------------------------

# The table's ASR columns report what a scheme concedes to any judged setting
# (highest_asr), which is the right contract for tab:costs.  The introduction
# and the abstract instead name a single attack, the positional,
# sentence-anchored \eda of \S\ref{sec:attacks}.  The bundle also carries the
# bag-anchored arm that \S\ref{subsec:fixed-valid-sets} introduces, and on
# \samark the two arms sit 0.2 points apart at q=0.90, so a column maximum can
# flip between them without changing any printed digit.  The prose macros below
# pin the arm, so an introduction number can never come from an attack the
# reader meets only in \S5.
PROSE_EDA_SEGMENTATION = "sentence-nltk"

# Keys of the four published schemes the introduction reports \eda against.
PRIOR_SCHEME_KEYS = ("base", "kbase", "pmark-online", "samark")


def positional_eda_rows(per_sample_rows: list[dict],
                        configuration: Configuration) -> list[dict]:
    """Return the configuration's judged rows for the positional-anchor \\eda."""
    return [
        row for row in judged_attack_rows(per_sample_rows, configuration,
                                          frozenset({"adaptive"}),
                                          scheme_specific=False)
        if row.get("anchor") == POSITIONAL_ANCHOR
        and segmentation_signature(row.get("attacker_seg"))
        == segmentation_signature(PROSE_EDA_SEGMENTATION)
    ]


def corpus_perplexity(row: dict) -> float:
    """Return exp of the token-weighted mean negative log-likelihood.

    Averaging per-document perplexities linearly would let one short
    high-surprisal document set a configuration's value.
    """
    total_nll = num(row, "gen_ppl_total_nll")
    tokens = num(row, "gen_ppl_scored_tokens")
    if not tokens or tokens != tokens:
        return float("nan")
    return math.exp(total_nll / tokens)


def record(rows: list[dict], per_sample_rows: list[dict],
           configuration: Configuration) -> tuple[
        Configuration, list[float], list[float], str]:
    """Return raw metric values and provenance for one table configuration."""
    marked = require_one(rows, configuration, "watermark")

    # PMark online has no human row; its analytic thresholds sit on the
    # watermark row.  Everything else uses its empirical human null.
    if configuration.mask == "online":
        null = marked
        threshold_source = "analytic standard-normal null"
    else:
        null = require_one(rows, configuration, "human")
        threshold_source = f"{HUMAN_NULL_DOCUMENTS:,}-document empirical human null"

    cells_by_column = {
        name: judged_attack_rows(per_sample_rows, configuration, families)
        for name, families in ATTACKER_COLUMNS
    }
    attack_results = {
        f"{name}_at{fpr}_q{int(quality_bar * 100)}": (
            highest_asr(cells, configuration, fpr, quality_bar)
            if cells else (float("nan"), "none")
        )
        for name, cells in cells_by_column.items()
        for quality_bar in ASR_QUALITY_BARS
        for fpr in (1, 5)
    }
    values = []
    ci_half_widths = []
    for metric in METRICS:
        if metric.key in attack_results:
            value = attack_results[metric.key][0]
        elif metric.key == "gen_ppl_corpus":
            value = corpus_perplexity(marked)
        else:
            value = num(marked if metric.source == "marked" else null, metric.key)
        values.append(value)
        source = marked if metric.source == "marked" else null
        ci_half_widths.append(
            num(source, metric.ci_key) if metric.ci_key else float("nan")
        )
    for metric, value, ci_half_width in zip(METRICS, values, ci_half_widths):
        if math.isnan(value):
            raise SystemExit(f"{configuration.key}: missing {metric.key}")
        if metric.ci_key and math.isnan(ci_half_width):
            raise SystemExit(f"{configuration.key}: missing {metric.ci_key}")
    attack_notes = "; ".join(
        f"q={quality_bar:.2f} {name}: "
        + ", ".join(
            f"{fpr}%={attack_results[f'{name}_at{fpr}_q{int(quality_bar * 100)}'][1]}"
            for fpr in (1, 5)
        )
        for quality_bar in ASR_QUALITY_BARS
        for name, _ in ATTACKER_COLUMNS
    )
    note = (f"{configuration.key:14s} n={marked['n_samples']:<4s} "
            f"threshold={threshold_source}; ASR settings: {attack_notes}")
    return configuration, values, ci_half_widths, note


def metric_extrema(records: list[
        tuple[Configuration, list[float], list[float], str]
]) -> tuple[list[float], list[float]]:
    """Find metric-wise best and worst values across the displayed rows.

    A metric is compared over the rows that have it, so a partly-absent column
    still marks its available cells.  A wholly absent one yields NaN and neither
    mark.
    """
    best_values = []
    worst_values = []
    for index, metric in enumerate(METRICS):
        present = [values[index] for _, values, _, _ in records
                   if not math.isnan(values[index])]
        if not present:
            best_values.append(float("nan"))
            worst_values.append(float("nan"))
            continue
        best_values.append((max if metric.higher_is_better else min)(present))
        worst_values.append((min if metric.higher_is_better else max)(present))
    return best_values, worst_values


ABSENT = "---"  # a metric this configuration has no value for


def format_metric(value: float, ci_half_width: float, metric: Metric) -> str:
    """Format a table value while preserving the source metric's unit."""
    if math.isnan(value):
        return ABSENT
    if metric.ci_key:
        mean = 100.0 * value
        lower = 100.0 * (value - ci_half_width)
        upper = 100.0 * (value + ci_half_width)
        return (f"{mean:.{metric.places}f} "
                f"[{lower:.{metric.places}f},{upper:.{metric.places}f}]")
    if metric.formatting == "proportion":
        value *= 100.0
    elif metric.formatting not in {"decimal", "percent_points"}:
        raise AssertionError(f"unknown formatting {metric.formatting}")
    return f"{value:.{metric.places}f}"


def row_text(item: tuple[Configuration, list[float], list[float], str],
             best_values: list[float], worst_values: list[float]) -> str:
    """Format one row, bolding best and underlining worst metric values."""
    configuration, values, ci_half_widths, _ = item
    formatted_metrics = []
    for metric, value, ci_half_width, best, worst in zip(
            METRICS, values, ci_half_widths, best_values, worst_values):
        text = format_metric(value, ci_half_width, metric)
        if (metric.emphasize
                and not math.isnan(value)
                and math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12)):
            text = rf"\textbf{{{text}}}"
        if (metric.emphasize
                and not math.isnan(value)
                and math.isclose(value, worst, rel_tol=0.0, abs_tol=1e-12)):
            text = rf"\underline{{{text}}}"
        formatted_metrics.append(text)
    segmentation = segmentation_signature(configuration.segmentation)
    if segmentation == ("sentence",):
        unit = "sent"
    elif segmentation and segmentation[0] == "semspan":
        unit = "span"
    else:
        raise SystemExit(
            f"{configuration.key}: no cost-table EDA-S unit label for "
            f"segmentation '{configuration.segmentation}'"
        )
    anchor = {
        POSITIONAL_ANCHOR: "pos",
        BAG_ANCHOR: "bag",
    }[matched_anchor(configuration.mask)]
    eda_configuration = {
        ("pos", "sent"): r"\edaconfigpossent",
        ("bag", "sent"): r"\edaconfigbagsent",
        ("bag", "span"): r"\edaconfigbagspan",
    }.get((anchor, unit))
    if eda_configuration is None:
        raise SystemExit(
            f"{configuration.key}: no cost-table EDA-S configuration macro "
            f"for ({anchor}, {unit})"
        )
    fields = [configuration.label, eda_configuration]
    fields.extend(
        " / ".join(formatted_metrics[index] for index in column)
        for column in DISPLAY_COLUMNS
    )
    return " & ".join(fields) + r" \\"


def main() -> None:
    bundle = bundle_path()
    rows = read_cell_summary(bundle)
    per_sample_rows = read_per_sample(bundle)

    landed = {
        configuration.key for configuration in PUBLISHED + LADDER_CONFIGS
        if present(rows, configuration)
    }
    missing = [
        configuration for configuration in PUBLISHED + LADDER_CONFIGS
        if configuration.key not in landed
    ]
    published_records = [
        record(rows, per_sample_rows, configuration)
        for configuration in PUBLISHED
        if configuration.key in landed
    ]
    published_sort_index = next(
        index for index, metric in enumerate(METRICS)
        if metric.key == PUBLISHED_SORT_METRIC
    )
    published_records.sort(
        key=lambda item: item[1][published_sort_index], reverse=True
    )
    # One macro per block so the table can rule between families; emphasis
    # still compares every displayed row.
    ladder_blocks = {
        macro: [
            record(rows, per_sample_rows, configuration)
            for configuration in configurations
            if configuration.key in landed
        ]
        for macro, configurations in (
            ("lshladdercostrows", LSH_LADDER),
            ("kmeansladdercostrows", KMEANS_LADDER),
        )
    }
    displayed_records = published_records + [
        item for records in ladder_blocks.values() for item in records
    ]
    table_best, table_worst = (metric_extrema(displayed_records)
                               if displayed_records else ([], []))

    body = [r"% Generated by python -m visualization extract -- do not edit.",
            result_command("costasrqualitybar", ASR_QUALITY_BARS[0],
                           percent=True, places=0),
            result_command("resultQualityRequirement", ASR_QUALITY_BARS[0],
                           percent=True, places=0),
            result_command("resultPrimaryFpr", PRIMARY_FPR / 100.0,
                           percent=True, places=0)]

    records_by_key = {
        configuration.key: values
        for configuration, values, _, _ in displayed_records
    }
    configs_by_key = {
        configuration.key: configuration
        for configuration, _, _, _ in displayed_records
    }

    def value(key: str, metric: str) -> float:
        try:
            return records_by_key[key][METRIC_INDEX[metric]]
        except KeyError as error:
            raise SystemExit(
                f"cannot emit prose result: missing displayed row {key}"
            ) from error

    adaptive_metric = f"adaptive_at{PRIMARY_FPR}_q90"
    dipper_metric = f"dipper_at{PRIMARY_FPR}_q90"
    prose_results = {
        "resultSemEdasAsr": value("base", adaptive_metric),
        "resultKSemEdasAsr": value("kbase", adaptive_metric),
        "resultPmarkEdasAsr": value("pmark-online", adaptive_metric),
        "resultPmarkDipperAsr": value("pmark-online", dipper_metric),
        "resultSemFidelity": value("base", "llm_quality"),
        "resultKSemFidelity": value("kbase", "llm_quality"),
        "resultSwordFidelity": value("span", "llm_quality"),
        "resultKSwordFidelity": value("kspan", "llm_quality"),
        "resultSemCleanTpr": value("base", "TPR_at5"),
        "resultKSemCleanTpr": value("kbase", "TPR_at5"),
        "resultSamarkCleanTpr": value("samark", "TPR_at5"),
        # \S\ref{subsec:eval-costs} reads \samark's detection at both operating
        # points, because its self-anchored null costs it far more at 1%.
        "resultSamarkCleanTprOne": value("samark", "TPR_at1"),
        "resultSwordCleanTpr": value("span", "TPR_at5"),
        "resultKSwordCleanTpr": value("kspan", "TPR_at5"),
    }
    # The four published schemes, each under the positional-anchor \eda alone.
    prior_eda = {}
    eda_notes = []
    for key in PRIOR_SCHEME_KEYS:
        if key not in configs_by_key:
            raise SystemExit(
                f"cannot emit the prior-scheme \\eda range: missing row {key}"
            )
        cells = positional_eda_rows(per_sample_rows, configs_by_key[key])
        if not cells:
            raise SystemExit(
                f"{key}: the bundle carries no positional-anchor, "
                f"{PROSE_EDA_SEGMENTATION} adaptive rows, so the introduction "
                "cannot report the \\eda of the attack section against it"
            )
        asr, attack = highest_asr(cells, configs_by_key[key], PRIMARY_FPR,
                                  ASR_QUALITY_BARS[0])
        prior_eda[key] = asr
        eda_notes.append(f"{key}={100.0 * asr:.1f}% ({attack})")
    prose_results["resultPriorEdapAsrMin"] = min(prior_eda.values())
    prose_results["resultPriorEdapAsrMax"] = max(prior_eda.values())
    prose_results["resultSamarkDipperAsr"] = value("samark", dipper_metric)
    body.extend(
        result_command(name, result, percent=True)
        for name, result in prose_results.items()
    )
    point_results = {
        "resultSwordEdasAsrReduction": 100.0 * (
            value("base", adaptive_metric) - value("span", adaptive_metric)
        ),
        "resultKSwordEdasAsrReduction": 100.0 * (
            value("kbase", adaptive_metric) - value("kspan", adaptive_metric)
        ),
        "resultSwordFidelityCost": 100.0 * (
            value("base", "llm_quality") - value("span", "llm_quality")
        ),
        "resultKSwordFidelityCost": 100.0 * (
            value("kbase", "llm_quality") - value("kspan", "llm_quality")
        ),
    }
    body.extend(
        result_command(name, result)
        for name, result in point_results.items()
    )
    body.append(result_command(
        "resultSamarkDipperGap",
        100.0 * (prior_eda["samark"] - prose_results["resultSamarkDipperAsr"]),
    ))
    body.append(result_command(
        "resultSamarkThreshold", value("samark", "thr5"), places=2
    ))
    body.append(
        r"\newcommand{\resultPriorEdapAsrRange}{"
        r"\resultPriorEdapAsrMin--\resultPriorEdapAsrMax}"
    )
    body.append(r"\newcommand{\publishedcostrows}{%")
    notes = [item[3] for item in published_records]
    # An empty block keeps its macro defined, so the table keeps structure.
    for item in published_records:
        body.append(row_text(item, table_best, table_worst))

    body.append("}")
    for macro, records in ladder_blocks.items():
        body.append(rf"\newcommand{{\{macro}}}{{%")
        # An absent family renders no rows but keeps the block.
        for item in records:
            body.append(row_text(item, table_best, table_worst))
            notes.append(item[3])
        body.append("}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "costs.tex", "w") as fh:
        fh.write("\n".join(body) + "\n")

    notes.insert(0, "columns: EDA-S config (anchor rule, segmentation unit), "
                    "TPR@1% / 5%, threshold@1% / 5%, then ASR@q=0.90 "
                    "at 1% / 5% for "
                    + ", ".join(name for name, _ in ATTACKER_COLUMNS)
                    + "; then fidelity mean and 95% CI, mean provider draws "
                      "per output sentence, PPL, semantic entropy, "
                      "repeated trigrams, sentence duplication")
    notes.insert(1, "all rate columns are percentages; thresholds are z-score cutoffs")
    notes.insert(2, f"base schemes sorted by descending {PUBLISHED_SORT_METRIC}")
    notes.insert(3, "quality metrics: means over marked documents; fidelity "
                    "cells show the mean's 95% t confidence interval")
    notes.insert(4, "ASR: strongest observed judged setting within each "
                    "attacker column at each quality bar "
                    + ", ".join(f"{quality_bar:.2f}" for quality_bar in ASR_QUALITY_BARS))
    notes.insert(5, "attacker columns: "
                    + "; ".join(f"{name}={'/'.join(sorted(families))}"
                                for name, families in ATTACKER_COLUMNS))
    notes.insert(6, "prose \\eda range: positional anchor, "
                    f"{PROSE_EDA_SEGMENTATION} attacker segmentation only, "
                    f"@{PRIMARY_FPR}% q={ASR_QUALITY_BARS[0]:.2f}; "
                    + ", ".join(eda_notes))
    for configuration in missing:
        notes.append(f"MISSING  {configuration.key:12s} ({configuration.label}): "
                     "not in this bundle")
    write_manifest(OUT_DIR / "COSTS_MANIFEST.txt", bundle, notes)
    print(f"costs: wrote {len(published_records)}/{len(PUBLISHED)} published and "
          f"{sum(len(r) for r in ladder_blocks.values())}/{len(LADDER_CONFIGS)} "
          "ladder rows to "
          f"{OUT_DIR / 'costs.tex'}")
    for configuration in missing:
        print(f"  MISSING  {configuration.key:12s} ({configuration.label})")


if __name__ == "__main__":
    main()
