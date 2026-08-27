"""Build appendix attack-channel trajectory tables."""

from __future__ import annotations

import collections
from pathlib import Path

from .common import (
    DATA_ROOT,
    FRONTIER_SCHEMES,
    LADDER,
    bundle_path,
    mean,
    num,
    prune_unwritten,
    read_cell_summary,
    read_per_sample,
    segmentation_signature,
    z_retention,
)


OUT_DIR = DATA_ROOT / "attack_channels"
QUALITY_BAR = 0.90
SCHEME_KEYS = ("ksemstamp",)
CHANNEL_COLUMNS = ("anchor_reword", "anchor_reorder", "anchor_reseg")
LADDERS = {
    "ksemstamp": LADDER,
}
CELL_IDENTITY = (
    "scheme", "family", "mask", "sampling", "segmentation", "flag_scope",
    "msig", "generation_num_candidates",
)


def as_int(row: dict, field: str) -> int:
    value = num(row, field)
    if value != value:
        raise SystemExit(f"missing {field} for {row['attack']}")
    return int(value)


def stage_rows(rows: list[dict], scheme) -> list[dict]:
    return [
        row for row in rows
        if scheme.matches(row)
        and row["stage"] == "attack"
        and row["z_source"] == "direct"
    ]


def cell_identity(row: dict) -> tuple[object, ...]:
    """Return one identity across CSV summaries and Parquet sample rows."""
    identity = []
    for field in CELL_IDENTITY:
        if field in {"msig", "generation_num_candidates"}:
            value = num(row, field)
            identity.append(None if value != value else value)
            continue
        value = row.get(field)
        identity.append("" if value is None else str(value))
    return tuple(identity)


def retention_by_cell(rows: list[dict]) -> dict[tuple[tuple[object, ...], str], float]:
    pairs: dict[tuple[tuple[object, ...], str], list[tuple[float, float]]] = (
        collections.defaultdict(list)
    )
    for row in rows:
        if row["stage"] != "attack" or row["z_source"] != "direct":
            continue
        z, z_wm = num(row, "z"), num(row, "z_wm")
        if z == z and z_wm == z_wm:
            pairs[(cell_identity(row), row["attack"])].append((z, z_wm))
    return {key: z_retention(value) for key, value in pairs.items()}


def evading_measures_by_cell(
    rows: list[dict],
) -> dict[
    tuple[tuple[object, ...], str],
    tuple[float, float, float, float, int, float, int, int],
]:
    values: dict[tuple[tuple[object, ...], str], list[list[float]]] = {}
    quality_passes: dict[tuple[tuple[object, ...], str], list[float]] = (
        collections.defaultdict(list)
    )
    for row in rows:
        if row["stage"] != "attack" or row["z_source"] != "direct":
            continue
        if num(row, "evades_at5") < 0.5:
            continue
        key = (cell_identity(row), row["attack"])
        quality = num(row, "llm_judge")
        quality_passes[key].append(1.0 if quality >= QUALITY_BAR else 0.0)
        if quality != quality:
            continue
        values.setdefault(key, [[], [], [], []])
        for index, column in enumerate(CHANNEL_COLUMNS):
            value = num(row, column)
            if value == value:
                values[key][index].append(value)
        values[key][3].append(quality)
    result = {}
    for key, passes in quality_passes.items():
        measures = values.get(key, [[], [], [], []])
        result[key] = (
            mean(measures[0]), mean(measures[1]), mean(measures[2]),
            mean(measures[3]), len(measures[0]), mean(passes),
            int(sum(passes)), len(passes),
        )
    return result


def dipper_trajectories(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    dipper = [row for row in rows if row["attack_family"] == "dipper"]
    order = sorted(
        (row for row in dipper if as_int(row, "dipper_lex") == 20),
        key=lambda row: as_int(row, "dipper_order"),
    )
    lexical = sorted(
        (row for row in dipper if as_int(row, "dipper_order") == 20),
        key=lambda row: as_int(row, "dipper_lex"),
    )
    if len(order) < 2 or len(lexical) < 2:
        raise SystemExit("Dipper requires both order and lexical trajectories")
    return [("dipper-order", order), ("dipper-lexical", lexical)]


def sentence_points(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    trajectories: list[tuple[str, list[dict]]] = []
    for family in ("pegasus", "parrot"):
        family_rows = [row for row in rows if row["attack_family"] == family]
        plain = [row for row in family_rows if row["attack_setting"] == "default"]
        bigram = [
            row for row in family_rows
            if "bigram" in str(row.get("attack_setting"))
        ]
        if len(plain) != 1 or len(bigram) != 1:
            raise SystemExit(
                f"{family} requires one default and one bigram point"
            )
        trajectories.extend([(family, plain), (f"{family}-bigram", bigram)])
    return trajectories


def adaptive_trajectory(rows: list[dict], rung, anchor: str, attacker_seg: str,
                        bag_agg: str, name: str,
                        required: bool = True) -> list[dict]:
    matches = [
        row for row in rows
        if rung.matches(row)
        and row["stage"] == "attack"
        and row["z_source"] == "direct"
        and row["attack_family"] == "adaptive"
        and row["anchor"] == anchor
        and segmentation_signature(row["attacker_seg"])
            == segmentation_signature(attacker_seg)
        and row["candidate_agg"] == "min"
        and row["bag_agg"] == bag_agg
    ]
    matches.sort(key=lambda row: as_int(row, "K"))
    if required and len(matches) < 2:
        raise SystemExit(f"adaptive attack requires a {name} budget trajectory")
    return matches


def adaptive_trajectories(rows: list[dict], base, ladder) -> list[tuple[str, list[dict]]]:
    return [
        ("adaptive-pos-sent", adaptive_trajectory(
            rows, base, "positional", "sentence-nltk", "", "positional sentence"
        )),
        ("adaptive-bag-sent", adaptive_trajectory(
            rows, ladder[2], "bag", "sentence-nltk", "min", "bag sentence"
        )),
        ("adaptive-bag-semspans", adaptive_trajectory(
            rows, ladder[4], "bag", "semspan-nltk-max15-win5", "min",
            "bag semantic-span"
        )),
    ]


def setting_label(series: str, row: dict) -> str:
    if series == "pegasus":
        return "Peg"
    if series == "pegasus-bigram":
        return "Peg+B"
    if series == "parrot":
        return "Par"
    if series == "parrot-bigram":
        return "Par+B"
    if series == "dipper-order":
        return f"O{as_int(row, 'dipper_order')}"
    if series == "dipper-lexical":
        return f"L{as_int(row, 'dipper_lex')}"
    return f"K{as_int(row, 'K')}"


def write_trajectories(
    path: Path, trajectories, retentions, evading,
) -> int:
    """Write one trajectory table and return its data-row count."""
    count = 0
    with open(path, "w") as fh:
        header = (
            "series setting endpoint reword reorder reseg zret "
            "evading_reword evading_reorder evading_reseg evading_quality evading_n "
            "evading_quality_pass evading_quality_pass_numer "
            "evading_quality_pass_denom"
        )
        fh.write(header + "\n")
        for series, trajectory in trajectories:
            for index, row in enumerate(trajectory):
                if len(trajectory) == 1:
                    endpoint = "point"
                elif index == 0:
                    endpoint = "start"
                elif index == len(trajectory) - 1:
                    endpoint = "end"
                else:
                    endpoint = "none"
                values = " ".join(f"{num(row, column):.6f}" for column in CHANNEL_COLUMNS)
                retention = retentions.get((cell_identity(row), row["attack"]))
                if retention is None:
                    raise SystemExit(f"missing z retention for {series} {row['attack']}")
                measures = evading.get((cell_identity(row), row["attack"]))
                if measures is None:
                    measures = (
                        float("nan"), float("nan"), float("nan"),
                        float("nan"), 0, float("nan"), 0, 0,
                    )
                fh.write(
                    f"{series} {setting_label(series, row)} {endpoint} "
                    f"{values} {retention:.6f} "
                    + " ".join(
                        f"{value:.6f}"
                        for value in measures
                    )
                    + "\n"
                )
                count += 1
    return count


def write_scheme(rows: list[dict], scheme, ladder, retentions, evading) -> tuple[Path, int]:
    base_rows = stage_rows(rows, scheme)
    trajectories = [
        *sentence_points(base_rows),
        *dipper_trajectories(base_rows),
        *adaptive_trajectories(rows, scheme, ladder),
    ]
    path = OUT_DIR / f"{scheme.key}.dat"
    return path, write_trajectories(path, trajectories, retentions, evading)


def main() -> None:
    bundle = bundle_path()
    rows = read_cell_summary(bundle)
    sample_rows = read_per_sample(bundle)
    retentions = retention_by_cell(sample_rows)
    evading = evading_measures_by_cell(sample_rows)
    targets = {
        scheme.key: scheme
        for scheme in FRONTIER_SCHEMES
        if scheme.key in SCHEME_KEYS
    }
    if set(targets) != set(SCHEME_KEYS):
        raise SystemExit("published SemStamp configurations are missing from FRONTIER_SCHEMES")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = [
        "Generated by python -m visualization extract -- do not edit.",
        f"bundle: {bundle.name}",
        "Each row summarizes one attack setting.",
        "Edit-signature columns are means of anchor-derived measures.",
        "Endpoints mark the first and final native-effort settings.",
        "zret is normalized z retention from paired corpus sums.",
        "evading columns use detector-evasive outputs at the 5% FPR operating point.",
        f"evading_quality_pass is P(llm_judge >= {QUALITY_BAR:.2f} | evades_at5); missing judge scores fail the threshold.",
        "evading_quality_pass_numer and evading_quality_pass_denom record each conditional rate's counts.",
        "Adaptive rows use their rung's matched attacker; adaptive-base rows use positional anchors and sentence segmentation where available.",
        "Connected rows trace native attack effort, not a causal attribution.",
        "",
    ]
    for key in SCHEME_KEYS:
        path, count = write_scheme(
            rows, targets[key], LADDERS[key], retentions, evading
        )
        manifest.append(f"{path.name}: {count} trajectory points")
    expected = {OUT_DIR / f"{key}.dat" for key in SCHEME_KEYS}
    for path in prune_unwritten(OUT_DIR, expected):
        manifest.append(f"pruned stale output: {path.name}")

    # Prose-facing values for the appendix discussion of the k-means panel.
    # Generate them beside the plotted trajectories so the narrative cannot
    # retain values or orderings from an earlier result bundle.
    kbase_rows = stage_rows(rows, targets["ksemstamp"])
    dipper = dict(dipper_trajectories(kbase_rows))
    dipper_order = dipper["dipper-order"]
    dipper_lexical = dipper["dipper-lexical"]
    adaptive = dict(adaptive_trajectories(
        rows, targets["ksemstamp"], LADDERS["ksemstamp"]
    ))

    def evading_measures(row: dict) -> tuple[float, ...]:
        measures = evading.get((cell_identity(row), row["attack"]))
        if measures is None:
            raise SystemExit(f"missing evading measures for {row['attack']}")
        return measures

    def by_budget(trajectory: list[dict], budgets: set[int]) -> list[dict]:
        selected = [row for row in trajectory if as_int(row, "K") in budgets]
        if len(selected) != len(budgets):
            raise SystemExit(
                f"expected adaptive budgets {sorted(budgets)}, found "
                f"{[as_int(row, 'K') for row in selected]}"
            )
        return selected

    positional_low = by_budget(
        adaptive["adaptive-pos-sent"], {1, 2, 4, 8}
    )
    comparison_budgets = {4, 8, 16, 32, 64}
    bag_sentence_all = adaptive["adaptive-bag-sent"]
    bag_semspans_all = adaptive["adaptive-bag-semspans"]
    bag_sentence = by_budget(bag_sentence_all, comparison_budgets)
    bag_semspans = by_budget(bag_semspans_all, comparison_budgets)
    positional_quality = [evading_measures(row)[5] for row in positional_low]
    dipper_quality = [evading_measures(row)[5] for row in dipper_order]
    bag_sentence_reseg = [
        evading_measures(row)[2] for row in bag_sentence_all
    ]
    bag_semspan_reseg = [
        evading_measures(row)[2] for row in bag_semspans_all
    ]
    quality_pairs = [
        (evading_measures(span_row)[5], evading_measures(sentence_row)[5])
        for span_row, sentence_row in zip(bag_semspans, bag_sentence)
    ]
    lower_count = sum(span < sentence for span, sentence in quality_pairs)
    if lower_count == len(quality_pairs):
        raise SystemExit(
            "the appendix comparison no longer needs its high-budget exception"
        )

    def percent_command(name: str, value: float) -> str:
        return rf"\newcommand{{\{name}}}{{{100.0 * value:.1f}\%}}"

    dipper_start = evading_measures(dipper_order[0])
    dipper_end = evading_measures(dipper_order[-1])
    dipper_lexical_end = evading_measures(dipper_lexical[-1])
    constants = [
        r"% Generated by python -m visualization extract -- do not edit.",
        percent_command("resultDipperEvadingReorderStart", dipper_start[1]),
        percent_command("resultDipperEvadingReorderEnd", dipper_end[1]),
        percent_command("resultDipperEvadingRewordStart", dipper_start[0]),
        percent_command("resultDipperEvadingRewordEnd", dipper_end[0]),
        percent_command("resultDipperOrderEvadingQualityPassEnd", dipper_end[5]),
        percent_command("resultDipperLexicalEvadingRewordEnd", dipper_lexical_end[0]),
        percent_command("resultDipperLexicalEvadingQualityPassEnd", dipper_lexical_end[5]),
        percent_command("resultEdapEvadingQualityPassMin", min(positional_quality)),
        percent_command("resultEdapEvadingQualityPassMax", max(positional_quality)),
        percent_command("resultDipperEvadingQualityPassMin", min(dipper_quality)),
        percent_command("resultDipperEvadingQualityPassMax", max(dipper_quality)),
        percent_command("resultBagSentenceResegMin", min(bag_sentence_reseg)),
        percent_command("resultBagSentenceResegMax", max(bag_sentence_reseg)),
        percent_command("resultBagSemspanResegMin", min(bag_semspan_reseg)),
        percent_command("resultBagSemspanResegMax", max(bag_semspan_reseg)),
        rf"\newcommand{{\resultBagSemspanQualityLowerCount}}{{{lower_count}}}",
        rf"\newcommand{{\resultBagSemspanQualityBudgetCount}}{{{len(quality_pairs)}}}",
        percent_command("resultBagSentenceQualityHighBudget", quality_pairs[-1][1]),
        percent_command("resultBagSemspanQualityHighBudget", quality_pairs[-1][0]),
    ]
    (OUT_DIR / "constants.tex").write_text("\n".join(constants) + "\n")
    (OUT_DIR / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    print(f"attack_channels: wrote {len(SCHEME_KEYS)} tables under {OUT_DIR}")


if __name__ == "__main__":
    main()
