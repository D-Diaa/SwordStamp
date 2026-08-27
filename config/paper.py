"""Authoritative experiment specification for the SwordStamp paper artifact.

The runners, compiler, visualizations, and tests import this module so the
additive ladders cannot drift between stages.
"""

from __future__ import annotations

from dataclasses import dataclass


PROVIDER_CANDIDATES = 64
PROMPT_DATASETS = (
    ("c4-val-def-256", 256),
    ("c4-val-def-256b", 256),
    ("c4-val-def-512", 512),
)
DATASETS = tuple(name for name, _ in PROMPT_DATASETS)
CENTER_DATASET = "c4-center-8192"
CENTER_DOCUMENTS = 8192
HUMAN_NULL_DATASET = "c4-human-def"
HUMAN_NULL_DOCUMENTS = 1024
PAPER_CORPORA = (
    (CENTER_DATASET, CENTER_DOCUMENTS),
    *PROMPT_DATASETS,
    (HUMAN_NULL_DATASET, HUMAN_NULL_DOCUMENTS),
)
ORACLE_KS = (4, 8, 16, 32, 64)

SENTENCE = "sentence-nltk"
MAIN_SEMSPAN = "semspan-spacy-max15-win5"
NONE_PRESET = "config/presets/sentence_none.yaml"


@dataclass(frozen=True)
class PaperRung:
    """One ordered cell in an additive watermark-design ladder."""

    key: str
    family: str
    scheme: str
    mask: str
    sampling: str
    segmentation: str
    preset: str
    num_candidates: int = PROVIDER_CANDIDATES


_COMPONENTS = (
    ("base", "context", "rejection", SENTENCE),
    ("bestn", "context", "best-of-n", SENTENCE),
    ("fixed", "fixed", "best-of-n", SENTENCE),
    ("diverse", "fixed_diverse", "best-of-n", SENTENCE),
    ("span", "fixed_diverse", "best-of-n", MAIN_SEMSPAN),
)

_FAMILIES = {
    "lsh": ("SemStamp", ""),
    "kmeans": ("k-SemStamp", "k"),
}


def _preset(family: str, mask: str, sampling: str, segmentation: str) -> str:
    segmentation_type = "semspan" if segmentation == MAIN_SEMSPAN else "sentence"
    return f"config/presets/{segmentation_type}_{family}_{mask}_{sampling}.yaml"


def ladder(family: str) -> tuple[PaperRung, ...]:
    """Return the five additive paper cells for one partition family."""

    scheme, prefix = _FAMILIES[family]
    return tuple(
        PaperRung(
            key=f"{prefix}{key}",
            family=family,
            scheme=scheme,
            mask=mask,
            sampling=sampling,
            segmentation=segmentation,
            preset=_preset(family, mask, sampling, segmentation),
        )
        for key, mask, sampling, segmentation in _COMPONENTS
    )


LADDERS = {family: ladder(family) for family in _FAMILIES}
PAPER_RUNGS = tuple(rung for family in _FAMILIES for rung in LADDERS[family])


@dataclass(frozen=True)
class Comparison:
    """One fixed comparison-system cell reported in the paper."""

    key: str
    family: str
    mask: str
    sampling: str
    segmentation: str
    msig: int
    num_candidates: int = PROVIDER_CANDIDATES


PMARK = Comparison("pmark-online", "pmark", "online", "rejection", SENTENCE, 4)
SAMARK = Comparison("samark", "samark", "flags-run", "rejection", SENTENCE, 2)
COMPARISONS = (PMARK, SAMARK)


def validate() -> None:
    """Fail fast if a future edit makes the paper registry inconsistent."""

    if tuple(len(LADDERS[family]) for family in _FAMILIES) != (5, 5):
        raise ValueError("each paper family must have exactly five additive rungs")
    if len({rung.key for rung in PAPER_RUNGS}) != len(PAPER_RUNGS):
        raise ValueError("paper rung keys must be unique")
    if sum(size for _, size in PAPER_CORPORA) != 10240:
        raise ValueError("paper C4 corpus sizes must total 10240 unique documents")
    for family, rungs in LADDERS.items():
        if tuple(rung.family for rung in rungs) != (family,) * 5:
            raise ValueError(f"mixed families in {family} ladder")
        if tuple(rung.num_candidates for rung in rungs) != (
            PROVIDER_CANDIDATES,
        ) * 5:
            raise ValueError(f"candidate budget drift in {family} ladder")
    comparison_specs = tuple(
        (comparison.msig, comparison.num_candidates)
        for comparison in COMPARISONS
    )
    if comparison_specs != (
        (4, PROVIDER_CANDIDATES),
        (2, PROVIDER_CANDIDATES),
    ):
        raise ValueError("comparison signature widths or candidate budgets drifted")


validate()
