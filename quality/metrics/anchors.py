"""Measure rewording, reordering, and resegmentation with patience anchors."""
from __future__ import annotations
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from segmentation import segment
from .common import _confidence_interval

CHANNELS = ("reword", "reword_novel", "reorder", "merge", "split", "reseg",
            "coverage", "n_anchors")

_TOK = re.compile(r"\w+")


@lru_cache(maxsize=8192)  # References recur across attack leaves.
def _profile(text: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Return lowercase tokens and their NLTK sentence indices."""
    toks: list[str] = []
    units: list[int] = []
    if text and str(text).strip():
        for ui, u in enumerate(segment(str(text), type="sentence", backend="nltk")):
            for t in _TOK.findall(u.display.lower()):
                toks.append(t)
                units.append(ui)
    return tuple(toks), tuple(units)


@lru_cache(maxsize=1)
def _rouge1_scorer():
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)


def _reword(ref: str, cand: str) -> float:
    """Return one minus ROUGE-1 F1, or NaN when both texts are empty."""
    if not _TOK.search(ref) and not _TOK.search(cand):
        return np.nan
    # Recall covers the reference; precision covers the candidate.
    f1 = _rouge1_scorer().score(ref, cand)["rouge1"].fmeasure
    return 1.0 - float(f1)


def _novel_bigrams(toks_r, toks_c) -> float:
    """Return the candidate's novel-bigram rate."""
    if len(toks_r) < 2 or len(toks_c) < 2:
        return np.nan
    ref_ng = set(zip(toks_r, toks_r[1:]))
    cand_ng = set(zip(toks_c, toks_c[1:]))
    return len(cand_ng - ref_ng) / len(cand_ng)


def anchor_channels(ref: str, cand: str) -> dict:
    """All anchor_* channels for one (reference, candidate) text pair."""
    ref_s = str(ref) if ref else ""
    cand_s = str(cand) if cand else ""
    toks_r, units_r = _profile(ref_s)
    toks_c, units_c = _profile(cand_s)
    o = {c: np.nan for c in CHANNELS}

    o["reword"] = _reword(ref_s, cand_s)
    o["reword_novel"] = _novel_bigrams(toks_r, toks_c)

    cnt_r, cnt_c = Counter(toks_r), Counter(toks_c)
    once_r = {t for t, k in cnt_r.items() if k == 1}
    once_c = {t for t, k in cnt_c.items() if k == 1}
    shared = once_r & once_c
    pos_r = {t: i for i, t in enumerate(toks_r) if t in shared}
    pos_c = {t: i for i, t in enumerate(toks_c) if t in shared}

    m = len(shared)
    o["n_anchors"] = float(m)
    if once_r:
        o["coverage"] = m / len(once_r)
    if m < 2:
        return o

    # anchors in reference order; y* = their candidate-side coordinates
    order = sorted(shared, key=lambda t: pos_r[t])
    xu = np.array([units_r[pos_r[t]] for t in order])
    yp = np.array([pos_c[t] for t in order])
    yu = np.array([units_c[pos_c[t]] for t in order])

    # Ignore within-unit inversions; those are rewording.
    iu, ju = np.triu_indices(m, 1)
    cross = xu[iu] != xu[ju]
    if cross.any():
        o["reorder"] = float((yp[iu] > yp[ju])[cross].mean())

    # A merge erases a witnessed reference boundary.
    bx = xu[:-1] != xu[1:]
    if bx.any():
        o["merge"] = float((yu[:-1] == yu[1:])[bx].mean())

    # A split introduces a witnessed candidate boundary.
    perm = np.argsort(yp)
    yu2, xu2 = yu[perm], xu[perm]
    by = yu2[:-1] != yu2[1:]
    if by.any():
        o["split"] = float((xu2[:-1] == xu2[1:])[by].mean())

    # Missing witnessed boundaries contribute a vacuous rate of 1.
    rec = 1.0 - o["merge"] if bx.any() else 1.0
    prec = 1.0 - o["split"] if by.any() else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    o["reseg"] = 1.0 - f1
    return o


def _aggregate(per: dict) -> dict:
    out = {}
    for c in CHANNELS:
        valid = per[c][~np.isnan(per[c])]
        out[f"anchor_{c}"] = float(valid.mean()) if valid.size else float("nan")
        out[f"anchor_{c}_ci"] = _confidence_interval(valid) if valid.size else float("nan")
        out[f"anchor_{c}_per_sample"] = per[c]
    return out


def evaluate_anchor_structure(gens, refs) -> dict:
    """Aggregate every anchor channel across paired documents."""
    n = len(gens)
    per = {c: np.full(n, np.nan) for c in CHANNELS}
    for i in range(n):
        ch = anchor_channels(refs[i], gens[i])
        for c in CHANNELS:
            per[c][i] = ch[c]
    return _aggregate(per)
