"""Encoder-driven semantic spans inside sentence-backend boundaries.

This stays separate from ``_spacy`` because semspan can wrap either the NLTK or
spaCy sentence backend; spaCy contributes only its static English stop-word
list here, not the sentence pipeline or the semantic boundary decisions.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
# Word list only -- importing this does not load a spaCy pipeline.
from spacy.lang.en.stop_words import STOP_WORDS as _STOP_WORDS

from segmentation import (
    DEFAULT_SEMCUT_MAX_WORDS,
    DEFAULT_SEMCUT_WINDOW,
    normalize_text,
    validate_semcut_policy,
)
from segmentation._unit import Unit

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
_EDGE_CHARS = " \t\n\r,;:!?()[]{}\"'"

def _require_encoder(encode):
    if encode is None:
        raise ValueError(
            "semspan requires an explicit semantic encoder; construct a "
            "segmentation.Segmenter with the defender watermark encoder or "
            "the adaptive attack surrogate encoder"
        )
    return encode


def _is_content(word: str) -> bool:
    """Return whether a token counts toward span size."""
    return bool(_WORD_RE.match(word))


def _n_content(toks) -> int:
    return sum(1 for t in toks if _is_content(t[0]))


def _is_stop(word: str) -> bool:
    """Return whether a token is a closed-class (stop) word."""
    m = _WORD_RE.match(word)
    return bool(m) and m.group(0).lower() in _STOP_WORDS


def _legal_gaps(toks, win: int) -> List[int]:
    """Gaps leaving >= win content words per side; win doubles as the floor."""
    out = []
    left = 0
    total = _n_content(toks)
    for g in range(1, len(toks)):
        if _is_content(toks[g - 1][0]):
            left += 1
        if left >= win and (total - left) >= win:
            out.append(g)
    # Function words attach rightward -- a determiner, preposition, auxiliary or
    # conjunction opens a phrase and never closes one, so a unit ending on one
    # has been severed from its complement. Drop those gaps unless they are all
    # that is available.
    return [g for g in out if not _is_stop(toks[g - 1][0])] or out


def _split_many(spans: List[Tuple[int, list]], encode, maxw: int,
                win: int) -> List[Tuple[int, list]]:
    """Split spans levelwise with one encode call per level."""
    done: List[Tuple[int, list]] = []
    active = [(k, t) for k, t in spans if _n_content(t) > maxw]
    done.extend((k, t) for k, t in spans if _n_content(t) <= maxw)

    while active:
        windows: List[str] = []
        index: List[Tuple[int, int, int]] = []
        for ai, (_, toks) in enumerate(active):
            for g in _legal_gaps(toks, win):
                index.append((ai, g, len(windows)))
                windows.append(" ".join(t[0] for t in toks[:g][-win:]))
                windows.append(" ".join(t[0] for t in toks[g:][:win]))
        if not windows:
            done.extend(active)
            break
        # Encode repeated window strings once.
        uniq: dict = {}
        order: List[str] = []
        for w in windows:
            if w not in uniq:
                uniq[w] = len(order)
                order.append(w)
        emb = encode(order)
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
        rows = np.fromiter((uniq[w] for w in windows), dtype=np.int64, count=len(windows))
        left, right = emb[rows[0::2]], emb[rows[1::2]]
        sims = (left * right).sum(1)

        best: dict = {}
        for slot, (ai, g, _) in enumerate(index):
            s = float(sims[slot])
            if ai not in best or s < best[ai][0]:
                best[ai] = (s, g)
        nxt = []
        for ai, (key, toks) in enumerate(active):
            if ai not in best:
                done.append((key, toks))
                continue
            g = best[ai][1]
            for part in (toks[:g], toks[g:]):
                (nxt if _n_content(part) > maxw else done).append((key, part))
        active = nxt
    return done


def _tokens(text: str, base: int) -> list:
    """Return whitespace tokens with character offsets."""
    return [(m.group(0), base + m.start(), base + m.end())
            for m in re.finditer(r"\S+", text)]


def _units_from_pieces(sent_display: str, sent_base: int, pieces: List[list]) -> List[Unit]:
    """Convert token runs into lossless units."""
    pieces = [p for p in pieces if p]
    if not pieces:
        return []
    pieces.sort(key=lambda p: p[0][1])
    end = sent_base + len(sent_display)
    # Own inter-unit whitespace on the right, matching the sentence backends
    # and the causal LM tokenizer.  With Llama-style tokenizers, ``" word"``
    # is one token; assigning its space to the left unit would make a resumed
    # generation prompt end in a standalone space token instead.
    bounds = [sent_base] + [p[-1][2] for p in pieces[:-1]] + [end]
    units = []
    for i, p in enumerate(pieces):
        display = sent_display[bounds[i] - sent_base: bounds[i + 1] - sent_base]
        raw = sent_display[p[0][1] - sent_base: p[-1][2] - sent_base]
        normalized = normalize_text(raw).strip(_EDGE_CHARS)
        if not normalized:
            normalized = normalize_text(display).strip(_EDGE_CHARS)
        units.append(Unit("semspan", normalized, display))
    return units


def segment_semspans_batch(texts: List[str], sentence_backend: str = "nltk",
                           encode=None, *,
                           max_words: int = DEFAULT_SEMCUT_MAX_WORDS,
                           window: int = DEFAULT_SEMCUT_WINDOW) -> List[List[Unit]]:
    """Split texts with one encode call per recursion level."""
    from segmentation import segment
    if not texts:
        return []
    if not any(text and text.strip() for text in texts):
        return [[] for _ in texts]
    validate_semcut_policy(max_words, window)
    encode = _require_encoder(encode)

    sent_units: List[List] = []
    spans: List[Tuple[int, list]] = []
    meta: List[Tuple[int, str, int]] = []
    for ti, text in enumerate(texts):
        rows = []
        if text and text.strip():
            for u in segment(text, type="sentence", backend=sentence_backend):
                key = len(meta)
                meta.append((ti, u.display, 0))
                rows.append(key)
                if u.display.strip():
                    spans.append((key, _tokens(u.display, 0)))
                else:
                    spans.append((key, []))
        sent_units.append(rows)

    pieces = _split_many(spans, encode, max_words, window)
    by_key: dict = {}
    for key, toks in pieces:
        by_key.setdefault(key, []).append(toks)

    out: List[List[Unit]] = []
    for ti, keys in enumerate(sent_units):
        units: List[Unit] = []
        for key in keys:
            _, display, base = meta[key]
            got = by_key.get(key) or []
            if not display.strip():
                if units:
                    last = units[-1]
                    units[-1] = Unit(last.type, last.normalized,
                                     last.display + display)
                continue
            units.extend(_units_from_pieces(display, base, got) or
                         [Unit("semspan", normalize_text(display).strip(_EDGE_CHARS),
                               display)])
        out.append(units)
    return out


def segment_semspans(text: str, sentence_backend: str = "nltk",
                     encode=None, *,
                     max_words: int = DEFAULT_SEMCUT_MAX_WORDS,
                     window: int = DEFAULT_SEMCUT_WINDOW) -> List[Unit]:
    """Split text into bounded semantic spans."""
    return segment_semspans_batch(
        [text], sentence_backend, encode=encode,
        max_words=max_words, window=window,
    )[0]


def first_semspan_batch(texts: List[str], sentence_backend: str = "nltk",
                        encode=None, *,
                        max_words: int = DEFAULT_SEMCUT_MAX_WORDS,
                        window: int = DEFAULT_SEMCUT_WINDOW) -> List[Unit]:
    """Return first semantic spans without materializing right branches."""
    from segmentation import segment
    if not texts:
        return []
    if not any(text and text.strip() for text in texts):
        return [Unit() for _ in texts]
    validate_semcut_policy(max_words, window)
    encode = _require_encoder(encode)

    heads: List[Tuple[int, str, list]] = []
    for ti, text in enumerate(texts):
        if not text or not text.strip():
            continue
        sents = segment(text, type="sentence", backend=sentence_backend)
        first = next((s for s in sents if s.display.strip()), None)
        if first is None:
            continue
        heads.append((ti, first.display, _tokens(first.display, 0)))

    # Descend only through left branches.
    active = [(i, toks) for i, (_, _, toks) in enumerate(heads)
              if _n_content(toks) > max_words]
    settled = {i: toks for i, (_, _, toks) in enumerate(heads)
               if _n_content(toks) <= max_words}
    while active:
        windows: List[str] = []
        index: List[Tuple[int, int]] = []
        for slot, (i, toks) in enumerate(active):
            for g in _legal_gaps(toks, window):
                index.append((slot, g))
                windows.append(" ".join(t[0] for t in toks[:g][-window:]))
                windows.append(" ".join(t[0] for t in toks[g:][:window]))
        if not windows:
            settled.update({i: toks for i, toks in active})
            break
        uniq: dict = {}
        order: List[str] = []
        for w in windows:
            if w not in uniq:
                uniq[w] = len(order)
                order.append(w)
        emb = encode(order)
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
        rows = np.fromiter((uniq[w] for w in windows), dtype=np.int64, count=len(windows))
        sims = (emb[rows[0::2]] * emb[rows[1::2]]).sum(1)
        best: dict = {}
        for k, (slot, g) in enumerate(index):
            s = float(sims[k])
            if slot not in best or s < best[slot][0]:
                best[slot] = (s, g)
        nxt = []
        for slot, (i, toks) in enumerate(active):
            if slot not in best:
                settled[i] = toks
                continue
            left = toks[:best[slot][1]]
            (nxt.append((i, left)) if _n_content(left) > max_words
             else settled.__setitem__(i, left))
        active = nxt

    out = [Unit()] * len(texts)
    for i, (ti, display, _) in enumerate(heads):
        toks = settled.get(i)
        if not toks:
            continue
        units = _units_from_pieces(display, 0, [toks])
        if units:
            end = toks[-1][2]
            first = units[0]
            out[ti] = Unit(first.type, first.normalized, display[:end])
    return out
