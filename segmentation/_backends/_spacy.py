"""Lossless spaCy sentence backend."""

from __future__ import annotations

from typing import List

from segmentation import normalize_text, insert_missing_punctuation
from segmentation._unit import Unit


_nlp = None


def _get_nlp():
    """Load and cache the spaCy English model."""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_md", exclude=["ner"])
    return _nlp


def _build_units(text: str, nlp) -> List[Unit]:
    """Build sentence units from punctuation-fixed text."""
    punct_fixed = insert_missing_punctuation(text)
    doc = nlp(punct_fixed)
    sents = list(doc.sents)
    if not sents:
        return []

    units: List[Unit] = []
    prev_end = 0
    pending_display = ""
    for sent in sents:
        end_char = sent.end_char
        raw_display = punct_fixed[prev_end:end_char]
        prev_end = end_char

        if not sent.text.strip():
            # Attach whitespace-only spans to the next unit.
            pending_display += raw_display
            continue

        display = pending_display + raw_display
        pending_display = ""
        normalized = normalize_text(sent.text)
        units.append(Unit("sentence", normalized, display))

    # Preserve trailing whitespace.
    tail = punct_fixed[prev_end:] + pending_display
    if tail and units:
        last = units[-1]
        units[-1] = Unit(last.type, last.normalized, last.display + tail)

    return units


def segment_sentences_spacy(text: str) -> List[Unit]:
    """Split text into sentence units with spaCy."""
    return _build_units(text, _get_nlp())
